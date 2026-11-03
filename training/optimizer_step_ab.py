"""Opt-in, ephemeral benchmark around the pinned native TrainerController.

Never used by train/smoke. Each variant runs in a fresh subprocess; no imitation
training loop, checkpoint publication, tracker, or alternate sample selection.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time
from types import FunctionType, MethodType
from unittest.mock import patch

from training.train import REPO_ROOT, load_training_config, sha256_file, validate_resume_checkpoint

WARMUP_STEPS = 2
MEASURED_STEPS = 5
BASE_STEP = 3000
NAMESPACE = Path('/vol/runs/m3-04/non-canonical/optimizer_step_ab')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def fingerprint(value):
    """Content identity, including dtype/shape; never pickle object addresses."""
    import torch
    digest = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(str((str(tensor.dtype), tuple(tensor.shape))).encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=lambda key: (type(key).__name__, repr(key))):
                visit(key)
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            digest.update(type(item).__name__.encode())
            for child in item:
                visit(child)
        else:
            digest.update(repr((type(item).__name__, item)).encode())
        digest.update(b'\x00')

    visit(value)
    return digest.hexdigest()


def json_safe(value):
    """Keep failure evidence writable even when a loss/gradient is NaN/Inf."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def canonical_inventory(checkpoint):
    """Hash ALL canonical files and links, including latest/best destinations."""
    root = checkpoint.parent
    result = {}
    for path in sorted(root.rglob('*')):
        key = str(path.relative_to(root))
        if path.is_symlink():
            result[key] = {'link': str(path.readlink())}
        elif path.is_file():
            result[key] = {'sha256': sha256_file(path)}
        else:
            result[key] = {'directory': True}
    require(bool(result), 'Empty canonical inventory')
    return result


def validated_config(check_features):
    """Read the same sealed feature/map contracts as the existing diagnostic."""
    from huggingface_hub import snapshot_download
    from training.train import feature_paths
    from training.native_vocab_mapping import PROFILE_PATH, _context, validate_native_vocab_mapping
    cfg = load_training_config(REPO_ROOT / 'configs/train_config.yaml')
    profile = json.loads(PROFILE_PATH.read_text())
    cfg['model']['target_model_path'] = snapshot_download(
        cfg['model']['target_model_path'], revision=profile['target_revision'], local_files_only=True)
    draft = cfg['model']['draft_model_config']
    cfg, key, contract = feature_paths(cfg, draft)
    manifest = json.loads((Path(cfg['data']['hidden_states_path']) / 'artifact_manifest.json').read_text())
    payload = {'config': cfg, 'artifact_key': key, 'contract': contract,
               'resolved_draft': json.loads(Path(draft).read_text()),
               'target_revision': profile['target_revision'], 'capture_layers': manifest['capture_layers']}
    check_features(payload)
    cfg['model']['vocab_mapping_path'] = str(_context(payload)[3])
    validate_native_vocab_mapping(payload)
    cfg['training']['resume_from'] = str(Path(cfg['output_dir']) / (cfg['run_id'] + '-step3000'))
    validate_resume_checkpoint(cfg)
    return cfg


def short_forward(attention, variant, compile_fn):
    """Rebind ONLY the short attention symbol; mask/long path stay pinned."""
    require(variant in {'raw', 'compiled_default'}, 'Unknown variant')
    original = type(attention).forward
    namespace = dict(original.__globals__)
    raw = namespace['flex_attention']
    selected = raw if variant == 'raw' else compile_fn(raw)
    calls = []

    def invoke(*args, **kwargs):
        query = kwargs.get('query', args[0] if args else None)
        require(query.shape[-2] <= 128, 'Short-path binding reached long attention')
        calls.append(list(query.shape))
        return selected(*args, **kwargs)

    namespace['flex_attention'] = invoke
    fn = FunctionType(original.__code__, namespace, original.__name__, original.__defaults__, original.__closure__)
    fn.__kwdefaults__ = original.__kwdefaults__
    return MethodType(fn, attention), calls


def deny_checkpoint_write(*args, **kwargs):
    raise RuntimeError('Benchmark forbids all checkpoint and best/latest publication')


def stats(steps):
    values = [step['optimizer_step_time_s'] for step in steps]
    require(len(values) == MEASURED_STEPS, 'Incomplete measured window')
    mean = statistics.mean(values)
    return {'mean_optimizer_step_time_s': mean, 'median_optimizer_step_time_s': statistics.median(values),
            'min_optimizer_step_time_s': min(values), 'max_optimizer_step_time_s': max(values),
            'individual_optimizer_step_times_s': values, 'steps_per_hour': 3600 / mean}


def classify(raw, compiled):
    """Conservative consistency rule; no invented percentage adoption gate."""
    if not raw.get('valid') or not compiled.get('valid'):
        return 'BENCHMARK INVALID'
    a, b = raw['steps'], compiled['steps']
    if any(not step['stable'] for step in b):
        return 'COMPILED REGRESSION'
    if statistics.mean(s['optimizer_step_time_s'] for s in b) > statistics.mean(s['optimizer_step_time_s'] for s in a):
        return 'COMPILED REGRESSION'
    if all(y['optimizer_step_time_s'] < x['optimizer_step_time_s'] for x, y in zip(a, b)):
        return 'COMPILED END-TO-END WIN'
    return 'NO MEANINGFUL END-TO-END GAIN'


def authenticate_pair(reference, observed):
    for key in ('state_identity', 'data_plan', 'gpu_uuid'):
        require(reference[key] == observed[key], 'A/B ' + key + ' mismatch before B update')


def validate_measurements(report):
    require(len(report['warmup']) == WARMUP_STEPS, 'Incomplete warm-up')
    require(len(report['steps']) == MEASURED_STEPS, 'Incomplete measured window')
    require(all(step['stable'] for step in report['warmup'] + report['steps']), 'Nonfinite training evidence')
    require(any(step['short_calls'] for step in report['steps']),
            'No short attention in measured TRAIN sequence; no substitution')
    require(all(step['compile_counters_before'] == step['compile_counters_after'] for step in report['steps']),
            'Compilation occurred during measurement; cannot claim steady state')


class WindowRecorder:
    """Observe native loader/core calls; synchronize only optimizer boundaries."""

    def __init__(self, torch, core, short_calls):
        self.torch, self.core, self.short_calls = torch, core, short_calls
        self.rows = []
        self.pending = []
        self.wait = 0.0
        self.started = None

    def begin_fetch(self):
        if self.started is None:
            self.torch.cuda.synchronize()
            self.started = time.perf_counter()
            self.short_start = len(self.short_calls)
            self.compile_before = compile_counters()

    def observe_step(self, original, batch, ctx=None):
        start, end = (self.torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        result = original(batch, ctx=ctx)
        end.record()
        self.pending.append((batch, result, start, end))
        if not result.optimizer_stepped:
            require(len(self.pending) < 4, 'Native accumulation boundary missing')
            return result
        self.torch.cuda.synchronize()
        elapsed = time.perf_counter() - self.started
        require(len(self.pending) == 4 and self.core.accumulation_remainder == 0,
                'One optimizer step must contain exactly four microbatches')
        metrics = [dict(item[1].materialize_metrics()) for item in self.pending]
        require(all(len(item[0].sample_ids) == 2 for item in self.pending), 'Microbatch drift')
        grad = result.grad_norm
        finite = all(math.isfinite(float(v)) for row in metrics for v in row.values())
        stable = finite and grad is not None and math.isfinite(float(grad))
        # Native BF16Optimizer computes an FP32 norm over every present gradient
        # and rejects nonfinite norms BEFORE Adam/scheduler/gradient clearing.
        row = {'optimizer_step_time_s': elapsed, 'optimizer_steps_per_hour': 3600 / elapsed,
               'data_wait_time_s': self.wait, 'train_compute_time_s': elapsed - self.wait,
               'microbatch_cuda_compute_time_s': [s.elapsed_time(e) / 1000 for _, _, s, e in self.pending],
               'microbatch_metrics': metrics, 'total_loss': statistics.mean(m['loss'] for m in metrics),
               'grad_norm': grad, 'all_loss_values_finite': finite,
               'all_relevant_gradients_finite': stable, 'stable': stable,
               'data_identity': [{'sample_ids': list(b.sample_ids), 'tensors_sha256': fingerprint(b.tensors)}
                                 for b, _, _, _ in self.pending],
               'short_calls': self.short_calls[self.short_start:],
               'compile_counters_before': self.compile_before, 'compile_counters_after': compile_counters()}
        self.rows.append(row)
        self.pending, self.wait, self.started = [], 0.0, None
        require(stable, 'Nonfinite loss/gradient detected')
        if len(self.rows) == WARMUP_STEPS:
            self.torch.cuda.reset_peak_memory_stats()
        return result


def compile_counters():
    from torch._dynamo.utils import counters
    # Forward graph counts AND AOTAutograd/Inductor (including backward).
    return {group: dict(counters[group]) for group in ('stats', 'aot_autograd', 'inductor')}


def compiler_timings():
    from torch._dynamo.utils import compilation_time_metrics
    return {name: list(values) for name, values in compilation_time_metrics.items()}


class ObservedLoader:
    def __init__(self, loader, recorder):
        self.loader, self.recorder = loader, recorder

    def __getattr__(self, name):
        return getattr(self.loader, name)

    def __iter__(self):
        iterator = iter(self.loader)
        while True:
            self.recorder.begin_fetch()
            started = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                return
            self.recorder.wait += time.perf_counter() - started
            yield batch


def run_variant(config, variant, report, private_dir, reference=None):
    """Native assembly + native fit; process-local state is discarded on exit."""
    import torch
    from specforge.config import Config
    from specforge.algorithms.eagle3.providers import create_registration
    from specforge.training.assembly import build_training_run
    from specforge.training.backend import FSDPTrainingBackend
    from specforge.training.checkpoint import CheckpointManager
    from training.flex_attention_ab import record_diagnostic_hardware

    runtime = record_diagnostic_hardware(report)
    require(all(runtime['checks'].values()), 'Requires known single A100 80GB / CC8 / CUDA13')
    report['performance_comparable_to_historical_sxm4'] = report['performance_comparable_to_canonical_sxm4']
    cfg = Config.model_validate(config)
    t = cfg.training
    require((t.batch_size, t.accumulation_steps, t.ttt_length, t.learning_rate, cfg.data.max_length,
             t.strategy, t.attention_backend, cfg.model.torch_dtype, cfg.mode) ==
            (2, 4, 5, 1e-4, 2048, 'eagle3', 'flex_attention', 'bfloat16', 'offline'), 'Frozen contract drift')
    require(not cfg.model.draft_checkpoint_path and not cfg.profiling.enabled, 'Unexpected draft/profiler')
    cfg.validate_world_size(1)
    # Match native CLI initialization (including NO_SHARD/DDP and seed). Resume
    # restores the checkpoint RNG later, exactly as in the canonical CLI.
    from accelerate.utils import set_seed
    from specforge.cli import _bootstrap_single_process_env
    from specforge.distributed import init_distributed
    os.environ['FSDP_SHARDING'] = t.fsdp_sharding
    if t.ddp_bucket_cap_mb is None:
        os.environ.pop('SPECFORGE_DDP_BUCKET_CAP_MB', None)
    else:
        os.environ['SPECFORGE_DDP_BUCKET_CAP_MB'] = str(t.ddp_bucket_cap_mb)
    set_seed(t.seed)
    _bootstrap_single_process_env()
    require(int(os.environ['WORLD_SIZE']) == 1, 'Benchmark requires world size one')
    init_distributed(timeout=t.dist_timeout, tp_size=t.tp_size,
                     sp_ulysses_size=t.sp_ulysses_size, sp_ring_size=t.sp_ring_size)
    require(torch.distributed.get_world_size() == 1, 'Actual world size mismatch')
    # Change storage/tracking only. Keep schedule horizon and train config intact
    # until native resume has validated its complete persisted contract.
    cfg.output_dir = str(private_dir)
    cfg.tracking.report_to = 'none'
    native_read = CheckpointManager.read_resume_state
    native_load = FSDPTrainingBackend.load_state_dict
    source_identity = {}

    def read(path, **kwargs):
        require(str(path) == config['training']['resume_from'], 'Unexpected resume path')
        state = native_read(path, map_location='cpu', require_full_state=True)
        require((state['global_step'], state['world_size'], state['strategy']) == (3000, 1, 'eagle3'),
                'Checkpoint identity mismatch')
        require(state['backend']['optimizer'].get('fp32_params') is not None, 'Missing FP32 optimizer masters')
        source_identity.update(full_state_sha256=fingerprint(state),
                               draft_sha256=fingerprint(state['draft_state_dict']),
                               counters={k: state[k] for k in ('global_step', 'epoch', 'epoch_batch', 'epoch_samples')})
        return state

    def restore(backend, state):
        native_load(backend, state)
        # Adam implementation flags are natively restored to this runtime's
        # implementation; authenticate tensors, counters, masters and schedule.
        saved, live = state['optimizer'], backend.optimizer.state_dict()
        for key in ('fp32_params', 'scheduler_state_dict'):
            require(fingerprint(saved[key]) == fingerprint(live[key]), 'Restored optimizer mismatch: ' + key)
        require(fingerprint(saved['optimizer_state_dict']['state']) ==
                fingerprint(live['optimizer_state_dict']['state']), 'Restored Adam state mismatch')
        require(fingerprint(state['rng']) == fingerprint(backend._rng_state()), 'Restored RNG mismatch')
        source_identity.update(optimizer_sha256=fingerprint(live), rng_sha256=fingerprint(backend._rng_state()))

    load_started = time.perf_counter()
    with patch.object(CheckpointManager, 'read_resume_state', side_effect=read), \
         patch.object(FSDPTrainingBackend, 'load_state_dict', restore), \
         patch.object(CheckpointManager, 'save', deny_checkpoint_write), \
         patch.object(CheckpointManager, 'update_best', deny_checkpoint_write):
        trainer = build_training_run(cfg, algorithm=create_registration()).trainer
        report['startup_model_load_s'] = time.perf_counter() - load_started
        controller, core = trainer._controller, trainer.core
        report['state_identity'] = source_identity
        require(controller.global_step == BASE_STEP and core.accumulation_remainder == 0, 'Resume boundary drift')
        # Authenticate loaded draft weights after native wrapping/restoration.
        draft = trainer.backend.optimizer.model
        loaded_weights = draft.state_dict()
        # Full state reader also authenticates native permitted omitted weights.
        check = native_read(config['training']['resume_from'], map_location='cpu', require_full_state=True)
        require(all(k in loaded_weights and torch.equal(v.cpu(), loaded_weights[k].cpu())
                    for k, v in check['draft_state_dict'].items()), 'Loaded draft differs from checkpoint')
        del check, loaded_weights
        require(source_identity['rng_sha256'] == fingerprint(trainer.backend._rng_state()), 'RNG changed after restore')
        require({k: getattr(controller, {'epoch_batch': '_epoch_batch', 'epoch_samples': '_epoch_samples'}.get(k, k))
                 for k in source_identity['counters']} == source_identity['counters'], 'Restored data counters mismatch')
        # Native epoch plan + native seek; do not select shorter examples. Hash
        # the exact bounded files before BOTH runs, warming the same file set.
        trainer._loader.set_epoch(controller.epoch)
        offset = controller._epoch_samples
        refs = trainer._loader._refs[offset:offset + 2 * 4 * (WARMUP_STEPS + MEASURED_STEPS)]
        require(len(refs) == 56, 'Bounded window crosses epoch; stop instead of changing order')
        plan = []
        for ref in refs:
            require(ref.feature_store_uri.startswith('file://'), 'Not immutable offline TRAIN features')
            path = Path(ref.feature_store_uri.removeprefix('file://'))
            plan.append({'sample_id': ref.sample_id, 'path': str(path), 'sha256': sha256_file(path)})
        report['data_plan'] = plan
        if reference is not None:
            authenticate_pair(reference, report)
        attention = draft.midlayer.self_attn
        require(all(getattr(draft.config, k, None) is True for k in
                    ('fc_norm', 'norm_output', 'use_post_norm_feedback')) and draft.config.num_hidden_layers == 1,
                'EAGLE-3.1 architecture mismatch')
        attention.forward, calls = short_forward(attention, variant, torch.compile)
        recorder = WindowRecorder(torch, core, calls)
        original_step = core.train_step
        core.train_step = lambda batch, ctx=None: recorder.observe_step(original_step, batch, ctx)
        # The native schedule has already been built/restored with its ORIGINAL
        # horizon. Only bound the controller's execution, never optimizer config.
        controller.max_steps = BASE_STEP + WARMUP_STEPS + MEASURED_STEPS
        controller.save_checkpoint = deny_checkpoint_write
        controller.save_interval = 0
        controller.eval_interval = 0
        controller.logger = None
        trainer.save_checkpoint = deny_checkpoint_write
        torch.cuda.reset_peak_memory_stats()
        try:
            # Deliberately bypass Trainer.fit's unconditional final publication;
            # use its exact existing controller/core/data path instead.
            require(source_identity['rng_sha256'] == fingerprint(trainer.backend._rng_state()),
                    'RNG changed before first training microbatch')
            report['initial_identity_authenticated'] = True
            final_step = controller.fit(ObservedLoader(trainer._loader, recorder))
        except Exception as exc:
            report['training_exception'] = bool(report.get('initial_identity_authenticated')) and (
                isinstance(exc, (RuntimeError, FloatingPointError)) or str(exc) == 'Nonfinite loss/gradient detected')
            raise
        finally:
            trainer._loader.close()
            controller.close_profiler()
            report.update(warmup=recorder.rows[:WARMUP_STEPS], steps=recorder.rows[WARMUP_STEPS:],
                          compiler_reported_timings_s=compiler_timings(),
                          memory={'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                                  'peak_reserved_bytes': torch.cuda.max_memory_reserved()})
        require(final_step == BASE_STEP + WARMUP_STEPS + MEASURED_STEPS, 'Incomplete bounded run')
        require(len(recorder.rows) == WARMUP_STEPS + MEASURED_STEPS, 'Wrong step count')
        consumed = [sample for row in recorder.rows for batch in row['data_identity'] for sample in batch['sample_ids']]
        require(consumed == [item['sample_id'] for item in plan], 'Consumed order differs from authenticated plan')
        require(all(sha256_file(item['path']) == item['sha256'] for item in plan), 'TRAIN features changed during run')
        report['cold_warmup_optimizer_step_s'] = report['warmup'][0]['optimizer_step_time_s']
        report['training_warmup_optimizer_step_s'] = report['warmup'][1]['optimizer_step_time_s']
        report['compile_cold_s'] = None
        report['cold_compile_note'] = 'Not isolated; first two full optimizer steps include cold compilation and training warm-up'
        report['cold_and_training_warmup_s'] = sum(r['optimizer_step_time_s'] for r in report['warmup'])
        validate_measurements(report)
        report.update(stats(report['steps']))
        report.update(valid=True, no_oom=True, no_runtime_or_compiler_exception=True)


def run_benchmark(config, evidence, report):
    """Two independent processes on the same Modal allocation, never A -> B."""
    checkpoint = Path(config['training']['resume_from'])
    require(evidence.parent.resolve() == NAMESPACE.resolve(), 'Evidence outside unique benchmark namespace')
    before = canonical_inventory(checkpoint)
    report.update(classification='BENCHMARK INVALID', canonical_state_unchanged=False,
                  config_identity=config, checkpoint={'path': str(checkpoint), 'before': before},
                  decision_rule='All five paired compiled times faster, finite/stable; no percentage adoption threshold',
                  timing_method='Optimizer-boundary CUDA synchronization + wall clock; per-microbatch CUDA events; identity hashing outside timing',
                  cache_policy='Fresh subprocess and empty private Inductor/Triton/CUDA cache for each variant; no remote compiler cache')
    config_path = evidence / 'input.json'
    with config_path.open('x', encoding='utf-8') as stream:
        json.dump(config, stream)
    try:
        for variant in ('raw', 'compiled_default'):
            with tempfile.TemporaryDirectory(prefix='optimizer-step-ab-') as tmp:
                environment = dict(os.environ, WANDB_MODE='disabled', TORCHINDUCTOR_CACHE_DIR=tmp + '/inductor',
                                   TRITON_CACHE_DIR=tmp + '/triton', CUDA_CACHE_PATH=tmp + '/cuda',
                                   TORCHINDUCTOR_FX_GRAPH_REMOTE_CACHE='0', TORCHINDUCTOR_AUTOGRAD_REMOTE_CACHE='0')
                result_path = evidence / (variant + '.json')
                with (evidence / (variant + '.log')).open('x', encoding='utf-8') as log:
                    command = [sys.executable, '-m', 'training.optimizer_step_ab',
                                             '--config', str(config_path), '--variant', variant,
                                             '--result', str(result_path), '--private-dir', tmp]
                    if variant == 'compiled_default':
                        command.extend(['--reference', str(evidence / 'raw.json')])
                    result = subprocess.run(command,
                                            env=environment, stdout=log, stderr=subprocess.STDOUT, timeout=1500)
                report[variant] = json.loads(result_path.read_text()) if result_path.exists() else {'valid': False}
                if variant == 'raw' and 'performance_comparable_to_historical_sxm4' in report[variant]:
                    report['performance_comparable_to_historical_sxm4'] = report[variant]['performance_comparable_to_historical_sxm4']
                if (variant == 'compiled_default' and report[variant].get('training_exception')
                        and report[variant].get('initial_identity_authenticated')):
                    # B was authenticated against the successful A before its
                    # first update. An actual training/compiler failure is a
                    # regression, without inventing incomplete timing numbers.
                    report['classification'] = 'COMPILED REGRESSION'
                    return
                require(result.returncode == 0 and report[variant].get('valid'), variant + ' failed; inspect evidence')
        a, b = report['raw'], report['compiled_default']
        require(a['state_identity'] == b['state_identity'], 'A/B initial state mismatch')
        hardware_keys = ('device_name', 'device_count', 'total_memory_bytes', 'compute_capability', 'torch_cuda')
        require(all(a[k] == b[k] for k in hardware_keys), 'Hardware mismatch')
        require(a['gpu_uuid'] == b['gpu_uuid'], 'Physical GPU changed')
        for x, y in zip(a['warmup'] + a['steps'], b['warmup'] + b['steps']):
            require(x['data_identity'] == y['data_identity'] and x['short_calls'] == y['short_calls'], 'A/B data/order/shape mismatch')
        report['data_identity'] = {'identical': True, 'includes_warmup': True, 'microbatches_per_variant': 28}
        report['performance_comparable_to_historical_sxm4'] = a['performance_comparable_to_historical_sxm4']
        report['speedup'] = a['mean_optimizer_step_time_s'] / b['mean_optimizer_step_time_s']
        report['percent_reduction'] = 100 * (1 - b['mean_optimizer_step_time_s'] / a['mean_optimizer_step_time_s'])
        report['classification'] = classify(a, b)
    finally:
        after = canonical_inventory(checkpoint)
        report['checkpoint']['after'] = after
        report['canonical_state_unchanged'] = before == after
        if before != after:
            report['classification'] = 'BENCHMARK INVALID'
            raise RuntimeError('Canonical checkpoint state changed')


def main():
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--variant', choices=['raw', 'compiled_default'], required=True)
    parser.add_argument('--result', required=True)
    parser.add_argument('--private-dir', required=True)
    parser.add_argument('--reference')
    args = parser.parse_args()
    report = {'valid': False, 'variant': args.variant, 'no_oom': True,
              'no_runtime_or_compiler_exception': True}
    try:
        report['gpu_uuid'] = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
        require(bool(report['gpu_uuid']), 'Missing GPU UUID')
        reference = json.loads(Path(args.reference).read_text()) if args.reference else None
        require((reference is not None) == (args.variant == 'compiled_default'), 'Compiled worker requires RAW identity')
        run_variant(json.loads(Path(args.config).read_text()), args.variant, report, Path(args.private_dir), reference)
    except Exception as exc:
        report.update(exception={'type': type(exc).__name__, 'message': str(exc)}, valid=False,
                      no_oom=not isinstance(exc, torch.cuda.OutOfMemoryError),
                      no_runtime_or_compiler_exception=False)
        raise
    finally:
        try:
            import torch.distributed as dist
            if dist.is_initialized():
                from specforge.distributed import destroy_distributed
                destroy_distributed()
        finally:
            with Path(args.result).open('x', encoding='utf-8') as stream:
                json.dump(json_safe(report), stream, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
