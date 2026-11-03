"""CPU contracts for the bounded native-controller benchmark (no Modal calls)."""
import ast
import copy
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from training import optimizer_step_ab as ab


def test_fingerprint_covers_values_dtype_shape_and_nested_state():
    a = {'weights': torch.arange(8, dtype=torch.bfloat16), 'state': {'step': 3000}}
    assert ab.fingerprint(a) == ab.fingerprint(copy.deepcopy(a))
    b = copy.deepcopy(a)
    b['weights'][0] = 9
    assert ab.fingerprint(a) != ab.fingerprint(b)
    b = copy.deepcopy(a)
    b['weights'] = b['weights'].reshape(2, 4)
    assert ab.fingerprint(a) != ab.fingerprint(b)
    b = copy.deepcopy(a)
    b['state']['step'] += 1
    assert ab.fingerprint(a) != ab.fingerprint(b)


def test_only_short_attention_changes_mask_and_long_symbols_are_identical():
    # Function globals mimic the exact pinned q_len branch. Assert objects,
    # rather than merely search for torch.compile text.
    namespace = {
        'flex_attention': lambda **kw: 'raw',
        'create_block_mask': lambda **kw: 'raw-mask',
        'compile_friendly_create_block_mask': lambda **kw: 'long-mask',
        'compile_friendly_flex_attention': lambda **kw: 'long-attention',
    }
    exec('def forward(self, query):\n'
         '    if query.shape[-2] <= 128:\n'
         '        return create_block_mask(), flex_attention(query=query)\n'
         '    return compile_friendly_create_block_mask(), compile_friendly_flex_attention(query=query)\n', namespace)
    attention = type('Attention', (), {'forward': namespace['forward']})()
    compiled = []

    def compile_fn(fn, *args, **kwargs):
        assert not args and not kwargs
        assert fn is namespace['flex_attention']
        compiled.append(fn)
        return lambda **kw: 'compiled-default'

    raw, raw_calls = ab.short_forward(attention, 'raw', compile_fn)
    assert not compiled
    candidate, candidate_calls = ab.short_forward(attention, 'compiled_default', compile_fn)
    assert len(compiled) == 1
    for fn in (raw, candidate):
        for key in ('create_block_mask', 'compile_friendly_create_block_mask', 'compile_friendly_flex_attention'):
            assert fn.__func__.__globals__[key] is namespace[key]
        assert fn(torch.zeros(2, 28, 129, 128)) == ('long-mask', 'long-attention')
    assert raw(torch.zeros(2, 28, 120, 128)) == ('raw-mask', 'raw')
    assert candidate(torch.zeros(2, 28, 120, 128)) == ('raw-mask', 'compiled-default')
    assert raw_calls == candidate_calls == [[2, 28, 120, 128]]
    assert attention.forward.__func__ is namespace['forward']


class FakeCuda:
    def __init__(self):
        self.actions = []

    def synchronize(self):
        self.actions.append('sync')

    def reset_peak_memory_stats(self):
        self.actions.append('reset-peak')

    def Event(self, *, enable_timing):
        assert enable_timing
        actions = self.actions

        class Event:
            def record(self):
                actions.append('event')

            def elapsed_time(self, other):
                assert actions[-1] == 'sync'
                return 2.0
        return Event()


def recorder_fixture(monkeypatch, *, boundary=4, finite=True):
    monkeypatch.setattr(ab, 'compile_counters', lambda: {'graphs': 1})
    cuda = FakeCuda()
    core = SimpleNamespace(accumulation_remainder=0)
    recorder = ab.WindowRecorder(SimpleNamespace(cuda=cuda), core, [])
    count = 0

    def native_step(batch, ctx=None):
        nonlocal count
        count += 1
        cuda.actions.append('native-forward-backward-update' if count % boundary == 0 else 'native-forward-backward')
        return SimpleNamespace(optimizer_stepped=count % boundary == 0,
                               materialize_metrics=lambda: {'loss': 1.0 if finite else float('nan')},
                               grad_norm=0.5 if count % boundary == 0 else None)

    batches = [SimpleNamespace(sample_ids=[str(2 * i), str(2 * i + 1)], tensors={'input_ids': torch.tensor([i])})
               for i in range(28)]
    return cuda, recorder, native_step, batches


def test_four_microbatches_boundary_sync_and_warmup_exclusion(monkeypatch):
    cuda, recorder, native_step, batches = recorder_fixture(monkeypatch)
    for i, batch in enumerate(batches):
        recorder.begin_fetch()
        recorder.observe_step(native_step, batch)
        assert len(recorder.rows) == (i + 1) // 4
    assert cuda.actions.count('sync') == 14  # start/end of 7 optimizer steps ONLY
    assert cuda.actions.count('native-forward-backward-update') == 7
    assert cuda.actions.count('native-forward-backward') == 21
    assert cuda.actions.count('reset-peak') == 1
    for row in recorder.rows:
        assert len(row['microbatch_cuda_compute_time_s']) == 4
        assert len(row['data_identity']) == 4
        assert row['stable'] and row['all_relevant_gradients_finite']
        assert row['optimizer_step_time_s'] >= row['data_wait_time_s']
    warmup, measured = recorder.rows[:ab.WARMUP_STEPS], recorder.rows[ab.WARMUP_STEPS:]
    warmup[0]['optimizer_step_time_s'] = 1e6
    summary = ab.stats(measured)
    assert summary['max_optimizer_step_time_s'] < 1e6
    assert len(summary['individual_optimizer_step_times_s']) == 5


@pytest.mark.parametrize('boundary', [1, 2, 3, 5])
def test_accumulation_drift_is_rejected(monkeypatch, boundary):
    _, recorder, native, batches = recorder_fixture(monkeypatch, boundary=boundary)
    with pytest.raises(ValueError, match='four|boundary'):
        for batch in batches:
            recorder.begin_fetch()
            recorder.observe_step(native, batch)


def test_nonfinite_loss_is_rejected_and_recorded(monkeypatch):
    _, recorder, native, batches = recorder_fixture(monkeypatch, finite=False)
    with pytest.raises(ValueError, match='Nonfinite'):
        for batch in batches[:4]:
            recorder.begin_fetch()
            recorder.observe_step(native, batch)
    assert not recorder.rows[0]['stable']
    assert json.loads(json.dumps(ab.json_safe(recorder.rows), allow_nan=False))[0]['total_loss'] == 'nan'


def result(times, *, stable=True):
    return {'valid': True, 'state_identity': {'checkpoint': 'immutable'}, 'gpu_uuid': 'GPU-one',
            'device_name': 'NVIDIA A100 80GB PCIe', 'device_count': 1, 'total_memory_bytes': 80_000_000_000,
            'compute_capability': [8, 0], 'torch_cuda': '13.0',
            'performance_comparable_to_historical_sxm4': False,
            'warmup': [{'data_identity': [1], 'short_calls': []}] * 2,
            'steps': [{'optimizer_step_time_s': t, 'stable': stable, 'data_identity': [i], 'short_calls': [120]}
                      for i, t in enumerate(times)],
            'mean_optimizer_step_time_s': sum(times) / len(times)}


@pytest.mark.parametrize('times,stable,classification', [
    ([3, 3, 3, 3, 3], True, 'COMPILED END-TO-END WIN'),
    ([3, 3, 3, 3, 4], True, 'NO MEANINGFUL END-TO-END GAIN'),
    ([5, 5, 5, 5, 5], True, 'COMPILED REGRESSION'),
    ([3, 3, 3, 3, 3], False, 'COMPILED REGRESSION'),
])
def test_classification_uses_every_step_without_percentage_gate(times, stable, classification):
    assert ab.classify(result([4] * 5), result(times, stable=stable)) == classification
    assert ab.classify({'valid': False}, result(times)) == 'BENCHMARK INVALID'


def test_inventory_detects_model_pointer_and_new_checkpoint_changes(tmp_path):
    checkpoint = tmp_path / 'run-step3000'
    checkpoint.mkdir()
    (checkpoint / 'training_state.pt').write_bytes(b'weights')
    (tmp_path / 'run.best_meta.json').write_text('best')
    before = ab.canonical_inventory(checkpoint)
    (checkpoint / 'training_state.pt').write_bytes(b'changed')
    assert before != ab.canonical_inventory(checkpoint)
    (checkpoint / 'training_state.pt').write_bytes(b'weights')
    (tmp_path / 'run-step3001').mkdir()
    assert before != ab.canonical_inventory(checkpoint)


def test_orchestrator_fresh_workers_separate_caches_same_checkpoint_and_pcie_scope(tmp_path, monkeypatch):
    namespace = tmp_path / 'non-canonical' / 'optimizer_step_ab'
    evidence = namespace / 'unique'
    evidence.mkdir(parents=True)
    monkeypatch.setattr(ab, 'NAMESPACE', namespace)
    checkpoint = tmp_path / 'canonical' / 'run-step3000'
    checkpoint.mkdir(parents=True)
    (checkpoint / 'state.pt').write_bytes(b'immutable')
    config = {'training': {'resume_from': str(checkpoint)}}
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        variant = command[command.index('--variant') + 1]
        assert json.loads(Path(command[command.index('--config') + 1]).read_text()) == config
        path = Path(command[command.index('--result') + 1])
        path.write_text(json.dumps(result([4] * 5 if variant == 'raw' else [3] * 5)))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(ab.subprocess, 'run', run)
    report = {}
    ab.run_benchmark(config, evidence, report)
    assert len(calls) == 2
    assert calls[0][0][calls[0][0].index('--variant') + 1] == 'raw'
    assert '--reference' not in calls[0][0] and '--reference' in calls[1][0]
    for variable in ('TORCHINDUCTOR_CACHE_DIR', 'TRITON_CACHE_DIR', 'CUDA_CACHE_PATH'):
        assert calls[0][1]['env'][variable] != calls[1][1]['env'][variable]
    assert all(c[1]['env']['WANDB_MODE'] == 'disabled' for c in calls)
    assert report['canonical_state_unchanged']
    assert not report['performance_comparable_to_historical_sxm4']
    assert report['speedup'] == 4 / 3
    assert report['percent_reduction'] == 25


def test_authenticated_compiled_training_failure_is_regression_without_fake_timings(tmp_path, monkeypatch):
    namespace = tmp_path / 'ab'
    evidence = namespace / 'unique'
    evidence.mkdir(parents=True)
    monkeypatch.setattr(ab, 'NAMESPACE', namespace)
    checkpoint = tmp_path / 'canonical' / 'run-step3000'
    checkpoint.mkdir(parents=True)
    (checkpoint / 'state.pt').write_bytes(b'immutable')

    def run(command, **kwargs):
        variant = command[command.index('--variant') + 1]
        payload = result([4] * 5) if variant == 'raw' else {
            'valid': False, 'initial_identity_authenticated': True, 'training_exception': True,
            'exception': {'type': 'OutOfMemoryError'}}
        Path(command[command.index('--result') + 1]).write_text(json.dumps(payload))
        return SimpleNamespace(returncode=0 if variant == 'raw' else 1)

    monkeypatch.setattr(ab.subprocess, 'run', run)
    report = {}
    ab.run_benchmark({'training': {'resume_from': str(checkpoint)}}, evidence, report)
    assert report['classification'] == 'COMPILED REGRESSION'
    assert report['canonical_state_unchanged']
    assert 'speedup' not in report


def test_checkpoint_and_tracking_guards_are_native_and_fail_closed():
    source = inspect.getsource(ab.run_variant)
    assert 'build_training_run(cfg, algorithm=create_registration()).trainer' in source
    assert 'controller.fit(ObservedLoader(trainer._loader, recorder))' in source
    assert 'trainer.fit(' not in source
    assert source.index('build_training_run(cfg') < source.index('controller.max_steps =')
    assert "cfg.tracking.report_to = 'none'" in source
    assert "cfg.output_dir = str(private_dir)" in source
    for name in ('save', 'update_best'):
        assert f"patch.object(CheckpointManager, '{name}', deny_checkpoint_write)" in source
    for obj in ('controller', 'trainer'):
        assert f'{obj}.save_checkpoint = deny_checkpoint_write' in source
    assert source.index('authenticate_pair(reference, report)') < source.index('controller.fit(')
    assert 'native_load(backend, state)' in source
    assert "fingerprint(state['rng']) == fingerprint(backend._rng_state())" in source
    assert "('fp32_params', 'scheduler_state_dict')" in source
    with pytest.raises(RuntimeError, match='forbids'):
        ab.deny_checkpoint_write()
    for forbidden in ('max-autotune', 'reduce-overhead', 'FlexKernelOptions', 'BLOCK_M', 'dynamic=True'):
        assert forbidden not in inspect.getsource(ab)


@pytest.mark.parametrize('key', ['state_identity', 'data_plan', 'gpu_uuid'])
def test_identity_failure_stops_before_b_can_update(key):
    reference = {'state_identity': {'optimizer': 'one', 'rng': 'same'}, 'data_plan': [1, 2], 'gpu_uuid': 'GPU-one'}
    ab.authenticate_pair(reference, copy.deepcopy(reference))
    changed = copy.deepcopy(reference)
    changed[key] = 'different'
    with pytest.raises(ValueError, match='before B update'):
        ab.authenticate_pair(reference, changed)


@pytest.mark.parametrize('fault', ['compile', 'no-short', 'incomplete', 'unstable'])
def test_measurement_guards_fail_closed(fault):
    step = {'stable': True, 'short_calls': [120], 'compile_counters_before': {'graphs': 1},
            'compile_counters_after': {'graphs': 1}}
    report = {'warmup': [copy.deepcopy(step) for _ in range(2)],
              'steps': [copy.deepcopy(step) for _ in range(5)]}
    ab.validate_measurements(report)
    if fault == 'compile':
        report['steps'][3]['compile_counters_after']['graphs'] += 1
    elif fault == 'no-short':
        for row in report['steps']:
            row['short_calls'] = []
    elif fault == 'incomplete':
        report['steps'].pop()
    else:
        report['warmup'][0]['stable'] = False
    with pytest.raises(ValueError):
        ab.validate_measurements(report)


def test_explicit_modal_dispatch_without_altering_canonical_function_bodies():
    # Existing authenticated-base tests additionally hash train/smoke/preflight.
    source = (Path(__file__).resolve().parents[1] / 'modal_pipeline.py').read_text()
    tree = ast.parse(source)
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'optimizer_step_ab')
    assert not any(k.arg == 'secrets' for k in function.decorator_list[0].keywords)
    assert "if step == \"optimizer_step_ab\":" in source
    assert 'optimizer_step_ab.remote(execution)' in source
    assert 'evidence.mkdir(parents=True, exist_ok=False)' in ast.get_source_segment(source, function)
