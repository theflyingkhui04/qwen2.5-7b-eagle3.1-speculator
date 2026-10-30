"""Opt-in stage observations for the isolated native profiler command only.

No model, Torch, Modal, or SpecForge imports occur until the explicit child
entrypoint. Hooks delegate unchanged calls and are removed on exit. The native
profiler, loader iterator, training loop, and checkpoint implementation stay intact.
"""
import importlib.abc
import importlib.machinery
import json
import os
import subprocess
import sys
import time
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from unittest.mock import patch

NAMESPACE = Path('/vol/runs/m3-04/non-canonical/profiler')


def canonical_inventory(checkpoint):
    """Content identity of exact step3000 and its three canonical run pointers.

    Stored SHA-256 values authenticate the baseline, not current mutable Volume
    bytes. Re-read only this checkpoint's contents; never walk its parent or
    follow latest/best into another checkpoint. Keep the existing inventory
    record format so both the authenticated baseline and pre/post checks agree.
    """
    from training.train import sha256_file

    checkpoint = Path(checkpoint)
    run_id = 'm3-04-qwen2.5-7b-eagle3.1'
    if (checkpoint.name != run_id + '-step3000' or checkpoint.is_symlink()
            or not checkpoint.is_dir()):
        raise ValueError('Profiler inventory requires the exact regular step3000 directory')
    root = checkpoint.parent
    result = {checkpoint.name: {'directory': True}}

    def record(path):
        key = path.relative_to(root).as_posix()
        if path.is_symlink():
            result[key] = {'link': str(path.readlink())}
        elif path.is_file():
            result[key] = {'sha256': sha256_file(path)}
        elif path.is_dir():
            result[key] = {'directory': True}
        elif not path.exists():
            result[key] = {'missing': True}
        else:
            raise ValueError('Unsupported protected artifact type: ' + str(path))

    for path in sorted(checkpoint.rglob('*')):
        record(path)
    for suffix in ('-latest', '-best', '.best_meta.json'):
        record(root / (run_id + suffix))
    return result


class StageLog:
    def __init__(self, run_dir, *, role='parent'):
        directory = Path(run_dir).resolve()
        root = NAMESPACE.resolve()
        if directory == root or not directory.is_relative_to(root):
            raise ValueError('Stage output requires a unique non-canonical profiler directory')
        self.path = directory / 'profiler_stages.jsonl'
        self.role = role
        self.started = time.monotonic()
        self.stack = []
        self.failed_stage = None

    def emit(self, stage, status, **fields):
        if any(not isinstance(v, (str, int, float, bool, type(None))) for v in fields.values()):
            raise TypeError('Stage fields must be scalar metadata, never tensors or state')
        event = dict(kind='profiler_stage', stage=stage, status=status, role=self.role,
                     pid=os.getpid(), timestamp=datetime.now(timezone.utc).isoformat(),
                     elapsed_s=round(time.monotonic() - self.started, 6), **fields)
        line = json.dumps(event, sort_keys=True)
        print(line, flush=True)
        # Child stdout already belongs to training.log. This transition-only file
        # is the fallback for reading child stages without interactive exec.
        if self.path.parent.is_dir():
            with self.path.open('a', encoding='utf-8') as stream:
                stream.write(line + '\n')
                stream.flush()

    @contextmanager
    def span(self, name, **fields):
        self.emit(name, 'begin', **fields)
        self.stack.append(name)
        try:
            yield
        except BaseException as exc:
            if isinstance(exc, SystemExit) and exc.code in (None, 0):
                self.emit(name, 'end', **fields)
                raise
            if self.failed_stage is None:
                self.failed_stage = name
            raise
        else:
            self.emit(name, 'end', **fields)
        finally:
            self.stack.pop()

    def error(self, exc):
        self.emit(self.failed_stage or (self.stack[-1] if self.stack else 'entry'),
                  'error', exception_type=type(exc).__name__, message=str(exc)[:1024])


def run_child(command, log_path, stages):
    """Observe subprocess.run's own spawn/wait without replacing its semantics."""
    original_popen = subprocess.Popen
    child_pid = None
    waiting = ExitStack()

    def spawn(*args, **kwargs):
        nonlocal child_pid
        with stages.span('subprocess_spawn'):
            child = original_popen(*args, **kwargs)
        child_pid = child.pid
        stages.emit('subprocess_child', 'started', child_pid=child.pid,
                    log_path=str(log_path))
        waiting.enter_context(stages.span('subprocess_wait', child_pid=child.pid))
        return child

    with Path(log_path).open('w') as log:
        with waiting, patch.object(subprocess, 'Popen', spawn):
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        stages.emit('subprocess_exit', 'end', child_pid=child_pid,
                    exit_code=result.returncode)
    return result.returncode


class NativeStages:
    """Observe a small allowlist after natural imports, without eager imports."""
    targets = {
        'specforge.cli', 'specforge.distributed', 'specforge.launch',
        'specforge.training.assembly', 'specforge.training.trainer',
        'specforge.training.checkpoint', 'specforge.training.backend',
        'specforge.training.controller', 'specforge.training.profiling',
    }

    def __init__(self, stages):
        self.stages = stages
        self.patches = ExitStack()
        self.seen = set()
        self.first_batch = False
        self.first_step = False
        self.waiting_for_batch = False

    def wrap(self, owner, name, stage=None, observer=None):
        key = (id(owner), name)
        if key in self.seen:
            return
        self.seen.add(key)
        descriptor = owner.__dict__[name]
        kind = type(descriptor)
        original = descriptor.__func__ if kind in (classmethod, staticmethod) else descriptor

        @wraps(original)
        def observed(*args, **kwargs):
            if observer is not None:
                return observer(original, args, kwargs)
            with self.stages.span(stage):
                return original(*args, **kwargs)

        replacement = kind(observed) if kind in (classmethod, staticmethod) else observed
        self.patches.enter_context(patch.object(owner, name, replacement))

    def instrument(self, module):
        name = module.__name__
        if name == 'specforge.cli':
            self.wrap(module, '_train', 'native_train')
        elif name == 'specforge.distributed':
            self.wrap(module, 'init_distributed', 'distributed_init')
        elif name == 'specforge.launch':
            self.wrap(module, 'build_offline_runtime', 'offline_runtime_build')
        elif name == 'specforge.training.assembly':
            for function, stage in [('build_training_run', 'run_build'),
                                    ('build_model_bundle', 'model_build'),
                                    ('_ensure_offline_vocab_mapping', 'vocab_mapping_auth')]:
                self.wrap(module, function, stage)
            self.wrap(module._ConfiguredOptimizerFactory, '__call__', 'optimizer_build')
        elif name == 'specforge.training.trainer':
            self.wrap(module.Trainer, '__init__', observer=self.trainer_init)
            self.wrap(module.Trainer, 'fit', 'fit')
            self.wrap(module.FeatureDataLoader, '__init__', 'dataloader_build')
            self.wrap(module.FeatureDataLoader, '__iter__', observer=self.loader_iter)
        elif name == 'specforge.training.checkpoint':
            self.wrap(module.CheckpointManager, 'read_resume_state', 'resume_state_read')
        elif name == 'specforge.training.backend':
            self.wrap(module.FSDPTrainingBackend, 'prepare_model', 'backend_prepare')
            self.wrap(module.FSDPTrainingBackend, 'load_state_dict', 'optimizer_rng_restore')
        elif name == 'specforge.training.controller':
            self.wrap(module.TrainerCore, 'train_step', observer=self.train_step)
        elif name == 'specforge.training.profiling':
            self.wrap(module.StepProfiler, 'before_micro_step', observer=self.before_micro_step)
            self.wrap(module.StepProfiler, '_stop_and_export', 'profiler_stop_and_export')

    def trainer_init(self, original, args, kwargs):
        # Trainer's model is keyword-only in the pin. Observe its existing draft
        # load without adding another checkpoint read or changing load arguments.
        draft = kwargs['model'].draft_model
        with self.stages.span('trainer_build'):
            with patch.object(draft, 'load_state_dict', self._draft_load(draft.load_state_dict)):
                return original(*args, **kwargs)

    def _draft_load(self, original):
        @wraps(original)
        def observed(*args, **kwargs):
            with self.stages.span('draft_weights_restore'):
                return original(*args, **kwargs)
        return observed

    def loader_iter(self, original, args, kwargs):
        if not self.waiting_for_batch:
            self.waiting_for_batch = True
            self.stages.emit('first_microbatch_wait', 'begin')
        # Return the original generator itself: no next(), yield, buffering,
        # prefetch, close, or loader behavior is substituted.
        return original(*args, **kwargs)

    def before_micro_step(self, original, args, kwargs):
        profiler = args[0]
        completed = args[1] if len(args) > 1 else kwargs['completed_steps']
        if not self.first_batch:
            self.first_batch = True
            self.stages.emit('first_microbatch_wait', 'end', completed_steps=completed)
        options = profiler.options
        starting = (options.enabled and not profiler._done and profiler._profiler is None
                    and options.start_step <= completed < options.start_step + options.num_steps)
        if starting:
            with self.stages.span('profiler_start', completed_steps=completed,
                                  update=completed + 1):
                return original(*args, **kwargs)
        return original(*args, **kwargs)

    def train_step(self, original, args, kwargs):
        ctx = kwargs.get('ctx', args[2] if len(args) > 2 else None)
        first = not self.first_step
        self.first_step = True
        if first:
            with self.stages.span('first_train_step'):
                result = original(*args, **kwargs)
        else:
            result = original(*args, **kwargs)
        if result.optimizer_stepped and ctx is not None and ctx.global_step + 1 in (3001, 3003):
            self.stages.emit('optimizer_update', 'end', update=ctx.global_step + 1)
        return result

    @contextmanager
    def installed(self):
        observer = self

        class Loader(importlib.abc.Loader):
            def __init__(self, delegate):
                self.delegate = delegate

            def create_module(self, spec):
                return self.delegate.create_module(spec)

            def exec_module(self, module):
                self.delegate.exec_module(module)
                observer.instrument(module)

            def __getattr__(self, name):
                return getattr(self.delegate, name)

        class Finder(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname not in observer.targets:
                    return None
                spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
                if spec is not None and spec.loader is not None:
                    spec.loader = Loader(spec.loader)
                return spec

        finder = Finder()
        try:
            for name in self.targets:
                if name in sys.modules:
                    self.instrument(sys.modules[name])
            sys.meta_path.insert(0, finder)
            yield
        finally:
            if finder in sys.meta_path:
                sys.meta_path.remove(finder)
            self.patches.close()


def main(config_path):
    # This is the opt-in child bootstrap, not a replacement trainer/profiler.
    stages = StageLog(Path(config_path).parent, role='child')
    try:
        with stages.span('child_entry'):
            with stages.span('child_config_guard'):
                import yaml
                raw = yaml.safe_load(Path(config_path).read_text())
                if (Path(raw['output_dir']).resolve() != stages.path.parent
                        or raw['tracking']['report_to'] != 'none'
                        or (raw['training']['total_steps'], raw['training']['max_steps'],
                            raw['training']['batch_size'], raw['training']['accumulation_steps'])
                        != (15000, 3003, 2, 4)
                        or raw['profiling'] != dict(enabled=True, start_step=3001,
                                                    num_steps=2, record_shapes=False)):
                    raise ValueError('Instrumentation requires the isolated bounded profiler config')
            with NativeStages(stages).installed():
                with stages.span('native_cli_import'):
                    from specforge.cli import cli
                with patch.object(sys, 'argv', ['specforge', 'train', '-c', str(config_path)]):
                    # Same Click entrypoint and arguments as the console script.
                    cli()
        stages.emit('child_complete', 'end')
    except BaseException as exc:
        if isinstance(exc, SystemExit) and exc.code in (None, 0):
            stages.emit('child_complete', 'end')
            raise
        stages.error(exc)
        raise


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('Usage: python -m training.profiler_stages <isolated execution.yaml>')
    main(sys.argv[1])
