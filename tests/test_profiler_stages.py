"""CPU-only checks for opt-in observations; no Modal calls or native training."""
import builtins
import importlib
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from training import profiler_stages as diagnostic

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def stages(tmp_path, monkeypatch):
    root = tmp_path / 'non-canonical' / 'profiler'
    run = root / 'unique-run'
    run.mkdir(parents=True)
    monkeypatch.setattr(diagnostic, 'NAMESPACE', root)
    return diagnostic.StageLog(run)


def events(stages):
    return [json.loads(line) for line in stages.path.read_text().splitlines()]


def test_json_stdout_flush_and_transition_only_file(stages, monkeypatch):
    emit = Mock(wraps=builtins.print)
    monkeypatch.setattr(builtins, 'print', emit)
    with stages.span('checkpoint_auth'):
        pass
    rows = events(stages)
    assert [row['status'] for row in rows] == ['begin', 'end']
    assert all(row['kind'] == 'profiler_stage' and row['pid'] > 0 for row in rows)
    assert all('timestamp' in row and row['elapsed_s'] >= 0 for row in rows)
    assert all(call.kwargs == {'flush': True} for call in emit.call_args_list)
    assert [json.loads(call.args[0]) for call in emit.call_args_list] == rows
    with pytest.raises(TypeError):
        stages.emit('bad', 'begin', tensor_or_state={})
    assert len(events(stages)) == 2


def test_stage_path_rejects_canonical_root_and_symlink_escape(stages, tmp_path):
    with pytest.raises(ValueError):
        diagnostic.StageLog(tmp_path / 'checkpoints')
    with pytest.raises(ValueError):
        diagnostic.StageLog(diagnostic.NAMESPACE)
    link = diagnostic.NAMESPACE / 'escape'
    try:
        link.symlink_to(tmp_path, target_is_directory=True)
    except OSError:
        pytest.skip('OS does not permit directory symlinks')
    with pytest.raises(ValueError):
        diagnostic.StageLog(link / 'run')


def test_parent_entry_can_flush_before_output_directory_exists(stages):
    run = diagnostic.NAMESPACE / 'not-created'
    diagnostic.StageLog(run).emit('entry', 'begin')
    assert not run.exists()


def test_import_is_opt_in_and_has_no_heavy_import_or_patch():
    code = """
import sys
before=list(sys.meta_path)
import training.profiler_stages
assert list(sys.meta_path)==before
assert not any(n=='specforge' or n.startswith('specforge.') for n in sys.modules)
assert 'torch' not in sys.modules and 'modal' not in sys.modules
"""
    subprocess.run([sys.executable, '-c', code], cwd=ROOT, check=True)


def test_hooks_follow_natural_import_and_restore(stages, tmp_path, monkeypatch):
    package = tmp_path / 'specforge'
    package.mkdir()
    (package / '__init__.py').write_text('')
    (package / 'cli.py').write_text('def _train(value):\n    return value\n')
    monkeypatch.syspath_prepend(str(tmp_path))
    for name in ('specforge', 'specforge.cli'):
        monkeypatch.delitem(sys.modules, name, raising=False)
    before = list(sys.meta_path)
    with diagnostic.NativeStages(stages).installed():
        assert 'specforge.cli' not in sys.modules
        module = importlib.import_module('specforge.cli')
        original = module._train.__wrapped__
        token = object()
        assert module._train(token) is token
        assert module.__spec__.loader.get_source('specforge.cli')
    assert module._train is original
    assert sys.meta_path == before
    assert [(r['stage'], r['status']) for r in events(stages)] == [
        ('native_train', 'begin'), ('native_train', 'end')]
    monkeypatch.delitem(sys.modules, 'specforge.cli')
    monkeypatch.delitem(sys.modules, 'specforge')


def test_classmethod_state_identity_arguments_and_exception_preserved(stages):
    state = {'global_step': 3000, 'backend': object()}
    error = RuntimeError('read failed')
    calls = []

    class CheckpointManager:
        @classmethod
        def read_resume_state(cls, path, **kwargs):
            calls.append((cls, path, kwargs))
            if path == 'bad':
                raise error
            return state

    descriptor = CheckpointManager.__dict__['read_resume_state']
    module = ModuleType('specforge.training.checkpoint')
    module.CheckpointManager = CheckpointManager
    hooks = diagnostic.NativeStages(stages)
    hooks.instrument(module)
    try:
        assert CheckpointManager.read_resume_state('step3000', map_location='cpu', require_full_state=True) is state
        with pytest.raises(RuntimeError) as caught:
            CheckpointManager.read_resume_state('bad')
        assert caught.value is error
        assert calls[0] == (CheckpointManager, 'step3000', {'map_location': 'cpu', 'require_full_state': True})
        assert stages.failed_stage == 'resume_state_read'
    finally:
        hooks.patches.close()
    assert CheckpointManager.__dict__['read_resume_state'] is descriptor


def test_bounded_window_observation_does_not_substitute_iterator_or_calls(stages):
    hooks = diagnostic.NativeStages(stages)
    iterator = iter(range(12))
    assert hooks.loader_iter(lambda: iterator, (), {}) is iterator
    options = SimpleNamespace(enabled=True, start_step=3001, num_steps=2, record_shapes=False)
    profiler = SimpleNamespace(options=options, _done=False, _profiler=None)
    calls = []

    def before(obj, completed):
        calls.append(('before', completed))
        if completed == 3001 and obj._profiler is None:
            obj._profiler = object()

    def train(core, batch, ctx):
        calls.append(('train', batch, ctx))
        return results[batch]

    results = [SimpleNamespace(optimizer_stepped=(i + 1) % 4 == 0) for i in range(12)]
    for i in iterator:
        completed = 3000 + i // 4
        hooks.before_micro_step(before, (profiler, completed), {})
        ctx = SimpleNamespace(global_step=completed, total_steps=15000)
        assert hooks.train_step(train, (None, i), {'ctx': ctx}) is results[i]
    assert len(calls) == 24
    assert profiler.options is options
    rows = events(stages)
    assert len([r for r in rows if r['stage'] == 'first_train_step']) == 2
    assert len([r for r in rows if r['stage'] == 'first_microbatch_wait']) == 2
    assert [(r['status'], r['update']) for r in rows if r['stage'] == 'profiler_start'] == [('begin', 3002), ('end', 3002)]
    assert [r['update'] for r in rows if r['stage'] == 'optimizer_update'] == [3001, 3003]


def test_draft_restore_wrapper_keeps_arguments_and_is_removed(stages):
    result = object()
    calls = []

    class Draft:
        def load_state_dict(self, state, **kwargs):
            calls.append((state, kwargs))
            return result

    draft = Draft()
    state = object()

    def initialize(trainer, *, model):
        assert model.draft_model.load_state_dict(state, strict=False) is result

    hooks = diagnostic.NativeStages(stages)
    assert hooks.trainer_init(initialize, (None,), {'model': SimpleNamespace(draft_model=draft)}) is None
    assert calls == [(state, {'strict': False})]
    assert 'load_state_dict' not in draft.__dict__


def test_subprocess_pid_wait_exit_and_existing_redirection(stages, tmp_path):
    log = tmp_path / 'training.log'
    command = [sys.executable, '-c', "import sys; print('stdout'); print('stderr',file=sys.stderr); sys.exit(7)"]
    assert diagnostic.run_child(command, log, stages) == 7
    assert 'stdout' in log.read_text() and 'stderr' in log.read_text()
    rows = events(stages)
    assert [(r['stage'], r['status']) for r in rows] == [
        ('subprocess_spawn', 'begin'), ('subprocess_spawn', 'end'),
        ('subprocess_child', 'started'), ('subprocess_wait', 'begin'),
        ('subprocess_wait', 'end'), ('subprocess_exit', 'end')]
    assert rows[2]['child_pid'] == rows[-1]['child_pid'] > 0
    assert rows[-1]['exit_code'] == 7


@pytest.mark.parametrize('failure', [RuntimeError('original failure'), SystemExit(0), SystemExit(9)])
def test_child_main_preserves_native_argv_exit_and_error(stages, monkeypatch, failure):
    module = ModuleType('specforge.cli')
    calls = []

    def train(*args, **kwargs):
        pass

    def cli():
        calls.append(list(sys.argv))
        raise failure

    module._train = train
    module.cli = cli
    monkeypatch.setitem(sys.modules, 'specforge', ModuleType('specforge'))
    monkeypatch.setitem(sys.modules, 'specforge.cli', module)
    config = stages.path.parent / 'execution.yaml'
    import yaml
    config.write_text(yaml.safe_dump({
        'output_dir': str(config.parent), 'tracking': {'report_to': 'none'},
        'training': {'total_steps': 15000, 'max_steps': 3003,
                     'batch_size': 2, 'accumulation_steps': 4},
        'profiling': {'enabled': True, 'start_step': 3001, 'num_steps': 2,
                      'record_shapes': False}}))
    with pytest.raises(type(failure)) as caught:
        diagnostic.main(str(config))
    assert caught.value is failure
    assert calls == [['specforge', 'train', '-c', str(config)]]
    rows = events(stages)
    if isinstance(failure, SystemExit) and failure.code == 0:
        assert rows[-1]['stage'] == 'child_complete'
        assert not any(r['status'] == 'error' for r in rows)
    else:
        assert rows[-1]['status'] == 'error'
        assert rows[-1]['exception_type'] == type(failure).__name__
    assert module._train is train


def test_existing_canonical_entrypoints_are_not_edited(historical_modal_source):
    # Stable identities recorded by the authenticated profiler readiness plan.
    import hashlib
    expected = {
        'modal_pipeline.py': '1a5fdc4b058fb9434fe795a2fede8c8da69e56296107a8268d6dd481f08d0053',
        'training/train.py': '620a3ab528aac9cec2f722a8c334b67bae04eafddba21b3046c088681e29cad2',
        'training/optimizer_step_ab.py': '6ae1c86012d1098ba67225781fd412fa2a997cfdba80944ef1845f4fe7a03d67',
        'training/flex_attention_ab.py': 'ae117f5e7c02e2b7e79f3215ab918af995e19ac2e416886be05ca66ba97c52f1',
        'configs/train_config.yaml': 'd135759b45aa1a4207d6f38ef91ae1bd8d8ca75ae1d4b21b92e83777bb992361',
    }
    for name, digest in expected.items():
        data = historical_modal_source if name == 'modal_pipeline.py' else (ROOT / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == digest


@pytest.mark.parametrize('unsafe', ['output', 'tracking'])
def test_child_rejects_canonical_destination_or_tracking_before_cli(stages, monkeypatch, unsafe):
    import yaml
    raw = {'output_dir': str(stages.path.parent), 'tracking': {'report_to': 'none'},
           'training': {'total_steps': 15000, 'max_steps': 3003, 'batch_size': 2,
                        'accumulation_steps': 4},
           'profiling': {'enabled': True, 'start_step': 3001, 'num_steps': 2,
                         'record_shapes': False}}
    if unsafe == 'output':
        raw['output_dir'] = str(stages.path.parent.parent.parent / 'checkpoints')
    else:
        raw['tracking']['report_to'] = 'wandb'
    config = stages.path.parent / 'execution.yaml'
    config.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match='isolated bounded profiler'):
        diagnostic.main(str(config))
    assert events(stages)[-1]['stage'] == 'child_config_guard'


@pytest.fixture
def canonical_tree(tmp_path, monkeypatch):
    """Real content files plus portable link metadata (Windows needs no privilege)."""
    run = 'm3-04-qwen2.5-7b-eagle3.1'
    checkpoint = tmp_path / (run + '-step3000')
    checkpoint.mkdir()
    for name in ('training_state.pt', 'training_state_rank0.pt'):
        (checkpoint / name).write_bytes(b'original-content')
    for suffix in ('-step2000', '-smoke-one-step1', '-smoke-one-step2'):
        sibling = tmp_path / (run + suffix)
        sibling.mkdir()
        (sibling / 'unrelated.pt').write_bytes(b'unrelated')
    meta = tmp_path / (run + '.best_meta.json')
    meta.write_text('{"step":3000}')
    links = {}
    for suffix in ('-latest', '-best'):
        path = tmp_path / (run + suffix)
        path.touch()
        links[path] = checkpoint.name
    is_symlink, readlink = Path.is_symlink, Path.readlink
    monkeypatch.setattr(Path, 'is_symlink', lambda p: p in links and p.exists() or is_symlink(p))
    monkeypatch.setattr(Path, 'readlink', lambda p: Path(links[p]) if p in links else readlink(p))
    return checkpoint, links, meta


def test_inventory_scope_and_reads_are_bounded(canonical_tree, monkeypatch):
    from training import train
    checkpoint, links, meta = canonical_tree
    traversed, hashed = [], []
    original_rglob, original_hash = Path.rglob, train.sha256_file

    def rglob(path, pattern):
        traversed.append(path)
        assert path == checkpoint, 'Must never traverse the parent/sibling tree'
        return original_rglob(path, pattern)

    def sha(path):
        hashed.append(path)
        assert path.is_relative_to(checkpoint) or path == meta
        return original_hash(path)

    monkeypatch.setattr(Path, 'rglob', rglob)
    monkeypatch.setattr(train, 'sha256_file', sha)
    snapshot = diagnostic.canonical_inventory(checkpoint)
    assert traversed == [checkpoint]
    assert set(hashed) == {checkpoint / 'training_state.pt', checkpoint / 'training_state_rank0.pt', meta}
    assert set(snapshot) == {checkpoint.name, *(checkpoint.name + '/' + n for n in
                            ('training_state.pt', 'training_state_rank0.pt')),
                            meta.name, *(p.name for p in links)}
    assert all(snapshot[p.name] == {'link': checkpoint.name} for p in links)


@pytest.mark.parametrize('artifact', ['training_state.pt', 'training_state_rank0.pt',
                                    'latest', 'best', 'best_meta'])
def test_every_protected_artifact_mutation_changes_snapshot(canonical_tree, artifact):
    import os
    checkpoint, links, meta = canonical_tree
    before = diagnostic.canonical_inventory(checkpoint)
    assert diagnostic.canonical_inventory(checkpoint) == before
    if artifact in ('latest', 'best'):
        pointer = next(p for p in links if p.name.endswith('-' + artifact))
        links[pointer] = checkpoint.name.replace('-step3000', '-step2000')
    else:
        path = meta if artifact == 'best_meta' else checkpoint / artifact
        stat = path.stat()
        data = path.read_bytes()
        path.write_bytes(bytes([data[0] ^ 1]) + data[1:])
        # Old hashes/size/mtime cannot stand in for a fresh content observation.
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        assert path.stat().st_size == stat.st_size
        assert path.stat().st_mtime_ns == stat.st_mtime_ns
    assert diagnostic.canonical_inventory(checkpoint) != before


def test_sibling_mutation_is_excluded_but_protected_entries_and_removal_are_detected(canonical_tree):
    checkpoint, links, meta = canonical_tree
    before = diagnostic.canonical_inventory(checkpoint)
    for sibling in checkpoint.parent.iterdir():
        if sibling.is_dir() and sibling != checkpoint:
            (sibling / 'unrelated.pt').write_bytes(b'changed')
            (sibling / 'new-file').write_bytes(b'new')
    assert diagnostic.canonical_inventory(checkpoint) == before
    empty = checkpoint / 'new-empty-directory'
    empty.mkdir()
    assert diagnostic.canonical_inventory(checkpoint) != before
    empty.rmdir()
    added = checkpoint / 'new-file'
    added.write_bytes(b'new')
    assert diagnostic.canonical_inventory(checkpoint) != before
    added.unlink()
    meta.unlink()
    assert diagnostic.canonical_inventory(checkpoint)[meta.name] == {'missing': True}
    assert diagnostic.canonical_inventory(checkpoint) != before


def test_missing_or_wrong_resume_directory_fails_closed(canonical_tree):
    checkpoint, _, _ = canonical_tree
    with pytest.raises(ValueError, match='exact regular step3000'):
        diagnostic.canonical_inventory(checkpoint.parent / checkpoint.name.replace('3000', '2000'))
    for path in checkpoint.iterdir():
        path.unlink()
    checkpoint.rmdir()
    with pytest.raises(ValueError, match='exact regular step3000'):
        diagnostic.canonical_inventory(checkpoint)
