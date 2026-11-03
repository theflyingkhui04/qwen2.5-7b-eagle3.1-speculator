"""Targeted CPU-only tests for canonical padding, resume and scan safety."""
import ast
import copy
import hashlib
import inspect
import io
import itertools
import json
from pathlib import Path
import pickle
import zipfile

import pytest
import torch

from training import short_path_scan as scan

FIXTURES = Path(__file__).parent / 'fixtures'


@pytest.fixture
def native():
    namespace = {'torch': torch}
    exec((FIXTURES / 'specforge_short_path_metadata.py.txt').read_text(encoding='utf-8'), namespace)
    return namespace['_distributed_sampler_indices'], namespace['paddingtensor2D']


@pytest.fixture
def observed():
    return json.loads((FIXTURES / 'step3000_failed_window.json').read_text())


def feature_paths():
    # The native offline reader indexes the lexicographically sorted inventory.
    return sorted(f'/features/rows_{i // 2000 * 2000}-{i // 2000 * 2000 + 2000}/data_{i}.ckpt'
                  for i in range(40000))


def test_capture_pair_padding_precedes_training_pair_padding(native):
    _, pad = native
    paths = [f'/features/rows_0-2000/data_{i}.ckpt' for i in range(6)]
    lengths = [64, 200, 64, 64, 4096, 30]
    rows = [{'input_ids': list(range(n)), 'loss_mask': [1] * n} for n in lengths]
    widths = scan.effective_widths(paths, rows.__getitem__, 6, 2048)
    assert widths == [200, 200, 64, 64, 2048, 2048]
    # Raw rows 0 and 2 are individually <=128, but training sees width 200.
    assert scan.padded_q_len([widths[0], widths[2]], pad) == 200
    assert scan.padded_q_len([widths[2], widths[3]], pad) == 64
    assert scan.padded_q_len([128, 80], pad) == 128
    assert scan.padded_q_len([129, 128], pad) == 129
    assert not torch.cuda.is_initialized()


def test_step3000_resume_and_first_28_native_samples_match_gpu_evidence(native, observed):
    sampler, pad = native
    paths = feature_paths()
    widths = [256] * 40000
    for row in observed['width_rows']:
        for index, length in zip(row['feature_indices'], row['effective_lengths']):
            widths[index] = length
    before = copy.deepcopy(observed)
    first = list(itertools.islice(scan.remaining_batches(observed['state'], observed['config'], paths, widths,
                                                        sampler, pad), 28))
    assert [r['sample_ids'] for r in first] == [r['sample_ids'] for r in observed['width_rows']]
    assert [r['q_len'] for r in first] == [r['q_len'] for r in observed['width_rows']]
    assert first[0]['feature_indices'] == [24245, 1509]
    assert first[0]['epoch_batch'] == 12000
    assert first[0]['relative_optimizer_step_offset'] == 0
    assert first[0]['global_optimizer_step'] == 3001
    assert [r['accumulation_slot'] for r in first] == [0, 1, 2, 3] * 7
    assert scan.summarize(first, observed)['failed_28_microbatch_window']['all_long']
    assert before == observed


@pytest.mark.parametrize('field,value', [('epoch_samples', 23998), ('epoch_batch', 11999),
                                         ('world_size', 2), ('batch_size', 4),
                                         ('offline_sampler_version', 2), ('effective_total_steps', 14999)])
def test_resume_identity_or_partial_accumulation_mismatch_stops(observed, field, value):
    state = copy.deepcopy(observed['state'])
    state[field] = value
    with pytest.raises(ValueError):
        scan.validate_resume(state, observed['config'], 40000)


def test_saved_seed_is_used_for_every_epoch_and_only_first_epoch_is_seeked(observed):
    calls = []
    def sampler(size, **kwargs):
        calls.append(kwargs)
        return list(range(size))
    # Observe iteration semantics without allocating any padding tensors here.
    rows = list(scan.remaining_batches(observed['state'], observed['config'], feature_paths(), [256] * 40000,
                                      sampler, lambda self, tensor, width: tensor))
    assert len(rows) == 48000
    assert len(rows) * 2 == 96000
    assert [x['seed'] for x in calls] == [42, 42, 42]
    assert [x['epoch'] for x in calls] == [0, 1, 2]
    assert rows[7999]['epoch'] == 0 and rows[7999]['epoch_batch'] == 19999
    assert rows[8000]['epoch'] == 1 and rows[8000]['epoch_batch'] == 0
    assert rows[28000]['epoch'] == 2 and rows[28000]['epoch_batch'] == 0
    assert rows[-1]['global_optimizer_step'] == 15000


def test_accumulation_histogram_has_exactly_four_microbatches_per_step(observed):
    rows = copy.deepcopy(observed['width_rows'])
    for offset in range(28, 48):
        short = offset - 28 < ((offset - 28) // 4)
        rows.append({'sample_ids': ['unused-a', 'unused-b'], 'short_path': short, 'q_len': 128 if short else 256})
    # Explicit 0,1,2,3,4 short microbatches in the five added optimizer steps.
    for group in range(5):
        for slot in range(4):
            row = rows[28 + group * 4 + slot]
            row.update(short_path=slot < group, q_len=128 if slot < group else 256)
    result = scan.summarize(rows, observed)
    assert result['optimizer_step_coverage']['short_microbatches_per_step_histogram'] == {
        '0': 8, '1': 1, '2': 1, '3': 1, '4': 1}
    assert result['short_path_microbatches'] == 10
    assert result['optimizer_step_coverage']['with_short_path'] == 4


def test_disagreement_with_failed_window_stops_before_reporting_counts(observed):
    rows = copy.deepcopy(observed['width_rows'])
    rows[0].update(q_len=128, short_path=True)
    with pytest.raises(ValueError, match='disagree with GPU evidence'):
        scan.summarize(rows, observed)
    rows = copy.deepcopy(observed['width_rows'])
    rows[0]['sample_ids'].reverse()
    with pytest.raises(ValueError, match='ordering'):
        scan.summarize(rows, observed)


def test_checkpoint_header_reader_is_inert_and_never_loads_weights():
    metadata = {'global_step': 3000, 'epoch': 0, 'epoch_samples': 24000, 'draft_state_dict': {'discard': 'weight'}}
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_STORED) as archive:
        archive.writestr('training_state.pt/data.pkl', pickle.dumps(metadata))
    assert scan.checkpoint_metadata(stream.getvalue()) == {k: v for k, v in metadata.items() if k != 'draft_state_dict'}
    import os
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_STORED) as archive:
        archive.writestr('training_state.pt/data.pkl', pickle.dumps({'bad': os.system}))
    with pytest.raises(ValueError, match='Unapproved checkpoint global'):
        scan.checkpoint_metadata(stream.getvalue())


def test_pinned_source_mismatch_fails_closed():
    with pytest.raises(ValueError, match='Pinned upstream source mismatch'):
        scan.native_helpers(b'changed', b'changed')


def test_scanner_is_read_only_and_has_no_model_attention_or_network_calls():
    source = inspect.getsource(scan)
    tree = ast.parse(source)
    calls = {n.func.attr if isinstance(n.func, ast.Attribute) else n.func.id
             for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, (ast.Name, ast.Attribute))}
    assert not calls & {'open', 'write_text', 'write_bytes', 'save', 'cuda', 'forward', 'backward',
                        'create_block_mask', 'flex_attention', 'build_training_run', 'from_pretrained', 'commit'}
    # The sole load is the allowlisted metadata Unpickler above, not torch.load.
    assert 'torch.load' not in {ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}
    assert not any(isinstance(n, ast.Import) and any(a.name.startswith('modal') for a in n.names) for n in ast.walk(tree))
    assert not torch.cuda.is_initialized()


def test_canonical_source_files_are_identical_to_failed_gpu_run(observed, historical_modal_source):
    root = Path(__file__).resolve().parents[1]
    for path, expected in observed['canonical_source_sha256'].items():
        data = (historical_modal_source if path == 'modal_pipeline.py'
                else (root / path).read_bytes().replace(b'\r\n', b'\n'))
        assert hashlib.sha256(data).hexdigest() == expected, path
