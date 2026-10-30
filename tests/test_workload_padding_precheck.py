"""CPU tests for identical-stream regrouping and padding accounting."""
import copy
from pathlib import Path

import pytest
import torch

from training.short_path_scan import compare_groupings, padded_q_len, raw_token_length


@pytest.fixture
def inputs():
    namespace = {'torch': torch}
    exec((Path(__file__).parent / 'fixtures/specforge_short_path_metadata.py.txt').read_text(), namespace)
    lengths = [64, 128, 129, 129, 200, 200, 64, 64]
    rows = [{'epoch': 0, 'feature_indices': [i, i + 1], 'sample_ids': [str(i), str(i + 1)],
             'effective_lengths': lengths[i:i + 2], 'q_len': max(lengths[i:i + 2])} for i in range(0, 8, 2)]
    return rows, [(0, i) for i in range(8)], dict(enumerate(lengths)), namespace['paddingtensor2D']


def test_exact_regrouping_and_padding(inputs):
    before = copy.deepcopy(inputs[:3])
    report = compare_groupings(*inputs, short_threshold=128)
    assert inputs[:3] == before
    a, b = report['variants']['A'], report['variants']['B']
    assert a['total_token_slots'] == 1042
    assert b['total_token_slots'] == 1316
    assert a['non_pad_tokens'] == b['non_pad_tokens'] == 978
    assert a['padding_tokens'] == 64 and b['padding_tokens'] == 338
    assert report['absolute_delta'] == 274
    assert report['delta_pct'] == pytest.approx(100 * 274 / 1042)
    assert a['short_microbatches'] == 2 and b['short_microbatches'] == 0
    assert report['changed_A_pair_classifications'] == 2
    assert report['windows_with_changed_classification'] == 1
    assert [x['q_len'] for x in report['windows'][0]['A']] == [128, 129, 200, 64]
    assert [x['q_len'] for x in report['windows'][0]['B']] == [129, 200]
    assert not torch.cuda.is_initialized()


def test_capture_padding_is_not_non_pad_or_supervision(inputs):
    inputs[2][0] = 30
    report = compare_groupings(*inputs, short_threshold=128)
    assert report['variants']['A']['non_pad_tokens'] == 944
    assert report['variants']['A']['padding_tokens'] == 98


@pytest.mark.parametrize('fault', ['reorder', 'duplicate', 'drop', 'width', 'short_window'])
def test_accounting_fails_closed(inputs, fault):
    rows, expected, raw, pad = inputs
    if fault == 'reorder':
        expected.reverse()
    elif fault == 'duplicate':
        rows[1]['feature_indices'] = rows[0]['feature_indices']
        expected[:] = [(r['epoch'], i) for r in rows for i in r['feature_indices']]
    elif fault == 'drop':
        expected.append((0, 8))
    elif fault == 'width':
        rows[0]['q_len'] = 1
    else:
        rows.pop()
    with pytest.raises(ValueError):
        compare_groupings(rows, expected, raw, pad, short_threshold=128)


def test_same_ids_across_epochs_are_valid_distinct_occurrences(inputs):
    rows, expected, raw, pad = inputs
    second = copy.deepcopy(rows)
    for row in second:
        row['epoch'] = 1
    report = compare_groupings(rows + second, expected + [(1, i) for i in range(8)], raw, pad, short_threshold=128)
    assert report['sample_accounting']['scanned_samples'] == 16
    assert report['sample_accounting']['duplicate_drop_check'] == 'PASS'


def test_native_quartet_padding_boundary(inputs):
    pad = inputs[-1]
    assert padded_q_len([1, 64, 127, 128], pad) == 128
    assert padded_q_len([1, 64, 128, 129], pad) == 129


@pytest.mark.parametrize('ids', [[1, 2, 3], [[1, 2, 3]], torch.tensor([[1, 2, 3]])])
def test_raw_cache_tokens_flatten_native_singleton_batch_dimension(ids):
    assert raw_token_length({'input_ids': ids}, 2048) == 3
    assert raw_token_length({'input_ids': ids}, 2) == 2
