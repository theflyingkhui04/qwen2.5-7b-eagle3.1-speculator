"""Read-only CPU analysis of the native remaining TRAIN microbatch widths.

No model, attention masks, FlexAttention, optimizer, Modal, or training imports.
The caller supplies authenticated metadata/cache and exact pinned source files.
"""
import ast
from collections import Counter, OrderedDict
import hashlib
import io
import pickle
import struct

from training.native_vocab_mapping import cache_record_for_feature, cache_sample_index

PINNED_BLOBS = {
    'launch': '3a0b19c88147dabaa9d2f10735749d52050b7f28',
    'data_utils': 'fbfcda4bf11981dd3d751c657c17c0a2ee5221a7',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def checkpoint_metadata(header):
    """Read only first ZIP entry data.pkl, discarding every tensor/storage.

    Explicitly allow only inert tensor rebuild placeholders and OrderedDict;
    never call torch.load, materialize weights, or execute checkpoint globals.
    """
    require(header[:4] == b'PK\x03\x04', 'Not a native ZIP checkpoint')
    compression = struct.unpack_from('<H', header, 8)[0]
    name_length, extra_length = struct.unpack_from('<HH', header, 26)
    name = header[30:30 + name_length]
    require(compression == 0 and name.endswith(b'/data.pkl'), 'Unexpected checkpoint header')
    allowed = {('torch._utils', '_rebuild_tensor_v2'), ('torch._tensor', '_rebuild_from_type_v2'),
               ('torch', 'Tensor'), ('torch', 'BoolStorage'), ('torch', 'LongStorage'),
               ('torch', 'BFloat16Storage')}

    class MetadataOnly(pickle.Unpickler):
        def persistent_load(self, value):
            return None

        def find_class(self, module, name):
            if (module, name) == ('collections', 'OrderedDict'):
                return OrderedDict
            require((module, name) in allowed, 'Unapproved checkpoint global')
            return lambda *args, **kwargs: None

    state = MetadataOnly(io.BytesIO(header[30 + name_length + extra_length:])).load()
    require(isinstance(state, dict), 'Malformed checkpoint metadata')
    return {key: value for key, value in state.items() if key != 'draft_state_dict'}


def native_helpers(launch_source, data_utils_source):
    """Execute only two unchanged, SHA-authenticated upstream CPU helpers."""
    import torch
    nodes = []
    for key, content in [('launch', launch_source), ('data_utils', data_utils_source)]:
        digest = hashlib.sha1(b'blob ' + str(len(content)).encode() + b'\0' + content).hexdigest()
        require(digest == PINNED_BLOBS[key], 'Pinned upstream source mismatch: ' + key)
        tree = ast.parse(content)
        if key == 'launch':
            nodes.append(next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                              and n.name == '_distributed_sampler_indices'))
        else:
            cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'DataCollatorWithPadding')
            nodes.append(next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'paddingtensor2D'))
    namespace = {'torch': torch}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<pinned CPU metadata helpers>', 'exec'), namespace)
    return namespace['_distributed_sampler_indices'], namespace['paddingtensor2D']


def validate_resume(state, config, dataset_size):
    expected = {'global_step': 3000, 'world_size': 1, 'batch_size': 2, 'accumulation_steps': 4,
                'strategy': 'eagle3', 'dataset_size': dataset_size, 'source_dataset_size': dataset_size,
                'num_epochs': config['training']['num_epochs'], 'offline_sampler_version': 1,
                'eagle3_ttt_length': 5, 'eagle3_attention_backend': 'flex_attention',
                'run_id': config['run_id'], 'tp_size': 1, 'sp_ulysses_size': 1, 'sp_ring_size': 1}
    for key, value in expected.items():
        require(state.get(key) == value, 'Resume contract mismatch: ' + key)
    require(config['training']['batch_size'] == 2 and config['training']['accumulation_steps'] == 4,
            'Canonical batch configuration mismatch')
    require(config['training']['strategy'] == 'eagle3' and config['training']['attention_backend'] == 'flex_attention',
            'Canonical strategy/backend mismatch')
    epoch, samples = state['epoch'], state['epoch_samples']
    require(isinstance(state['sampler_seed'], int), 'Missing sampler seed')
    require(isinstance(epoch, int) and 0 <= epoch < state['num_epochs'], 'Invalid saved epoch')
    require(isinstance(samples, int) and 0 <= samples <= dataset_size and samples % 8 == 0,
            'Resume must be at a complete optimizer boundary')
    require(dataset_size % 8 == 0, 'Dataset has incomplete accumulation windows')
    require(state.get('epoch_batch', samples // 2) == samples // 2, 'epoch_batch disagrees with epoch_samples')
    require((epoch * dataset_size + samples) // 8 == 3000, 'Step3000 position mismatch')
    require(state['effective_total_steps'] == state['num_epochs'] * dataset_size // 8,
            'Unrecognized training horizon')


def effective_widths(paths, get_row, num_rows, max_length):
    """Reuse accepted capture-pair reconstruction, then native truncation."""
    return [min(cache_record_for_feature(path, get_row, num_rows)['input_ids'].numel(), max_length)
            for path in paths]


def raw_token_length(row, max_length):
    """Count flattened cache tokens, including native cache shape [1, n]."""
    import torch
    size = torch.as_tensor(row['input_ids'], dtype=torch.int64).numel()
    require(size > 0, 'Empty cache token row')
    return min(size, max_length)


def padded_q_len(widths, paddingtensor2D):
    """Native collator input-id padding only; no attention mask is constructed."""
    import torch
    require(len(widths) in (2, 4) and all(w > 0 for w in widths), 'Expected two or four nonempty feature widths')
    tensors = [torch.empty((1, width), device='meta', dtype=torch.int64) for width in widths]
    # Same max_length and paddingtensor2D calls as the pinned collator. Meta
    # tensors carry only shapes, avoiding token/hidden-state/attention allocation.
    maximum = max(t.shape[1] for t in tensors)
    padded = [paddingtensor2D(None, t, maximum) for t in tensors]
    require(all(t.shape == (1, maximum) for t in padded), 'Native padding shape mismatch')
    return maximum


def distribution(values):
    values = sorted(values)
    require(bool(values), 'Empty distribution')
    def percentile(p):
        rank = (len(values) - 1) * p / 100
        low = int(rank)
        return values[low] + (values[min(low + 1, len(values) - 1)] - values[low]) * (rank - low)
    return {'min': values[0], **{f'p{p}': percentile(p) for p in (25, 50, 75, 90, 95, 99)},
            'median': percentile(50), 'max': values[-1], 'method': 'linear/inclusive'}


def compare_groupings(rows, expected_occurrences, raw_lengths, paddingtensor2D, *, short_threshold):
    """Compare native pairs vs quartets without changing the epoch/sample stream.

    raw_lengths maps feature index to cache length after token truncation.
    Capture padding is retained in effective_lengths and counted as padding,
    never confused with loss_mask (which marks supervision, not non-pad tokens).
    The caller authenticates the path threshold from the pinned model source.
    """
    require(len(rows) % 4 == 0 and bool(rows), 'Incomplete optimizer windows')
    actual = [(r['epoch'], i) for r in rows for i in r['feature_indices']]
    require(actual == expected_occurrences, 'Sample ordering/drop mismatch')
    require(len(set(actual)) == len(actual), 'Duplicate epoch/sample occurrence')
    windows = []
    variants = {name: [] for name in ('A', 'B')}
    for start in range(0, len(rows), 4):
        block = rows[start:start + 4]
        require(len({r['epoch'] for r in block}) == 1, 'Window crosses epoch boundary')
        indices = [i for r in block for i in r['feature_indices']]
        ids = [i for r in block for i in r['sample_ids']]
        lengths = [n for r in block for n in r['effective_lengths']]
        require(len(indices) == len(ids) == len(lengths) == 8, 'Expected eight samples/update')
        native_raw = [raw_lengths[i] for i in indices]
        require(all(0 < raw <= width for raw, width in zip(native_raw, lengths)), 'Invalid raw/native lengths')
        groups = {}
        for name, size in [('A', 2), ('B', 4)]:
            groups[name] = []
            for offset in range(0, 8, size):
                q_len = padded_q_len(lengths[offset:offset + size], paddingtensor2D)
                non_pad = sum(native_raw[offset:offset + size])
                batch = {'sample_ids': ids[offset:offset + size], 'q_len': q_len,
                         'non_pad_tokens': non_pad, 'token_slots': size * q_len,
                         'padding_tokens': size * q_len - non_pad, 'short_path': q_len <= short_threshold}
                if name == 'A':
                    require(q_len == block[offset // 2]['q_len'], 'A native width mismatch')
                groups[name].append(batch)
                variants[name].append(batch)
        require([i for b in groups['A'] for i in b['sample_ids']] ==
                [i for b in groups['B'] for i in b['sample_ids']] == ids, 'A/B ordered IDs mismatch')
        slots = {name: sum(b['token_slots'] for b in batches) for name, batches in groups.items()}
        changed = sum(groups['A'][i]['short_path'] != groups['B'][i // 2]['short_path'] for i in range(4))
        windows.append({'optimizer_window_offset': start // 4, 'global_optimizer_step': 3001 + start // 4,
                        'epoch': block[0]['epoch'], 'sample_ids': ids, 'feature_indices': indices,
                        'raw_lengths_after_truncation': native_raw, 'native_lengths': lengths,
                        **groups, 'non_pad_tokens': sum(native_raw), 'native_token_slots': sum(lengths),
                        'A_padding_tokens': slots['A'] - sum(native_raw),
                        'B_padding_tokens': slots['B'] - sum(native_raw),
                        'A_token_slots': slots['A'], 'B_token_slots': slots['B'],
                        'delta': slots['B'] - slots['A'], 'delta_pct': 100 * (slots['B'] / slots['A'] - 1),
                        'changed_A_pair_classifications': changed})
    summary = {}
    for name, batches in variants.items():
        slots = sum(b['token_slots'] for b in batches)
        non_pad = sum(b['non_pad_tokens'] for b in batches)
        short = sum(b['short_path'] for b in batches)
        summary[name] = {'microbatch': 2 if name == 'A' else 4, 'accumulation': 4 if name == 'A' else 2,
                         'microbatches': len(batches), 'total_token_slots': slots, 'non_pad_tokens': non_pad,
                         'padding_tokens': slots - non_pad, 'padding_overhead_pct_of_non_pad': 100 * (slots / non_pad - 1),
                         'capture_padding_tokens': sum(w['native_token_slots'] - w['non_pad_tokens'] for w in windows),
                         'training_padding_tokens': slots - sum(w['native_token_slots'] for w in windows),
                         'padding_fraction_pct_of_slots': 100 * (slots - non_pad) / slots,
                         'short_microbatches': short, 'long_microbatches': len(batches) - short,
                         'short_pct': 100 * short / len(batches), 'long_pct': 100 * (len(batches) - short) / len(batches),
                         'windows_with_short': sum(any(b['short_path'] for b in w[name]) for w in windows),
                         'q_len_distribution': distribution([b['q_len'] for b in batches]),
                         'microbatch_token_slots_distribution': distribution([b['token_slots'] for b in batches]),
                         'window_token_slots_distribution': distribution([w[name + '_token_slots'] for w in windows]),
                         'window_padding_distribution': distribution([sum(b['padding_tokens'] for b in w[name]) for w in windows]),
                         'microbatch_non_pad_distribution': distribution([b['non_pad_tokens'] for b in batches]),
                         'microbatch_padding_distribution': distribution([b['padding_tokens'] for b in batches])}
    delta = summary['B']['total_token_slots'] - summary['A']['total_token_slots']
    return {'sample_accounting': {'same_ordered_samples': 'PASS', 'eight_samples_per_update': 'PASS',
                                 'duplicate_drop_check': 'PASS', 'scanned_samples': len(actual),
                                 'scanned_optimizer_windows': len(windows),
                                 'samples_per_epoch': dict(Counter(epoch for epoch, _ in actual)),
                                 'identity_scope': '(epoch, feature index); repeats across epochs are intentional'},
            'variants': summary, 'absolute_delta': delta,
            'delta_pct': 100 * delta / summary['A']['total_token_slots'],
            'delta_distribution': distribution([w['delta'] for w in windows]),
            'delta_pct_distribution': distribution([w['delta_pct'] for w in windows]),
            'changed_A_pair_classifications': sum(w['changed_A_pair_classifications'] for w in windows),
            'windows_with_changed_classification': sum(w['changed_A_pair_classifications'] > 0 for w in windows),
            'worst_windows': sorted(windows, key=lambda w: (-w['delta'], w['optimizer_window_offset']))[:5],
            'windows': windows}


def remaining_batches(state, config, paths, widths, sampler, paddingtensor2D):
    validate_resume(state, config, len(paths))
    require(len(paths) == len(widths) and paths == sorted(paths) and len(set(paths)) == len(paths),
            'Feature inventory/order mismatch')
    relative = 0
    for epoch in range(state['epoch'], state['num_epochs']):
        order = sampler(len(paths), dp_rank=0, dp_size=1, seed=state['sampler_seed'], epoch=epoch, shuffle=True)
        require(len(order) == len(paths) and sorted(order) == list(range(len(paths))), 'Invalid native sampler permutation')
        offset = state['epoch_samples'] if epoch == state['epoch'] else 0
        for position in range(offset, len(order), 2):
            indices = order[position:position + 2]
            lengths = [widths[i] for i in indices]
            q_len = padded_q_len(lengths, paddingtensor2D)
            yield {'relative_microbatch_offset': relative, 'relative_optimizer_step_offset': relative // 4,
                   'optimizer_step_number_after_resume': relative // 4 + 1, 'global_optimizer_step': 3001 + relative // 4,
                   'accumulation_slot': relative % 4, 'epoch': epoch, 'epoch_batch': position // 2,
                   'sample_ids': [f"{config['run_id']}:{i:08d}" for i in indices], 'feature_indices': indices,
                   'capture_rows': [cache_sample_index(paths[i], len(paths)) for i in indices],
                   'effective_lengths': lengths, 'q_len': q_len, 'short_path': q_len <= 128}
            relative += 1


def summarize(rows, benchmark):
    require(len(rows) >= 28 and len(rows) % 4 == 0, 'Incomplete remaining workload')
    raw = benchmark['raw']
    observed = [batch['sample_ids'] for step in raw['warmup'] + raw['steps'] for batch in step['data_identity']]
    require(len(observed) == 28 and observed == [r['sample_ids'] for r in rows[:28]],
            'First 28 microbatches disagree with failed benchmark ordering')
    require(all(not step['short_calls'] for step in raw['warmup'] + raw['steps']), 'Unexpected benchmark short calls')
    require(all(r['q_len'] > 128 for r in rows[:28]),
            'SCAN INVALID: first 28 widths disagree with GPU evidence; stop')
    require([s for r in rows[:28] for s in r['sample_ids']] == [item['sample_id'] for item in raw['data_plan']],
            'Benchmark plan identity mismatch')
    widths = sorted(r['q_len'] for r in rows)

    def percentile(p):
        rank = (len(widths) - 1) * p / 100
        low = int(rank)
        return widths[low] + (widths[min(low + 1, len(widths) - 1)] - widths[low]) * (rank - low)

    histogram = Counter(sum(r['short_path'] for r in rows[start:start + 4]) for start in range(0, len(rows), 4))
    short = sum(r['short_path'] for r in rows)
    total_steps = len(rows) // 4
    return {'total_remaining_samples': len(rows) * 2, 'total_remaining_microbatches': len(rows),
            'short_path_microbatches': short, 'long_path_microbatches': len(rows) - short,
            'short_path_fraction': short / len(rows),
            'q_len_distribution': {'min': widths[0], **{f'p{p}': percentile(p) for p in (25, 50, 75, 90, 95, 99)},
                                   'median': percentile(50), 'max': widths[-1], 'percentile_method': 'linear/inclusive'},
            'bands': {label: sum(low < w <= high for w in widths) for label, low, high in
                      [('<=64', 0, 64), ('65-128', 64, 128), ('129-256', 128, 256),
                       ('257-512', 256, 512), ('513-1024', 512, 1024), ('>1024', 1024, float('inf'))]},
            'optimizer_step_coverage': {'total': total_steps, 'with_short_path': total_steps - histogram[0],
                                        'with_short_path_fraction': 1 - histogram[0] / total_steps,
                                        'all_four_long': histogram[0],
                                        'short_microbatches_per_step_histogram': {str(i): histogram[i] for i in range(5)}},
            'first_20_short_path_locations': [r for r in rows if r['short_path']][:20],
            'failed_28_microbatch_window': {'all_long': True, 'sample_order_identical': True, 'rows': rows[:28]}}
