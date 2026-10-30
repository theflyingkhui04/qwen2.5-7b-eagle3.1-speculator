"""Local-only evidence writer for the bounded 2x4 versus 4x2 CPU precheck.

Run: python -m training.workload_padding_precheck --inputs runs/short-path-scan-20261028
"""
import argparse
import ast
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import subprocess
from unittest.mock import patch
import uuid

from training import native_vocab_mapping as mapping
from training.short_path_scan import (checkpoint_metadata, compare_groupings, effective_widths,
                                      native_helpers, raw_token_length, remaining_batches, require, summarize)
from training.train import REPO_ROOT, SPECFORGE_SHA


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import torch
    require(not torch.cuda.is_initialized(), 'CUDA was initialized')
    root = args.inputs.resolve()
    def load(path):
        return json.loads(path.read_text(encoding='utf-8'))
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    def git(*args):
        return subprocess.check_output(['git', '-c', f'safe.directory={REPO_ROOT.as_posix()}', *args],
                                       cwd=REPO_ROOT, text=True).strip()
    branch, head, status = git('branch', '--show-current'), git('rev-parse', 'HEAD'), git('status', '--short')
    profile_path = REPO_ROOT / 'configs/native_vocab_mapping_proof.json'
    profile = load(profile_path)
    benchmark = load(root / 'benchmark.json')
    state = checkpoint_metadata((root / 'checkpoint_header.bin').read_bytes())
    config = benchmark['config_identity']
    manifest = load(root / 'artifact_manifest.json')
    require(digest(root / 'artifact_manifest.json') == profile['train_manifest_sha256'], 'Manifest hash mismatch')
    require(profile['specforge_sha'] == benchmark['versions']['specforge_sha'] == SPECFORGE_SHA,
            'SpecForge identity mismatch')
    require(len(manifest['files']) == profile['train_samples'] == state['dataset_size'] == 40000,
            'Dataset count mismatch')
    require(manifest['target_revision'] == profile['target_revision'], 'Target revision mismatch')
    require(manifest['contract']['specforge_sha'] == SPECFORGE_SHA and
            manifest['contract']['max_length'] == profile['max_length'] == 2048, 'Manifest contract mismatch')
    require(state['sampler_seed'] == 42 and state['epoch'] == 0 and state['epoch_samples'] == 24000
            and state['epoch_batch'] == 12000, 'Requested resume mismatch')
    require({k: state[k] for k in benchmark['raw']['state_identity']['counters']} ==
            benchmark['raw']['state_identity']['counters'], 'Checkpoint/benchmark mismatch')
    sampler, padding = native_helpers((root / 'pinned_launch.py').read_bytes(),
                                      (root / 'pinned_data_utils.py').read_bytes())
    source_root = REPO_ROOT / 'profiler_analysis/results/mlp'
    source_manifest = load(source_root / 'source_manifest.json')
    source = source_root / 'sources/llama3_eagle.py.txt'
    identity = next(x for x in source_manifest if x['file'] == source.name)
    require(identity['ref'] == SPECFORGE_SHA and digest(source) == identity['sha256'], 'Model source mismatch')
    tree = ast.parse(source.read_bytes())
    conditions = [node for node in ast.walk(tree) if isinstance(node, ast.If)
                  and ast.unparse(node.test) == 'q_len <= 128']
    require(len(conditions) == 1, 'Unrecognized FlexAttention source condition')
    condition = {'source': identity, 'line': conditions[0].lineno, 'condition': ast.unparse(conditions[0].test)}
    protected = [profile_path, root / 'artifact_manifest.json', root / 'checkpoint_header.bin',
                 root / 'benchmark.json', root / 'pinned_launch.py', root / 'pinned_data_utils.py',
                 root / 'ordering_runtime_audit.json', source, source_root / 'source_manifest.json',
                 REPO_ROOT / 'configs/train_config.yaml', REPO_ROOT / 'configs/eagle3.1_qwen2.5_7b.json',
                 REPO_ROOT / 'modal_pipeline.py', REPO_ROOT / 'training/native_vocab_mapping.py',
                 REPO_ROOT / 'training/short_path_scan.py', Path(__file__).resolve()]
    protected += sorted((root / 'cache/processed_dataset').glob('*.pkl'))
    fingerprints = {str(p): digest(p) for p in protected}
    with patch.object(mapping, 'VOLUME_ROOT', root):
        table = mapping._load_cache(root / 'cache', profile)
    @lru_cache(maxsize=128)
    def get_row(index):
        return {key: table[key][index].as_py() for key in ('input_ids', 'loss_mask')}
    paths = [config['data']['hidden_states_path'] + '/' + item['path'] for item in manifest['files']]
    widths = effective_widths(paths, get_row, len(paths), profile['max_length'])
    raw_lengths = {i: raw_token_length(get_row(mapping.cache_sample_index(path, len(paths))), profile['max_length'])
                   for i, path in enumerate(paths)}
    rows = list(remaining_batches(state, config, paths, widths, sampler, padding))
    print('Authenticated cache and reconstructed widths; comparing 12000 windows', flush=True)
    baseline = summarize(rows, benchmark)  # Includes independent first-28 GPU identity gate.
    expected = []
    audit = load(root / 'ordering_runtime_audit.json')
    for epoch in range(state['epoch'], state['num_epochs']):
        order = sampler(len(paths), dp_rank=0, dp_size=1, seed=state['sampler_seed'], epoch=epoch, shuffle=True)
        hashed = hashlib.sha256(torch.tensor(order, dtype=torch.int64).numpy().astype('<i8').tobytes()).hexdigest()
        require(hashed == audit['epoch_permutation_sha256_int64_little_endian'][str(42 + epoch)],
                'Native permutation differs from prior cross-runtime audit')
        offset = state['epoch_samples'] if epoch == state['epoch'] else 0
        expected.extend((epoch, i) for i in order[offset:])
    require(len(expected) == 96000 and len(rows) == 48000, 'Remaining workload mismatch')
    report = compare_groupings(rows, expected, raw_lengths, padding, short_threshold=128)
    require(report['sample_accounting']['scanned_optimizer_windows'] == 12000, 'Window count mismatch')
    require(report['variants']['A']['short_microbatches'] == baseline['short_path_microbatches'], 'Baseline mismatch')
    draft = load(REPO_ROOT / 'configs/eagle3.1_qwen2.5_7b.json')
    proxy = {}
    for name, variant in report['variants'].items():
        slots = variant['microbatch_token_slots_distribution']['max']
        proxy[name] = {'max_padded_q_len': variant['q_len_distribution']['max'],
                       'max_microbatch_token_slots': slots,
                       'hidden_B_q_H_elements': slots * draft['hidden_size'],
                       'capture_B_q_3H_elements': slots * 3 * draft['hidden_size'],
                       'mlp_B_q_I_elements': slots * draft['intermediate_size'],
                       'draft_logits_B_q_Vdraft_elements': slots * draft['draft_vocab_size']}
    proxy['scope'] = ('Static individual tensor dimensions only, not simultaneous residency or peak GPU VRAM. '
                      'B*q*H, B*q*3H and B*q*I follow pinned draft source; logits use draft_vocab_size=16000. '
                      'Target vocab_size=152064 is separate. No attention matrices allocated.')
    require(fingerprints == {str(p): digest(p) for p in protected}, 'Protected input changed')
    require(not torch.cuda.is_initialized(), 'CUDA initialized during scan')
    report.update(execution_branch=branch, head=head, historical_handoff_head='2d79c14a1a5efb118d84f3e477e424e54f962881',
                  working_tree_before_scan=status, specforge_sha=SPECFORGE_SHA, resume_state=state, seed=42,
                  timestamp=datetime.now(timezone.utc).isoformat(), source_and_input_sha256=fingerprints,
                  protected_local_inputs_unchanged=True, path_condition=condition, memory_pressure_proxy=proxy,
                  non_pad_definition='Truncated raw cache token length; excludes capture and training padding; not supervised token count.',
                  ordering_runtime_audit=audit, verdict='CPU PRECHECK PASS', no_gpu_claim=True,
                  command='python -m training.workload_padding_precheck --inputs ' + args.inputs.as_posix())
    output = REPO_ROOT / 'runs' / ('workload-padding-precheck-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
                                  + '-' + uuid.uuid4().hex[:8] + '.json')
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps({'artifact': str(output), **{k: v for k, v in report.items() if k in
                     ('sample_accounting', 'variants', 'absolute_delta', 'delta_pct', 'changed_A_pair_classifications',
                      'windows_with_changed_classification', 'memory_pressure_proxy', 'worst_windows', 'verdict')}}, indent=2))


if __name__ == '__main__':
    main()
