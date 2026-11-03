## EXECUTIVE VERDICT

Historical audit and dated follow-ups. The latest production-integration implementation and its CPU validation are recorded at the end of this document.

`ROOT CAUSE NOT YET PROVEN`

Audit date: 2026-10-26 (+07). Project HEAD: `3ce14947212c07604382b69219118ea24191c90b`, branch `fix/m3-04-specforge-current-migration`.
Upstream source was fetched through the GitHub connector at `fddcb4ad37733a821708170df28cac2f7bb6ea05`, not main.

**BLOCKER / FACT:** The pinned colocated offline assembly always counts TRAIN feature tokens before constructing the trainer when no explicit vocabulary map is configured. Even a native mapping-cache hit does not skip that count. Synthetic CPU execution reproduces this exactly.

**BLOCKER / INFERENCE:** This serial startup scan is the leading explanation for the supplied run evidence. There is no captured Python/native stack, phase marker, or completed scan timing establishing that PID 8 was inside that operation at the sampled time. The actual Volume mmap/page-fault cost is also unknown. Consequently the complete runtime causal chain is not proven.

**BLOCKER / UNKNOWN:** The prepared TRAIN mapping has not been shown tensor-identical to the map selected by the native offline feature-counting path. A synthetic counterexample demonstrates why switching paths without that check can change the selected vocabulary at tied frequencies.

Only audit documentation and CPU tests were added. Production configuration, orchestration, upstream source and immutable artifacts are unchanged. No Modal command, GPU workload, paid workload, Volume mutation, commit, push, merge or PR was performed. `M3-04_CODEX_CONTEXT.md` was not read or modified.

## RUNTIME EVIDENCE SUMMARY

The following are **BLOCKER / OBSERVATION**, supplied in the incident attachment; they were not independently collected during this local audit:

- App `ap-ZiQDphG7T0Df23fDrsJAE2`: preflight passed, then `train_specforge` started.
- PID 8 ran `specforge train -c /vol/runs/m3-04/20261025T232919Z-e01428d6/execution.yaml` for roughly an hour.
- `ps` showed `Dl`; `/proc/8/status` showed disk sleep, about 3.95 GB RSS and seven threads; no child process was observed.
- GPU utilization was 0%, with approximately 3187 MiB allocated; no optimizer, loss, eval or canonical checkpoint evidence was observed.
- FD 54 referenced TRAIN `rows_18000-20000/data_19097.ckpt`, approximately 20 MB, through the physical Modal Volume path.
- FD offset remained 4096 across five seconds. Independent metadata, first-4-KiB and sampled-offset reads succeeded. Those bounded reads do not establish full-file integrity or mmap performance.
- In approximately five seconds, `rchar` rose by 189814 bytes and `syscr` by 83; `read_bytes` remained zero. This shows activity, not a hard deadlock, and does not measure all mmap or backing-store traffic.
- The user stopped the app. The resulting `RemoteError` is a termination consequence. Incident status remains `BLOCKED WITH EVIDENCE`, not `FAILED TRAINING`.

## EXACT PINNED SPECFORGE CALL PATH

All paths/line numbers below refer to the pinned SHA. The local Windows environment is Python 3.11.0 / Torch 2.6.0+cu124; these tests execute source excerpts on CPU, not the full Modal runtime with Torch 2.13.0.

Common entry:

1. [`specforge/cli.py:355–367`](https://github.com/sgl-project/SpecForge/blob/fddcb4ad37733a821708170df28cac2f7bb6ea05/specforge/cli.py#L355): `main` → Click `cli.main` → `train`.
2. [`cli.py:231–264`](https://github.com/sgl-project/SpecForge/blob/fddcb4ad37733a821708170df28cac2f7bb6ea05/specforge/cli.py#L231): `load_config` → `resolve_run` → `build_launch_plan`; worker branch → `_train(bind_run(...))`. The `--plan` branch returns without assembling a trainer.
3. [`config/schema.py`, `load_config` / `Config.from_file`](https://github.com/sgl-project/SpecForge/blob/fddcb4ad37733a821708170df28cac2f7bb6ea05/specforge/config/schema.py): YAML parsing, legacy migration and typed validation. `Config.mode` is offline when `data.hidden_states_path` is nonempty.
4. [`application/composition.py:33–57,136–148`](https://github.com/sgl-project/SpecForge/blob/fddcb4ad37733a821708170df28cac2f7bb6ea05/specforge/application/composition.py#L33): `resolve_run` → registry resolution → `bind_run`; `_train` initializes distributed state and calls `build_application_run(resolved).run()` (`cli.py:137–152`).
5. `build_application_run` → [`training/assembly.py:576–640`](https://github.com/sgl-project/SpecForge/blob/fddcb4ad37733a821708170df28cac2f7bb6ea05/specforge/training/assembly.py#L576): `build_training_run` → `build_model_bundle` → `_ensure_offline_vocab_mapping`.

**First unconditional feature-content path for this reconstructed configuration:**

`_ensure_offline_vocab_mapping` (`assembly.py:476–515`)
→ `list_feature_files` + `os.stat` for identity
→ [`count_effective_feature_tokens`, `training/vocab_mapping.py`](https://github.com/sgl-project/SpecForge/blob/fddcb4ad37733a821708170df28cac2f7bb6ea05/specforge/training/vocab_mapping.py)
→ `list_feature_files` again
→ for each sorted TRAIN path, [`feature_store.py:278–283`, `load_feature_file`](https://github.com/sgl-project/SpecForge/blob/fddcb4ad37733a821708170df28cac2f7bb6ea05/specforge/runtime/data_plane/feature_store.py#L278)
→ `.ckpt`: `torch.load(path, weights_only=False, mmap=True)`
→ use `input_ids` and `loss_mask` tensor values to count supervised tokens
→ `_install_dataset_vocab_mapping` (`assembly.py:424–473`), **then** check native mapping cache.

**Later steady-state path:**

`build_training_run` (`assembly.py:643`)
→ [`launch.py:541–663`, `build_offline_runtime`](https://github.com/sgl-project/SpecForge/blob/fddcb4ad37733a821708170df28cac2f7bb6ea05/specforge/launch.py#L541)
→ [`algorithms/eagle3/data.py:30–48`, `build_offline_reader`](https://github.com/sgl-project/SpecForge/blob/fddcb4ad37733a821708170df28cac2f7bb6ea05/specforge/algorithms/eagle3/data.py#L30)
→ `OfflineManifestReader.read` → `__iter__` → `_ref_for` (metadata-only by default)
→ `_shard_offline_refs` / `_distributed_sampler_indices`
→ `_assemble_trainer` → [`Trainer.__init__`, `training/trainer.py:165–180`](https://github.com/sgl-project/SpecForge/blob/fddcb4ad37733a821708170df28cac2f7bb6ea05/specforge/training/trainer.py#L165)
→ `FeatureDataLoader`
→ at iteration, [`feature_dataloader.py:359–405`](https://github.com/sgl-project/SpecForge/blob/fddcb4ad37733a821708170df28cac2f7bb6ea05/specforge/runtime/data_plane/feature_dataloader.py#L359): `__iter__` → `_iter_refs` → `_iter_refs_prefetch` → `_prepare_batch` → `_timed_make_batch` → `_make_batch` → `_materialize`
→ `LocalFeatureStore.get` → `_get_from_file` (`feature_store.py:447–525`)
→ `load_feature_file` → `torch.load(..., mmap=True)`.

There is no PyTorch Dataset `__getitem__` in this canonical path. The retained legacy `OfflineEagle3Dataset` is not the selected dataset.

## BEFORE-STEP-1 FULL-CORPUS OPERATIONS

Rows are **FACT** from pinned source unless explicitly conditional. Scope is full-corpus operations, not every model-weight read at startup.

| Operation | Files touched | Bytes/tensors read | Cached? | Runs every retry? | Evidence |
|---|---|---|---|---|---|
| Project raw/prepared audit, CPU preflight | Three raw JSONL files, 45000 rows total including audit-only test | Raw hashes and tokenization/conversation audit; test is not used for model or mapping | No training feature cache involved | Yes | `modal_pipeline.py:preflight`, `validate_raw`, `audit_prepared` |
| Project feature validation, CPU preflight | 40000 TRAIN + 3000 eval metadata entries, two manifests | Manifest JSON, file names, sizes/types; no checkpoint-content rehash | Existing manifests/inventory reused | Yes | `modal_pipeline.py:check_features`, `feature_metadata` |
| Offline map identity | 40000 TRAIN, sequential sorted-path stat calls | Absolute paths, sizes, mtime_ns; no tensor values | Identity rebuilt in memory | Yes without explicit mapping | `assembly.py:493–504` |
| Offline token count | 40000 TRAIN, another listing and serial load loop on each rank | Entire checkpoint object/storage descriptors; `.ckpt` tensors mmap-backed; only ids/mask values accessed by counter | Count itself is not cached | **Yes even on native map-cache hit** | `assembly.py:505–515`, `vocab_mapping.py:count_effective_feature_tokens` |
| Native map install/cache lookup | One derived `.pt` map if present; otherwise creates map after count | Small `t2d` / `d2t` | `<cache_dir>/vocab_mapping/<sha256>.pt` | Lookup/install every assembly; write only on miss | `assembly.py:424–473` |
| TRAIN reference inventory | 40000 sorted names via recursive `os.walk` | Metadata-only SampleRefs by default; no feature content | In-memory refs | Yes | `launch.py:595–607`, `offline_reader.py:72–163` |
| Eval reference inventory | 3000 sorted names | Same; no eval token count | In-memory refs retained by eval factory | Yes | `launch.py:244–289,614–624` |
| Opt-in eager feature validation | Conditional 40000 TRAIN + 3000 eval | `load_feature_file`, tensor shape/dtype/numel metadata; mmap for `.ckpt`, full decompression for `.gz` | None | Only if `SPECFORGE_VALIDATE_OFFLINE_FEATURES=1` | `offline_reader.py:117–129` |
| TRAIN shuffle, rank partition, length and schedule | All 40000 in-memory refs/indices | `torch.randperm(seed + epoch)` and list lengths; no feature files | In memory, rebuilt each epoch | Yes; deterministic on resume | `launch.py:176–241`, `trainer.py` fixed-plan validation |
| Eval rank partition | All 3000 refs, no shuffle | Metadata list | In-memory factory | Yes | `launch.py:272–274` |

Directory enumeration/stat and count are serial in the trainer process, not DataLoader workers. With the canonical single rank they execute once per assembly, rather than in multiple processes. No processed-dataset pickle/Arrow cache is consulted by this colocated offline assembly. No persistent feature index/cache is constructed. Reader refs, shuffle plan and batch-chunk lists are in-memory only. First-batch prefetch materializes a bounded number of records, not the whole corpus.

Native cache key is SHA256 of `dataset_identity + ':' + target_vocab_size + ':' + draft_vocab_size`; identity is sorted JSON containing `kind=offline-features-v1`, a sorted list of `(absolute path, size, mtime_ns)` and `max_length`. Preparation does not generate this cache. Its existence in the stopped run is **UNKNOWN**. It is persisted only after counting finishes, and even an existing file would not bypass counting. Absolute-path/mtime changes may change the cache key; that is a cache characteristic, not reopening the closed mount-identity bug.

Default reader validation is off. The stopped process environment was not captured, so an external override cannot be ruled out; project image code does not set one.

## VOCAB MAPPING ANALYSIS

**BLOCKER / FACT:** `model.vocab_mapping_path` defaults to the empty string (`schema.py:307`) and project config never sets it. The trainer does not auto-discover `<hidden_states_path>/vocab_mapping/vocab_mapping.pt`. It derives a map from TRAIN features, truncates ids/masks to `max_length`, selects nonzero-mask tokens, then uses the late native cache lookup. Eval does not choose a vocabulary; it uses the same model buffers.

An explicit path is supported: [`algorithms/model_providers.py:86–113`](https://github.com/sgl-project/SpecForge/blob/fddcb4ad37733a821708170df28cac2f7bb6ea05/specforge/algorithms/model_providers.py#L86) loads it during `build_eagle3_draft`, before `_ensure_offline_vocab_mapping`, whose early return then bypasses both identity and token scans.

Preparation uses a different mechanism: [`prepare_hidden_states.py:821–870`](https://github.com/sgl-project/SpecForge/blob/fddcb4ad37733a821708170df28cac2f7bb6ea05/scripts/prepare_hidden_states.py#L821) builds/reuses tokenized TRAIN data and generates `<output_path>/vocab_mapping/vocab_mapping.pt` from that dataset. Eval preparation separately generates an eval map; it must never be used to select training vocabulary. The prepared dataset cache is `<cache_dir>/processed_dataset`; its MD5 key encodes data path, max length, chat template, target path, sample limit and preformatted flag. The HF generator cache is under `/opt/SpecForge/cache/hf_dataset`, a separate location. These are preparation caches, not the offline training map cache.

**BLOCKER / FACT:** Both mapping builders use `Counter.most_common`. Equal-frequency ties retain first-seen order. Preparation iterates its shuffled tokenized dataset; offline training sorts checkpoint paths lexicographically, which is not numeric sample order (`data_10.ckpt` precedes `data_2.ckpt`). CPU reproduction shows different selected tokens for the same token multiset when a tie crosses the draft-vocab cutoff. This does **not** prove the actual 16000-token maps differ, but their shapes alone do not prove equality. Binary mask equivalence, no skipped captures and exact token correspondence must also hold for reuse.

The provided prep evidence establishes TRAIN/EVAL counts and map dimensions. It does not supply map checksums/content, the stopped run's native cache inventory, or tensor equality. No current Volume cache existence was independently inspected. The already-recorded full feature hashes/inventory remain intact; no new cache format was added.

The smallest next CPU discriminator for safe map reuse is a read-only equality check on the existing TRAIN artifacts: count the cached tokenized TRAIN rows in the exact lexicographic feature-file order, verify row-to-feature ids/masks correspondence from retained preparation evidence, apply the pinned selector, and compare both `t2d` and `d2t` against the prepared TRAIN map. If correspondence cannot be verified, report the test inconclusive; do not assume it or substitute eval data. This is not a new cache or benchmark. These actual artifacts are not present locally, so the test was not run here.

## DATALOADER ANALYSIS

**NON-BLOCKING / FACT:** The selected source is `OfflineManifestReader` → metadata-only `SampleRef` list → `LocalFeatureStore` → custom `FeatureDataLoader`.

- Discovery uses recursive `os.walk`, suffix filtering for `.ckpt` / `.ckpt.gz`, then lexicographic absolute-path sorting, not recursive glob.
- EAGLE3 provider default is **4** (`algorithms/eagle3/providers.py:195`); config field default is `None` (`schema.py:436`), resolved by `assembly._dataloader_num_workers:518–524`.
- Workers are lazy **threads**, created by `ThreadPoolExecutor` at offline iteration, not subprocesses. No `ps --ppid 8` children is fully compatible with active loader threads, or with the earlier serial scan. Seven total threads does not discriminate these phases.
- Offline prefetch has at most four submitted batches, preserves ordered yields, and refills one batch after each consumed result. Canonical batch size 2 means up to eight records across the initial four tasks. There is no PyTorch `prefetch_factor` or `persistent_workers` option; the executor lives for that iterator and shuts down on exit.
- TRAIN ref order reproduces `DistributedSampler(drop_last=False)` with seed + epoch; each epoch can read random file positions across the Volume. Resume recreates the same deterministic plan then seeks batches. Eval is unshuffled. No new sampling choice was made.
- TRAIN pin memory is enabled when worker count > 0 and CUDA is available (`trainer.py:165–180`). Eval factory passes no pin-memory flag, so it defaults to false (`launch.py:277–287`).
- Each materialized record freshly invokes `load_feature_file`. The uncompressed path deserializes the full record's Python/tensor descriptors with mmap storage, then chooses requested keys; it does not eagerly read every hidden-tensor byte in `torch.load`. Normal training clones/materializes required tensors and collates them, so hidden values are eventually read. There is no application tensor cache for repeated file reads; filesystem/page caching is separate.
- `.ckpt.gz` is supported but the supplied open path is `.ckpt`, and project preparation does not request compression. Gzip reads/decompresses the entire stream into `BytesIO`, with no mmap. It is not a demonstrated explanation for this open file.
- EAGLE3 normalization slices at 2048, clones/masks the final loss entry, and uses `DataCollatorWithPadding` for batch padding/concatenation (`algorithms/eagle3/data.py:10–27,82–87`; `data/utils.py:32–169`). Typically one feature file represents one captured sample; the project's manifest counts enforce 40000/3000, and preparation writes per-sample records (`prepare_hidden_states.py:693–729`).

**Reconstructed execution config, not a downloaded execution.yaml:** `feature_paths` and `resolved_config` set TRAIN `/vol/hidden_states/m3-04/train/a113d35ad76e959e31dd`, eval `/vol/hidden_states/m3-04/eval/a113d35ad76e959e31dd`, cache `/vol/datasets/m3-04/processed/a113d35ad76e959e31dd/cache`; no explicit map or worker field, therefore map `''`, resolved workers 4. Strategy eagle3, batch 2, accumulation 4, max_length 2048, draft_vocab_size 16000, offline train/eval, BF16. The target becomes the resolved HF snapshot path; `wandb_dir` is the run-evidence `wandb` directory. These follow exact project resolver code and existing identity regression, not remote file inspection.

## ACTIVE HYPOTHESES

1. **BLOCKER / HYPOTHESIS — serial pre-trainer token scan is the observed stall.** For: unconditionally present, touches every TRAIN record even on native cache hits, matches low GPU and an open TRAIN file. Against/limit: no sampled stack or phase timing. Minimal discriminator: attribute the blocked load/page fault to `count_effective_feature_tokens` versus `_iter_refs_prefetch` using an all-thread stack at the stall. The local test already proves scan extent/cache ordering, not the original process phase.
2. **BLOCKER / HYPOTHESIS — Volume mmap/page faults make feature access pathological, possibly during actual loading.** For: mmap is the exact default, D-state and stable FD offset are compatible with page faults; successful `dd` does not exercise the same access path. Against/limit: no page-fault/native-stack evidence or representative mmap timing; no optimizer log alone does not locate the phase. Minimal discriminator: pair the same all-thread stack with kernel wait/native read context; no worker-count change is justified.
3. **BLOCKER / HYPOTHESIS — the existing prepared TRAIN map can safely bypass the scan.** For: generated from TRAIN by the same pinned selector. Against/limit: different first-seen token order can change tied cutoff selection; actual tensors are unavailable. Minimal discriminator: the read-only CPU mapping-equivalence test specified above, using the existing TRAIN processed cache and prepared map.

The first two are parts of a possible causal chain, not established alternatives with measured likelihoods; the third addresses the safety of the smallest potential fix. Corruption is **NON-BLOCKING / HYPOTHESIS**, not retained as an active explanation without new evidence. Full local staging/repacking and performance architecture changes are **OUT-OF-SCOPE / FACT** under the task constraints.

## ROOT CAUSE

The exact **source defect** is proven: native mapping-cache lookup occurs after a mandatory serial TRAIN token scan. Missing explicit mapping plumbing exposes this path. That defect is not sufficient to certify the precise cause of the hour-long runtime stall or the correctness of reusing the prepared map.

Performance model (**INFERENCE**, explicit assumptions): 40000 opens × average archive metadata plus accessed ids/mask pages, divided by effective throughput, plus open/stat/page-fault latency. If both token arrays are int64 and every sequence is length 2048, token payload is 32768 bytes per record, about 1.31 GB across TRAIN, plus metadata/page overhead. One hour corresponds to about 11.1 records/s or 0.36 MB/s of that assumed token payload. This is plausible for costly small accesses but not a measurement. Actual lengths, tensor dtypes, page granularity, readahead and storage throughput are unknown.

Multiplying 40000 × 20 MB gives about 800 GB and 222 MB/s for one hour **only if** every record is 20 MB and all storage bytes are physically read. Neither condition is established; mmap does not justify that model. The 5-second `rchar` delta (~38 KB/s) does not include all possible mmap/backend traffic and cannot be extrapolated to corpus throughput. The file name 19097 does not establish scan percentage because path sorting is lexical.

## PROPOSED MINIMAL FIX

No production fix is proposed until runtime phase attribution and mapping equivalence are established. A supported explicit map would bypass the scan, but pointing it at the prepared TRAIN map without equality evidence risks changing the frozen token mapping. A cache-before-count wrapper would help an existing native cache hit, but that cache's existence is unknown and such a wrapper would not avoid the first scan on a miss. Neither is implemented speculatively.

## CHANGED FILES

- `tests/fixtures/specforge_offline_startup.py.txt`: exact pinned reader, feature-load, preparation mapping and worker-resolver bodies; GPU dependencies injected only by tests.
- `tests/test_offline_startup_audit.py`: seven bounded CPU discriminators, reusing the existing pinned assembly fixture.
- `docs/m3-04-offline-startup-audit.md`: this evidence report.

Production code changes: none. Suggested human commit message: `test(m3-04): audit pinned offline startup scans and mapping reuse risks`.

## TARGETED TESTS

`python -m pytest tests/test_offline_startup_audit.py tests/test_vocab_contract.py -v`

11 passed in 88.08s; final targeted rerun: **11 passed in 34.25s**. Tests prove all-N TRAIN loads before and after a native cache hit; prepared fixed-name map ignored without explicit configuration; explicit-path early return; counter does not consume hidden values; mmap argument; lazy reader; mapping tie counterexample; provider worker resolution. Existing tests verify 152064/16000 shapes, TRAIN-only selection and deterministic mapping.

The explicit-path test covers the assembly guard, not a real production map's validity, end-to-end startup throughput or successful optimizer progress. Smoke/full/resume/test-split protections have not changed and remain covered by the existing suite.

## FULL CPU SUITE

`python -m pytest tests/ -v`: **102 passed in 114.37s**, no failures or skips. Modal APIs and accelerator checks in the suite are mocked; this did not launch Modal or allocate a GPU.

## DIFF CHECK

`git -c safe.directory=D:/Document/Four-year/do_an/qwen2.5-7b-eagle3.1-speculator diff --check`: exit 0, no output. The per-command safe.directory override is necessary for the sandbox account; no global Git settings were changed. New files are untracked, so supplemental `git diff --no-index --check -- NUL <file>` checks cover all three; a trailing blank line in the new fixture was corrected before the final check.

## SCIENTIFIC CONTRACT CHECK

| Frozen item | Affected? |
|---|---|
| Raw data | NO |
| Split | NO |
| Sample count (40000/3000; test 2000 audit-only) | NO |
| Target Qwen2.5-7B-Instruct | NO |
| SpecForge SHA | NO |
| BF16 | NO |
| Resolver capture layers [1,13,24] | NO |
| Draft vocab 16000 / target vocab 152064 | NO |
| Token mapping definition | NO |
| Epochs 3 | NO |
| Batch 2 | NO |
| Accumulation 4 | NO |
| LR 1e-4 | NO |
| Max length 2048 | NO |
| TTT 5 | NO |
| Eval interval 1000 | NO |
| Save interval 1000 | NO |
| Max checkpoints 2 | NO |
| Best metric eval/simulated_acc_len | NO |

EAGLE-3.1 fc_norm, norm_output, post-norm feedback, one decoder layer, run identity and output path are unchanged. No hardcoded layer override was introduced.

## REMAINING GAP

`Canonical Modal GPU training has NOT been rerun.`

The supplied latest run has no demonstrated optimizer-step, train-loss, numerical-stability, eval, step1000 checkpoint, best checkpoint or packaged canonical weights. M3-04 is not complete. Actual execution YAML, runtime stack/environment and native cache inventory were not locally available. CPU/source evidence cannot prove mmap performance on the original mounted Volume or equality of the actual mapping artifacts.

## VERDICT

`NOT READY — runtime stall phase is unproven; prepared/native TRAIN mapping equivalence is unverified`.

## NEXT ACTION

Run the single read-only CPU mapping-equivalence test described above on the existing TRAIN processed cache and prepared TRAIN map, preserving native lexicographic feature order and checking correspondence. It must use already-accessible artifacts; obtaining them through a paid job is not authorized by this report.

## Follow-up: exact native mapping from TRAIN cache (2026-10-26)

**FACT / CPU proof: `EXACTLY EQUIVALENT` for cache-derived C versus native B.**
The initial runtime audit above remains historical evidence: the original stalled process's precise phase remains unproven. This follow-up establishes the mapping semantics needed to design a later project-local integration. No production integration was implemented.

### Input identity and reference

- Artifact: `a113d35ad76e959e31dd`; TRAIN features: 40000.
- TRAIN cache: `f60968e1f91ca5759dffaed4147cf79e`, eight Arrow IPC shards with `.pkl` names, each 5000 rows; total 282293528 bytes.
- Target revision: `a09a35458c702b33eeacc393d103063234e8bc28`; capture layers `[1,13,24]`.
- `max_length=2048`; target vocabulary 152064; draft vocabulary 16000.
- SpecForge SHA: `fddcb4ad37733a821708170df28cac2f7bb6ea05`.
- Manifest SHA256: `6e58292f0a054167d08df1d78bbf6cd7bc3b779482658f067e7bdc3d045b1fb2`.

Native B had previously existed only in process memory. It was independently reconstructed from all 40000 actual feature token payloads using the already-audited bounded byte-range procedure and pinned counter/selector, then saved to a temporary local file. B was not derived from preparation mapping A or candidate C. A versus B reproduced the previous 56 differing `t2d` entries and 10873 differing `d2t` entries; A remains prohibited as a substitute.

### Three orderings and correspondence

1. Physical cache order concatenates shard ordinal 00000 through 00007, preserving each shard's 5000 rows.
2. Numeric sample identity N maps to global cache row N, shard `N // 5000`, offset `N % 5000`. The pinned writer maps N to `rows_<2000*(N//2000)>-<2000*(N//2000)+2000>/data_N.ckpt`.
3. Native feature order comes from the original pinned `list_feature_files`: recursive discovery followed by full-path lexicographic sorting. Its start is `data_0`, `data_1`, `data_10`, `data_100`, `data_1000`, not numeric order; its final group is `rows_8000-10000`.

The proof verified all numeric identities 0 through 39999, live inventory names/sizes against the manifest, and all 40000 raw `input_ids`/`loss_mask` pairs against the new test-only cache reconstruction helper. Capture batch size 2 pads both arrays with zeros to the longer row in each adjacent pair; all 11878228 padding tokens were included in the comparison. The inventory and manifest/preparation-map hashes were stable across the read.

Native ordered-path SHA256: `8767b0e6cf5af8440035a58b19091bf32fa605b1097e7ebf7e8813d7650631b2`.

B and C ordered token-payload SHA256: `5dc9da497f6d7e1a704f8eb1bb2e7d765266b39f5ecdbf6cda86a3949c9a3dae`. Its canonical stream is each full POSIX path plus a NUL, then for ids and mask in that order a little-endian uint64 element count and contiguous int64 tensor bytes. This matches the previous direct-feature investigation's fingerprint.

### Exact construction and equality

The exact pinned function bodies were authenticated against existing fixtures by AST comparison. Both B and C execute `count_effective_feature_tokens` and `process_token_dict_to_mappings` unchanged. C replaces only the proof-process feature-read binding with the test-only cache-row lookup; no production module or upstream file is patched.

The counter flattens ids/masks, rejects mismatched lengths, truncates both at 2048, selects with `loss_mask.to(dtype=bool)`, counts each file's unique token IDs, validates selected IDs against the target vocabulary, and retains Counter first-encounter order. It does not reset the last mask entry or replace nonzero-mask handling with `mask == 1`. The selector uses `Counter.most_common(16000)` and sorts selected token IDs only after selection, then constructs `d2t` offsets and the target-sized boolean `t2d`.

The full ordered `(token, count)` sequences matched, as did the exact tensor inputs. The final acceptance was independent elementwise comparison of C against B reloaded from its temporary reference file:

| Check | Cache-derived C | Native B | Equal? |
|---|---|---|---|
| `t2d` shape | `(152064,)` | `(152064,)` | Yes |
| `d2t` shape | `(16000,)` | `(16000,)` | Yes |
| Selected set | 16000 tokens | 16000 tokens | Yes |
| Exact `t2d` | 0 differing entries | Reference | `torch.equal=True` |
| Exact `d2t` | 0 differing entries | Reference | `torch.equal=True` |

Canonical tensor-byte SHA256 values are identical for B and C:

- `t2d`: `2c10e7206a2cbfd763acfc4538967f0b52c3656d2a30da3d7b151ccf73e2fc80`.
- `d2t`: `d1678443a756c734089e0792b2487b17b2d5db6ce451b324eae5b8ee78b26dc0`.

For these hashes, `t2d` is contiguous uint8 boolean values and `d2t` is contiguous little-endian int64. Shapes/dtypes are checked separately; `.pt` serialization metadata is not used to establish tensor equality.

### Tie-sensitive regression and checks

Cutoff remains 64, with 164 tokens tied there; 80790 distinct effective TRAIN tokens and 16749122 supervised-token occurrences were counted. C preserves the complete native Counter encounter order, including cutoff ties.

The focused synthetic construction test independently writes feature records and reconstructs them from cache. Equal-frequency tokens 2, 9 and 7 are encountered in native order 2, 9, 7; draft size 2 selects `[2,9]` with offsets `[2,8]`. Numeric file order instead selects `[2,7]`; frequency-then-token-ID tie breaking also produces the wrong selection. A cache row permutation trips the correspondence gate. Additional assertions cover nonbinary nonzero mask 2, truncation, invalid target IDs, wrong row groups and out-of-range sample indices. Tests exercise construction rather than asserting only a stored production hash.

- `python -m pytest tests/test_offline_startup_audit.py tests/test_vocab_contract.py -v`: **16 passed in 62.04s** after correcting Windows path-separator handling in the test helper.
- `python -m pytest tests/ -v`: **107 passed in 119.81s**, no failures or skips.
- `git diff --check`: no whitespace errors. Supplemental untracked-file checks cover the edited audit/test files without staging.

### Performance observation and preserved evidence

**OBSERVATION**, local Python 3.11.0 / Torch 2.6.0+cu124 CPU; no CUDA context was initialized. This executes the pinned integer counting/selection source, not the full Modal Torch 2.13 runtime.

| Phase | Observation |
|---|---|
| Fetch eight existing TRAIN cache shards | 10.625 s wall; 282293528 bytes |
| Derive C Counter from cache, including provenance hashing | 115.938 s wall |
| Derive C Counter and exact selector output | 135.844 s wall; 131.890625 s process CPU |
| Feature content read by C derivation | **0 files / 0 bytes** |
| Independently reconstruct/verify reference B | 40000 feature prefixes, 36864 bytes each; 1474560000 bytes total |
| Complete proof, including reference reconstruction | 823.75 s wall |

The proof as a whole did touch all feature prefixes to reconstruct missing B. C itself depends only on the existing cache and authenticated metadata/order; it does not read feature content. These timings are supporting observations, not a GPU startup performance guarantee.

Temporary local evidence directory: `C:/Users/daibu/AppData/Local/Temp/m3-04-native-cache-proof-bqr32nma`.
It contains `native_B.pt`, `candidate_C.pt`, `candidate_counter.pt` with ordered Counter items, and `comparison.json`. No production artifact location or Volume was written, and no permanent artifact format was established.
An independent CPU readback loaded both saved mappings, repeated elementwise comparisons and recomputed their canonical tensor hashes against the report: **PASS**.

### Implication and remaining gap

**FACT:** Existing processed TRAIN cache can reproduce exact native canonical B without scanning feature contents during mapping derivation. A later minimal integration can authenticate this cache and manifest, reproduce native order, execute the pinned counter/selector, persist the exact B-equivalent mapping, and supply supported `model.vocab_mapping_path` so upstream bypasses its pre-step scan. This is proposed only; production behavior and configuration remain unchanged.

`NO scientific contract changes.` All raw hashes, datasets/splits/counts, target/revision, SpecForge SHA, offline train/eval, BF16, resolver capture layers, vocabulary sizes, mapping definition, one decoder layer, fc_norm/norm_output/post-norm feedback, epochs, batch/accumulation/effective batch, LR, max length, TTT, eval/save intervals, checkpoint retention, best metric, run identity and output location remain frozen.

`Canonical Modal GPU training has NOT been rerun.` M3-04 remains not done: canonical optimizer/loss/numerical-stability/eval/checkpoint/best-checkpoint/packaging evidence is still missing. This proof authorizes no GPU retry or production modification.

`READY TO DESIGN MINIMAL PRODUCTION INTEGRATION`

Next action: Design the smallest project-local integration that persists the exact native-equivalent mapping and supplies it to pinned SpecForge without changing scientific semantics.

## Production integration ready for review (2026-10-26)

**FACT:** Project-local production integration is implemented and CPU-verified. The historical D-state has not been conclusively attributed to the token scan. The proven startup behavior is that an empty explicit mapping path causes pinned SpecForge to read all TRAIN features before its native map-cache lookup; a nonempty supported `model.vocab_mapping_path` bypasses that scan.

### Implementation and authentication

`training/native_vocab_mapping.py` promotes the already-proven cache-row reconstruction helper, retains the native counter semantics, and calls the upstream selector directly. It does not patch SpecForge. It derives only from authenticated TRAIN cache, traversed in the pinned native lexicographic feature order. It never opens feature tensor contents, uses no eval/test rows, and preserves batch-2 padding and first-encounter tie handling.

`configs/native_vocab_mapping_proof.json` records the accepted artifact key, target revision, pinned SHA, TRAIN manifest digest, cache identity, eight exact cache shard names/sizes/row counts/digests, vocabulary dimensions, max length and both accepted canonical tensor-byte hashes. These are proof bindings, not a new dataset/cache format. Capture layers are authenticated through the feature manifest and current resolver payload; no layer-index override is introduced.

CPU `preflight(mode="features")` first retains the existing feature checks, then authenticates the profile, namespace, manifest and TRAIN cache bytes. On a miss it counts cached rows and applies the pinned selector. It writes standard EAGLE `t2d`/`d2t` tensors plus a JSON sidecar binding the complete feature contract, resolver capture layers, target revision and accepted proof. Tensor shape, dtype, selected-mask/offset consistency and exact hashes are checked before publication and after reload. Existing malformed, partial or stale seals are rejected, never silently replaced.

On a sealed hit, preflight authenticates the eight cache files and reloads/validates the mapping without recounting. Mapping derivation never reads the 40000 `.ckpt` contents. The execution gate rechecks namespace, manifest, sidecar and canonical tensor bytes without reading source cache contents or invoking token counting. Both resolved config writes occur only after validation. Resume resolves the same mapping artifact. Offline eval consumes the same model mapping buffers.

The new full integration is deliberately restricted to the accepted full proof. Unknown full artifact/cache/revision combinations fail closed. Smoke keeps its existing bounded native mapping derivation, requires its own smoke namespace and rejects an explicit full mapping. Prepared mapping A is never selected or used as a fallback.

### Persisted paths and config binding

Canonical mapping, to be created by the CPU preflight when the reviewed workflow is next run:

```text
/vol/datasets/m3-04/processed/a113d35ad76e959e31dd/cache/vocab_mapping/native_canonical.pt
```

The sidecar is `native_canonical.json` in the same directory. Inputs remain the existing `cache/processed_dataset/f60968e1f91ca5759dffaed4147cf79e_00000_of_00008.pkl` through `00007_of_00008.pkl`. No new processed-data conversion, native upstream patch or dependency change was added.

The explicit field is set before `resolved.yaml` is saved, checked again before `execution.yaml`, and preserved by `resolved_config`, including resume. Local canonical execution-config evidence contains:

```yaml
model:
  vocab_mapping_path: /vol/datasets/m3-04/processed/a113d35ad76e959e31dd/cache/vocab_mapping/native_canonical.pt
```

Actual production-selector/persistence verification ran locally against the accepted full ordered Counter and independently saved native B. Both persisted/reloaded tensors were elementwise identical to B:

| Tensor | Shape | Canonical bytes SHA256 | Equal to native B? |
|---|---|---|---|
| `t2d` | `(152064,)` | `2c10e7206a2cbfd763acfc4538967f0b52c3656d2a30da3d7b151ccf73e2fc80` | Yes, 0 differences |
| `d2t` | `(16000,)` | `d1678443a756c734089e0792b2487b17b2d5db6ce451b324eae5b8ee78b26dc0` | Yes, 0 differences |

Local evidence directory: `C:/Users/daibu/AppData/Local/Temp/m3-04-production-mapping-n2510iru`, containing `native_canonical.pt`, its sidecar, `execution.yaml` and `production_verification.json`. This is local verification output, not a claim that the Volume mapping already exists. The accepted full feature/cache equivalence proof was reused, not repeated. End-to-end cache reading, authentication, persistence, preflight ordering and execution consumption were exercised with independent synthetic CPU artifacts; the production selector/persistence additionally used the accepted full canonical Counter/B.

### Regression and review scope

Changed/added for this integration:

- `modal_pipeline.py`: CPU preflight binding, delayed resolved-config write, execution validation gate.
- `training/native_vocab_mapping.py`: authenticated cache-derived native mapping producer/validator.
- `configs/native_vocab_mapping_proof.json`: accepted input/output proof identities.
- `tests/test_native_vocab_mapping.py`: cache construction/authentication/persistence/config/consumer regressions, fail-closed mutations and full canonical mapping identity.
- `tests/test_offline_startup_audit.py`: imports the promoted helper and compares the production counter to the unchanged pinned counter.
- `tests/fixtures/native_train_counts.json`: accepted ordered Counter plus native selected IDs, about 951 KB; regression reference only, never used by production. It preserves the complete 80790-token Counter instead of approximating frequencies or keeping only cutoff tokens.
- This audit document: integration evidence and remaining limits.

The preexisting pinned source excerpt fixture is reused unchanged. No raw data, feature manifest, extraction bridge, draft/base training config, export implementation or requirements were changed. No commit, push, merge or Trello operation occurred.

Targeted command:

```text
python -m pytest tests/test_native_vocab_mapping.py tests/test_offline_startup_audit.py tests/test_vocab_contract.py tests/test_modal.py tests/test_train.py -v
84 passed in 181.28s
```

Full suite:

```text
python -m pytest tests/ -v
131 passed in 189.68s
```

Coverage includes missing/stale/malformed mappings, wrong artifact/cache/revision/contract, prepared-A path rejection, cache mutations on reuse, smoke/full/test namespace separation, native tie ordering, nonzero masks/truncation/invalid IDs, exact full canonical hashes and elementwise tensors, persisted reload, resume, real preflight-to-config ordering, and the pinned explicit-path early-return guard. The CPU tests prohibit `.ckpt` content reads during production derivation. The ordinary feature metadata checks remain intact.

### Modal inspection and scientific status

No Modal commands were necessary in this integration turn: artifact/cache paths and accepted proof identities were already available locally. No new compute/container, GPU/smoke training, hidden-state generation, Volume mutation, file transfer, deployment or repeated full feature proof was performed. Runtime verification in tests uses local CPU source excerpts and mocks; no claim is made that the full Modal Torch 2.13 runtime has run this integration.

`NO scientific contract changes.` All frozen dataset/split/sample/raw-hash, target/revision, SHA, offline train/eval, BF16, resolver capture, target/draft vocab, one-layer architecture/norm/feedback, token-mapping, epoch/batch/accumulation/effective-batch/LR/max-length/TTT, eval/save/retention/best-metric, run-ID and output contracts remain unchanged. The canonical artifact key remains `a113d35ad76e959e31dd`.

`Canonical Modal GPU training has NOT been rerun.` Optimizer progress, canonical loss/stability/eval, checkpoints, best checkpoint and weight packaging remain unverified. The new mapping has not been written to Volume in this task.

`READY FOR REVIEW BEFORE GPU RETRY`

Next action: Human reviews the minimal mapping integration and CPU evidence before committing or scheduling a GPU retry.
