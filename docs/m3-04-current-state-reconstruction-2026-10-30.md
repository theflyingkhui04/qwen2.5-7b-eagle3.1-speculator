# M3-04 current-state reconstruction

Audit ngày 30/10/2026, múi giờ Asia/Bangkok. Phạm vi là repository local tại `D:/Document/Four-year/do_an/qwen2.5-7b-eagle3.1-speculator`, đối chiếu GitHub branch trực tiếp và SpecForge đúng SHA. FACT là điều kiểm tra được từ source/artifact; OBSERVATION là kết quả quan sát; INFERENCE là suy luận; HYPOTHESIS chưa được chứng minh; RECOMMENDATION là đề xuất; UNKNOWN là chưa xác minh.

## 1. Executive verdict

**READY WITH RISKS**

FACT: Canonical implementation là `modal_pipeline.py` → `training/train.py` và `training/native_vocab_mapping.py` → CLI `specforge train -c execution.yaml`, với data bridge và bốn config được ghi trong phần 3. Scientific contract vẫn là 2×4, BF16, offline, draft output 16000 trên target vocab 152064. Full CPU suite sau commit A/B đạt 365 passed trong 100,03 giây; compileall PASS. Không phát hiện canonical correctness blocker mới.

OBSERVATION: Identity/provenance và diagnostic tooling đã được commit A/B; tại snapshot trước commit tài liệu C, source closure `e7df71b` ahead hai commit so với remote-tracking ref trước fetch; inventory hiện tại khác inventory profiler được review trước đây. Canonical train ghi source identity vào evidence, không có acceptance comparison với một request đã review. Hàm so sánh identity chỉ được gọi trong launcher profiler lưu dưới ignored `runs/` và tests. Do đó không được nói canonical train đã được một acceptance gate xác thực toàn bộ source hiện tại với một baseline review cũ.

INFERENCE: Có thể chuẩn bị canonical resume với source hiện tại; các rủi ro còn lại là tính tái lập từ Git, phạm vi identity rộng hơn execution closure, tài liệu điều tra còn chỉ dẫn chạy thử nghiệm cũ và chứng cứ resume/Volume chỉ là snapshot lịch sử. Đây là rủi ro quản trị/evidence, chưa phải bằng chứng training sai. Không có quyết định thay protocol nào cần PM chọn trong lượt này. M3-04 chưa DONE.

FACT lịch sử: Trong audit gốc không sửa executable source/config/test, không chạy Modal/GPU/training/diagnostic, không truy cập Volume, không xóa file, không commit/push/PR/Trello. Sau đó commit A/B đã bảo tồn source và tests như phần 2; tại snapshot bàn giao trước commit C chưa push remote. Kết luận khoa học không thay đổi.

## 2. Git baseline

### Snapshot source sau commit A/B, trước commit tài liệu C

| Trường | Trạng thái |
|---|---|
| Execution/source closure trước commit tài liệu C | `e7df71b11893fb57aaab324bb6595750a10894ce` |
| Commit A | `5ce271f` — `fix(m3-04): separate execution identity from worktree provenance` |
| Commit B | `e7df71b` — `chore(m3-04): preserve closed diagnostic tooling and regression evidence` |
| Remote-tracking HEAD trước fetch | `2d79c14a1a5efb118d84f3e477e424e54f962881`; chưa xác minh remote live |
| Ahead / behind remote-tracking ref tại snapshot trước commit C | `2 / 0` |
| A+B | 19 files changed, 5964 insertions, 14 deletions |
| Source/test tại snapshot trước commit C | Không có diff tracked; hai tài liệu này nằm ngoài execution identity và phục vụ reconstruction / PM handoff |
| Kiểm tra | Full CPU suite 365 passed trong 100,03 giây; compileall PASS; diff A+B `--check` PASS |

### Baseline lịch sử tại đầu audit gốc

| Trường | FACT tại đầu audit |
|---|---|
| Local branch | `fix/m3-04-specforge-current-migration` |
| Local HEAD | `2d79c14a1a5efb118d84f3e477e424e54f962881` |
| Local origin ref | `2d79c14a1a5efb118d84f3e477e424e54f962881` |
| Remote branch HEAD trực tiếp | Cùng SHA, xác minh bằng GitHub connector `fetch_commit` với branch ref |
| Ahead / behind local origin ref | `0 / 0` |
| Tracked modified | `modal_pipeline.py`, `tests/test_modal.py` |
| Staged | Không có; `git diff --cached` rỗng |
| Untracked | 269 file; trong đó 233 file dưới `profiler_analysis/` |
| Tổng dirty trước báo cáo | 271 file, liệt kê từng file ở phần 7 |
| Diff tracked | 224 insertions, 14 deletions; 122 dòng diff ở pipeline, 116 dòng thêm ở test |
| Commit HEAD | `fix(m3-04): extend canonical timeout and authenticate full resume` |

FACT lịch sử: Không có SOURCE STATE DIVERGED giữa local HEAD và remote HEAD tại đầu audit gốc. Working tree khi đó khác HEAD. `git ls-remote` qua shell thất bại do kết nối github.com:443; connector GitHub sau đó trả remote commit thành công. Git ownership khác tài khoản sandbox được xử lý bằng `git -c safe.directory=<repo>` riêng từng lệnh; không đổi global config, không fetch/reset/checkout.

Nguồn remote lịch sử: [commit baseline](https://github.com/theflyingkhui04/qwen2.5-7b-eagle3.1-speculator/commit/2d79c14a1a5efb118d84f3e477e424e54f962881). Baseline đã đọc status, branch, rev-parse hai ref, rev-list ahead/behind, log 20 và all refs, diff/stat/cached, tracked/untracked inventory; thêm inventory file ẩn và ignored evidence liên quan.

## 3. Canonical production dependency closure

FACT: `main(step="train")` kiểm tra scope/resume → `execution_source()` → `preflight.remote(mode="features")` → verify runtime/toolchain, resolve target/tokenizer/capture, raw/prepared audit, `feature_paths`, `check_features`, native mapping preparation, resume validation, CLI CPU `--plan` → `train_specforge.remote` → runtime check, mapping seal check, resolved config, resume validation lần hai, bảo vệ output namespace, write execution evidence → `run_specforge_training` → CLI pinned SpecForge. Đây là trace source; không phải execution đã chạy trong lượt audit.

| Path | Reachable from `--step train`? | Tracked? | Identity-covered? | Role |
|---|---|---|---|---|
| `modal_pipeline.py` | Có; import/app registration và train dispatch | Có | Có | Entrypoint, image, CPU/GPU orchestration |
| `training/__init__.py` | Có khi import package | Có | Có | Chỉ docstring, không diagnostic import/side effect |
| `training/train.py` | Có | Có | Có | Config, feature identity, resume guard, CLI/runtime |
| `training/native_vocab_mapping.py` | Có | Có | Có | TRAIN cache mapping và authentication |
| `data/__init__.py` | Có trong preflight data import | Có | Có, commit A | Package data |
| `data/prepare_specforge.py` | Có trong CPU preflight | Có | Có | Raw audit, lossless serialization |
| `training/prepare.py` | Không execute khi train; đọc bytes cho feature contract | Có | Có | Capture bridge, chỉ chạy khi prepare step |
| `configs/train_config.yaml` | Có | Có | Có | Canonical recipe |
| `configs/eagle3.1_qwen2.5_7b.json` | Có | Có | Có | Draft/resolver input |
| `configs/m3_04_dataset.json` | Có | Có | Có | Raw counts/hash |
| `configs/native_vocab_mapping_proof.json` | Có | Có | Có | Accepted TRAIN mapping proof |
| `requirements.txt` | Không execute bởi train; image pip cài checkout trực tiếp | Có | Có | Dependency declaration |
| `training/flex_attention_ab.py` | Không import; upload/hash có | Có, commit B | Có | Explicit diagnostic step |
| `training/flex_compile_mode_ab.py` | Không import; upload/hash có | Có, commit B | Có | Compile-mode branch của Flex diagnostic |
| `training/optimizer_step_ab.py` | Không import; upload/hash có | Có, commit B | Có | Explicit optimizer diagnostic step |
| `training/profiler_stages.py` | Không import; upload/hash có | Có, commit B | Có | Opt-in `python -m` bootstrap |
| `training/short_path_scan.py` | Không import; upload/hash có | Có, commit B | Có | CPU metadata scanner |
| `training/workload_padding_precheck.py` | Không import; upload/hash có | Có, commit B | Có | CPU grouping precheck |
| `data/prepare_hybrid_dataset.py` | Không; không upload | Có | Không | Raw generator lịch sử, không gọi lại |
| `export/package_and_upload.py` | Không; không upload | Có | Không | Native-export packaging riêng |
| `run_all.sh` | Wrapper bên ngoài, không được Python train gọi | Có | Không | Shell entrypoint, không forward cờ resume |
| `tests/**`, `docs/**`, `worklogs/**`, `task-requirements/**` | Không; không upload | Mixed | Không | Test/docs/evidence |
| `runs/**`, `profiler_analysis/**`, root `.diff`/PNG/TXT | Không trong train | Ignored hoặc untracked | Không mặc định | Artifacts; chỉ ba file explicit khi truyền artifact_dir |
| `/opt/SpecForge` | Có trong container | Upstream pin | Không trong local hash dict; SHA/runtime guard riêng | Canonical trainer/loss/eval/checkpoint/backend |

FACT: Không thấy wildcard import, package registry hay dynamic import từ train sang diagnostic. Dynamic `importlib` trong `training/prepare.py` chỉ nạp preparation script upstream khi prepare được gọi. `profiler_stages` có import hook/patch nhưng chỉ cài trong context `NativeStages.installed()` của bootstrap diagnostic; canonical không gọi nó. Hai diagnostic function mới được Modal đăng ký lúc import entrypoint, không dispatch nếu chọn train; body import diagnostic là lazy. Upload không đồng nghĩa execute.

FACT về execution identity:

- `execution_identity(source)` chỉ chọn `branch`, `head`, `source_sha256`; thiếu field gây KeyError. `working_tree` là provenance, mã hóa lossless porcelain-v1-z bytes thành hex.
- Inventory hiện tại gồm pipeline, requirements, hai data bridge files, recursive `training/**/*.py`, recursive tất cả file configs. Path set và digest đều tham gia equality; thêm/xóa Python/config đổi identity. Bytecode và training log/diff không thuộc scope. Sửa source bytes, kể cả LF/CRLF, đổi identity.
- `.diff`, PNG, log, report ngoài inventory không gây false reject chỉ vì đổi status. Nếu đặt artifact vào configs thì có hash vì toàn bộ config directory nằm trong scope; đây là upload scope hiện tại.
- `artifact_dir=None` ở canonical. Chỉ caller profiler explicit mới thêm `future-gpu-command.txt`, `remote-command.txt`, `diagnostic.yaml`. Request chứa baseline không tự hash chính nó. Old request thiếu inventory không được nâng cấp ngầm.
- Không thấy executable local Python được upload thuộc training/data bridge mà thiếu khỏi inventory. Runtime packages, upstream checkout và raw/features/checkpoint có cơ chế identity riêng; local hash không đóng kín dependency distribution/container digest. Runtime versions được verify nhưng dependency resolution không phải toàn bộ lockfile.
- **Acceptance scope:** canonical train chỉ record execution; không gọi `execution_identity` để so sánh expected/current. Profiler launcher trong ignored runs mới assert equality. Không thể lấy test helper pass làm bằng chứng canonical acceptance gate đang chạy.

OBSERVATION lịch sử so với baseline `2d79c14`: AST của `save_json`, `feature_metadata`, `feature_inventory`, `check_features`, `preflight`, `prepare_hidden_states`, `train_specforge`, `smoke_specforge` giống baseline `2d79c14`, gồm decorators. `main` đổi để thêm explicit diagnostic selection; `execution_source` đổi inventory/provenance; thêm helper identity và hai function diagnostic. Nhánh train vẫn dispatch như baseline `2d79c14`.

OBSERVATION: So với `profiler_analysis/execution.json`, pipeline và tất cả config/data/native train source hashes khớp chính xác. `training/short_path_scan.py` khác hash lịch sử, và `training/workload_padding_precheck.py` được thêm sau inventory đó. Vì vậy current full identity khác baseline profiler; không được tái dùng request đó cho current inventory hoặc nói “toàn bộ execution identity không đổi”. Hai thay đổi chỉ thuộc CPU diagnostic và không import trong canonical train. Tests historical hash tái dựng hai helper cũ qua `tests/conftest.py`; đây là assertion về các bytes còn lại, không phải actual source hash hiện tại.

## 4. Scientific contract check

| Contract | Current value | Changed? | Evidence |
|---|---|---|---|
| Raw immutable, train/val/test | 40000 / 3000 / 2000 | Không | FACT: đếm dòng và SHA-256 local cả ba split khớp manifest |
| Giữ mọi turn trước token truncation | `prepared_row` duyệt toàn bộ conversation; validate exact ChatML; không truncate | Không | FACT: data bridge source, adapter tests |
| Target | `Qwen/Qwen2.5-7B-Instruct` | Không | FACT: YAML |
| Target revision | Proof yêu cầu `a09a35458c702b33eeacc393d103063234e8bc28` | Không | FACT: proof JSON, mapping `_context`; preflight resolve model mặc định trước, mismatch fail closed |
| SpecForge | `fddcb4ad37733a821708170df28cac2f7bb6ea05` | Không | FACT: constant/image/requirements/runtime guard |
| Runtime versions | 0.2.0 / Torch 2.13.0 / SGLang 0.5.18 / FlashInfer 0.6.17 / Transformers 5.12.1 | Không trong guard | FACT: `verify_runtime`; UNKNOWN current remote image installation |
| CUDA | Image 13.0.3 devel; NVCC và Torch CUDA 13.0 | Không | FACT: image và CPU toolchain guard |
| Target vocab / draft vocab | 152064 / 16000 | Không | FACT: draft JSON, proof, vocab tests |
| Mapping source | Native-equivalent TRAIN cache, eval reuse | Không | FACT: native mapping code; không prepared mapping tie-order cũ |
| Mapping tensor identity | t2d `(152064,)`, d2t `(16000,)` | Không | FACT: validation/proof; raw tensor byte hashes giữ như phần 6 |
| Draft architecture | Một decoder; fc_norm/norm_output/use_post_norm_feedback=true | Không | FACT: JSON; metadata feedback không phải switch riêng upstream |
| Capture | Resolver ba tầng; historical `[1,13,24]` | Không hardcode | FACT: preflight resolver và manifest lưu sẵn |
| Data mode | Offline train + offline eval | Không | FACT: hidden_states paths và eval path |
| Dtype | bfloat16 | Không | FACT: config/proof |
| Batch / accumulation / effective batch | 2 / 4 / 8 | Không | FACT: YAML, world size 1 |
| Epochs / max length / TTT / LR | 3 / 2048 / 5 / 0.0001 | Không | FACT: YAML |
| Eval / save intervals | 1000 / 1000 completed optimizer updates | Không | FACT: YAML và pinned controller |
| Best metric | `eval/simulated_acc_len`, maximize | Không | FACT: local constant và pinned CheckpointManager default |
| Topology / attention | nnodes=1, nproc=1; flex_attention | Không | FACT: YAML |
| Full feature key | `a113d35ad76e959e31dd` khi resolve đúng snapshot path lịch sử | Không trong reconstructed contract | FACT: tái dựng `feature_paths` với target từ manifest, exact contract equality |
| Smoke isolation | `/smoke/<key>/train,eval`; bounds 64/16 mặc định | Không | FACT: feature_selection/feature_paths |
| Global horizon | 15000; còn 12000 sau step3000 | Không | FACT: YAML không max_steps/total_steps, pinned schedule; historical checkpoint metadata |
| Canonical timeout | 79200 giây (22 giờ) | Đã commit ở `2d79c14` | FACT: decorator, tests |

OBSERVATION: YAML vẫn chứa placeholder `/current`, nhưng canonical preflight luôn thay bằng hash namespace qua `feature_paths` trước `check_features` và trước CLI plan/train. Không dùng `/current` làm identity. Chạy helper trực tiếp với target model ID chưa resolve cho key `43c813b9912d719e22b8`; đây không phải canonical preflight. Dùng đúng snapshot path trong manifest cho key canonical chính xác. Identity gắn absolute resolved snapshot path; nếu mount/cache path khác sẽ fail mapping proof thay vì âm thầm dùng features khác.

FACT: SHA raw local: train `075592a4f9ae92cf942eb767dcd6bf208326ef2732fa5f10915d8795b724f7ca`; val `86d77222894abbb6e92f9e5ea1b22350f24531ea31a75fbf17258634355e331b`; test `8859387d789314f8e97ee21436e725a6ab8aef9570e9df422d4a7251d3240ff9`.

## 5. Historical fixes vs current source

| Fix | Current status | Commit/local | Evidence |
|---|---|---|---|
| SpecForge migration | PRESENT | `70afa83` | Unified CLI, pinned checkout, offline/resolver/adapter; legacy custom training removed |
| Bounded smoke features | PRESENT | `70afa83` | Smoke selection, count bounds, namespace, preparation CLI sample bound |
| CUDA development toolchain | PRESENT | `70afa83` | CUDA devel image, host tools, CUDA_HOME, NVCC guard |
| Draft vocab 16000 | PRESENT | `70afa83` | JSON and vocab-contract tests; no full-vocab loss rewrite |
| Native mapping integration | PRESENT | `63eed80` | Native proof/cache/token counting/persist/reload; `vocab_mapping_path` binding |
| Metadata-only feature preflight | PRESENT | `0309a9b` | No full feature tensor rehash at every training startup |
| Mount alias feature path fix | PRESENT | `3ce1494` | Resolve shared root, reject artifact symlink escape |
| BF16 export dtype compatibility | PRESENT | `3ce1494` | Export accepts native serialized `dtype` or `torch_dtype`, keeps BF16 |
| Canonical timeout extension | PRESENT | `2d79c14` | Train 22h; smoke/preparation remain 6h |
| Full-state explicit resume authentication | PRESENT | `2d79c14` | CPU + execution guard, exact step path, full-state reader |
| Identity/provenance separation | PRESENT | `5ce271f` | New helper, recursive inventory/data init và identity tests |
| Flex/compile/optimizer/profiler diagnostics | PRESENT | `e7df71b` | Explicit noncanonical paths, not production adoption |
| Legacy target_layer=28/custom loss/old launcher docs | SUPERSEDED | Tracked historical docs/worklogs | Current source/AGENTS migration contract thay thế |

FACT lịch sử tại audit gốc: Các commit khi đó xác minh qua `git log --all`, pickaxe `-S draft_vocab_size`, `-S vocab_mapping_path`, `-S resume_from`, `-S execution_identity`, `-G timeout=.*`, commit stats và current source. Các fix gộp trong migration commit không được giả định có commit riêng. Startup-audit “BLOCKER” về prepared tie-order là historical finding đã được native integration giải quyết; không reopen.

## 6. Checkpoint/resume readiness from current source

FACT: Explicit command đã review được source hỗ trợ với `--step train --resume --resume-from <canonical-root>/<run_id>-step3000`. Audit không thực thi lệnh đó. `resume_from` không kèm `resume` bị reject; resume chỉ cho train/full. Nếu chỉ truyền `--resume`, code còn hỗ trợ latest alias, rồi freeze concrete directory. Vì vậy “explicit step3000” là lựa chọn vận hành đã review, không phải yêu cầu duy nhất CLI cho phép.

| Nội dung | Current-source proof | Historical evidence / remaining gap |
|---|---|---|
| Path/run/step | Namespace, resolved directory, regex run-step, state run_id/global_step, strategy/world=1; no smoke/other run | Checkpoint step3000 metadata lưu local; UNKNOWN current Volume |
| Optimizer | Native full reader + non-None optimizer; trainer restores backend after wrap | Historical RAW worker records loaded-state identity; không load full checkpoint trong audit |
| Scheduler / FP32 masters | Native optimizer load restores scheduler and masters when present; scheduler load requires saved scheduler dict | Local preflight không kiểm sâu scheduler/master fields. Missing masters trong upstream có warning fallback từ BF16; CPU mocked resume tests không chứng minh full runtime state |
| RNG | Local requires nonempty RNG; native backend restoration owns actual streams | Historical worker RNG digest; chưa xác minh current rank shard |
| Data position | Pinned trainer restores global_step, epoch, epoch_samples; derives batch and rejects inconsistent epoch_batch | Saved state: step3000, epoch0, samples24000, batch12000, seed42 |
| Horizon | Không đặt max_steps/total_steps ở canonical resolved_config; pinned resolve_total_steps từ full data | 40000 ÷ 2 × 3 ÷ 4 = 15000; còn 15000 − 3000 = 12000; next update3001 |
| Schedule comparison | Native trainer compares effective_total_steps, legacy scheduler horizon if needed, objective/topology/provenance | Không rút horizon xuống 12000, không reset scheduler |
| Mapping | Accepted proof + full TRAIN cache shard hashes ở CPU; execution revalidates seal/tensor hashes không feature scan | Current Volume tensors UNKNOWN |
| Feature/target | Raw contract/capture/revision manifests; mapping proof requires target revision/key | Preflight resolves target revision, mismatch fail closed; current target service không được gọi |
| Output/latest/best | Existing run refused without resume; upstream save/latest/best/rotation; local writes evidence/W&B dir riêng | Historical best/latest evidence tồn tại, current links UNKNOWN |

FACT: Canonical mapping path là `/vol/datasets/m3-04/processed/a113d35ad76e959e31dd/cache/vocab_mapping/native_canonical.pt`; t2d byte hash `2c10e7206a2cbfd763acfc4538967f0b52c3656d2a30da3d7b151ccf73e2fc80`, d2t byte hash `d1678443a756c734089e0792b2487b17b2d5db6ce451b324eae5b8ee78b26dc0`. CPU preflight authenticate cache trên reuse, execution chỉ reload mapping và validate tensor/binding; không trở lại full feature scan. Feature preflight kiểm names/sizes/manifests, không phát hiện same-size tensor corruption; đây là giới hạn đã chủ ý của metadata-only gate.

FACT: Đã đọc source upstream đúng pin qua GitHub connector: [trainer](https://github.com/sgl-project/SpecForge/blob/fddcb4ad37733a821708170df28cac2f7bb6ea05/specforge/training/trainer.py), [checkpoint](https://github.com/sgl-project/SpecForge/blob/fddcb4ad37733a821708170df28cac2f7bb6ea05/specforge/training/checkpoint.py), [schedule](https://github.com/sgl-project/SpecForge/blob/fddcb4ad37733a821708170df28cac2f7bb6ea05/specforge/training/schedule.py), [controller](https://github.com/sgl-project/SpecForge/blob/fddcb4ad37733a821708170df28cac2f7bb6ea05/specforge/training/controller.py). Optimizer source copy nằm ở `profiler_analysis/results/aten_vec/sources/specforge_optimizer.py.txt`. Không dùng current main để diễn giải pin.

INFERENCE: Current source giữ semantics resume đã review. CPU suite không thay thế full checkpoint authentication và optimizer/RNG/scheduler restoration tại runtime. Không claim current durable checkpoint hoặc latest/best link đã được đọc lại trong lượt này.

## 7. Dirty-file classification tại baseline lịch sử

FACT: Bảng dưới liệt kê từng file trong baseline: 2 tracked modified + 269 untracked, không gộp thư mục. Reachability “Không; upload/hash” nghĩa là không execute/import trên train nhưng có ảnh hưởng identity. Mọi recommendation trong bảng là đề xuất, chưa stage/move/delete. “Archive” không có nghĩa được phép xóa bản local cuối cùng.

| Path | Classification | Canonical reachability | Recommendation | Reason |
|---|---|---|---|---|
| `M3-04_CODEX_CONTEXT.md` | obsolete | Không | KEEP LOCAL / ARCHIVE OUTSIDE GIT | Prompt migration cũ; không phải current instructions hoặc runtime result |
| `docs/m3-04-current-state-reconstruction-2026-10-30.md` | canonical-documentation | Không | Commit cùng audit docs | Báo cáo mới của audit sau baseline, ngoài execution inventory |
| `docs/m3-04-optimizer-step-ab.md` | canonical-documentation | Không | Giữ ở commit diagnostic/docs | Report điều tra; chỉ dẫn GPU cũ cần đánh dấu historical trước publication |
| `docs/m3-04-short-path-frequency.md` | scientific-evidence | Không | Giữ ở commit diagnostic/docs | Report điều tra; chỉ dẫn GPU cũ cần đánh dấu historical trước publication |
| `m304-compile-mode-ab.diff` | temporary-patch | Không | SUPERSEDED; delete sau review/bảo tồn successor | Current source/tests successor giữ insight; xem patch audit phần9 |
| `m304-compile-mode-hook.diff` | temporary-patch | Không | SUPERSEDED; delete sau review/bảo tồn successor | Current source/tests successor giữ insight; xem patch audit phần9 |
| `m304-execution-identity-fix.diff` | temporary-patch | Không | SUPERSEDED; delete sau review/bảo tồn successor | Current source/tests successor giữ insight; xem patch audit phần9 |
| `m304-flex-ab-full.diff` | temporary-patch | Không | SUPERSEDED; delete sau review/bảo tồn successor | Current source/tests successor giữ insight; xem patch audit phần9 |
| `m304-flex-ab.diff` | temporary-patch | Không | SUPERSEDED; delete sau review/bảo tồn successor | Current source/tests successor giữ insight; xem patch audit phần9 |
| `m304-flex-length-fix-full.diff` | temporary-patch | Không | SUPERSEDED; delete sau review/bảo tồn successor | Current source/tests successor giữ insight; xem patch audit phần9 |
| `m304-flex-length-fix.diff` | temporary-patch | Không | SUPERSEDED; delete sau review/bảo tồn successor | 0 byte, không evidence |
| `m304-forward-isolation.diff` | temporary-patch | Không | SUPERSEDED; delete sau review/bảo tồn successor | Current source/tests successor giữ insight; xem patch audit phần9 |
| `m304-identity-artifact-gate.diff` | temporary-patch | Không | EXTRACT FIRST | Ignored auth bootstrap chưa tracked |
| `m304-optimizer-step-ab-full.diff` | temporary-patch | Không | SUPERSEDED; delete sau review/bảo tồn successor | Current source/tests successor giữ insight; xem patch audit phần9 |
| `m304-optimizer-step-ab.diff` | temporary-patch | Không | SUPERSEDED; delete sau review/bảo tồn successor | Current source/tests successor giữ insight; xem patch audit phần9 |
| `m304-pcie-diagnostic-policy.diff` | temporary-patch | Không | SUPERSEDED; delete sau review/bảo tồn successor | Current source/tests successor giữ insight; xem patch audit phần9 |
| `m304-profiler-bounded-auth.diff` | temporary-patch | Không | SUPERSEDED; delete sau review/bảo tồn successor | Current source/tests successor giữ insight; xem patch audit phần9 |
| `m304-ttt0-reference.diff` | temporary-patch | Không | SUPERSEDED; delete sau review/bảo tồn successor | Current source/tests successor giữ insight; xem patch audit phần9 |
| `m304_eval_history.txt` | generated-artifact | Không | SAFE TO DELETE AFTER HUMAN REVIEW | 0 byte, không eval evidence |
| `mintty.2026-10-28_07-44-08.png` | generated-artifact | Không | Exclude Git; archive hoặc delete sau review | Đã xem: versions/NVCC/download output lặp structured execution evidence |
| `modal_pipeline.py` | canonical-production | Có | COMMIT NOW, chia hunks A/B | Identity helper và explicit diagnostic dispatch cùng file; canonical bodies giữ nguyên |
| `profiler_analysis/execution.json` | scientific-evidence | Không; không upload/hash | Giữ; small evidence commit riêng | Raw execution/stage/log/trace duy nhất của profiler; giữ hash/provenance |
| `profiler_analysis/profile_rank0.trace.json.gz` | scientific-evidence | Không; không upload/hash | DO NOT DELETE; archive ngoài Git | Raw execution/stage/log/trace duy nhất của profiler; giữ hash/provenance |
| `profiler_analysis/profiler_stages.jsonl` | scientific-evidence | Không; không upload/hash | Giữ; small evidence commit riêng | Raw execution/stage/log/trace duy nhất của profiler; giữ hash/provenance |
| `profiler_analysis/results/all_kernel_contamination.csv` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/all_kernel_names.csv` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/all_kernel_names.sql` | analysis-tooling | Không; không upload/hash | Chọn vào reproducibility commit; archive dependencies | SQL lưu phép truy vấn; giữ để giải thích/tái lập derived number |
| `profiler_analysis/results/all_kernel_names.stderr.txt` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/all_kernels_summary.csv` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/all_mm_context.csv` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/all_mm_context.sql` | analysis-tooling | Không; không upload/hash | Chọn vào reproducibility commit; archive dependencies | SQL lưu phép truy vấn; giữ để giải thích/tái lập derived number |
| `profiler_analysis/results/all_mm_context.stderr.txt` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/analysis.md` | scientific-evidence | Không; không upload/hash | Giữ; small structured evidence/docs commit riêng | Attribution, validation hoặc source identity; stale candidate recommendations không reopen gate |
| `profiler_analysis/results/annotations.csv` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/annotations.sql` | analysis-tooling | Không; không upload/hash | Chọn vào reproducibility commit; archive dependencies | SQL lưu phép truy vấn; giữ để giải thích/tái lập derived number |
| `profiler_analysis/results/annotations.stderr.txt` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/aten_vec/amdahl_candidate.csv` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/aten_vec/aten_vec_analysis.md` | scientific-evidence | Không; không upload/hash | Giữ; small structured evidence/docs commit riêng | Attribution, validation hoặc source identity; stale candidate recommendations không reopen gate |
| `profiler_analysis/results/aten_vec/aten_vec_cpu_ops.csv` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/aten_vec/aten_vec_kernels.csv` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/aten_vec/aten_vec_owner.csv` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/aten_vec/aten_vec_per_kernel.csv` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/aten_vec/aten_vec_per_kernel_detailed.csv` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/aten_vec/aten_vec_phase.csv` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/aten_vec/aten_vec_summary.json` | scientific-evidence | Không; không upload/hash | Giữ; small structured evidence/docs commit riêng | Attribution, validation hoặc source identity; stale candidate recommendations không reopen gate |
| `profiler_analysis/results/aten_vec/candidate_compile_coverage.csv` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/aten_vec/extract_metadata.py` | analysis-tooling | Không; không upload/hash | Chọn vào reproducibility commit; archive dependencies | Offline analysis/query; một số cần Perfetto RPC session và generated inputs |
| `profiler_analysis/results/aten_vec/finalize.py` | analysis-tooling | Không; không upload/hash | Chọn vào reproducibility commit; archive dependencies | Offline analysis/query; một số cần Perfetto RPC session và generated inputs |
| `profiler_analysis/results/aten_vec/kernel_metadata.csv` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/aten_vec/kernel_metadata.sql` | analysis-tooling | Không; không upload/hash | Chọn vào reproducibility commit; archive dependencies | SQL lưu phép truy vấn; giữ để giải thích/tái lập derived number |
| `profiler_analysis/results/aten_vec/kernel_metadata.stderr.txt` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/aten_vec/metadata_keys.csv` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/aten_vec/preliminary.json` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/aten_vec/semantic_operations.csv` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/aten_vec/sources/accumulate_grad.cpp.txt` | fixture | Không; không upload/hash | Archive pinned source cache; exclude production commit | Copied upstream source, ref/hash manifest; dependency của analysis, không production vendoring |
| `profiler_analysis/results/aten_vec/sources/embedding.cpp.txt` | fixture | Không; không upload/hash | Archive pinned source cache; exclude production commit | Copied upstream source, ref/hash manifest; dependency của analysis, không production vendoring |
| `profiler_analysis/results/aten_vec/sources/engine.cpp.txt` | fixture | Không; không upload/hash | Archive pinned source cache; exclude production commit | Copied upstream source, ref/hash manifest; dependency của analysis, không production vendoring |
| `profiler_analysis/results/aten_vec/sources/input_buffer.cpp.txt` | fixture | Không; không upload/hash | Archive pinned source cache; exclude production commit | Copied upstream source, ref/hash manifest; dependency của analysis, không production vendoring |
| `profiler_analysis/results/aten_vec/sources/specforge_optimizer.py.txt` | fixture | Không; không upload/hash | Archive pinned source cache; exclude production commit | Copied upstream source, ref/hash manifest; dependency của analysis, không production vendoring |
| `profiler_analysis/results/aten_vec/validation.json` | scientific-evidence | Không; không upload/hash | Giữ; small structured evidence/docs commit riêng | Attribution, validation hoặc source identity; stale candidate recommendations không reopen gate |
| `profiler_analysis/results/aten_vec/write_report.py` | analysis-tooling | Không; không upload/hash | Chọn vào reproducibility commit; archive dependencies | Offline analysis/query; một số cần Perfetto RPC session và generated inputs |
| `profiler_analysis/results/aten_vec_attribution.csv` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/aten_vec_attribution_context.csv` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/aten_vec_attribution_kernels.csv` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/attribute.py` | analysis-tooling | Không; không upload/hash | Chọn vào reproducibility commit; archive dependencies | Offline analysis/query; một số cần Perfetto RPC session và generated inputs |
| `profiler_analysis/results/attribution_coverage.json` | scientific-evidence | Không; không upload/hash | Giữ; small structured evidence/docs commit riêng | Attribution, validation hoặc source identity; stale candidate recommendations không reopen gate |
| `profiler_analysis/results/bottleneck_table.csv` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/bottlenecks.csv` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/compiled_ancestors.csv` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/compiled_ancestors.sql` | analysis-tooling | Không; không upload/hash | Chọn vào reproducibility commit; archive dependencies | SQL lưu phép truy vấn; giữ để giải thích/tái lập derived number |
| `profiler_analysis/results/compiled_ancestors.stderr.txt` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/compiled_context.csv` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/compiled_context.sql` | analysis-tooling | Không; không upload/hash | Chọn vào reproducibility commit; archive dependencies | SQL lưu phép truy vấn; giữ để giải thích/tái lập derived number |
| `profiler_analysis/results/compiled_context.stderr.txt` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/correlation_setup.csv` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/correlation_setup.sql` | analysis-tooling | Không; không upload/hash | Chọn vào reproducibility commit; archive dependencies | SQL lưu phép truy vấn; giữ để giải thích/tái lập derived number |
| `profiler_analysis/results/correlation_setup.stderr.txt` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/cpu_correlations_check.csv` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/cpu_correlations_check.sql` | analysis-tooling | Không; không upload/hash | Chọn vào reproducibility commit; archive dependencies | SQL lưu phép truy vấn; giữ để giải thích/tái lập derived number |
| `profiler_analysis/results/cpu_correlations_check.stderr.txt` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/cpu_duplicate_ids.csv` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/cpu_duplicate_ids.sql` | analysis-tooling | Không; không upload/hash | Chọn vào reproducibility commit; archive dependencies | SQL lưu phép truy vấn; giữ để giải thích/tái lập derived number |
| `profiler_analysis/results/cpu_duplicate_ids.stderr.txt` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/crosscheck.py` | analysis-tooling | Không; không upload/hash | Chọn vào reproducibility commit; archive dependencies | Offline analysis/query; một số cần Perfetto RPC session và generated inputs |
| `profiler_analysis/results/extract.py` | analysis-tooling | Không; không upload/hash | Chọn vào reproducibility commit; archive dependencies | Offline analysis/query; một số cần Perfetto RPC session và generated inputs |
| `profiler_analysis/results/family_contamination.csv` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/flow_schema.csv` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/flow_schema.sql` | analysis-tooling | Không; không upload/hash | Chọn vào reproducibility commit; archive dependencies | SQL lưu phép truy vấn; giữ để giải thích/tái lập derived number |
| `profiler_analysis/results/flow_schema.stderr.txt` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/gemm_contamination.csv` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/gemm_cpu_op.csv` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/gemm_module_attribution.csv` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/gemm_module_totals.csv` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/gemm_per_kernel.csv` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/gemm_phase_attribution.csv` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/gemm_runtime_control.csv` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/gemm_runtime_control.sql` | analysis-tooling | Không; không upload/hash | Chọn vào reproducibility commit; archive dependencies | SQL lưu phép truy vấn; giữ để giải thích/tái lập derived number |
| `profiler_analysis/results/gemm_runtime_control.stderr.txt` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/gemm_sequence_threads.csv` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/gemm_sequence_threads.sql` | analysis-tooling | Không; không upload/hash | Chọn vào reproducibility commit; archive dependencies | SQL lưu phép truy vấn; giữ để giải thích/tái lập derived number |
| `profiler_analysis/results/gemm_sequence_threads.stderr.txt` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/host_ancestors.csv` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/host_ancestors.sql` | analysis-tooling | Không; không upload/hash | Chọn vào reproducibility commit; archive dependencies | SQL lưu phép truy vấn; giữ để giải thích/tái lập derived number |
| `profiler_analysis/results/host_ancestors.stderr.txt` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/host_args.csv` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/host_args.sql` | analysis-tooling | Không; không upload/hash | Chọn vào reproducibility commit; archive dependencies | SQL lưu phép truy vấn; giữ để giải thích/tái lập derived number |
| `profiler_analysis/results/host_args.stderr.txt` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/identifiable_attention_kernels.csv` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/import_stats.csv` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/import_stats.sql` | analysis-tooling | Không; không upload/hash | Chọn vào reproducibility commit; archive dependencies | SQL lưu phép truy vấn; giữ để giải thích/tái lập derived number |
| `profiler_analysis/results/import_stats.stderr.txt` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/kernel_attribution_detail.csv` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/kernel_family.csv` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/kernel_time_ceilings.csv` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/kernel_to_cpu_op.csv` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/kernel_to_cpu_op.sql` | analysis-tooling | Không; không upload/hash | Chọn vào reproducibility commit; archive dependencies | SQL lưu phép truy vấn; giữ để giải thích/tái lập derived number |
| `profiler_analysis/results/kernel_to_cpu_op.stderr.txt` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/meaningful_spans.csv` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/meaningful_spans.sql` | analysis-tooling | Không; không upload/hash | Chọn vào reproducibility commit; archive dependencies | SQL lưu phép truy vấn; giữ để giải thích/tái lập derived number |
| `profiler_analysis/results/meaningful_spans.stderr.txt` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/mlp/adjacent_norm_cost.csv` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/mlp/amdahl_ceilings.csv` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/mlp/analysis_console.json` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/mlp/analyze_mlp.py` | analysis-tooling | Không; không upload/hash | Chọn vào reproducibility commit; archive dependencies | Offline analysis/query; một số cần Perfetto RPC session và generated inputs |
| `profiler_analysis/results/mlp/authentication.json` | scientific-evidence | Không; không upload/hash | Giữ; small structured evidence/docs commit riêng | Attribution, validation hoặc source identity; stale candidate recommendations không reopen gate |
| `profiler_analysis/results/mlp/backward_nodes.csv` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/mlp/backward_nodes.sql` | analysis-tooling | Không; không upload/hash | Chọn vào reproducibility commit; archive dependencies | SQL lưu phép truy vấn; giữ để giải thích/tái lập derived number |
| `profiler_analysis/results/mlp/backward_nodes.stderr.txt` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/mlp/backward_role_evidence.csv` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/mlp/compile_coverage.csv` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/mlp/compile_regions.csv` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/mlp/compile_regions.sql` | analysis-tooling | Không; không upload/hash | Chọn vào reproducibility commit; archive dependencies | SQL lưu phép truy vấn; giữ để giải thích/tái lập derived number |
| `profiler_analysis/results/mlp/compile_regions.stderr.txt` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/mlp/completion_audit.json` | scientific-evidence | Không; không upload/hash | Giữ; small structured evidence/docs commit riêng | Attribution, validation hoặc source identity; stale candidate recommendations không reopen gate |
| `profiler_analysis/results/mlp/extract_mlp.py` | analysis-tooling | Không; không upload/hash | Chọn vào reproducibility commit; archive dependencies | Offline analysis/query; một số cần Perfetto RPC session và generated inputs |
| `profiler_analysis/results/mlp/forward_layout_ops.csv` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/mlp/metadata_key_inventory.csv` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/mlp/microbatch_cost.csv` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/mlp/mlp_analysis.md` | scientific-evidence | Không; không upload/hash | Giữ; small structured evidence/docs commit riêng | Attribution, validation hoặc source identity; stale candidate recommendations không reopen gate |
| `profiler_analysis/results/mlp/mlp_ancestor_examples.md` | scientific-evidence | Không; không upload/hash | Giữ; small structured evidence/docs commit riêng | Attribution, validation hoặc source identity; stale candidate recommendations không reopen gate |
| `profiler_analysis/results/mlp/mlp_forward_ops.csv` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/mlp/mlp_forward_ops.sql` | analysis-tooling | Không; không upload/hash | Chọn vào reproducibility commit; archive dependencies | SQL lưu phép truy vấn; giữ để giải thích/tái lập derived number |
| `profiler_analysis/results/mlp/mlp_forward_ops.stderr.txt` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/mlp/mlp_gemm_shapes.csv` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/mlp/mlp_join_validation.csv` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/mlp/mlp_join_validation.sql` | analysis-tooling | Không; không upload/hash | Chọn vào reproducibility commit; archive dependencies | SQL lưu phép truy vấn; giữ để giải thích/tái lập derived number |
| `profiler_analysis/results/mlp/mlp_join_validation.stderr.txt` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/mlp/mlp_launch_buckets.csv` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/mlp/mlp_launch_buckets.sql` | analysis-tooling | Không; không upload/hash | Chọn vào reproducibility commit; archive dependencies | SQL lưu phép truy vấn; giữ để giải thích/tái lập derived number |
| `profiler_analysis/results/mlp/mlp_launch_buckets.stderr.txt` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/mlp/mlp_launch_metadata.csv` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/mlp/mlp_module_metadata.csv` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/mlp/mlp_module_metadata.sql` | analysis-tooling | Không; không upload/hash | Chọn vào reproducibility commit; archive dependencies | SQL lưu phép truy vấn; giữ để giải thích/tái lập derived number |
| `profiler_analysis/results/mlp/mlp_module_metadata.stderr.txt` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/mlp/mlp_per_kernel.csv` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/mlp/mlp_phase_attribution.csv` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/mlp/mlp_projection_attribution.csv` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/mlp/mlp_shape_role_summary.csv` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/mlp/mlp_summary.json` | scientific-evidence | Không; không upload/hash | Giữ; small structured evidence/docs commit riêng | Attribution, validation hoặc source identity; stale candidate recommendations không reopen gate |
| `profiler_analysis/results/mlp/module_mapping_evidence.csv` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/mlp/operator_arg_keys.csv` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/mlp/operator_arg_keys.sql` | analysis-tooling | Không; không upload/hash | Chọn vào reproducibility commit; archive dependencies | SQL lưu phép truy vấn; giữ để giải thích/tái lập derived number |
| `profiler_analysis/results/mlp/operator_arg_keys.stderr.txt` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/mlp/pointwise_compile_coverage.csv` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/mlp/query_mlp.py` | analysis-tooling | Không; không upload/hash | Chọn vào reproducibility commit; archive dependencies | Offline analysis/query; một số cần Perfetto RPC session và generated inputs |
| `profiler_analysis/results/mlp/relevant_metadata.csv` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/mlp/relevant_metadata.sql` | analysis-tooling | Không; không upload/hash | Chọn vào reproducibility commit; archive dependencies | SQL lưu phép truy vấn; giữ để giải thích/tái lập derived number |
| `profiler_analysis/results/mlp/relevant_metadata.stderr.txt` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/mlp/selection.csv` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/mlp/selection.sql` | analysis-tooling | Không; không upload/hash | Chọn vào reproducibility commit; archive dependencies | SQL lưu phép truy vấn; giữ để giải thích/tái lập derived number |
| `profiler_analysis/results/mlp/selection.stderr.txt` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/mlp/source_manifest.json` | scientific-evidence | Không; không upload/hash | Giữ; small structured evidence/docs commit riêng | Attribution, validation hoặc source identity; stale candidate recommendations không reopen gate |
| `profiler_analysis/results/mlp/sources/eagle3_adapters.py.txt` | fixture | Không; không upload/hash | Archive pinned source cache; exclude production commit | Copied upstream source, ref/hash manifest; dependency của analysis, không production vendoring |
| `profiler_analysis/results/mlp/sources/eagle3_model.py.txt` | fixture | Không; không upload/hash | Archive pinned source cache; exclude production commit | Copied upstream source, ref/hash manifest; dependency của analysis, không production vendoring |
| `profiler_analysis/results/mlp/sources/llama3_eagle.py.txt` | fixture | Không; không upload/hash | Archive pinned source cache; exclude production commit | Copied upstream source, ref/hash manifest; dependency của analysis, không production vendoring |
| `profiler_analysis/results/mlp/sources/specforge_utils.py.txt` | fixture | Không; không upload/hash | Archive pinned source cache; exclude production commit | Copied upstream source, ref/hash manifest; dependency của analysis, không production vendoring |
| `profiler_analysis/results/mlp/sources/torch_FunctionsManual.cpp.txt` | fixture | Không; không upload/hash | Archive pinned source cache; exclude production commit | Copied upstream source, ref/hash manifest; dependency của analysis, không production vendoring |
| `profiler_analysis/results/mlp/sources/torch_Linear.cpp.txt` | fixture | Không; không upload/hash | Archive pinned source cache; exclude production commit | Copied upstream source, ref/hash manifest; dependency của analysis, không production vendoring |
| `profiler_analysis/results/mlp/sources/torch_derivatives.yaml.txt` | fixture | Không; không upload/hash | Archive pinned source cache; exclude production commit | Copied upstream source, ref/hash manifest; dependency của analysis, không production vendoring |
| `profiler_analysis/results/mlp/sources/torch_gen_autograd_functions.py.txt` | fixture | Không; không upload/hash | Archive pinned source cache; exclude production commit | Copied upstream source, ref/hash manifest; dependency của analysis, không production vendoring |
| `profiler_analysis/results/mlp/sources/torch_inductor_config.py.txt` | fixture | Không; không upload/hash | Archive pinned source cache; exclude production commit | Copied upstream source, ref/hash manifest; dependency của analysis, không production vendoring |
| `profiler_analysis/results/mlp/sources/torch_load_derivatives.py.txt` | fixture | Không; không upload/hash | Archive pinned source cache; exclude production commit | Copied upstream source, ref/hash manifest; dependency của analysis, không production vendoring |
| `profiler_analysis/results/mlp/sources/torch_profiler.py.txt` | fixture | Không; không upload/hash | Archive pinned source cache; exclude production commit | Copied upstream source, ref/hash manifest; dependency của analysis, không production vendoring |
| `profiler_analysis/results/mlp/surrounding_cost.csv` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/mlp/surrounding_kernels.csv` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/mlp/validate_mlp.py` | analysis-tooling | Không; không upload/hash | Chọn vào reproducibility commit; archive dependencies | Offline analysis/query; một số cần Perfetto RPC session và generated inputs |
| `profiler_analysis/results/mlp/validation.json` | scientific-evidence | Không; không upload/hash | Giữ; small structured evidence/docs commit riêng | Attribution, validation hoặc source identity; stale candidate recommendations không reopen gate |
| `profiler_analysis/results/mlp/write_report.py` | analysis-tooling | Không; không upload/hash | Chọn vào reproducibility commit; archive dependencies | Offline analysis/query; một số cần Perfetto RPC session và generated inputs |
| `profiler_analysis/results/mm_backward_children.csv` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/mm_backward_children.sql` | analysis-tooling | Không; không upload/hash | Chọn vào reproducibility commit; archive dependencies | SQL lưu phép truy vấn; giữ để giải thích/tái lập derived number |
| `profiler_analysis/results/mm_backward_children.stderr.txt` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/mm_context_args.csv` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/mm_context_args.sql` | analysis-tooling | Không; không upload/hash | Chọn vào reproducibility commit; archive dependencies | SQL lưu phép truy vấn; giữ để giải thích/tái lập derived number |
| `profiler_analysis/results/mm_context_args.stderr.txt` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/optimizer_kernels.csv` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/query.py` | analysis-tooling | Không; không upload/hash | Chọn vào reproducibility commit; archive dependencies | Offline analysis/query; một số cần Perfetto RPC session và generated inputs |
| `profiler_analysis/results/reduction_owners.csv` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/report.py` | analysis-tooling | Không; không upload/hash | Chọn vào reproducibility commit; archive dependencies | Offline analysis/query; một số cần Perfetto RPC session và generated inputs |
| `profiler_analysis/results/report_console.json` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/runtime_links.csv` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/runtime_links.sql` | analysis-tooling | Không; không upload/hash | Chọn vào reproducibility commit; archive dependencies | SQL lưu phép truy vấn; giữ để giải thích/tái lập derived number |
| `profiler_analysis/results/runtime_links.stderr.txt` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/sample_args.csv` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/sample_args.sql` | analysis-tooling | Không; không upload/hash | Chọn vào reproducibility commit; archive dependencies | SQL lưu phép truy vấn; giữ để giải thích/tái lập derived number |
| `profiler_analysis/results/sample_args.stderr.txt` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/sample_stack.csv` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/sample_stack.sql` | analysis-tooling | Không; không upload/hash | Chọn vào reproducibility commit; archive dependencies | SQL lưu phép truy vấn; giữ để giải thích/tái lập derived number |
| `profiler_analysis/results/sample_stack.stderr.txt` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/setup.csv` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/setup.sql` | analysis-tooling | Không; không upload/hash | Chọn vào reproducibility commit; archive dependencies | SQL lưu phép truy vấn; giữ để giải thích/tái lập derived number |
| `profiler_analysis/results/setup.stderr.txt` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/summarize.py` | analysis-tooling | Không; không upload/hash | Chọn vào reproducibility commit; archive dependencies | Offline analysis/query; một số cần Perfetto RPC session và generated inputs |
| `profiler_analysis/results/summary.json` | scientific-evidence | Không; không upload/hash | Giữ; small structured evidence/docs commit riêng | Attribution, validation hoặc source identity; stale candidate recommendations không reopen gate |
| `profiler_analysis/results/summary_console.json` | scientific-evidence | Không; không upload/hash | Giữ; small structured evidence/docs commit riêng | Attribution, validation hoặc source identity; stale candidate recommendations không reopen gate |
| `profiler_analysis/results/sync_copy.csv` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/sync_copy.sql` | analysis-tooling | Không; không upload/hash | Chọn vào reproducibility commit; archive dependencies | SQL lưu phép truy vấn; giữ để giải thích/tái lập derived number |
| `profiler_analysis/results/sync_copy.stderr.txt` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/top_kernels.csv` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/trace_identity.json` | scientific-evidence | Không; không upload/hash | Giữ; small structured evidence/docs commit riêng | Attribution, validation hoặc source identity; stale candidate recommendations không reopen gate |
| `profiler_analysis/results/track_threads.csv` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/track_threads.sql` | analysis-tooling | Không; không upload/hash | Chọn vào reproducibility commit; archive dependencies | SQL lưu phép truy vấn; giữ để giải thích/tái lập derived number |
| `profiler_analysis/results/track_threads.stderr.txt` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/tracks.csv` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/tracks.sql` | analysis-tooling | Không; không upload/hash | Chọn vào reproducibility commit; archive dependencies | SQL lưu phép truy vấn; giữ để giải thích/tái lập derived number |
| `profiler_analysis/results/tracks.stderr.txt` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/triton_reduction/analyze.py` | analysis-tooling | Không; không upload/hash | Chọn vào reproducibility commit; archive dependencies | Offline analysis/query; một số cần Perfetto RPC session và generated inputs |
| `profiler_analysis/results/triton_reduction/compile_contamination.csv` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/triton_reduction/dominant_compiled_evidence.csv` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/triton_reduction/extract_metadata.py` | analysis-tooling | Không; không upload/hash | Chọn vào reproducibility commit; archive dependencies | Offline analysis/query; một số cần Perfetto RPC session và generated inputs |
| `profiler_analysis/results/triton_reduction/kernel_metadata.csv` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/triton_reduction/kernel_metadata.sql` | analysis-tooling | Không; không upload/hash | Chọn vào reproducibility commit; archive dependencies | SQL lưu phép truy vấn; giữ để giải thích/tái lập derived number |
| `profiler_analysis/results/triton_reduction/kernel_metadata.stderr.txt` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/triton_reduction/metadata_keys.csv` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/triton_reduction/per_kernel.csv` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/triton_reduction/prior_hashes.json` | scientific-evidence | Không; không upload/hash | Giữ; small structured evidence/docs commit riêng | Attribution, validation hoặc source identity; stale candidate recommendations không reopen gate |
| `profiler_analysis/results/triton_reduction/report.py` | analysis-tooling | Không; không upload/hash | Chọn vào reproducibility commit; archive dependencies | Offline analysis/query; một số cần Perfetto RPC session và generated inputs |
| `profiler_analysis/results/triton_reduction/semantic_operations.csv` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/triton_reduction/source_manifest.json` | scientific-evidence | Không; không upload/hash | Giữ; small structured evidence/docs commit riêng | Attribution, validation hoặc source identity; stale candidate recommendations không reopen gate |
| `profiler_analysis/results/triton_reduction/sources/flex_attention.py.txt` | fixture | Không; không upload/hash | Archive pinned source cache; exclude production commit | Copied upstream source, ref/hash manifest; dependency của analysis, không production vendoring |
| `profiler_analysis/results/triton_reduction/triton_reduction_analysis.md` | scientific-evidence | Không; không upload/hash | Giữ; small structured evidence/docs commit riêng | Attribution, validation hoặc source identity; stale candidate recommendations không reopen gate |
| `profiler_analysis/results/triton_reduction/triton_reduction_cpu_ops.csv` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/triton_reduction/triton_reduction_kernels.csv` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/triton_reduction/triton_reduction_owner.csv` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/triton_reduction/triton_reduction_phase.csv` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/triton_reduction/triton_reduction_summary.json` | scientific-evidence | Không; không upload/hash | Giữ; small structured evidence/docs commit riêng | Attribution, validation hoặc source identity; stale candidate recommendations không reopen gate |
| `profiler_analysis/results/triton_reduction/validation.json` | scientific-evidence | Không; không upload/hash | Giữ; small structured evidence/docs commit riêng | Attribution, validation hoặc source identity; stale candidate recommendations không reopen gate |
| `profiler_analysis/results/triton_reduction_attribution.csv` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/triton_reduction_attribution_context.csv` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/triton_reduction_attribution_kernels.csv` | generated-artifact | Không; không upload/hash | Archive trước; exclude production commit | Derived CSV/JSON/console/query output; chưa chứng minh tái lập độc lập |
| `profiler_analysis/results/validate.py` | analysis-tooling | Không; không upload/hash | Chọn vào reproducibility commit; archive dependencies | Offline analysis/query; một số cần Perfetto RPC session và generated inputs |
| `profiler_analysis/results/validation.json` | scientific-evidence | Không; không upload/hash | Giữ; small structured evidence/docs commit riêng | Attribution, validation hoặc source identity; stale candidate recommendations không reopen gate |
| `profiler_analysis/results/wait_extract.py` | analysis-tooling | Không; không upload/hash | Chọn vào reproducibility commit; archive dependencies | Offline analysis/query; một số cần Perfetto RPC session và generated inputs |
| `profiler_analysis/trace_processor` | analysis-tooling | Không; không upload/hash | Chọn vào reproducibility commit; archive dependencies | Offline analysis/query; một số cần Perfetto RPC session và generated inputs |
| `profiler_analysis/training.log` | scientific-evidence | Không; không upload/hash | DO NOT DELETE; archive ngoài Git | Raw execution/stage/log/trace duy nhất của profiler; giữ hash/provenance |
| `tests/conftest.py` | regression-test | Không | Giữ; commit A/B theo dependency | CPU guards về identity, isolation, resume, sampler hoặc padding; conftest chỉ test-time |
| `tests/fixtures/legacy_execution_source.py.txt` | fixture | Không | Giữ cùng regression tests | Legacy identity / pinned metadata / failed-window inputs, không runtime evidence đầy đủ |
| `tests/fixtures/specforge_short_path_metadata.py.txt` | fixture | Không | Giữ cùng regression tests | Legacy identity / pinned metadata / failed-window inputs, không runtime evidence đầy đủ |
| `tests/fixtures/step3000_failed_window.json` | fixture | Không | Giữ cùng regression tests | Legacy identity / pinned metadata / failed-window inputs, không runtime evidence đầy đủ |
| `tests/test_execution_identity.py` | regression-test | Không | Giữ; commit A/B theo dependency | CPU guards về identity, isolation, resume, sampler hoặc padding; conftest chỉ test-time |
| `tests/test_flex_attention_ab.py` | regression-test | Không | Giữ; commit A/B theo dependency | CPU guards về identity, isolation, resume, sampler hoặc padding; conftest chỉ test-time |
| `tests/test_flex_compile_mode_ab.py` | regression-test | Không | Giữ; commit A/B theo dependency | CPU guards về identity, isolation, resume, sampler hoặc padding; conftest chỉ test-time |
| `tests/test_modal.py` | regression-test | Không | Giữ; commit A/B theo dependency | CPU guards về identity, isolation, resume, sampler hoặc padding; conftest chỉ test-time |
| `tests/test_optimizer_step_ab.py` | regression-test | Không | Giữ; commit A/B theo dependency | CPU guards về identity, isolation, resume, sampler hoặc padding; conftest chỉ test-time |
| `tests/test_profiler_stages.py` | regression-test | Không | Giữ; commit A/B theo dependency | CPU guards về identity, isolation, resume, sampler hoặc padding; conftest chỉ test-time |
| `tests/test_short_path_scan.py` | regression-test | Không | Giữ; commit A/B theo dependency | CPU guards về identity, isolation, resume, sampler hoặc padding; conftest chỉ test-time |
| `tests/test_workload_padding_precheck.py` | regression-test | Không | Giữ; commit A/B theo dependency | CPU guards về identity, isolation, resume, sampler hoặc padding; conftest chỉ test-time |
| `training/flex_attention_ab.py` | diagnostic-tooling | Không; upload/hash | SEPARATE DIAGNOSTIC COMMIT | Opt-in investigation; test cùng tên phụ thuộc; xóa làm mất tái lập và đổi identity |
| `training/flex_compile_mode_ab.py` | diagnostic-tooling | Không; upload/hash | SEPARATE DIAGNOSTIC COMMIT | Opt-in investigation; test cùng tên phụ thuộc; xóa làm mất tái lập và đổi identity |
| `training/optimizer_step_ab.py` | diagnostic-tooling | Không; upload/hash | SEPARATE DIAGNOSTIC COMMIT | Opt-in investigation; test cùng tên phụ thuộc; xóa làm mất tái lập và đổi identity |
| `training/profiler_stages.py` | diagnostic-tooling | Không; upload/hash | SEPARATE DIAGNOSTIC COMMIT | Opt-in investigation; test cùng tên phụ thuộc; xóa làm mất tái lập và đổi identity |
| `training/short_path_scan.py` | diagnostic-tooling | Không; upload/hash | SEPARATE DIAGNOSTIC COMMIT | Opt-in investigation; test cùng tên phụ thuộc; xóa làm mất tái lập và đổi identity |
| `training/workload_padding_precheck.py` | diagnostic-tooling | Không; upload/hash | SEPARATE DIAGNOSTIC COMMIT | Opt-in investigation; test cùng tên phụ thuộc; xóa làm mất tái lập và đổi identity |

FACT lịch sử: Báo cáo này là file mới do audit tạo sau baseline; classification `canonical-documentation`, không train reachable, không identity-covered; đề xuất commit cùng documentation audit. Tổng dirty sau khi thêm báo cáo là 272 file; hàng của báo cáo cũng đã được liệt kê trong bảng.

## 8. Diagnostic investigation closure

| Diagnostic | Original question | Result và gate status | Reusable value / tests / Git treatment |
|---|---|---|---|
| `flex_attention_ab` | Raw so với compiled short FlexAttention; numerical sensitivity, masks, TTT0 reference | FACT: Code có raw deltas, fixed screens, per-instance rebinding; không canonical adoption. UNKNOWN: raw kết quả Flex/compile đầy đủ không thấy trong tracked/untracked root artifacts; không suy pass từ code | RECOMMENDATION: separate diagnostic commit, giữ module + test_flex_attention_ab + dispatch tests; không xóa vì mất cách tái dựng điều tra |
| `flex_compile_mode_ab` | Default compile so với max-autotune/dynamic và reference sensitivity | FACT: Opt-in compile_mode_ab chỉ với Flex step; tốc độ không quyết định numerical classification. Gate optimization đã đóng theo yêu cầu lượt này; không chạy lại | Separate diagnostic commit với Flex dependency; test_flex_compile_mode_ab giữ isolation/reference policy |
| `optimizer_step_ab` | Compiled short attention có giúp native optimizer window 2 warm-up +5 measured không? | FACT: Saved benchmark BENCHMARK INVALID: 28 microbatch đều long, không substitution; không có B speedup chứng minh | Separate diagnostic commit; tests giữ native horizon, boundaries, state/cache/data identity, checkpoint protection |
| `short_path_scan` | Phân phối short-path trong remaining workload và tại sao benchmark không exercise nhánh? | FACT: scan 146/48000, 145/12000; gate tần suất đã đóng | Giữ CPU scanner + tests + metadata fixtures trong diagnostic/repro commit. Ignored analyze.py và input manifests cần archive cùng |
| `profiler_stages` | Stage nào gây chờ startup/profiler; bảo vệ bounded resume/protected artifacts? | FACT: Opt-in import hooks; guard total15000/max3003, 2×4, report_to none; execution snapshot exit0, canonical_unchanged true. Không có canonical instrumentation | Separate diagnostic commit; test_profiler_stages và historical fixture quan trọng; bootstrap/request trong ignored runs phải archive |
| `workload_padding_precheck` | Cùng 8 samples/update, 4×2 có giảm padding/short-path không? | FACT: 12000 windows cùng ordered samples; padded slots tăng33,096310%; không production speedup/correctness | Separate diagnostic/repro commit cùng scanner; giữ tests và structured output; gate 4×2 đóng |
| `profiler_analysis` | Attribution GEMM/MLP/ATEN_VEC/reductions và candidate còn khả thi? | OBSERVATION: tổng12302 kernels, GEMM52,0066% aggregate kernel time; later reduction report NO ACTIONABLE TRITON_REDUCTION TARGET. Older MLP/ATEN_VEC report còn hypothesis chưa phải accepted optimization | Archive raw + analysis chain; chọn scripts/SQL/small structured outputs vào reproducibility commit sau khi ghi dependencies. Không commit cache upstream/large dump vào production |

FACT: Final investigation instruction là `NO FURTHER ACTIONABLE CANDIDATE`; giữ gate đóng. Không tìm thấy một artifact tổng hợp độc lập mang đúng chuỗi kết luận cuối cho mọi nhánh trong repo; reduction closure hỗ trợ dừng nhánh cuối, còn reports cũ đề xuất candidate tiếp theo. Không biến stale recommendation thành task mới. HYPOTHESIS cũ MLP compile, embedding-add và reduction chỉ là lịch sử; không có canonical code adoption tương ứng.

## 9. Evidence preservation audit

| Claim | Artifact/source thực tế | Preservation decision |
|---|---|---|
| Raw splits nguyên vẹn | Local ignored data/raw/m3_04_drive + tracked manifest | FACT: hashes/counts kiểm lại; DO NOT DELETE raw; không regenerate |
| Mapping native và tie-order | `configs/native_vocab_mapping_proof.json`, helper/tests, native_train_counts fixture, offline startup/vocab fixtures | DO NOT DELETE; regression fixtures giữ exact pin/selection |
| Step3000 và resume position | Ignored `runs/short-path-scan-20261028/checkpoint_header.bin`, checkpoint_metadata.json, benchmark.json, listings, remote_verification.json | DO NOT DELETE; header không phải full checkpoint tensor proof |
| Short-path146/48000 và first28 window | scan.json, microbatches.jsonl, analyze.py, cache shards, pinned_launch.py/pinned_data_utils.py, ordering_runtime_audit.json, step3000_failed_window fixture, doc | DO NOT DELETE ignored scan bundle; fixture chỉ small window không thay whole scan |
| 4×2 padding +33,0963% | Ignored workload-padding-precheck JSON, source/input hashes, source script/tests | DO NOT DELETE large output duy nhất trước compact summary + archived input bundle; không cần đưa 41,8MB vào production commit |
| Profiler successful và canonical unchanged | profiler_analysis/execution.json, profiler_stages.jsonl, training.log, raw trace | DO NOT DELETE local bản cuối; structured evidence giữ hardware-only comparability |
| Numeric profiler attribution | Raw trace + CSVs/SQL + validation/summary JSON + analysis scripts + source manifests | Archive trọn dependency chain trước giảm generated tables; kernel percentage không phải canonical step speedup |
| Source review/authentication | Ignored runs profiler requests/commands/configs, regenerate-bounded-profiler-identity.py, local identity tests | DO NOT DELETE only executable auth bootstrap; artifact-specific gate diff cần extract/archive trước |
| Toolchain versions trong PNG | mintty screenshot lặp versions/NVCC và “Finished downloading”; execution.json cũng có versions/toolchain | OBSERVATION: ảnh không chứa metric canonical độc nhất; exclude Git, archive tùy nhu cầu |
| Eval history TXT | `m304_eval_history.txt` có0 byte | FACT: không thể chứng minh eval; safe delete after review |
| Legacy training success | logs/training_full.log, logs/training_summary_report.md, worklogs/tasks | Historical only; không chứng minh M3-04 migration DONE |

FACT: Raw trace SHA-256 vừa kiểm lại = `08e8ffff3f28a16662fb809f601ada68f2fd406dc86b96465ade22fedcedb469`, khớp stored validations. MLP source_manifest có11 local source copies và hash đều khớp; SpecForge ref đúng pin, PyTorch ref v2.13.0. Triton manifest tham chiếu hai source copy nằm ở MLP sources và một flex_attention copy tại triton_reduction/sources. Không xem chúng là production vendored dependencies.

OBSERVATION: Nhiều script query/report gọi Perfetto RPC `127.0.0.1:9001` và dựa warm parsed session cùng generated CSVs. Vì vậy “deterministic analysis script” không đồng nghĩa chạy độc lập chỉ với trace. Giữ trace_processor wrapper, scripts/SQL, outputs cần thiết và ghi bootstrap/tool version trước khi bỏ cache; chưa chạy lại pipeline analysis trong audit. Upstream cached source có thể tải lại từ refs/manifests, nhưng giữ archive nếu script/report đọc chúng trực tiếp. Không commit hàng nghìn dòng cache vào canonical fix.

### Audit từng temporary patch

FACT: Đã đọc target/hunk additions của cả14 patch; dùng `git apply --reverse --check` chỉ để kiểm tra, không apply. Reverse-check pass chứng minh patch hiện có trong source byte context; reverse-check fail ở patch cũ được đối chiếu với helper/test successor, không tự coi fail là “chưa tích hợp”.

| Patch | Stage / current coverage | Classification | Evidence duy nhất / insight |
|---|---|---|---|
| `m304-flex-ab.diff` | Initial explicit dispatch + tests, sau đó thêm optimizer/compile options | SUPERSEDED | Current pipeline và dispatch tests giữ behavior |
| `m304-flex-ab-full.diff` | Initial replay implementation; thay bằng length/reference/isolation fixes | SUPERSEDED | Current module/tests giữ selection, deltas, isolation; snapshot cũ không phải accepted implementation |
| `m304-flex-length-fix.diff` | File0 byte | SUPERSEDED | Không chứa evidence |
| `m304-flex-length-fix-full.diff` | Feature length/padding selection; newer module + boundary tests | SUPERSEDED | Test feature selection/padding còn; không dùng raw length cũ |
| `m304-forward-isolation.diff` | Per-instance forward/mask isolation trước reference additions | SUPERSEDED | Current isolated_forward/run_forward_isolation + tests thay signatures cũ |
| `m304-ttt0-reference.diff` | Explicit TTT0 reference, trước PCIe/compile hooks | SUPERSEDED | Current dense/reference functions và tests giữ numerical investigation |
| `m304-pcie-diagnostic-policy.diff` | Diagnostic hardware recording/policy | SUPERSEDED | Current hardware policy/tests ghi PCIe noncanonical, identity-only comparability |
| `m304-compile-mode-ab.diff` | Opt-in mode module/test/dispatch | SUPERSEDED | Mọi added line module/tests còn; dispatch expanded thêm optimizer |
| `m304-compile-mode-hook.diff` | Integrate compile experiment vào replay | SUPERSEDED | Reverse-check pass |
| `m304-optimizer-step-ab.diff` | Explicit optimizer dispatch tests | SUPERSEDED | Reverse-check pass |
| `m304-optimizer-step-ab-full.diff` | Full native benchmark + tests + doc | SUPERSEDED | Reverse-check pass |
| `m304-profiler-bounded-auth.diff` | Stage instrumentation/authentication; old direct whole-source hash test | SUPERSEDED | Module added lines còn; test chuyển sang historical_modal_source để tách identity helpers, current identity tested riêng |
| `m304-execution-identity-fix.diff` | Identity/provenance + recursive inventory + historical tests | SUPERSEDED | Reverse-check pass tại audit gốc; successor identity/provenance đã được bảo tồn trong `5ce271f` |
| `m304-identity-artifact-gate.diff` | Sửa auth bootstrap dưới ignored runs | EXTRACT FIRST | Reverse-check pass và test có kiểm command, nhưng full bootstrap chỉ còn trong ignored run; archive/extract executable command trước bỏ patch |

RECOMMENDATION: SUPERSEDED có nghĩa không còn insight/code duy nhất trong patch hiện tại; chỉ delete sau khi source/tests/docs successor đã được commit hoặc archive đáng tin cậy. Không xóa đồng thời patch và untracked successor chưa bảo tồn.

## 10. Kiểm thử tại audit gốc

| Command | Exact result |
|---|---|
| `python -m compileall modal_pipeline.py training tests data export` | Exit0; syntax pass |
| `python -m pytest tests/ -q` với `CUDA_VISIBLE_DEVICES=''` | `354 passed, 1 skipped in 205.94s (0:03:25)`; exit0 |
| `python -m pytest tests/test_profiler_stages.py::test_stage_path_rejects_canonical_root_and_symlink_escape -q -rs` | `1 skipped in 37.13s`; `SKIPPED [1] tests/test_profiler_stages.py:56: OS does not permit directory symlinks`; chỉ chạy riêng để lấy skip reason |
| `git -c safe.directory=<repo> diff --check` | Exit0; chỉ warning LF sẽ thành CRLF trong pipeline, không whitespace error |

FACT: Không có GPU/Modal-dependent skip; sole skip là quyền symlink của OS trong test profiler path escape. Modal dispatch được mocked, hardware failure-path được mocked, các source/helper tests dùng CPU. Kết quả này thuộc worktree tại audit gốc, không lấy historical276/342; kết quả sau commit A/B là 365 passed trong 100,03 giây. Test runtime SpecForge một số dùng fixture/AST/mocks; không xác nhận full remote runtime hay serving.

## 11. Ranh giới commit đề xuất tại audit gốc

RECOMMENDATION lịch sử: Chia theo hunks và dependencies; không `git add modal_pipeline.py` toàn bộ vào canonical commit. Không có source patch hay staging nào được thực hiện trong audit.

### Commit A — canonical correctness và execution identity

Giữ phần `execution_identity`/`execution_source` trong pipeline và `tests/test_execution_identity.py`. Giữ report audit này để phân định recording và acceptance semantics. `tests/conftest.py`, fixture legacy_execution_source và các test adaptations trong test_profiler_stages/test_short_path_scan chỉ đi cùng các tests historical tương ứng: hoặc đã có ở baseline commit diagnostic trước, hoặc chia hunks sang commit B. Không tạo commit A khiến tests import modules chưa được đưa vào Git.

FACT: Tests identity coverage dùng actual extracted helper functions; current upload closure bao phủ training modules đang có. Một test optionally đọc ignored profiler PLAN và skip nếu không có; clone sạch không có full auth bootstrap. Ranh giới review cần ghi rõ dependency đó. Không thêm acceptance gate canonical trong lượt audit.

Commit message gợi ý: `fix(m3-04): tách execution identity khỏi working-tree provenance`.

### Commit B — diagnostic/repro tooling tách riêng

Giữ sáu training diagnostic modules; diagnostic imports/functions/step dispatch hunks của pipeline; diagnostic additions ở tests/test_modal; sáu test files tương ứng; shared conftest và ba fixtures; hai diagnostic docs. Có thể tách B1 Flex/compile/optimizer và B2 profiler/CPU scan nếu giữ import/fixture dependencies đầy đủ. Hash lịch sử vẫn giữ nguyên baseline; không cập nhật hash để che source drift.

Trước commit documentation, đánh dấu recommendations GPU cũ trong optimizer/short-path reports là historical và gate đã đóng, cùng ghi tổng hợp closure từ report này; audit này chỉ đề xuất, chưa sửa các docs cũ. Docs optimizer “READY FOR BOUNDED ... RUN” không được coi next action hiện tại.

Commit message gợi ý: `chore(m3-04): bảo tồn diagnostic tooling và regression evidence đã đóng`.

### Commit C — scientific evidence nhỏ, có dependencies tái lập

Có thể giữ `execution.json`, stage JSONL, source manifests, summary/validation JSON và reports cuối; scripts/SQL cần thiết cho reported numbers. Không stage toàn bộ profiler_analysis/results theo thư mục: nhiều report/query cần session và intermediate CSV. Compact summary của scan/padding và consolidated closure là phần cần extract từ ignored runs trước publication; chưa có trong tracked branch.

Commit message gợi ý: `docs(m3-04): bảo tồn evidence và trạng thái đóng điều tra hiệu năng`.

### Do not commit vào production fix

14 `.diff`, PNG, empty eval TXT, `M3-04_CODEX_CONTEXT.md` (prompt migration lịch sử, stale task directions), raw compressed trace, large training log, downloaded trace_processor, copied upstream source cache và bulk generated CSV/stderr/console dumps. Archive evidence cần thiết; không coi “do not commit” là “safe delete”. Không đổi `.gitignore` hàng loạt chỉ để status đẹp.

FACT lịch sử: Tại baseline gốc, hai file tracked dirty chứa cả identity fix lẫn diagnostic integration; commit A/B đã tách và bảo tồn các phần này. Sáu diagnostic modules không executable trên train nhưng hiện trong source hash/upload; bất kỳ tách/xóa/move chúng đều đổi identity và phải review baseline mới. Commit đổi HEAD cũng đổi identity theo thiết kế.

## 12. Files safe to delete after review

RECOMMENDATION: Có thể xóa sau human review và sau bảo tồn successor: 13 patch đánh dấu SUPERSEDED ở phần9, `m304_eval_history.txt`0 byte. Bản screenshot không có evidence độc nhất, có thể archive hoặc xóa sau review vì versions/toolchain đã có structured evidence. Các empty `*.stderr.txt` không mang claim độc nhất; có thể bỏ khỏi Git nhưng giữ audit package nếu cần.

RECOMMENDATION: Không đưa `m304-identity-artifact-gate.diff`, raw trace, ignored scan bundle, workload-padding JSON duy nhất hay intermediate analysis dependency vào danh sách delete ngay. Với copied upstream source, chỉ loại khỏi production commit; archive/cache removal cần đảm bảo script reproducibility và manifest trước. Không có deletion nào được thực hiện.

## 13. Remaining gaps before M3-04 DONE

| Phạm vi | Điều còn thiếu |
|---|---|
| Code readiness | FACT: CPU checks pass; source paths hiểu rõ. Cần review hai commit đã tạo và ghi nhận source identity cho lần chạy được duyệt; code không cần optimization mới |
| Execution readiness | UNKNOWN current Volume checkpoint/mapping/manifests/links và installed runtime; preflight đã có guards để kiểm khi lượt execution được phép. Cần full-state restore thực tế đúng step3000 và exact source snapshot |
| Canonical GPU training | Evidence updates3001–15000, đủ3 epochs, finite/stable logs, no unreported protocol drift, durable final state; audit không tìm thấy evidence completion đó |
| Eval | Canonical eval trên3000 validation features, metrics/history đúng completed updates, gồm eval/simulated_acc_len; empty eval TXT không phải evidence |
| Best checkpoint | Verified canonical best link + best_meta metric/step/value + selected weight artifact; historical links không chứng minh final best |
| Resume persistence | Durable full optimizer/scheduler/FP32 masters/RNG/data counters; xác minh reload tiếp tục đúng next step/schedule; CPU mocks và header-only metadata không đủ |
| Export | Native SpecForge export từ selected best, actual BF16 tensors/config/mapping packaging; current packaging unit tests chỉ code compatibility |
| Serving compatibility | Thử load đúng exported artifact trong pinned serving runtime vLLM/SGLang và kiểm output/acceptance theo protocol được duyệt; MODEL_CARD không claim đã pass |
| Report evidence | Immutable run/config/source/runtime/data/checkpoint IDs, W&B run/metrics, eval/best/export hashes, preserved raw evidence + reproducible derived summaries; docs legacy cần phân biệt historical DONE với M3-04 DONE |

FACT: Worklogs/task-requirements cũ ghi DONE cho các task implementation trước migration, target_layer28, CE+SmoothL1 hoặc launcher cũ. Những câu đó không override current source/AGENTS và không nghiệm thu training M3-04. Không sửa lịch sử để ép thành current result.

UNKNOWN: Actual balance/credit không được truy cập hoặc suy đoán. Historical planning12000 updates × supplied4,5074s/update ≈15,025 GPU hours chỉ là ESTIMATE dựa input lịch sử chưa có raw canonical timing độc lập; không phải scientific result hoặc quote current published price. Không đưa budget thành blocker kỹ thuật đã chứng minh.

## 14. Recommended next action

RECOMMENDATION: Preserve canonical 2×4 configuration and wait for sufficient execution budget before running the explicit step3000 canonical resume.
