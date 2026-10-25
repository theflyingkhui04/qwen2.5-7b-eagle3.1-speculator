# Migration M3-04 và cổng Modal CPU plan

Ngày 25/10/2026. Phạm vi: readiness implementation; chưa thực thi Modal, GPU,
hidden-state extraction, training, upload hoặc benchmark. Không phải M3-04 DONE.

## Quyết định và thay đổi

PM đã duyệt thay single-layer capture bằng EAGLE3 low/mid/high, offline train/eval,
BF16 và giữ mọi lượt raw trước truncation token. AGENTS.md đã được cập nhật đúng
quyết định mới. Target và hyperparameters không đổi.

SpecForge được ghim tại `fddcb4ad37733a821708170df28cac2f7bb6ea05`, version 0.2.0.
Python ≥3.11; Torch 2.13.0, Transformers 5.12.1, SGLang 0.5.18 do metadata upstream
quản lý. Image Modal lưu pip freeze để ghi lại transitive dependency resolution;
chưa build image nên chưa có version thực tế của các dependency không được upstream ghim.

Draft chuyển sang architecture đăng ký `LlamaForCausalLMEagle3`, model_type llama,
`fc_norm: true`, `norm_output: true`, một decoder layer. PM đã chốt
draft_vocab_size=16000, giữ vocab_size=152064 của target/tokenizer, theo recipe
Qwen2.5-7B EAGLE3 tại SHA đã ghim và để sửa lỗi migration full-vocab draft.
Rope/token IDs dùng recipe Qwen2.5
upstream. `use_post_norm_feedback` là metadata; feedback normalized thực sự được
thực hiện bởi `norm_output`, không có switch upstream riêng cho metadata đó.

Resolver nguyên bản đã chạy bằng kiểm tra CPU tách qua AST với draft migrated
và target depth 28: kết quả [1, 13, 24]. Production không hardcode IDs:
Modal dùng `resolve_offline_capture` trên target config thực và ghi kết quả.

## Nguồn đã đối chiếu

Tất cả đường SpecForge sau dùng cùng SHA bất biến:

- [Metadata và dependencies](https://github.com/sgl-project/SpecForge/blob/fddcb4ad37733a821708170df28cac2f7bb6ea05/pyproject.toml)
- [Typed schema](https://github.com/sgl-project/SpecForge/blob/fddcb4ad37733a821708170df28cac2f7bb6ea05/specforge/config/schema.py)
- [Resolver capture layers](https://github.com/sgl-project/SpecForge/blob/fddcb4ad37733a821708170df28cac2f7bb6ea05/specforge/algorithms/model_providers.py#L190)
- [Draft FC và normalized feedback](https://github.com/sgl-project/SpecForge/blob/fddcb4ad37733a821708170df28cac2f7bb6ea05/specforge/modeling/draft/llama3_eagle.py#L1576)
- [Canonical preparation](https://github.com/sgl-project/SpecForge/blob/fddcb4ad37733a821708170df28cac2f7bb6ea05/scripts/prepare_hidden_states.py)
- [Parser loại/cắt lượt ở raw mode](https://github.com/sgl-project/SpecForge/blob/fddcb4ad37733a821708170df28cac2f7bb6ea05/specforge/data/parse.py#L212)
- [Loader làm mất text](https://github.com/sgl-project/SpecForge/blob/fddcb4ad37733a821708170df28cac2f7bb6ea05/specforge/utils.py)
- [Qwen template registry](https://github.com/sgl-project/SpecForge/blob/fddcb4ad37733a821708170df28cac2f7bb6ea05/specforge/data/template.py)
- [Objective EAGLE3](https://github.com/sgl-project/SpecForge/blob/fddcb4ad37733a821708170df28cac2f7bb6ea05/specforge/algorithms/eagle3/model.py)
- [BF16 optimizer](https://github.com/sgl-project/SpecForge/blob/fddcb4ad37733a821708170df28cac2f7bb6ea05/specforge/optimizer.py)
- [Best/checkpoint rotation](https://github.com/sgl-project/SpecForge/blob/fddcb4ad37733a821708170df28cac2f7bb6ea05/specforge/training/checkpoint.py)
- [Target config](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/config.json)
- [Target tokenizer template](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/tokenizer_config.json)
- [Modal required_keys](https://modal.com/docs/sdk/py/latest/Secret)
- [W&B thuộc CoreWeave, API keys giữ nguyên](https://www.coreweave.com/companies/weights-and-biases)

## Adapter và cầu nối loader

Native raw parser sẽ thay đổi 3724 mẫu train (3755 lượt), 272 mẫu val (278 lượt);
test có 143 mẫu bị ảnh hưởng (143 lượt), chỉ được audit.
Các số này được đo trước tokenization bằng chính hàm parser upstream.

Adapter giữ nguyên ID, role tương ứng, nội dung và thứ tự mọi lượt.
Dùng tokenizer target render ChatML với system prompt của template qwen upstream;
so sánh toàn bộ chuỗi với serialization giữ nguyên từng message. Không sửa Unicode,
strip whitespace, merge message hoặc thêm tokenization policy.

Không thể giải quyết lỗi loader bằng file-only adapter: canonical script luôn gọi
`safe_conversations_generator`, hàm đó chỉ xuất conversations/tools, làm mất text.
Cầu nối project-local nạp nguyên script bằng importlib, thay duy nhất binding loader
trong module, rồi gọi main với `--is-preformatted`. Không sửa source upstream,
không custom extractor/evaluator. Regression kiểm tra bridge không sửa script nguồn.

Tất cả raw hash/count được kiểm tra fail-fast. Prepared output deterministic,
exclusive create, nằm ngoài raw. Namespace feature bao gồm hash adapter/bridge/draft,
SpecForge SHA và snapshot target bất biến. Feature manifest lưu hash từng .ckpt;
full train/eval phải có đủ 40000/3000 feature trước khi full train. Test không được capture.

## Runtime và checkpoint

Training dùng `specforge train -c <resolved.yaml>`. Chỉ feature preparation dùng
torchrun để thiết lập distributed environment cho canonical script.
Legacy launcher, W&B pre-init và checkpoint pruning thủ công đã được bỏ.
Checkpoint rotation/best/full resume do upstream quản lý.

Default objective là soft-target log-softmax distillation, không có SmoothL1.
AdamW trên FP32 master copies; cosine/warmup dùng mặc định upstream.
Eval/save interval 1000 completed optimizer updates; max_checkpoints 2, bảo vệ best.
Metric chọn best duy nhất: eval/simulated_acc_len; artifact <run_id>-best.

Modal full run giữ run_id m3-04-qwen2.5-7b-eagle3.1. Smoke có suffix riêng,
max_steps 2, eval/save mỗi step; không đổi TTT, LR, dtype hay batch/accumulation.
Resume dùng training.resume_from tới <run_id>-latest, không weights-only warm start.
Mặc định không ghi đè run đã có checkpoint; không có auto export/upload.

Native W&B project qwen2.5-7b-eagle3.1-speculator; secret wandb-secret yêu cầu
WANDB_API_KEY. Không đọc/in key. Không thêm CoreWeave auth.
Execution evidence và W&B logs nằm trong Volume. Ghi repo SHA/branch/source hashes,
resolved YAML/hash, target revision, draft, raw/feature hashes, package versions,
GPU/CUDA, Modal call ID, timestamp và best/latest paths. W&B run ID/metrics thực
sẽ nằm trong native W&B logs; chưa có bằng chứng execution.

## Kiểm chứng và giới hạn

Kiểm tra schema độc lập đã dùng nguyên schema.py và runtime/contracts.py của SHA
đã ghim trong namespace tạm, chỉ cần Pydantic/PyYAML. Config offline/world size 1
được chấp nhận; unknown field, eval không có interval và online train + offline
eval bị từ chối. Đây không phải chạy toàn bộ SpecForge CLI trên Windows.

Local tests kiểm tra toàn bộ 45000 raw rows bằng reference ChatML renderer;
count/hash và mọi lượt đều giữ nguyên. Tokenizer target thật sẽ được audit trên
toàn bộ snapshot trong Modal CPU plan; chưa được chạy tại thời điểm migration.

Lệnh targeted ở README.md: 33 passed, 0 failed, 0 skipped.
Lệnh full CPU `python -m pytest tests/ -v`: 45 passed, 0 failed, 0 skipped.
Modal definitions import thành công với SDK 1.6.2 mà không khởi chạy remote jobs.
Không coi unit tests là bằng chứng GPU readiness.

## Lệnh tiếp theo và bằng chứng mong đợi

```powershell
modal run modal_pipeline.py --step plan
```

CPU function, cùng image với GPU, không cấp A100. Kiểm tra installed SHA/versions,
raw Volume, target snapshot metadata, resolved architecture/capture, audit mọi lượt
bằng tokenizer thật, rồi chạy specforge --plan. Evidence lưu resolved.yaml,
evidence.json, runtime.freeze.txt, plan.log tại /vol/runs/m3-04/<timestamp-unique>/.

RUN STATUS: NOT RUN.

Sau khi review CPU plan và được ủy quyền riêng:

```powershell
modal run modal_pipeline.py --step prepare_hidden_states --feature-scope smoke --train-samples 64 --eval-samples 16
modal run modal_pipeline.py --step smoke
```

DO NOT RUN trong migration. Feature preparation cũng cần GPU và phải hoàn tất
trước smoke. Chưa chứng minh image build, CLI plan Linux, feature tensor layout,
numerical stability, forward/backward, optimizer update, một W&B run có metric,
checkpoint persistence, eval/best, hoặc export/serving compatibility.

Commit message gợi ý, không tự commit:
`fix: migrate M3-04 to pinned SpecForge offline EAGLE3.1 runtime`

## Bổ sung bounded feature smoke

Người dùng đã cung cấp CPU_PLAN_PASS tại /vol/runs/m3-04/20261025T114512Z-ee45bdc0,
capture layers [1, 13, 24]. Đây là bằng chứng của phiên bản trước bổ sung bounded scope;
không phải bằng chứng GPU extraction hoặc runtime của thay đổi bounded.

Scope smoke mặc định chọn 64 train/16 eval bằng upstream --num-samples trên prefix
prepared data trước canonical shuffle. Không thay raw, full split hoặc extractor.
Giữ mọi lượt của mẫu được chọn trước truncation token. Không dùng test để capture.

Smoke path: /vol/hidden_states/m3-04/smoke/<contract-hash>/train/ và /eval/.
Smoke hash bổ sung scope, hai sample limits, target revision, capture strategy/layers
vào identity hiện có (SHA upstream, draft, adapter, bridge, max length, template).
Mỗi manifest giữ raw counts 40000/3000, selected counts 64/16 và produced count
của split tương ứng. Checker kiểm tra đúng số lượng và từ chối namespace khác scope.

Full preparation vẫn dùng --step prepare_hidden_states không thêm flags.
Full contract không thêm các field scope/selected-count; vẫn yêu cầu đủ 40000/3000
và dùng namespace train/<hash>, eval/<hash>. Hash vẫn phản ánh source bridge hiện tại
như trước để tránh tái sử dụng feature khác implementation.

--step smoke mặc định chỉ dùng scope smoke 64/16. Nếu preparation dùng giới hạn khác,
truyền cùng --train-samples/--eval-samples khi smoke train. Full train từ chối smoke scope.
Smoke optimizer vẫn max_steps 2, eval/save interval 1; không đổi model/dtype/LR/TTT/batch.
Không sử dụng 16 eval samples làm bằng chứng chất lượng mô hình.

## Sửa môi trường CUDA cho FlashInfer JIT

Bằng chứng GPU do người dùng cung cấp: bounded train prefix 64 mẫu đã được xử lý,
nhưng FlashInfer JIT lỗi vì không có nvcc và /usr/local/cuda. Image debian_slim
chỉ cài Python packages không cung cấp CUDA development toolkit hệ thống.

Đổi base sang nvidia/cuda:13.0.3-cudnn-devel-ubuntu24.04, add_python 3.11;
CUDA_HOME=/usr/local/cuda, PATH có bin và nvvm/bin; cài build-essential/ninja-build.
CUDA development base cùng dòng với
[Dockerfile SGLang v0.5.18](https://github.com/sgl-project/sglang/blob/v0.5.18/docker/Dockerfile).
[Metadata phiên bản đó](https://github.com/sgl-project/sglang/blob/v0.5.18/python/pyproject.toml)
ghim FlashInfer 0.6.17 và Torch 2.13.0.
[FlashInfer JIT](https://github.com/flashinfer-ai/flashinfer/blob/v0.6.17/flashinfer/jit/cpp_ext.py)
tìm CUDA_HOME/CUDA_PATH hoặc which nvcc.
[Modal](https://modal.com/docs/guide/cuda) khuyến nghị development base cho compiler/JIT.

CPU preflight và runtime checks ghi SpecForge SHA/version, SGLang/FlashInfer versions,
which nvcc, nvcc --version, CUDA_HOME và torch.version.cuda. Kiểm tra headers,
gcc/g++/make/ninja và cùng major/minor CUDA 13.0 trước khi tải model/extraction.
Không gọi GPU, compile JIT hay in secrets trong check. CPU plan giữ nguyên logic;
evidence.json bổ sung cuda_toolchain trong versions, log có diagnostics khi lỗi.

Bridge gọi guard toolchain rõ ràng trước canonical preparation. Bridge source hash
vì vậy đổi theo identity hiện có; retry dùng namespace hash mới, không xóa/ghi đè
namespace thất bại cũ. Không thay sample selection, capture, backend hoặc protocol.
Image build và compiler/JIT trên Modal chưa được chạy lại trong lượt sửa này.

## PM chốt draft vocabulary 16000

PM decision: draft_vocab_size=16000.
Reason: pinned Qwen2.5-7B EAGLE3 reference + migration bug correction.
[Recipe tại SHA đã ghim](https://github.com/sgl-project/SpecForge/blob/fddcb4ad37733a821708170df28cac2f7bb6ea05/configs/qwen2.5-7b-eagle3.json)
giữ target/tokenizer vocab_size=152064 nhưng draft output vocab=16000.
Giá trị full-vocab draft do migration trước đó thêm vào là lỗi, không phải yêu cầu PM.
Smoke runtime do người dùng cung cấp đã lỗi trước optimizer update: loss nhận V=152064,
next_power_of_2(V)=262144 vượt MAX_FUSED_SIZE=131072. Head 16000 dùng block 16384.
Không sửa SpecForge, Triton, loss, SHA, BF16 hoặc các hyperparameters để xử lý lỗi này.

Pinned resolver giữ giá trị explicit 16000. Constructor upstream tạo embedding
152064 entries và lm_head Linear(3584,16000). Kiểm thử dùng excerpts nguyên bản
của pinned resolver/constructor, thực thi lm_head PyTorch thật trên meta device;
decoder/init nặng được stub, không chạy full SpecForge hoặc cấp GPU trên Windows.

Mapping canonical: t2d bool dài 152064, chọn 16000 token; d2t có 16000 offsets,
target_id=draft_id+d2t[draft_id]. Offline trainer đếm token có loss mask từ
data.hidden_states_path (TRAIN), không từ eval_hidden_states_path.
Config không chỉ định vocab_mapping_path: native trainer tự dựng/cache train mapping
và dùng chung cho eval. Không tải lại mapping full-vocab cũ hoặc chọn vocab từ eval.
Regression chạy nguyên pinned mapping/assembly excerpts với CPU tensors, kiểm tra
size, offsets, tính deterministic và loại trừ token chỉ xuất hiện trong eval.

Hidden-state tensors về thuật toán không phụ thuộc draft output vocabulary:
capture vẫn là input_ids/loss_mask và target aux/final hidden states.
Tuy nhiên draft config hash thuộc identity; thay vocab làm namespace mới.
check_features hiện yêu cầu contract bằng nhau và không có đường adoption/reuse
có kiểm tra tensor layout/provenance sang identity khác. Không sửa manifest hoặc
artifacts cũ để vượt check. Không thêm framework migration artifacts cho một config fix.

Quyết định vận hành bảo thủ: BOUNDED PREP MUST BE RERUN với 64 train/16 eval tại
namespace mới. Đây là giới hạn đường orchestration hiện có, không phải khẳng định
các tensor cũ sai. Raw và artifacts cũ giữ nguyên. Mapping 16000 được native tooling
dựng lại từ TRAIN; không dùng mapping full-vocab hoặc mapping của eval để train.
Chưa truy cập/validate artifacts Volume, chưa chạy Modal/GPU/preparation/smoke trong lượt fix.
Kiểm chứng lượt sửa vocab: targeted 33 passed; test identity sau hiệu chỉnh có
1 passed riêng; full CPU suite 78 passed, 0 failed, 0 skipped; git diff --check PASS.
