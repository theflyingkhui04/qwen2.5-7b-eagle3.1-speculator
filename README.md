# Qwen2.5-7B EAGLE-3.1 speculator

Repository huấn luyện draft cho Lane 4 của đề tài Hardware-Aware Speculative Decoding.
CPU Modal plan đã pass theo evidence người dùng cung cấp. Smoke đã vào training nhưng
lỗi tại loss đầu tiên do draft vocab full; config đã được sửa theo PM, chưa chạy lại.
Không dùng kết quả lịch sử để khẳng định convergence, speedup hay tương thích vLLM hiện tại.

## Hợp đồng đã được PM duyệt

- Target: `Qwen/Qwen2.5-7B-Instruct`; BF16; 1 decoder layer.
- SpecForge: `fddcb4ad37733a821708170df28cac2f7bb6ea05`, phiên bản 0.2.0.
- Draft: `configs/eagle3.1_qwen2.5_7b.json`, architecture `LlamaForCausalLMEagle3`.
- PM chốt `vocab_size=152064` cho target/tokenizer và `draft_vocab_size=16000`
  cho draft lm_head, theo recipe Qwen2.5-7B EAGLE3 đã ghim. Đây là sửa lỗi migration;
  không thay target tokenizer hoặc loss kernel.
- `fc_norm: true`, `norm_output: true`; feedback đã chuẩn hóa do `norm_output` thực hiện.
  `use_post_norm_feedback: true` được giữ làm metadata, không phải switch riêng trong upstream.
- Capture ba tầng low/mid/high do resolver upstream xác định từ target config.
  Với target có 28 layers, resolver đã kiểm tra trả `[1, 13, 24]`; code không hardcode các ID này.
- Offline train và offline eval. Best metric: `eval/simulated_acc_len`;
  artifact canonical: `<run_id>-best`, newest complete checkpoint: `<run_id>-latest`.
- 3 epochs, microbatch 2, accumulation 4, effective batch 8 trên một GPU;
  learning rate 1e-4, max length 2048, TTT 5.
- Eval/save mỗi 1000 optimizer updates; rotation native giữ 2 checkpoint mới nhất và bảo vệ best.

CLI production là `specforge train -c <resolved.yaml>`. SpecForge quản lý loss,
optimizer, checkpoint và một W&B tracker. Objective mặc định là distillation bằng
soft-target log-softmax loss qua các bước TTT; không có SmoothL1 trong objective này.
AdamW dùng FP32 master parameters, warmup/cosine theo mặc định upstream.
Mô tả CE + SmoothL1 trong worklogs là lịch sử.

## Dữ liệu bất biến và adapter

Raw đã có trong Volume `qwen-eagle3-volume`, gắn tại `/vol`:

```text
/vol/datasets/m3-04/raw/m3_04_drive/train.jsonl  40000 mẫu
/vol/datasets/m3-04/raw/m3_04_drive/val.jsonl     3000 mẫu
/vol/datasets/m3-04/raw/m3_04_drive/test.jsonl    2000 mẫu
```

Hash canonical nằm trong `configs/m3_04_dataset.json`. Không tải lại, upload lại,
resplit hoặc sửa raw. Test split chỉ được kiểm tra/serialize; không dùng để train/eval.

`data/prepare_specforge.py` giữ ID, thứ tự và nội dung mọi lượt, ánh xạ role
human → user, gpt → assistant, rồi dùng tokenizer của target để render Qwen ChatML.
Adapter kiểm tra toàn bộ chuỗi render có giữ nguyên mọi lượt, kể cả assistant ở đầu
hoặc assistant nối tiếp. Không tokenization, truncation hay filtering trong adapter.
System prompt dùng đúng template qwen của SpecForge: "You are a helpful assistant."

Loader JSONL của revision đã ghim loại trường text. `training/prepare.py` nạp
`scripts/prepare_hidden_states.py` và thay duy nhất binding loader trong module bằng
loader project-local giữ text. Không sửa/fork source upstream. Đường preformatted
bỏ qua bước parser loại/cắt lượt. Tokenization, truncation 2048 token, shuffle seed
42 của preparation upstream, mask và feature capture vẫn thuộc canonical tooling.

Artifacts dùng namespace theo hash của SpecForge SHA, draft, adapter/bridge,
target snapshot, dtype, max length và raw manifest:

```text
/vol/datasets/m3-04/processed/<key>/
/vol/hidden_states/m3-04/train/<key>/
/vol/hidden_states/m3-04/eval/<key>/
/vol/checkpoints/qwen2.5-7b-eagle3.1/
/vol/runs/m3-04/<timestamp-unique>/
```

File mới được tạo exclusive; feature path đã tồn tại sẽ bị từ chối.
Manifest lưu số mẫu, hash file prepared và từng feature. Raw được mở chỉ đọc.

## Môi trường và cổng CPU plan

Windows chỉ dùng Git, sửa code, kiểm thử CPU nhẹ và Modal CLI.
Không cài requirements training vào Windows. Image Modal dùng
`nvidia/cuda:13.0.3-cudnn-devel-ubuntu24.04` với Python 3.11, CUDA_HOME=/usr/local/cuda,
CUDA bin/nvvm trong PATH, build-essential và ninja-build cho FlashInfer JIT; checkout
đúng SHA rồi cài SpecForge từ checkout; dependency Torch/Transformers/SGLang do upstream quản lý.
Không dùng Vast.ai hoặc runtime CUDA thứ hai cho M3-04.

Lệnh tiếp theo dành cho người dùng, chưa được thực thi trong migration:

```powershell
modal run modal_pipeline.py --step plan
```

Đây là function CPU, dùng cùng image với GPU. Nó kiểm tra revision/package versions,
NVCC path/version, CUDA_HOME, host tools và torch.version.cuda (cùng CUDA 13.0),
raw count/hash, tải metadata/tokenizer của target snapshot có revision bất biến,
resolve draft/capture qua API upstream rồi chạy CLI `--plan`.
Không tải trọng số target hoặc trích xuất hidden states ở bước plan.
Config resolved, hash, revision target, source Git/hash, capture IDs và plan.log
được lưu tại `/vol/runs/m3-04/<timestamp-unique>/`.

Secret đã có: `wandb-secret`, required key `WANDB_API_KEY`.
Không tạo auth CoreWeave riêng. Native tracking dùng project
`qwen2.5-7b-eagle3.1-speculator`; tên full run `m3-04-qwen2.5-7b-eagle3.1`.
Không pre-initialize W&B. GPU run lưu W&B logs tại evidence directory trên Volume.

## Các bước GPU sau này, cần ủy quyền riêng

Chỉ sau CPU plan, review và ủy quyền GPU mới chạy bounded preparation:

```powershell
modal run modal_pipeline.py --step prepare_hidden_states --feature-scope smoke --train-samples 64 --eval-samples 16
```

Scope smoke mặc định 64 train/16 eval, chọn prefix deterministic từ prepared data
qua upstream --num-samples trước shuffle của upstream. Raw và mọi lượt trong mỗi
mẫu vẫn giữ nguyên trước truncation token. Test không được capture.
Smoke features nằm riêng tại /vol/hidden_states/m3-04/smoke/<contract-hash>/train/
và /eval/; scope, sample limits, target revision và capture contract tham gia hash.
Manifest phân biệt raw counts, selected counts và số feature thực sự sinh ra.
Thay giới hạn sẽ tạo identity khác; training smoke dùng cùng hai giới hạn nếu đổi mặc định.

Smoke yêu cầu manifests của scope smoke, không yêu cầu full features.
CPU kiểm tra trước khi cấp GPU.
Smoke giữ target/draft/dtype/batch/accumulation/LR/TTT, chỉ giới hạn 2 optimizer updates,
eval/save mỗi update và dùng run_id riêng:

```powershell
modal run modal_pipeline.py --step smoke
```

Full run và full-state resume, cũng cần ủy quyền riêng:

```powershell
modal run modal_pipeline.py --step prepare_hidden_states
modal run modal_pipeline.py --step train
modal run modal_pipeline.py --step train --resume
```

Preparation không chỉ định scope vẫn là full 40000 train/3000 eval.
Namespace, manifest checks và hyperparameters của full run giữ nguyên; smoke
không thể thỏa mãn full manifest. Số liệu smoke chỉ kiểm tra pipeline, không chứng minh chất lượng.

Không tự chọn best bằng train loss. Smoke PASS cần forward/backward/optimizer,
loss finite, không NaN/OOM, một W&B run có metrics, checkpoint trên Volume,
eval/simulated_acc_len và best/latest đúng nghĩa; process start chưa đủ.

## Export và kiểm thử

Runtime checkpoint chưa phải model directory phục vụ inference. Sau này dùng
`specforge export --to hf --checkpoint <run_id>-best --draft-config <draft.json>
--embedding-source <pinned-target-snapshot> --output-dir <new-export-dir>`.
Chạy trong image Modal Linux. Chưa upload HF hoặc xác nhận tương thích serving;
`export/package_and_upload.py` chỉ đóng gói directory đã qua native exporter.
MODEL_CARD hiện là bản mô tả ứng viên chưa huấn luyện.

```powershell
python -m pytest tests/test_train.py tests/test_configs.py tests/test_prepare_specforge.py tests/test_modal.py tests/test_cleanup_deprecated.py tests/test_export.py -v
python -m pytest tests/ -v
```

`run_all.sh` chỉ delegate sang Modal, mặc định plan.
Notebook cũ là chỉ dẫn chuyển tiếp sang Modal. Các worklogs, task-requirements
và logs cũ là bằng chứng/đặc tả lịch sử, không phải runbook M3-04 hiện tại.
Xem `docs/m3-04-migration.md` để đối chiếu nguồn và giới hạn kiểm chứng.
