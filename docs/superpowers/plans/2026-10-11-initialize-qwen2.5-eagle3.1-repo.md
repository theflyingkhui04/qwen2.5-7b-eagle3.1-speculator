# Kế Hoạch Triển Khai: Khởi Tạo Dự Án Huấn Luyện Đầu Nháp EAGLE-3.1 Cho Qwen2.5-7B

> **Dành cho agent thực thi:** SUB-SKILL BẮT BUỘC: Sử dụng `superpowers:subagent-driven-development` (khuyến nghị) hoặc `superpowers:executing-plans` để thực hiện từng task. Các bước sử dụng cú pháp checkbox (`- [ ]`) để theo dõi tiến độ.

**Mục tiêu:** Xây dựng toàn bộ pipeline mã nguồn, cấu hình, kiểm thử và script tự động hóa huấn luyện đầu nháp suy đoán EAGLE-3.1 cho `Qwen/Qwen2.5-7B-Instruct` theo đúng đặc tả kỹ thuật, sẵn sàng triển khai 1-click trên Vast.ai A100 và tương thích vLLM trên GPU Tesla T4.

**Kiến trúc:** Pipeline 4 bước khép kín: (1) Chuẩn bị dữ liệu lai ShareGPT + Hóa đơn tiếng Việt; (2) Trích xuất Hidden States tầng 28 từ Qwen2.5-7B; (3) Huấn luyện mạng EAGLE-3.1 (1 tầng Decoder + FC Normalization) với Dual Loss (CrossEntropy + 0.5 * Smooth L1); (4) Đóng gói chuẩn Hugging Face (`EagleQwen2ForCausalLM`) xuất ra FP16/BF16 và tự hủy máy Vast.ai.

**Tech Stack:** Python 3.10+, PyTorch 2.4.0, Hugging Face Transformers, Accelerate, Safetensors, Pytest, YAML, Bash.

**Spec:** [2026-10-10-qwen2.5-eagle3.1-training-spec.md](file:///d:/DoAn/qwen2.5-7b-eagle3.1-speculator/2026-10-10-qwen2.5-eagle3.1-training-spec.md)

## Ràng Buộc Toàn Cục (Global Constraints)

- **Mô hình mục tiêu (Target Model):** `Qwen/Qwen2.5-7B-Instruct`, kích thước ẩn `hidden_size = 3584`, số tầng `num_hidden_layers = 28`, số heads `28`.
- **Tầng trích xuất:** Tầng thứ 28 (tầng cuối cùng trước LM Head).
- **Kiến trúc đầu nháp:** 1 tầng Transformer Decoder tự hồi quy, 28 attention heads, có tầng FC Normalization chống trôi dạt vector ẩn.
- **Dữ liệu huấn luyện:** 30.000 mẫu (20k ShareGPT_Vicuna_unfiltered + 10k Tiếng Việt / MC-OCR).
- **Hàm mất mát:** `Total_Loss = Loss_CrossEntropy(token_pred) + 0.5 * Loss_SmoothL1(pred_hidden, target_hidden)`.
- **Đóng gói vLLM:** `architectures = ["EagleQwen2ForCausalLM"]`, xuất khẩu cấu hình `config.json` và trọng số `model.safetensors`.
- **Không dùng LaTeX:** Báo cáo, log và code tuân thủ văn bản thuần / Unicode trực quan.

---

### Task 1: Thiết Lập Scaffolding, `.gitignore` và `requirements.txt`

**Files:**
- Create: `.gitignore`
- Create: `requirements.txt`
- Create: `tests/__init__.py`
- Create: `tests/test_scaffold.py`

**Interfaces:**
- Produces: Môi trường cấu trúc chuẩn, danh mục dependencies cố định phiên bản cho Vast.ai A100.

- [ ] **Step 1: Viết test kiểm tra cấu trúc thư mục và dependencies**

```python
# tests/test_scaffold.py
from pathlib import Path

def test_essential_files_exist():
    root = Path(__file__).parent.parent
    assert (root / "requirements.txt").exists()
    assert (root / ".gitignore").exists()

def test_requirements_contains_pinned_packages():
    root = Path(__file__).parent.parent
    req_content = (root / "requirements.txt").read_text(encoding="utf-8")
    assert "torch" in req_content
    assert "transformers" in req_content
    assert "safetensors" in req_content
    assert "accelerate" in req_content
```

- [ ] **Step 2: Chạy test để xác nhận FAIL**

Run: `pytest tests/test_scaffold.py -v`
Expected: FAIL vì `requirements.txt` và `.gitignore` chưa tồn tại.

- [ ] **Step 3: Tạo `.gitignore` và `requirements.txt`**

Nội dung `.gitignore`:
```gitignore
# Byte-compiled / optimized / cache
__pycache__/
*.py[cod]
*$py.class
.pytest_cache/

# Data and Checkpoints
data/raw/
data/processed/
data/features/
checkpoints/
output/
*.pt
*.bin
*.safetensors
*.tar.gz

# Environment
.venv/
venv/
ENV/
*.env

# Vast.ai & Jupyter
.ipynb_checkpoints/
nohup.out
tmux.log
```

Nội dung `requirements.txt`:
```text
torch>=2.4.0
transformers>=4.45.0
accelerate>=0.34.0
safetensors>=0.4.4
huggingface_hub>=0.24.0
pyyaml>=6.0.2
tqdm>=4.66.5
numpy>=1.26.0
pytest>=8.0.0
```

- [ ] **Step 4: Chạy test để xác nhận PASS**

Run: `pytest tests/test_scaffold.py -v`
Expected: PASS.

- [ ] **Step 5: Git commit**

```bash
git add .gitignore requirements.txt tests/
git commit -m "chore: scaffold project structure, gitignore, and requirements"
```

---

### Task 2: Cấu Hình Kiến Trúc & Siêu Tham Số Huấn Luyện

**Files:**
- Create: `configs/eagle3_qwen2.5_7b.json`
- Create: `configs/train_config.yaml`
- Create: `tests/test_configs.py`

**Interfaces:**
- Produces: `configs/eagle3_qwen2.5_7b.json` chứa thông số mô hình (`hidden_size=3584`, `num_hidden_layers=1`, `num_attention_heads=28`, `target_layer=28`, `fc_norm=True`).
- Produces: `configs/train_config.yaml` chứa siêu tham số huấn luyện (`batch_size=16`, `learning_rate=3e-4`, `epochs=20`, `warmup_steps=500`, `smooth_l1_weight=0.5`).

- [ ] **Step 1: Viết test kiểm tra tính toàn vẹn của cấu hình**

```python
# tests/test_configs.py
import json
import yaml
from pathlib import Path

def test_model_config_integrity():
    root = Path(__file__).parent.parent
    config_path = root / "configs" / "eagle3_qwen2.5_7b.json"
    assert config_path.exists()
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["hidden_size"] == 3584
    assert data["num_hidden_layers"] == 1
    assert data["num_attention_heads"] == 28
    assert data["target_layer"] == 28
    assert data["use_fc_norm"] is True
    assert "EagleQwen2ForCausalLM" in data["architectures"]

def test_train_config_integrity():
    root = Path(__file__).parent.parent
    config_path = root / "configs" / "train_config.yaml"
    assert config_path.exists()
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert data["training"]["batch_size"] == 16
    assert data["training"]["gradient_accumulation_steps"] == 2
    assert float(data["training"]["learning_rate"]) == 3e-4
    assert data["training"]["epochs"] == 20
    assert data["training"]["warmup_steps"] == 500
    assert data["loss"]["smooth_l1_weight"] == 0.5
```

- [ ] **Step 2: Chạy test để xác nhận FAIL**

Run: `pytest tests/test_configs.py -v`
Expected: FAIL vì thư mục `configs/` chưa có file.

- [ ] **Step 3: Tạo `configs/eagle3_qwen2.5_7b.json` và `configs/train_config.yaml`**

Tạo `configs/eagle3_qwen2.5_7b.json`:
```json
{
  "architectures": [
    "EagleQwen2ForCausalLM"
  ],
  "model_type": "eagle_qwen2",
  "hidden_size": 3584,
  "intermediate_size": 18944,
  "num_hidden_layers": 1,
  "num_attention_heads": 28,
  "num_key_value_heads": 4,
  "hidden_act": "silu",
  "max_position_embeddings": 32768,
  "initializer_range": 0.02,
  "rms_norm_eps": 1e-6,
  "vocab_size": 152064,
  "target_model_name_or_path": "Qwen/Qwen2.5-7B-Instruct",
  "target_layer": 28,
  "use_fc_norm": true,
  "torch_dtype": "float16"
}
```

Tạo `configs/train_config.yaml`:
```yaml
model:
  config_path: "configs/eagle3_qwen2.5_7b.json"
  target_model: "Qwen/Qwen2.5-7B-Instruct"
  target_layer: 28

data:
  dataset_path: "data/processed/hybrid_dataset.json"
  features_dir: "data/features"
  max_seq_len: 2048
  chunk_size: 5000

training:
  batch_size: 16
  gradient_accumulation_steps: 2
  learning_rate: 3e-4
  min_learning_rate: 1e-5
  weight_decay: 0.01
  adam_beta1: 0.9
  adam_beta2: 0.95
  epochs: 20
  warmup_steps: 500
  checkpoint_interval_epochs: 5
  output_dir: "checkpoints"

loss:
  ce_weight: 1.0
  smooth_l1_weight: 0.5

safeguards:
  min_disk_free_gb: 50
  max_runtime_hours: 6.0
```

- [ ] **Step 4: Chạy test để xác nhận PASS**

Run: `pytest tests/test_configs.py -v`
Expected: PASS.

- [ ] **Step 5: Git commit**

```bash
git add configs/ tests/test_configs.py
git commit -m "feat: add EAGLE-3.1 architecture and training configs"
```

---

### Task 3: Pipeline Dữ Liệu Lai (Hybrid Dataset Pipeline)

**Files:**
- Create: `data/prepare_hybrid_dataset.py`
- Create: `tests/test_dataset.py`

**Interfaces:**
- `filter_and_format_conversation(conversation: list) -> dict`
- `build_hybrid_dataset(sharegpt_path: str, vietnamese_path: str, output_path: str, sharegpt_limit: int = 20000, vi_limit: int = 10000) -> dict`

- [ ] **Step 1: Viết test cho pipeline gộp và lọc dữ liệu lai**

```python
# tests/test_dataset.py
import json
import pytest
from pathlib import Path
from data.prepare_hybrid_dataset import (
    filter_and_format_conversation,
    build_hybrid_dataset,
)

def test_filter_and_format_conversation():
    raw_convo = [
        {"from": "human", "value": "Trích xuất thông tin hóa đơn sau: Cty ABC MST 0123456789"},
        {"from": "gpt", "value": '{"company": "Cty ABC", "tax_code": "0123456789"}'}
    ]
    res = filter_and_format_conversation(raw_convo)
    assert res is not None
    assert len(res["conversations"]) == 2
    assert res["conversations"][0]["role"] == "user"
    assert res["conversations"][1]["role"] == "assistant"

def test_build_hybrid_dataset(tmp_path):
    sharegpt_file = tmp_path / "sharegpt.json"
    vi_file = tmp_path / "vi_invoice.json"
    out_file = tmp_path / "hybrid.json"

    dummy_sg = [{"conversations": [{"from": "human", "value": f"Hi {i}"}, {"from": "gpt", "value": f"Hello {i}"}]} for i in range(5)]
    dummy_vi = [{"conversations": [{"from": "human", "value": f"Hóa đơn {i}"}, {"from": "gpt", "value": f"Thông tin {i}"}]} for i in range(5)]

    sharegpt_file.write_text(json.dumps(dummy_sg), encoding="utf-8")
    vi_file.write_text(json.dumps(dummy_vi), encoding="utf-8")

    stats = build_hybrid_dataset(
        sharegpt_path=str(sharegpt_file),
        vietnamese_path=str(vi_file),
        output_path=str(out_file),
        sharegpt_limit=3,
        vi_limit=2,
    )
    assert stats["total_samples"] == 5
    assert stats["sharegpt_count"] == 3
    assert stats["vietnamese_count"] == 2
    assert out_file.exists()
```

- [ ] **Step 2: Chạy test để xác nhận FAIL**

Run: `pytest tests/test_dataset.py -v`
Expected: FAIL vì `data/prepare_hybrid_dataset.py` chưa được định nghĩa.

- [ ] **Step 3: Viết mã `data/prepare_hybrid_dataset.py`**

Triển khai script chuẩn hóa role `user`/`assistant`, lọc bỏ mẫu rác, gộp và xáo trộn (shuffle) 20k ShareGPT + 10k Tiếng Việt/Hóa đơn. Tích hợp CLI với `argparse`.

- [ ] **Step 4: Chạy test để xác nhận PASS**

Run: `pytest tests/test_dataset.py -v`
Expected: PASS.

- [ ] **Step 5: Git commit**

```bash
git add data/prepare_hybrid_dataset.py tests/test_dataset.py
git commit -m "feat: implement hybrid dataset preparation pipeline"
```

---

### Task 4: Pipeline Trích Xuất Vector Ẩn (Feature Extraction)

**Files:**
- Create: `extraction/extract_hidden_states.py`
- Create: `tests/test_extraction.py`

**Interfaces:**
- `bucket_sequences(sequences: list[dict], bucket_limits: list[int]) -> dict[int, list[dict]]`
- `FeatureExtractor(target_model_name: str, target_layer: int, output_dir: str, chunk_size: int = 5000)`

- [ ] **Step 1: Viết test cho thuật toán phân nhóm (Bucketing) và quản lý chunk**

```python
# tests/test_extraction.py
from extraction.extract_hidden_states import bucket_sequences

def test_bucket_sequences():
    samples = [
        {"id": 1, "length": 120},
        {"id": 2, "length": 450},
        {"id": 3, "length": 700},
        {"id": 4, "length": 1800},
        {"id": 5, "length": 3000},  # vượt max 2048 -> loại bỏ
    ]
    buckets = bucket_sequences(samples, bucket_limits=[512, 1024, 2048])
    assert len(buckets[512]) == 2  # id 1, 2
    assert len(buckets[1024]) == 1  # id 3
    assert len(buckets[2048]) == 1  # id 4
    assert sum(len(v) for v in buckets.values()) == 4
```

- [ ] **Step 2: Chạy test để xác nhận FAIL**

Run: `pytest tests/test_extraction.py -v`
Expected: FAIL vì `extraction/extract_hidden_states.py` chưa tồn tại.

- [ ] **Step 3: Viết mã `extraction/extract_hidden_states.py`**

Triển khai:
- `bucket_sequences`
- Forward qua Qwen2.5-7B với `output_hidden_states=True`
- Lấy `hidden_states[target_layer]` (layer 28)
- Lưu `safetensors` từng chunk 5.000 mẫu để chống tràn RAM/VRAM
- Chốt chặn OOM và kiểm tra đĩa trước khi trích xuất.

- [ ] **Step 4: Chạy test để xác nhận PASS**

Run: `pytest tests/test_extraction.py -v`
Expected: PASS.

- [ ] **Step 5: Git commit**

```bash
git add extraction/ tests/test_extraction.py
git commit -m "feat: implement feature extraction and sequence bucketing pipeline"
```

---

### Task 5: Kiến Trúc Mạng EAGLE-3.1 Kèm FC Normalization

**Files:**
- Create: `training/model.py`
- Create: `tests/test_model.py`

**Interfaces:**
- Class `FCNormalization(hidden_size: int)`
- Class `Eagle3DecoderLayer(config: dict)`
- Class `Eagle3SpeculatorHead(config: dict)`:
  - `forward(hidden_states: Tensor, input_ids: Tensor) -> tuple[Tensor, Tensor]`
  (trả về `(predicted_logits, predicted_hidden_states)`)

- [ ] **Step 1: Viết test cho cấu trúc tầng Decoder, FC Norm và kích thước tensor đầu ra**

```python
# tests/test_model.py
import pytest
from unittest.mock import MagicMock
# Viết mock / structural test để verify cấu trúc lớp và logic toán học của model
from training.model import FCNormalization, Eagle3SpeculatorConfig

def test_fc_norm_initialization():
    config = Eagle3SpeculatorConfig(hidden_size=3584, use_fc_norm=True)
    norm = FCNormalization(hidden_size=3584)
    assert norm.hidden_size == 3584

def test_config_defaults():
    config = Eagle3SpeculatorConfig()
    assert config.hidden_size == 3584
    assert config.num_attention_heads == 28
    assert config.target_layer == 28
    assert config.use_fc_norm is True
```

- [ ] **Step 2: Chạy test để xác nhận FAIL**

Run: `pytest tests/test_model.py -v`
Expected: FAIL vì `training/model.py` chưa tồn tại.

- [ ] **Step 3: Viết mã `training/model.py`**

Cài đặt:
- `Eagle3SpeculatorConfig`
- `FCNormalization`: Lớp chuẩn hóa kết hợp Linear + RMSNorm chống Attention Drift
- `Eagle3DecoderLayer`: Attention (28 heads, `head_dim=128`, GQA 4 kv-heads) + SwiGLU MLP (`intermediate_size=18944`)
- `Eagle3SpeculatorHead`: Kết hợp Feature Fusion + Decoder Layer + FC Norm.

- [ ] **Step 4: Chạy test để xác nhận PASS**

Run: `pytest tests/test_model.py -v`
Expected: PASS.

- [ ] **Step 5: Git commit**

```bash
git add training/model.py tests/test_model.py
git commit -m "feat: implement EAGLE-3.1 speculator head with FC normalization"
```

---

### Task 6: Hàm Mất Mát Kết Hợp (Dual Loss Function)

**Files:**
- Create: `training/loss.py`
- Create: `tests/test_loss.py`

**Interfaces:**
- Class `EagleDualLoss(smooth_l1_weight: float = 0.5, ce_weight: float = 1.0)`:
  - `forward(logits, target_token_ids, pred_hidden, target_hidden) -> tuple[Tensor, dict]`

- [ ] **Step 1: Viết test cho logic tính toán Dual Loss**

```python
# tests/test_loss.py
from training.loss import EagleDualLoss

def test_dual_loss_weights():
    loss_fn = EagleDualLoss(smooth_l1_weight=0.5, ce_weight=1.0)
    assert loss_fn.smooth_l1_weight == 0.5
    assert loss_fn.ce_weight == 1.0
```

- [ ] **Step 2: Chạy test để xác nhận FAIL**

Run: `pytest tests/test_loss.py -v`
Expected: FAIL vì `training/loss.py` chưa tồn tại.

- [ ] **Step 3: Viết mã `training/loss.py`**

Cài đặt `EagleDualLoss`:
- `ce_loss = CrossEntropyLoss(logits.view(-1, vocab_size), target_token_ids.view(-1))`
- `reg_loss = SmoothL1Loss(pred_hidden, target_hidden)`
- `total_loss = ce_weight * ce_loss + smooth_l1_weight * reg_loss`
- Trả về `total_loss` và dict log chi tiết `{"ce_loss": ..., "reg_loss": ...}`.

- [ ] **Step 4: Chạy test để xác nhận PASS**

Run: `pytest tests/test_loss.py -v`
Expected: PASS.

- [ ] **Step 5: Git commit**

```bash
git add training/loss.py tests/test_loss.py
git commit -m "feat: implement dual loss function combining cross-entropy and smooth L1"
```

---

### Task 7: Vòng Lặp Huấn Luyện & Quản Lý Checkpoint

**Files:**
- Create: `training/train.py`
- Create: `tests/test_train.py`

**Interfaces:**
- `parse_training_args() -> argparse.Namespace`
- `save_checkpoint(model, optimizer, scheduler, epoch, step, output_dir)`
- `load_latest_checkpoint(checkpoint_dir) -> tuple[int, int]`

- [ ] **Step 1: Viết test cho checkpoint recovery và argument parser**

```python
# tests/test_train.py
import pytest
from pathlib import Path
from training.train import parse_training_args, get_latest_checkpoint

def test_parse_args_defaults():
    args = parse_training_args(["--config", "configs/train_config.yaml"])
    assert args.config == "configs/train_config.yaml"

def test_get_latest_checkpoint(tmp_path):
    (tmp_path / "checkpoint_epoch_5.pt").write_text("test")
    (tmp_path / "checkpoint_epoch_10.pt").write_text("test")
    latest = get_latest_checkpoint(tmp_path)
    assert latest is not None
    assert latest.name == "checkpoint_epoch_10.pt"
```

- [ ] **Step 2: Chạy test để xác nhận FAIL**

Run: `pytest tests/test_train.py -v`
Expected: FAIL vì `training/train.py` chưa tồn tại.

- [ ] **Step 3: Viết mã `training/train.py`**

Triển khai:
- Đọc config YAML
- Khởi tạo Optimizer AdamW (`lr=3e-4`, `weight_decay=0.01`)
- Cosine Annealing decay với 500 bước warm-up
- Gradient Accumulation (2 bước)
- Tự động resume checkpoint gần nhất nếu có
- Ghi log epoch, loss, GPU memory và dọn dẹp cache `torch.cuda.empty_cache()` sau mỗi epoch.

- [ ] **Step 4: Chạy test để xác nhận PASS**

Run: `pytest tests/test_train.py -v`
Expected: PASS.

- [ ] **Step 5: Git commit**

```bash
git add training/train.py tests/test_train.py
git commit -m "feat: implement main training loop with automatic checkpoint recovery"
```

---

### Task 8: Đóng Gói Checkpoint Chuẩn Hugging Face & Tải Lên Hub

**Files:**
- Create: `export/package_and_upload.py`
- Create: `tests/test_export.py`

**Interfaces:**
- `build_hf_config(base_config: dict, precision: str = "float16") -> dict`
- `package_model(checkpoint_path: str, output_dir: str, precision: str = "float16")`
- `upload_to_huggingface(export_dir: str, repo_id: str, token: str)`

- [ ] **Step 1: Viết test kiểm tra định dạng metadata xuất bản cho vLLM**

```python
# tests/test_export.py
from export.package_and_upload import build_hf_config

def test_build_hf_config_for_vllm():
    raw_cfg = {
        "hidden_size": 3584,
        "num_hidden_layers": 1,
        "num_attention_heads": 28,
        "target_model_name_or_path": "Qwen/Qwen2.5-7B-Instruct",
    }
    hf_cfg = build_hf_config(raw_cfg, precision="float16")
    assert hf_cfg["architectures"] == ["EagleQwen2ForCausalLM"]
    assert hf_cfg["torch_dtype"] == "float16"
    assert hf_cfg["hidden_size"] == 3584
```

- [ ] **Step 2: Chạy test để xác nhận FAIL**

Run: `pytest tests/test_export.py -v`
Expected: FAIL vì `export/package_and_upload.py` chưa tồn tại.

- [ ] **Step 3: Viết mã `export/package_and_upload.py`**

Cài đặt:
- Chuyển đổi trọng số `.pt` sang `model.safetensors`
- Tùy chọn ép kiểu sang `float16` cho tối ưu GPU Turing T4
- Viết `config.json` chuẩn vLLM nhận diện
- Tự động upload lên Hugging Face Hub bằng `HfApi`.

- [ ] **Step 4: Chạy test để xác nhận PASS**

Run: `pytest tests/test_export.py -v`
Expected: PASS.

- [ ] **Step 5: Git commit**

```bash
git add export/ tests/test_export.py
git commit -m "feat: implement Hugging Face model packaging and export pipeline"
```

---

### Task 9: Script Tự Động Hóa 1-Click Kèm Chốt Chặn Chi Phí & Tự Hủy Máy

**Files:**
- Create: `run_all.sh`
- Create: `tests/test_script_guards.py`

**Interfaces:**
- Produces: Bash script `run_all.sh` chạy 4 bước tuần tự, kiểm tra dung lượng ổ đĩa, kiểm soát thời gian tối đa 6.0 giờ, tự gọi `vastai stop instance` khi kết thúc.

- [ ] **Step 1: Viết test kiểm tra logic chốt chặn an toàn trong script bash**

```python
# tests/test_script_guards.py
from pathlib import Path

def test_run_all_script_contains_safeguards():
    root = Path(__file__).parent.parent
    script_path = root / "run_all.sh"
    assert script_path.exists()
    content = script_path.read_text(encoding="utf-8")
    assert "disk" in content.lower()
    assert "timeout" in content.lower() or "MAX_HOURS" in content
    assert "vastai stop instance" in content
```

- [ ] **Step 2: Chạy test để xác nhận FAIL**

Run: `pytest tests/test_script_guards.py -v`
Expected: FAIL vì `run_all.sh` chưa tồn tại.

- [ ] **Step 3: Viết mã `run_all.sh`**

Triển khai script Bash hoàn chỉnh với:
- `set -euo pipefail`
- Kiểm tra dung lượng đĩa trống >= 50 GB
- Giới hạn chạy tối đa 6.0 giờ
- Chạy lần lượt: `prepare_hybrid_dataset.py` -> `extract_hidden_states.py` -> `train.py` -> `package_and_upload.py`
- Tự động bắt tín hiệu tắt máy Vast.ai: `vastai stop instance "$CONTAINER_ID"` dù thành công hay lỗi.

- [ ] **Step 4: Chạy test để xác nhận PASS**

Run: `pytest tests/test_script_guards.py -v`
Expected: PASS.

- [ ] **Step 5: Git commit**

```bash
git add run_all.sh tests/test_script_guards.py
git commit -m "feat: add 1-click automation bash script with auto-stop safeguard"
```

---

### Task 10: Tài Liệu Hướng Dẫn & Jupyter Notebook

**Files:**
- Create: `notebooks/train_on_vastai.ipynb`
- Update: `README.md`
- Create: `tests/test_final_structure.py`

**Interfaces:**
- Produces: Hướng dẫn chi tiết triển khai trên Vast.ai & Colab, notebook chạy tương tác trực quan.

- [ ] **Step 1: Viết test kiểm tra cấu trúc toàn bộ repository theo đúng Spec**

```python
# tests/test_final_structure.py
from pathlib import Path

def test_full_repository_layout():
    root = Path(__file__).parent.parent
    required = [
        "README.md",
        "requirements.txt",
        "configs/eagle3_qwen2.5_7b.json",
        "configs/train_config.yaml",
        "data/prepare_hybrid_dataset.py",
        "extraction/extract_hidden_states.py",
        "training/model.py",
        "training/loss.py",
        "training/train.py",
        "export/package_and_upload.py",
        "notebooks/train_on_vastai.ipynb",
        "run_all.sh",
    ]
    for rel_path in required:
        assert (root / rel_path).exists(), f"Missing {rel_path}"
```

- [ ] **Step 2: Chạy test để xác nhận FAIL**

Run: `pytest tests/test_final_structure.py -v`
Expected: FAIL vì notebook và README chưa hoàn thiện.

- [ ] **Step 3: Tạo `notebooks/train_on_vastai.ipynb` và hoàn thiện `README.md`**

Viết notebook chuẩn JSON với các cell kiểm tra GPU, cài dependencies, chạy pipeline trích xuất, huấn luyện và đóng gói.
Cập nhật `README.md` hướng dẫn thuê máy Vast.ai A100, thiết lập HF token, chạy lệnh và tích hợp vào vLLM trên GPU T4.

- [ ] **Step 4: Chạy toàn bộ test suite để xác nhận PASS**

Run: `pytest tests/ -v`
Expected: Tất cả bài test đều PASS.

- [ ] **Step 5: Git commit**

```bash
git add notebooks/ README.md tests/test_final_structure.py
git commit -m "docs: finalize readme, jupyter notebook and full repository layout"
```
