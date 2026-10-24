# Kế Hoạch Triển Khai: Refactor Hệ Thống Sang SpecForge Cho EAGLE-3.1 (Qwen2.5-7B)

> **Dành cho agent thực thi:** SUB-SKILL BẮT BUỘC: Sử dụng `superpowers:subagent-driven-development` (khuyến nghị) hoặc `superpowers:executing-plans` để thực hiện từng task. Các bước sử dụng cú pháp checkbox (`- [ ]`) để theo dõi tiến độ.

**Goal:** Chuyển đổi toàn bộ pipeline huấn luyện của dự án `qwen2.5-7b-eagle3.1-speculator` sang framework chính thức `sgl-project/SpecForge`, ghim kiến trúc EAGLE-3.1 (1 tầng Decoder, FC Norm, unroll TTT đa bước, target layer 28 của Qwen2.5-7B), loại bỏ trích xuất offline thủ công và đảm bảo checkpoint nạp trực tiếp vào vLLM/SGLang.

**Architecture:** Thay thế tầng trích xuất offline thủ công (`extraction/`) và mô hình tự viết (`training/model.py`, `training/loss.py`) bằng SpecForge runner qua `training/train.py`. Dữ liệu lai được chuẩn hóa sang định dạng ShareGPT JSON Lines (`data/processed/hybrid_train.jsonl`). Pipeline rút ngắn thành 3 bước khép kín: (1) Chuẩn bị dữ liệu JSONL; (2) Huấn luyện unroll autoregressive với SpecForge; (3) Xác thực và tải checkpoint lên Hugging Face Hub.

**Tech Stack:** Python 3.10+, PyTorch >= 2.4.0, SpecForge (`git+https://github.com/sgl-project/SpecForge.git`), SGLang (`sglang[all]`), Transformers >= 4.45.0, Accelerate >= 0.34.0, PyYAML, Pytest, Modal.

**Spec:** [docs/2026-10-24-qwen2.5-eagle3.1-specforge-refactor-spec.md](file:///d:/DoAn/qwen2.5-7b-eagle3.1-speculator/docs/2026-10-24-qwen2.5-eagle3.1-specforge-refactor-spec.md)

## Global Constraints

- **Không tự động commit:** Tuyệt đối không tự động chạy lệnh `git commit`. Sau khi hoàn thành code, chỉ cung cấp commit message gợi ý để người dùng tự kiểm tra và commit.
- **Không dùng LaTeX:** Trình bày công thức bằng chữ thường và Unicode trực quan (e.g., alpha >= 0.70, loss = ce_loss + 0.5 * smooth_l1).
- **Ghim chuẩn EAGLE-3.1:** Chỉ dùng duy nhất file cấu hình `configs/eagle3.1_qwen2.5_7b.json` (bỏ `configs/eagle3_qwen2.5_7b.json`), giữ nguyên `fc_norm: true`, `norm_output: true`, `use_post_norm_feedback: true`, `target_layer: 28`, `hidden_size: 3584`, `eagle_version: "3.1"`.
- **Định dạng dữ liệu:** ShareGPT JSON Lines (`.jsonl`), mỗi dòng là JSON object chứa `id` và `conversations` (hoặc `messages`).
- **An toàn hạ tầng:** Giữ nguyên Disk Guard (>= 50GB) và Cost Ceiling Guard (<= 6.0h) trong `run_all.sh`.

---

### Task 1: Dọn Dẹp Các Module & Bài Test Cũ Không Còn Sử Dụng

**Files:**
- Delete: `extraction/extract_hidden_states.py`
- Delete: `extraction/__init__.py` (hoặc toàn bộ thư mục `extraction/`)
- Delete: `training/model.py`
- Delete: `training/loss.py`
- Delete: `configs/eagle3_qwen2.5_7b.json` (loại bỏ file trùng lặp cũ, giữ `configs/eagle3.1_qwen2.5_7b.json`)
- Delete: `tests/test_extraction.py`
- Delete: `tests/test_loss.py`
- Delete: `tests/test_model.py`
- Modify: `training/__init__.py`

**Interfaces:**
- Produces: Codebase sạch sẽ, không còn file code trích xuất và mạng tự viết thủ công.

- [ ] **Step 1: Viết test kiểm tra việc dọn dẹp các module cũ**

```python
# tests/test_cleanup_deprecated.py
from pathlib import Path

def test_deprecated_modules_and_tests_removed():
    root = Path(__file__).parent.parent
    deprecated_paths = [
        root / "extraction" / "extract_hidden_states.py",
        root / "training" / "model.py",
        root / "training" / "loss.py",
        root / "configs" / "eagle3_qwen2.5_7b.json",
        root / "tests" / "test_extraction.py",
        root / "tests" / "test_loss.py",
        root / "tests" / "test_model.py",
    ]
    for p in deprecated_paths:
        assert not p.exists(), f"File cũ chưa được dọn dẹp: {p}"
```

- [ ] **Step 2: Chạy test để xác nhận FAIL**

Run: `pytest tests/test_cleanup_deprecated.py -v`
Expected: FAIL vì các file cũ vẫn đang tồn tại.

- [ ] **Step 3: Thực hiện xóa các file cũ và cập nhật `training/__init__.py`**

Xóa các file:
- `extraction/extract_hidden_states.py`
- `extraction/__init__.py`
- `training/model.py`
- `training/loss.py`
- `configs/eagle3_qwen2.5_7b.json` (nếu tồn tại)
- `tests/test_extraction.py`
- `tests/test_loss.py`
- `tests/test_model.py`

Cập nhật `training/__init__.py`:
```python
# training/__init__.py
"""
Package training cho EAGLE-3.1 Speculator sử dụng SpecForge runner.
"""
```

- [ ] **Step 4: Chạy test để xác nhận PASS**

Run: `pytest tests/test_cleanup_deprecated.py -v`
Expected: PASS.

- [ ] **Step 5: Gợi ý Git Commit (Người dùng tự chạy)**

```bash
git add -u
git add tests/test_cleanup_deprecated.py
# Gợi ý commit:
# git commit -m "refactor: clean up deprecated extraction and custom model/loss modules"
```

---

### Task 2: Chuẩn Hóa Dữ Liệu Đầu Ra Sang ShareGPT JSONL trong `data/prepare_hybrid_dataset.py`

**Files:**
- Modify: `data/prepare_hybrid_dataset.py`
- Modify: `tests/test_dataset.py`

**Interfaces:**
- `filter_and_format_conversation(raw_convo: list[dict], sample_id: str = "") -> dict | None`
  - Output: `{"id": sample_id, "conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}`
- `build_hybrid_dataset(sharegpt_path: str, vietnamese_path: str, output_path: str, sharegpt_limit: int = 20000, vi_limit: int = 10000, seed: int = 42) -> dict`
  - Output: Ghi từng dòng JSON vào file `.jsonl` (không phải JSON array).

- [ ] **Step 1: Viết test mới kiểm tra định dạng ShareGPT JSONL**

Cập nhật `tests/test_dataset.py`:
```python
# tests/test_dataset.py
import json
import pytest
from pathlib import Path
from data.prepare_hybrid_dataset import (
    filter_and_format_conversation,
    build_hybrid_dataset,
)

def test_filter_and_format_conversation_sharegpt():
    raw_convo = [
        {"from": "human", "value": "Trích xuất thông tin hóa đơn sau: Cty ABC MST 0123456789"},
        {"from": "gpt", "value": '{"company": "Cty ABC", "tax_code": "0123456789"}'}
    ]
    res = filter_and_format_conversation(raw_convo, sample_id="sample_0")
    assert res is not None
    assert res["id"] == "sample_0"
    assert len(res["conversations"]) == 2
    assert res["conversations"][0]["from"] == "human"
    assert res["conversations"][1]["from"] == "gpt"

def test_filter_empty_or_invalid_conversation():
    assert filter_and_format_conversation([], sample_id="sample_bad") is None
    assert filter_and_format_conversation([{"from": "human", "value": ""}], sample_id="sample_bad") is None

def test_sanitizes_unusual_line_terminators():
    raw_convo = [
        {"from": "human", "value": "Line 1\u2028Line 2"},
        {"from": "gpt", "value": "Paragraph 1\u2029Paragraph 2"}
    ]
    res = filter_and_format_conversation(raw_convo, sample_id="sample_1")
    assert "\u2028" not in res["conversations"][0]["value"]
    assert "\u2029" not in res["conversations"][1]["value"]
    assert "Line 1\nLine 2" == res["conversations"][0]["value"]
    assert "Paragraph 1\n\nParagraph 2" == res["conversations"][1]["value"]

def test_build_hybrid_dataset_jsonl(tmp_path):
    sharegpt_file = tmp_path / "sharegpt.json"
    vi_file = tmp_path / "vi_invoice.json"
    out_file = tmp_path / "hybrid_train.jsonl"

    dummy_sg = [
        {"conversations": [{"from": "human", "value": f"Hi {i}"}, {"from": "gpt", "value": f"Hello {i}"}]}
        for i in range(5)
    ]
    dummy_vi = [
        {"conversations": [{"from": "human", "value": f"Hóa đơn {i}"}, {"from": "gpt", "value": f"Thông tin {i}"}]}
        for i in range(5)
    ]

    sharegpt_file.write_text(json.dumps(dummy_sg), encoding="utf-8")
    vi_file.write_text(json.dumps(dummy_vi), encoding="utf-8")

    stats = build_hybrid_dataset(
        sharegpt_path=str(sharegpt_file),
        vietnamese_path=str(vi_file),
        output_path=str(out_file),
        sharegpt_limit=3,
        vi_limit=2,
        seed=42,
    )
    assert stats["total_samples"] == 5
    assert stats["sharegpt_count"] == 3
    assert stats["vietnamese_count"] == 2
    assert out_file.exists()

    # Xác thực mỗi dòng là một JSON object độc lập
    with open(out_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    assert len(lines) == 5
    for idx, line in enumerate(lines):
        item = json.loads(line)
        assert "id" in item
        assert "conversations" in item
        assert len(item["conversations"]) == 2
        assert item["conversations"][0]["from"] == "human"
        assert item["conversations"][1]["from"] == "gpt"
```

- [ ] **Step 2: Chạy test để xác nhận FAIL**

Run: `pytest tests/test_dataset.py -v`
Expected: FAIL vì `data/prepare_hybrid_dataset.py` đang xuất JSON array thay vì JSONL và `conversations` dùng `role`/`content`.

- [ ] **Step 3: Cập nhật `data/prepare_hybrid_dataset.py`**

Chỉnh sửa `data/prepare_hybrid_dataset.py` để:
1. Chuẩn hóa `conversations` thành cặp `{"from": "human"|"gpt", "value": str}` kèm khóa `"id"`.
2. Ghi ra file theo định dạng JSON Lines (`.jsonl`).
3. Đổi output path mặc định thành `data/processed/hybrid_train.jsonl`.

```python
"""
Script prepare_hybrid_dataset.py
Gộp dữ liệu ShareGPT_Vicuna_unfiltered (20k) và Tiếng Việt/Hóa đơn MC-OCR (10k)
Chuẩn hóa cấu trúc hội thoại đa lượt và lưu định dạng JSON Lines (.jsonl) tương thích SpecForge.
"""

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def filter_and_format_conversation(raw_convo: List[Dict[str, Any]], sample_id: str = "") -> Optional[Dict[str, Any]]:
    """
    Chuẩn hóa một đoạn hội thoại sang định dạng ShareGPT cho SpecForge:
    {"id": sample_id, "conversations": [{"from": "human"|"gpt", "value": str}, ...]}
    Lọc bỏ các đoạn hội thoại rỗng hoặc chỉ có 1 lượt.
    """
    if not raw_convo or not isinstance(raw_convo, list) or len(raw_convo) < 2:
        return None

    formatted_turns = []
    for turn in raw_convo:
        if not isinstance(turn, dict):
            continue

        raw_role = turn.get("from") or turn.get("role") or ""
        content = turn.get("value") or turn.get("content") or ""

        if not str(content).strip():
            continue

        raw_role_lower = str(raw_role).lower()
        if raw_role_lower in ["human", "user"]:
            role = "human"
        elif raw_role_lower in ["gpt", "assistant", "bot"]:
            role = "gpt"
        else:
            continue

        clean_content = str(content).replace("\u2028", "\n").replace("\u2029", "\n\n").strip()
        formatted_turns.append({"from": role, "value": clean_content})

    # Cần tối thiểu 1 cặp human - gpt
    if len(formatted_turns) < 2:
        return None

    return {"id": sample_id, "conversations": formatted_turns}


def load_raw_dataset(file_path: str) -> List[Dict[str, Any]]:
    """Đọc dữ liệu thô từ file JSON hoặc JSONL."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {file_path}")

    if path.suffix.lower() == ".jsonl":
        samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if line_str:
                    samples.append(json.loads(line_str))
        return samples
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and "data" in data:
                return data["data"]
            else:
                return [data]


def build_hybrid_dataset(
    sharegpt_path: str,
    vietnamese_path: str,
    output_path: str,
    sharegpt_limit: int = 20000,
    vi_limit: int = 10000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Gộp và xáo trộn tập dữ liệu lai theo tỷ lệ xác định, xuất định dạng .jsonl.
    """
    logger.info("Đang nạp tập dữ liệu ShareGPT từ: %s", sharegpt_path)
    raw_sharegpt = load_raw_dataset(sharegpt_path)

    logger.info("Đang nạp tập dữ liệu Tiếng Việt / MC-OCR từ: %s", vietnamese_path)
    raw_vietnamese = load_raw_dataset(vietnamese_path)

    cleaned_sharegpt = []
    for idx, item in enumerate(raw_sharegpt):
        convo = item.get("conversations") or item.get("messages") or []
        s_id = item.get("id") or f"sharegpt_{idx}"
        formatted = filter_and_format_conversation(convo, sample_id=str(s_id))
        if formatted:
            cleaned_sharegpt.append(formatted)
        if len(cleaned_sharegpt) >= sharegpt_limit:
            break

    cleaned_vietnamese = []
    for idx, item in enumerate(raw_vietnamese):
        convo = item.get("conversations") or item.get("messages") or []
        v_id = item.get("id") or f"vi_{idx}"
        formatted = filter_and_format_conversation(convo, sample_id=str(v_id))
        if formatted:
            cleaned_vietnamese.append(formatted)
        if len(cleaned_vietnamese) >= vi_limit:
            break

    logger.info("Lọc được: %d mẫu ShareGPT, %d mẫu Tiếng Việt / MC-OCR", len(cleaned_sharegpt), len(cleaned_vietnamese))

    hybrid = cleaned_sharegpt + cleaned_vietnamese
    random.seed(seed)
    random.shuffle(hybrid)

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with open(out_file, "w", encoding="utf-8") as f:
        for sample in hybrid:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    logger.info("Đã ghi thành công %d mẫu vào %s (JSONL)", len(hybrid), str(out_file))

    return {
        "total_samples": len(hybrid),
        "sharegpt_count": len(cleaned_sharegpt),
        "vietnamese_count": len(cleaned_vietnamese),
        "output_path": str(out_file),
    }


def main():
    parser = argparse.ArgumentParser(description="Chuẩn bị tập dữ liệu lai cho SpecForge EAGLE-3.1")
    parser.add_argument("--sharegpt-path", type=str, default="data/raw/sharegpt.json", help="Đường dẫn file ShareGPT JSON")
    parser.add_argument("--vi-path", type=str, default="data/raw/vietnamese_invoices.json", help="Đường dẫn file Tiếng Việt JSON")
    parser.add_argument("--output-path", type=str, default="data/processed/hybrid_train.jsonl", help="Đường dẫn file xuất .jsonl")
    parser.add_argument("--sharegpt-limit", type=int, default=20000, help="Số lượng mẫu ShareGPT tối đa")
    parser.add_argument("--vi-limit", type=int, default=10000, help="Số lượng mẫu Tiếng Việt tối đa")
    parser.add_argument("--seed", type=int, default=42, help="Seed xáo trộn dữ liệu")
    args = parser.parse_args()

    build_hybrid_dataset(
        sharegpt_path=args.sharegpt_path,
        vietnamese_path=args.vi_path,
        output_path=args.output_path,
        sharegpt_limit=args.sharegpt_limit,
        vi_limit=args.vi_limit,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Chạy test để xác nhận PASS**

Run: `pytest tests/test_dataset.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Gợi ý Git Commit (Người dùng tự chạy)**

```bash
git add data/prepare_hybrid_dataset.py tests/test_dataset.py
# Gợi ý commit:
# git commit -m "feat: format hybrid dataset into ShareGPT jsonl for SpecForge"
```

---

### Task 3: Cập Nhật `requirements.txt` và Cấu Hình SpecForge EAGLE-3.1

**Files:**
- Modify: `requirements.txt`
- Modify: `configs/eagle3.1_qwen2.5_7b.json`
- Modify: `configs/train_config.yaml`
- Modify: `tests/test_scaffold.py`
- Modify: `tests/test_configs.py`

**Interfaces:**
- `configs/eagle3.1_qwen2.5_7b.json`: Cấu hình draft model chuẩn EAGLE-3.1 cho Qwen2.5-7B (`hidden_size: 3584`, `intermediate_size: 18944`, `num_attention_heads: 28`, `num_key_value_heads: 4`, `fc_norm: true`, `norm_output: true`, `eagle_version: "3.1"`).
- `configs/train_config.yaml`: Chứa `target_model_path`, `draft_model_config`, `train_data_path`, `output_dir`, `num_epochs`, `batch_size`, `gradient_accumulation_steps`, `learning_rate`, `max_seq_len`, `ttt_length`, `tp_size`.

- [ ] **Step 1: Viết test kiểm tra dependencies và cấu hình SpecForge EAGLE-3.1**

Cập nhật `tests/test_scaffold.py`:
```python
# tests/test_scaffold.py
from pathlib import Path

def test_essential_files_exist():
    root = Path(__file__).parent.parent
    assert (root / "requirements.txt").exists(), "requirements.txt must exist"
    assert (root / ".gitignore").exists(), ".gitignore must exist"

def test_requirements_contains_specforge_and_sglang():
    root = Path(__file__).parent.parent
    req_content = (root / "requirements.txt").read_text(encoding="utf-8")
    assert "torch" in req_content
    assert "transformers" in req_content
    assert "SpecForge" in req_content
    assert "sglang" in req_content
```

Cập nhật `tests/test_configs.py`:
```python
# tests/test_configs.py
import json
import yaml
from pathlib import Path

def test_eagle31_model_config_integrity():
    root = Path(__file__).parent.parent
    config_path = root / "configs" / "eagle3.1_qwen2.5_7b.json"
    assert config_path.exists(), "configs/eagle3.1_qwen2.5_7b.json must exist"
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["hidden_size"] == 3584
    assert data["intermediate_size"] == 18944
    assert data["num_hidden_layers"] == 1
    assert data["num_attention_heads"] == 28
    assert data["num_key_value_heads"] == 4
    assert data["target_layer"] == 28
    assert data["use_fc_norm"] is True
    assert data.get("use_post_norm_feedback") is True
    assert data.get("eagle_version") == "3.1"
    assert "Eagle3_1Qwen2ForCausalLM" in data["architectures"]
    assert "EagleQwen2ForCausalLM" in data["architectures"]

def test_specforge_train_config_integrity():
    root = Path(__file__).parent.parent
    config_path = root / "configs" / "train_config.yaml"
    assert config_path.exists(), "configs/train_config.yaml must exist"
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert data["target_model_path"] == "Qwen/Qwen2.5-7B-Instruct"
    assert data["draft_model_config"] == "configs/eagle3.1_qwen2.5_7b.json"
    assert data["train_data_path"] == "data/processed/hybrid_train.jsonl"
    assert data["output_dir"] == "checkpoints/qwen2.5-7b-eagle3.1"
    assert data["num_epochs"] == 3
    assert data["batch_size"] == 2
    assert data["gradient_accumulation_steps"] == 4
    assert float(data["learning_rate"]) == 1.0e-4
    assert data["max_seq_len"] == 2048
    assert data["ttt_length"] == 5
    assert data.get("tp_size", 1) == 1
```

- [ ] **Step 2: Chạy test để xác nhận FAIL**

Run: `pytest tests/test_scaffold.py tests/test_configs.py -v`
Expected: FAIL vì `requirements.txt` chưa có SpecForge/sglang và `train_config.yaml` chưa có định dạng SpecForge mới.

- [ ] **Step 3: Cập nhật `requirements.txt`, `configs/eagle3.1_qwen2.5_7b.json` và `configs/train_config.yaml`**

Nội dung `requirements.txt`:
```text
torch>=2.4.0
transformers>=4.45.0
accelerate>=0.34.0
datasets
sglang[all]
git+https://github.com/sgl-project/SpecForge.git
huggingface_hub
pyyaml
pytest
safetensors>=0.4.4
tqdm>=4.66.5
modal>=1.6.0
```

Nội dung `configs/eagle3.1_qwen2.5_7b.json`:
```json
{
  "architectures": [
    "Eagle3_1Qwen2ForCausalLM",
    "EagleQwen2ForCausalLM"
  ],
  "model_type": "eagle_qwen2",
  "eagle_version": "3.1",
  "hidden_size": 3584,
  "intermediate_size": 18944,
  "num_hidden_layers": 1,
  "num_attention_heads": 28,
  "num_key_value_heads": 4,
  "hidden_act": "silu",
  "max_position_embeddings": 32768,
  "initializer_range": 0.02,
  "rms_norm_eps": 1e-06,
  "vocab_size": 152064,
  "target_model_name_or_path": "Qwen/Qwen2.5-7B-Instruct",
  "target_layer": 28,
  "use_fc_norm": true,
  "use_post_norm_feedback": true,
  "torch_dtype": "float16"
}
```

Nội dung `configs/train_config.yaml`:
```yaml
target_model_path: "Qwen/Qwen2.5-7B-Instruct"
draft_model_config: "configs/eagle3.1_qwen2.5_7b.json"
train_data_path: "data/processed/hybrid_train.jsonl"
output_dir: "checkpoints/qwen2.5-7b-eagle3.1"
num_epochs: 3
batch_size: 2
gradient_accumulation_steps: 4
learning_rate: 1.0e-4
max_seq_len: 2048
ttt_length: 5
tp_size: 1
```

- [ ] **Step 4: Chạy test để xác nhận PASS**

Run: `pytest tests/test_scaffold.py tests/test_configs.py -v`
Expected: PASS.

- [ ] **Step 5: Gợi ý Git Commit (Người dùng tự chạy)**

```bash
git add requirements.txt configs/ tests/test_scaffold.py tests/test_configs.py
# Gợi ý commit:
# git commit -m "feat: update dependencies and configs for SpecForge EAGLE-3.1"
```

---

### Task 4: Viết Lại `training/train.py` Thành SpecForge Launcher

**Files:**
- Modify: `training/train.py`
- Modify: `tests/test_train.py`

**Interfaces:**
- `build_specforge_command(cfg: dict, num_gpus: int = 1) -> list[str]`
- `run_specforge_training(config_path: str, num_gpus: int = 1, dry_run: bool = False) -> list[str]`

- [ ] **Step 1: Viết test cho CLI builder và launcher của `training/train.py`**

Cập nhật `tests/test_train.py`:
```python
# tests/test_train.py
import pytest
from pathlib import Path
from training.train import parse_training_args, load_training_config, build_specforge_command, run_specforge_training

def test_parse_training_args_defaults():
    args = parse_training_args(["--config", "configs/train_config.yaml", "--gpus", "1"])
    assert args.config == "configs/train_config.yaml"
    assert args.gpus == 1

def test_build_specforge_command():
    cfg = {
        "target_model_path": "Qwen/Qwen2.5-7B-Instruct",
        "draft_model_config": "configs/eagle3.1_qwen2.5_7b.json",
        "train_data_path": "data/processed/hybrid_train.jsonl",
        "output_dir": "checkpoints/qwen2.5-7b-eagle3.1",
        "num_epochs": 3,
        "batch_size": 2,
        "gradient_accumulation_steps": 4,
        "learning_rate": 1e-4,
        "max_seq_len": 2048,
        "ttt_length": 5,
        "tp_size": 1,
    }
    cmd = build_specforge_command(cfg, num_gpus=2)
    assert cmd[0] == "torchrun"
    assert "--nproc_per_node=2" in cmd
    assert "-m" in cmd
    assert "specforge.train" in cmd
    assert "--model-path" in cmd
    assert "Qwen/Qwen2.5-7B-Instruct" in cmd
    assert "--draft-config" in cmd
    assert "configs/eagle3.1_qwen2.5_7b.json" in cmd
    assert "--data-path" in cmd
    assert "data/processed/hybrid_train.jsonl" in cmd
    assert "--output-dir" in cmd
    assert "checkpoints/qwen2.5-7b-eagle3.1" in cmd
    assert "--ttt-length" in cmd
    assert "5" in cmd

def test_run_specforge_training_dry_run():
    root = Path(__file__).parent.parent
    config_path = str(root / "configs" / "train_config.yaml")
    cmd = run_specforge_training(config_path, num_gpus=1, dry_run=True)
    assert cmd[0] == "torchrun"
    assert "--nproc_per_node=1" in cmd
```

- [ ] **Step 2: Chạy test để xác nhận FAIL**

Run: `pytest tests/test_train.py -v`
Expected: FAIL vì `training/train.py` chưa cài đặt `build_specforge_command` và `run_specforge_training(..., dry_run=True)`.

- [ ] **Step 3: Viết lại `training/train.py`**

```python
# training/train.py
"""
Script train.py
Launcher nạp cấu hình configs/train_config.yaml và khởi chạy tiến trình huấn luyện
SpecForge (sgl-project/SpecForge) cho đầu nháp EAGLE-3.1 (Qwen2.5-7B).
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_training_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Khởi chạy SpecForge Training Launcher cho EAGLE-3.1")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml", help="Đường dẫn file cấu hình YAML")
    parser.add_argument("--gpus", type=int, default=1, help="Số lượng GPU sử dụng (nproc_per_node)")
    parser.add_argument("--dry-run", action="store_true", help="Chỉ sinh câu lệnh mà không thực thi subprocess")
    return parser.parse_args(args)


def load_training_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_specforge_command(cfg: Dict[str, Any], num_gpus: int = 1) -> List[str]:
    """
    Chuyển đổi các siêu tham số từ train_config.yaml sang command line gọi torchrun specforge.train.
    """
    cmd = [
        "torchrun",
        f"--nproc_per_node={num_gpus}",
        "-m", "specforge.train",
        "--model-path", str(cfg["target_model_path"]),
        "--draft-config", str(cfg["draft_model_config"]),
        "--data-path", str(cfg["train_data_path"]),
        "--output-dir", str(cfg["output_dir"]),
        "--num-epochs", str(cfg.get("num_epochs", 3)),
        "--micro-batch-size", str(cfg.get("batch_size", 2)),
        "--grad-accum-steps", str(cfg.get("gradient_accumulation_steps", 4)),
        "--lr", str(cfg.get("learning_rate", 1.0e-4)),
        "--max-seq-len", str(cfg.get("max_seq_len", 2048)),
        "--ttt-length", str(cfg.get("ttt_length", 5)),
        "--tp-size", str(cfg.get("tp_size", 1)),
    ]
    return cmd


def run_specforge_training(config_path: str, num_gpus: int = 1, dry_run: bool = False) -> List[str]:
    cfg = load_training_config(config_path)
    cmd = build_specforge_command(cfg, num_gpus=num_gpus)
    logger.info("Lệnh thực thi SpecForge: %s", " ".join(cmd))

    if dry_run:
        logger.info("[DRY RUN] Bỏ qua thực thi subprocess thực tế.")
        return cmd

    subprocess.run(cmd, check=True)
    return cmd


def main():
    args = parse_training_args()
    run_specforge_training(config_path=args.config, num_gpus=args.gpus, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Chạy test để xác nhận PASS**

Run: `pytest tests/test_train.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Gợi ý Git Commit (Người dùng tự chạy)**

```bash
git add training/train.py tests/test_train.py
# Gợi ý commit:
# git commit -m "feat: implement SpecForge training launcher in training/train.py"
```

---

### Task 5: Tinh Chỉnh `export/package_and_upload.py` Cho Checkpoint SpecForge

**Files:**
- Modify: `export/package_and_upload.py`
- Modify: `tests/test_export.py`

**Interfaces:**
- `verify_package_integrity(export_dir: Any) -> bool`
- `attach_model_card(export_dir: str, model_card_path: str = "MODEL_CARD.md")`
- `upload_to_huggingface(export_dir: str, repo_id: str, token: str | None = None)`

- [ ] **Step 1: Viết test cho gói đóng gói và upload của SpecForge checkpoint**

Cập nhật `tests/test_export.py`:
```python
# tests/test_export.py
import json
import pytest
from pathlib import Path
from export.package_and_upload import (
    verify_package_integrity,
    parse_export_args,
    attach_model_card,
)

def test_verify_package_integrity(tmp_path):
    # Thiếu file -> False
    assert verify_package_integrity(tmp_path) is False

    # Đầy đủ config.json và model.safetensors -> True
    (tmp_path / "config.json").write_text(json.dumps({
        "architectures": ["Eagle3_1Qwen2ForCausalLM", "EagleQwen2ForCausalLM"],
        "eagle_version": "3.1"
    }))
    (tmp_path / "model.safetensors").write_bytes(b"dummy_weights")
    assert verify_package_integrity(tmp_path) is True

def test_attach_model_card(tmp_path):
    card_file = tmp_path / "MODEL_CARD.md"
    card_file.write_text("# Model Card Test", encoding="utf-8")
    export_dir = tmp_path / "export_dir"
    export_dir.mkdir()

    attach_model_card(str(export_dir), str(card_file))
    readme_in_export = export_dir / "README.md"
    assert readme_in_export.exists()
    assert "# Model Card Test" in readme_in_export.read_text(encoding="utf-8")

def test_parse_export_args():
    args = parse_export_args(["--repo-id", "theflyingkhui04/qwen2.5-7b-eagle3.1-speculator", "--output-dir", "checkpoints/qwen2.5-7b-eagle3.1"])
    assert args.repo_id == "theflyingkhui04/qwen2.5-7b-eagle3.1-speculator"
    assert args.output_dir == "checkpoints/qwen2.5-7b-eagle3.1"
```

- [ ] **Step 2: Chạy test để xác nhận FAIL**

Run: `pytest tests/test_export.py -v`
Expected: FAIL vì `attach_model_card` chưa được định nghĩa.

- [ ] **Step 3: Cập nhật `export/package_and_upload.py`**

```python
# export/package_and_upload.py
"""
Script package_and_upload.py
Xác thực tính toàn vẹn của thư mục checkpoint sinh ra từ SpecForge (config.json, model.safetensors),
đính kèm Model Card (README.md) và tự động tải lên Hugging Face Hub.
"""

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def verify_package_integrity(export_dir: Any) -> bool:
    """
    Kiểm định gói xuất: bắt buộc phải có config.json và model.safetensors.
    """
    exp_path = Path(export_dir)
    if not exp_path.exists():
        return False

    config_file = exp_path / "config.json"
    weights_file = exp_path / "model.safetensors"

    if not config_file.exists() or not weights_file.exists():
        return False

    try:
        with open(config_file, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        archs = cfg.get("architectures", [])
        if "EagleQwen2ForCausalLM" not in archs and "Eagle3_1Qwen2ForCausalLM" not in archs:
            return False
    except Exception:
        return False

    return True


def attach_model_card(export_dir: str, model_card_path: str = "MODEL_CARD.md") -> None:
    """
    Copy MODEL_CARD.md thành README.md trong thư mục checkpoint để hiển thị trên Hugging Face.
    """
    target = Path(export_dir) / "README.md"
    card_source = Path(model_card_path)
    if card_source.exists():
        shutil.copyfile(card_source, target)
        logger.info("Đã đính kèm Model Card vào: %s", target)
    else:
        logger.warning("Không tìm thấy %s để đính kèm.", model_card_path)


def parse_export_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Xác thực và tải checkpoint SpecForge lên Hugging Face")
    parser.add_argument("--output-dir", type=str, default="checkpoints/qwen2.5-7b-eagle3.1", help="Thư mục checkpoint cần kiểm tra")
    parser.add_argument("--repo-id", type=str, default="theflyingkhui04/qwen2.5-7b-eagle3.1-speculator", help="Repo ID trên Hugging Face")
    parser.add_argument("--model-card", type=str, default="MODEL_CARD.md", help="Đường dẫn file Model Card markdown")
    parser.add_argument("--upload", action="store_true", help="Bật cờ upload lên Hugging Face Hub")
    return parser.parse_args(args)


def upload_to_huggingface(export_dir: str, repo_id: str, token: Optional[str] = None):
    """
    Tải thư mục checkpoint lên Hugging Face Hub.
    """
    try:
        from huggingface_hub import HfApi
    except ImportError:
        logger.error("Cần cài đặt thư viện huggingface_hub để upload.")
        sys.exit(1)

    hf_token = token or os.getenv("HF_TOKEN")
    if not hf_token:
        logger.warning("Không tìm thấy HF_TOKEN! Bỏ qua bước tải lên Hugging Face.")
        return

    logger.info("Đang tải toàn bộ thư mục %s lên repo %s...", export_dir, repo_id)
    api = HfApi(token=hf_token)
    api.create_repo(repo_id=repo_id, exist_ok=True, repo_type="model")
    api.upload_folder(
        folder_path=export_dir,
        repo_id=repo_id,
        repo_type="model",
        commit_message="feat: upload EAGLE-3.1 draft head weights trained via SpecForge",
    )
    logger.info("Đã upload thành công lên: https://huggingface.co/%s", repo_id)


def main():
    args = parse_export_args()
    if not verify_package_integrity(args.output_dir):
        logger.error("Thư mục %s không hợp lệ hoặc thiếu checkpoint!", args.output_dir)
        sys.exit(1)

    attach_model_card(args.output_dir, args.model_card)

    if args.upload:
        upload_to_huggingface(export_dir=args.output_dir, repo_id=args.repo_id)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Chạy test để xác nhận PASS**

Run: `pytest tests/test_export.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Gợi ý Git Commit (Người dùng tự chạy)**

```bash
git add export/package_and_upload.py tests/test_export.py
# Gợi ý commit:
# git commit -m "feat: adapt packaging and upload script for SpecForge checkpoints"
```

---

### Task 6: Tối Ưu Pipeline Điều Phối 3 Bước (`run_all.sh`, `modal_pipeline.py`, `notebooks/train_on_vastai.ipynb`)

**Files:**
- Modify: `run_all.sh`
- Modify: `modal_pipeline.py`
- Modify: `notebooks/train_on_vastai.ipynb`
- Modify: `tests/test_script_guards.py`

**Interfaces:**
- Pipeline 3 bước:
  1. `python -m data.prepare_hybrid_dataset`
  2. `python -m training.train --config configs/train_config.yaml --gpus $NUM_GPUS`
  3. `python -m export.package_and_upload --output-dir checkpoints/qwen2.5-7b-eagle3.1`
- Không còn bước gọi `extraction/extract_hidden_states.py`.

- [ ] **Step 1: Cập nhật test kiểm tra chốt chặn an toàn trong `run_all.sh` 3 bước mới**

Cập nhật `tests/test_script_guards.py`:
```python
# tests/test_script_guards.py
from pathlib import Path

def test_run_all_script_safeguards():
    root = Path(__file__).parent.parent
    script_path = root / "run_all.sh"
    assert script_path.exists(), "run_all.sh must exist"

    content = script_path.read_text(encoding="utf-8")
    assert "pipefail" in content
    assert "disk" in content.lower()
    assert "vastai stop instance" in content
    assert "50" in content  # 50GB disk check
    assert "prepare_hybrid_dataset.py" in content
    assert "extract_hidden_states.py" not in content  # Đã loại bỏ trích xuất offline
    assert "train.py" in content
    assert "package_and_upload.py" in content
```

- [ ] **Step 2: Chạy test để xác nhận FAIL**

Run: `pytest tests/test_script_guards.py -v`
Expected: FAIL vì `run_all.sh` cũ vẫn chứa lệnh gọi `extract_hidden_states.py`.

- [ ] **Step 3: Cập nhật `run_all.sh`**

```bash
#!/usr/bin/env bash
# ==============================================================================
# Script: run_all.sh
# Pipeline Tự Động Hóa 1-Click Huấn Luyện Đầu Nháp EAGLE-3.1 Cho Qwen2.5-7B
# Sử dụng SpecForge (3 bước: Prepare Dataset -> SpecForge Train -> Package & Upload)
# Tích hợp: Disk Guard (>=50GB), Cost Ceiling Guard (<=6.0h), Auto-Stop Vast.ai
# ==============================================================================

set -euo pipefail

CONTAINER_ID="${CONTAINER_ID:-${VAST_CONTAINERLABEL:-""}}"
MAX_RUNTIME_HOURS=6.0
MIN_DISK_FREE_GB=50
NUM_GPUS="${NUM_GPUS:-1}"
LOG_FILE="training_pipeline.log"

echo "=== [EAGLE-3.1 SpecForge Pipeline Bắt Đầu] ===" | tee -a "$LOG_FILE"
echo "Thời gian bắt đầu: $(date)" | tee -a "$LOG_FILE"
echo "Container ID Vast.ai: ${CONTAINER_ID:-'(Chạy local/chưa cấu hình)'}" | tee -a "$LOG_FILE"

cleanup_and_stop() {
    local exit_code=$?
    echo "=== [Kết thúc tiến trình với mã thoát: $exit_code] ===" | tee -a "$LOG_FILE"
    echo "Thời gian kết thúc: $(date)" | tee -a "$LOG_FILE"

    if [ -n "$CONTAINER_ID" ]; then
        echo ">>> Kích hoạt tự hủy máy Vast.ai: vastai stop instance $CONTAINER_ID" | tee -a "$LOG_FILE"
        vastai stop instance "$CONTAINER_ID" || true
    fi
    exit $exit_code
}

trap cleanup_and_stop EXIT INT TERM

# Disk Overflow Guard
echo ">>> [Kiểm tra ổ đĩa] Yêu cầu tối thiểu: ${MIN_DISK_FREE_GB} GB..." | tee -a "$LOG_FILE"
FREE_DISK_GB=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')

if [ "$FREE_DISK_GB" -lt "$MIN_DISK_FREE_GB" ]; then
    echo "❌ LỖI: Dung lượng ổ đĩa khả dụng (${FREE_DISK_GB} GB) nhỏ hơn mức an toàn (${MIN_DISK_FREE_GB} GB)!" | tee -a "$LOG_FILE"
    exit 1
fi
echo "✅ Dung lượng đĩa khả dụng: ${FREE_DISK_GB} GB (Đạt chuẩn an toàn)" | tee -a "$LOG_FILE"

# Cost Ceiling Guard: Tự ngắt sau 6.0 giờ
(
    sleep 21600
    echo "❌ CẢNH BÁO: Quá trần thời gian 6.0 giờ! Tự động ngắt..." | tee -a "$LOG_FILE"
    if [ -n "$CONTAINER_ID" ]; then
        vastai stop instance "$CONTAINER_ID"
    fi
    kill -9 $$
) &
TIMER_PID=$!

# ==============================================================================
# 3 Bước Kỹ Thuật Chạy SpecForge
# ==============================================================================

# Bước 1: Chuẩn bị dữ liệu lai định dạng ShareGPT JSONL
echo ">>> [Bước 1/3] Chuẩn bị dữ liệu lai (ShareGPT 20k + Tiếng Việt 10k -> JSONL)..." | tee -a "$LOG_FILE"
python data/prepare_hybrid_dataset.py \
    --sharegpt-path "data/raw/sharegpt.json" \
    --vi-path "data/raw/vietnamese_invoices.json" \
    --output-path "data/processed/hybrid_train.jsonl" \
    --sharegpt-limit 20000 \
    --vi-limit 10000

# Bước 2: Huấn luyện với SpecForge (Ghép hidden trực tiếp VRAM & unroll TTT đa bước)
echo ">>> [Bước 2/3] Huấn luyện đầu nháp EAGLE-3.1 qua SpecForge Launcher..." | tee -a "$LOG_FILE"
python training/train.py \
    --config "configs/train_config.yaml" \
    --gpus "$NUM_GPUS"

# Bước 3: Đóng gói và upload lên Hugging Face Hub
echo ">>> [Bước 3/3] Đóng gói và upload lên Hugging Face Hub..." | tee -a "$LOG_FILE"
python export/package_and_upload.py \
    --output-dir "checkpoints/qwen2.5-7b-eagle3.1" \
    --repo-id "theflyingkhui04/qwen2.5-7b-eagle3.1-speculator" \
    --upload

kill "$TIMER_PID" || true
echo "=== [HOÀN TẤT THÀNH CÔNG TOÀN BỘ PIPELINE HUẤN LUYỆN SPECFORGE] ===" | tee -a "$LOG_FILE"
```

- [ ] **Step 4: Cập nhật `modal_pipeline.py`**

Điều chỉnh `modal_pipeline.py`:
- Cài đặt `git+https://github.com/sgl-project/SpecForge.git` và `sglang[all]`.
- Bỏ hàm `extract_features` và mount mã nguồn chỉ gồm `training`, `data`, `export`.
- Hàm `train_specforge` chạy trên GPU A100-80GB gọi `run_specforge_training`.

```python
"""
Modal.com Serverless Pipeline for Qwen2.5-7B EAGLE-3.1 Speculator Training (SpecForge)
"""

import os
import sys
from pathlib import Path
import modal

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

volume = modal.Volume.from_name("qwen-eagle3-volume", create_if_missing=True)

app_image = (
    modal.Image.debian_slim(python_version="3.11")
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .apt_install("git")
    .pip_install(
        "torch>=2.4.0",
        "transformers>=4.45.0",
        "accelerate>=0.34.0",
        "safetensors>=0.4.4",
        "datasets",
        "sglang[all]",
        "git+https://github.com/sgl-project/SpecForge.git",
        "pyyaml>=6.0.1",
        "tqdm>=4.66.0",
        "huggingface_hub>=0.24.0",
        "wandb>=0.16.0",
    )
    .add_local_python_source("training", "data", "export")
    .add_local_dir("configs", remote_path="/root/configs")
)

local_data_path = Path("data/processed/hybrid_train.jsonl")
if local_data_path.exists():
    app_image = app_image.add_local_file(
        str(local_data_path),
        remote_path="/root/data/processed/hybrid_train.jsonl",
    )

app = modal.App("qwen2-5-eagle3.1-specforge", image=app_image)


@app.function(
    gpu="A100-80GB",
    timeout=28800,
    volumes={"/vol": volume},
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def train_specforge(dry_run: bool = False):
    from training.train import run_specforge_training
    config_path = "/root/configs/train_config.yaml"
    return run_specforge_training(config_path, num_gpus=1, dry_run=dry_run)


@app.function(
    volumes={"/vol": volume},
    secrets=[modal.Secret.from_name("my-huggingface-secret", required_keys=["HF_TOKEN"])],
    timeout=1800,
)
def export_and_upload(repo_id: str = "theflyingkhui04/qwen2.5-7b-eagle3.1-speculator"):
    from export.package_and_upload import upload_to_huggingface, verify_package_integrity
    export_dir = "/vol/checkpoints/qwen2.5-7b-eagle3.1"
    if verify_package_integrity(export_dir):
        upload_to_huggingface(export_dir=export_dir, repo_id=repo_id)
    return repo_id


@app.local_entrypoint()
def main(step: str = "train", repo_id: str = "theflyingkhui04/qwen2.5-7b-eagle3.1-speculator"):
    if step == "train":
        train_specforge.remote()
    elif step == "export":
        export_and_upload.remote(repo_id=repo_id)
    elif step == "all":
        train_specforge.remote()
        export_and_upload.remote(repo_id=repo_id)
```

- [ ] **Step 5: Cập nhật `notebooks/train_on_vastai.ipynb`**

Cập nhật các command cell trong `notebooks/train_on_vastai.ipynb` để:
- Clone/cài đặt `SpecForge` và `sglang[all]`.
- Gọi lệnh 3 bước của `run_all.sh`.

- [ ] **Step 6: Chạy test để xác nhận PASS**

Run: `pytest tests/test_script_guards.py -v`
Expected: PASS.

- [ ] **Step 7: Gợi ý Git Commit (Người dùng tự chạy)**

```bash
git add run_all.sh modal_pipeline.py notebooks/train_on_vastai.ipynb tests/test_script_guards.py
# Gợi ý commit:
# git commit -m "feat: streamline 3-step orchestration pipeline with SpecForge"
```

---

### Task 7: Ghim Quy Tắc EAGLE-3.1 Vào Workspace Rules & Hoàn Thiện Toàn Diện Bộ Test

**Files:**
- Modify: `AGENTS.md`
- Modify: `tests/test_final_structure.py`
- Modify: `README.md`

**Interfaces:**
- `AGENTS.md`: Chứa quy tắc ghim kiến trúc EAGLE-3.1, quy tắc không commit tự động, và không dùng LaTeX.
- `tests/test_final_structure.py`: Kiểm định layout hoàn chỉnh của repo sau refactor sang SpecForge.

- [ ] **Step 1: Cập nhật `tests/test_final_structure.py`**

```python
# tests/test_final_structure.py
import json
import pytest
from pathlib import Path

def test_full_repository_layout():
    root = Path(__file__).parent.parent
    required = [
        "README.md",
        "requirements.txt",
        ".gitignore",
        "configs/eagle3.1_qwen2.5_7b.json",
        "configs/train_config.yaml",
        "data/prepare_hybrid_dataset.py",
        "training/train.py",
        "export/package_and_upload.py",
        "notebooks/train_on_vastai.ipynb",
        "run_all.sh",
    ]
    for rel_path in required:
        file_path = root / rel_path
        assert file_path.exists(), f"Thiếu file bắt buộc theo spec: {rel_path}"

    # Kiểm tra các module cũ đã được loại bỏ hoàn toàn
    deprecated = [
        "extraction/extract_hidden_states.py",
        "training/model.py",
        "training/loss.py",
        "configs/eagle3_qwen2.5_7b.json",
    ]
    for rel_path in deprecated:
        file_path = root / rel_path
        assert not file_path.exists(), f"File cũ chưa được loại bỏ: {rel_path}"

def test_notebook_is_valid_json():
    root = Path(__file__).parent.parent
    nb_path = root / "notebooks" / "train_on_vastai.ipynb"
    assert nb_path.exists()
    with open(nb_path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)
    assert "cells" in nb_data
    assert "metadata" in nb_data
    assert len(nb_data["cells"]) > 0

def test_readme_contains_specforge_and_eagle31():
    root = Path(__file__).parent.parent
    readme_path = root / "README.md"
    assert readme_path.exists()
    content = readme_path.read_text(encoding="utf-8")
    assert "EAGLE-3.1" in content
    assert "Qwen2.5-7B" in content
    assert "SpecForge" in content
```

- [ ] **Step 2: Chạy test để xác nhận FAIL**

Run: `pytest tests/test_final_structure.py -v`
Expected: FAIL vì `README.md` chưa cập nhật "SpecForge" hoặc `AGENTS.md` chưa cập nhật.

- [ ] **Step 3: Cập nhật `AGENTS.md` và `README.md`**

Cập nhật `AGENTS.md`:
```markdown
# Workspace Rules

- **Không tự động commit:** Tuyệt đối không tự động chạy lệnh `git commit`. Sau khi hoàn thành code, chỉ cung cấp commit message gợi ý để người dùng tự kiểm tra và commit.
- **Không dùng LaTeX:** Trình bày công thức bằng chữ thường và Unicode trực quan.
- **Ghim chuẩn Kiến trúc EAGLE-3.1:** Luôn ghim chuẩn kiến trúc EAGLE-3.1 (1 tầng Decoder, fc_norm, norm_output, unroll autoregressive đa bước qua SpecForge, target layer 28 của Qwen2.5-7B) và cấu hình chuẩn duy nhất tại `configs/eagle3.1_qwen2.5_7b.json`.
```

Cập nhật `README.md` để bổ sung mô tả về SpecForge và quy trình 3 bước mới.

- [ ] **Step 4: Chạy toàn bộ test suite để xác nhận 100% PASS**

Run: `pytest tests/ -v`
Expected: Tất cả bài test (Scaffold, Configs, Dataset, Train launcher, Export, Script guards, Final layout) đều PASS 100%.

- [ ] **Step 5: Gợi ý Git Commit (Người dùng tự chạy)**

```bash
git add AGENTS.md README.md tests/test_final_structure.py
# Gợi ý commit:
# git commit -m "docs: pin EAGLE-3.1 architecture rules and update README for SpecForge"
```
