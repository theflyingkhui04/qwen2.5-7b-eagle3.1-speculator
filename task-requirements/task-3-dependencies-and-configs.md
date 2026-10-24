# Đặc tả nhiệm vụ: Task 3 - Cập nhật thư viện và file cấu hình

- **Mã nhiệm vụ:** `TASK-03`
- **Tên nhiệm vụ:** Cập nhật môi trường phụ thuộc (dependencies) và chuẩn hóa cấu hình SpecForge
- **Trạng thái:** Chờ thực hiện (Pending)
- **Tài liệu tham chiếu:** [configs/eagle3.1_qwen2.5_7b.json](file:///d:/DoAn/qwen2.5-7b-eagle3.1-speculator/configs/eagle3.1_qwen2.5_7b.json), [configs/train_config.yaml](file:///d:/DoAn/qwen2.5-7b-eagle3.1-speculator/configs/train_config.yaml)

---

## 1. Mục tiêu kỹ thuật
1. Cập nhật `requirements.txt` với đầy đủ các thư viện cần thiết để chạy SpecForge và SGLang (`torch>=2.4.0`, `transformers>=4.45.0`, `accelerate>=0.34.0`, `sglang[all]`, `git+https://github.com/sgl-project/SpecForge.git`, `datasets`, `pyyaml`, `pytest`).
2. Cấu trúc lại file siêu tham số `configs/train_config.yaml` khớp với các tham số dòng lệnh của SpecForge runner (`ttt_length: 5`, `tp_size: 1`, `max_seq_len: 2048`, `train_data_path: data/processed/train.jsonl`).
3. Xác thực file cấu hình `configs/eagle3.1_qwen2.5_7b.json` đáp ứng đầy đủ tiêu chuẩn của họ Qwen2.5-7B (1 Decoder Layer, 28 query heads, 4 KV heads, hidden size 3584, vocab size 152064).

---

## 2. Chi tiết cấu hình chuẩn

### `configs/train_config.yaml`:
```yaml
target_model_path: "Qwen/Qwen2.5-7B-Instruct"
draft_model_config: "configs/eagle3.1_qwen2.5_7b.json"
train_data_path: "data/processed/train.jsonl"
val_data_path: "data/processed/val.jsonl"
output_dir: "checkpoints/qwen2.5-7b-eagle3.1"
num_epochs: 3
batch_size: 2
gradient_accumulation_steps: 4
learning_rate: 1.0e-4
max_seq_len: 2048
ttt_length: 5
tp_size: 1
```

---

## 3. Tiêu chuẩn nghiệm thu (Definition of Done - DoD)
- [ ] `requirements.txt` chứa đầy đủ các gói thư viện cần thiết cho SpecForge.
- [ ] `configs/train_config.yaml` chứa đủ các trường tham số và trỏ đúng đường dẫn dữ liệu.
- [ ] Bài test `tests/test_configs.py` xác thực tính hợp lệ của YAML và JSON đạt 100% PASS.
