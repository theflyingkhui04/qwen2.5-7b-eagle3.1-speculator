# Đặc tả nhiệm vụ: Task 5 - Đóng gói checkpoint và tải lên Hugging Face Hub

- **Mã nhiệm vụ:** `TASK-05`
- **Tên nhiệm vụ:** Tinh chỉnh module đóng gói checkpoint và tự động tải lên Hugging Face Hub
- **Trạng thái:** Đã hoàn thành (Done)
- **Tài liệu tham chiếu:** [worklogs/task-5.md](file:///d:/DoAn/qwen2.5-7b-eagle3.1-speculator/worklogs/task-5.md), [export/package_and_upload.py](file:///d:/DoAn/qwen2.5-7b-eagle3.1-speculator/export/package_and_upload.py)

---

## 1. Mục tiêu kỹ thuật
1. Tinh chỉnh `export/package_and_upload.py` để tương thích hoàn toàn với thư mục đầu ra của SpecForge.
2. Vì SpecForge đã tự động xuất ra file trọng số an toàn `model.safetensors` và `config.json` chuẩn Hugging Face, script này tập trung vào 3 nhiệm vụ:
   - Xác thực tính toàn vẹn của thư mục checkpoint (`model.safetensors`, `config.json`).
   - Đính kèm tự động file mô tả `MODEL_CARD.md` (chứa siêu dữ liệu, hướng dẫn triển khai vLLM trên Tesla T4).
   - Tải toàn bộ thư mục lên repo chỉ định trên Hugging Face Hub qua thư viện `huggingface_hub.HfApi`.

---

## 2. Tiêu chuẩn nghiệm thu (Definition of Done - DoD)
- [x] Script kiểm tra đầy đủ sự hiện diện của `config.json` và `model.safetensors` trước khi upload.
- [x] Tự động đọc và đẩy kèm model card chuẩn.
