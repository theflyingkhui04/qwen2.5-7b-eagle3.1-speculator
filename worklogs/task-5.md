# Task 5: Đóng gói checkpoint SpecForge và tự động tải lên Hugging Face Hub

- **Dự án:** `qwen2.5-7b-eagle3.1-speculator`
- **Mã nhiệm vụ:** `TASK-05`
- **Mô hình mục tiêu (Target Model):** `Qwen/Qwen2.5-7B-Instruct`
- **Kiến trúc đầu nháp (Draft Head):** EAGLE-3.1 Speculator (1 Decoder Layer, FC Normalization, multi-layer hidden fusion)
- **Framework huấn luyện:** `sgl-project/SpecForge`
- **Ngày lập:** 25/10/2026
- **Trạng thái:** Đã hoàn thành (Done & Verified)
- **Tài liệu tham chiếu:** [task-requirements/task-5-export-and-packaging.md](file:///d:/DoAn/qwen2.5-7b-eagle3.1-speculator/task-requirements/task-5-export-and-packaging.md), [export/package_and_upload.py](file:///d:/DoAn/qwen2.5-7b-eagle3.1-speculator/export/package_and_upload.py), [tests/test_export.py](file:///d:/DoAn/qwen2.5-7b-eagle3.1-speculator/tests/test_export.py), [MODEL_CARD.md](file:///d:/DoAn/qwen2.5-7b-eagle3.1-speculator/MODEL_CARD.md)

---

## 1. Bối cảnh kỹ thuật và mục tiêu nhiệm vụ

Trước đây, khi tự viết vòng lặp huấn luyện, mô hình xuất ra file `.pt` chứa cả optimizer và scheduler của PyTorch (~1.6GB) nên cần qua bước chuyển đổi thủ công sang safetensors.

Với framework `sgl-project/SpecForge`, các checkpoint xuất ra tại thư mục `output_dir` (`checkpoints/qwen2.5-7b-eagle3.1`) đã mang định dạng an toàn `model.safetensors` và `config.json` chuẩn Hugging Face.

Do đó, nhiệm vụ Task 5 tinh chỉnh [export/package_and_upload.py](file:///d:/DoAn/qwen2.5-7b-eagle3.1-speculator/export/package_and_upload.py) tập trung vào:
1. Xác thực tính toàn vẹn của thư mục checkpoint SpecForge (`config.json` hợp lệ, chứa kiến trúc `Eagle3_1Qwen2ForCausalLM` hoặc `EagleQwen2ForCausalLM`, và sự hiện diện của `model.safetensors`).
2. Tự động đọc và sao chép [MODEL_CARD.md](file:///d:/DoAn/qwen2.5-7b-eagle3.1-speculator/MODEL_CARD.md) thành `README.md` trong thư mục checkpoint để hiển thị tài liệu hướng dẫn và siêu dữ liệu trên Hugging Face.
3. Tải toàn bộ thư mục lên kho chứa trên Hugging Face Hub qua `huggingface_hub.HfApi`.
4. Cung cấp cờ `--dry-run` để kiểm tra cục bộ mà không đòi hỏi token Hugging Face thật.

---

## 2. Thiết kế giao diện và các hàm xử lý

Module [export/package_and_upload.py](file:///d:/DoAn/qwen2.5-7b-eagle3.1-speculator/export/package_and_upload.py) cung cấp các hàm chính:

1. `verify_package_integrity(export_dir: Any) -> bool`:
   - Kiểm tra sự tồn tại của `config.json` và `model.safetensors`.
   - Phân tích cú pháp JSON và kiểm tra mảng `architectures`.
2. `attach_model_card(export_dir: str, model_card_path: str = "MODEL_CARD.md") -> bool`:
   - Sao chép [MODEL_CARD.md](file:///d:/DoAn/qwen2.5-7b-eagle3.1-speculator/MODEL_CARD.md) thành `README.md` trong thư mục đích.
3. `upload_to_huggingface(export_dir: str, repo_id: str, token: Optional[str] = None, dry_run: bool = False) -> bool`:
   - Quản lý việc kết nối `HfApi`, tạo repo nếu chưa tồn tại (`repo_type="model"`) và đẩy toàn bộ thư mục bằng `upload_folder`.
   - Trong chế độ `--dry-run`, in log xác nhận và trả về `True` mà không gọi mạng.
4. `parse_export_args(args: Optional[List[str]] = None) -> argparse.Namespace`:
   - Hỗ trợ các cờ: `--output-dir` (hoặc `--checkpoint-dir`), `--repo-id`, `--model-card`, `--upload`, `--dry-run`.

---

## 3. Cập nhật Model Card chuẩn SpecForge

Đã tinh chỉnh [MODEL_CARD.md](file:///d:/DoAn/qwen2.5-7b-eagle3.1-speculator/MODEL_CARD.md) để phản ánh trung thực phương pháp huấn luyện mới:
- Framework: `sgl-project/SpecForge`.
- Dữ liệu: 40.000 mẫu `train.jsonl` (Tiếng Việt, Hóa đơn MC-OCR, CoT).
- Unroll đa bước: `ttt_length = 5`.
- Thời lượng: 3 epochs (~15.000 optimization steps với effective batch size 8).
- Hướng dẫn nạp vào runtime vLLM trên Tesla T4 / GPU đám mây.

---

## 4. Nhật ký kiểm thử và nghiệm thu kỹ thuật (Execution Log)

### 4.1. Bộ test tự động `tests/test_export.py`
Bộ test bao gồm 6 bài kiểm tra:
- `test_build_hf_config_for_vllm_float16`: Kiểm tra cấu trúc config tương thích vLLM.
- `test_verify_package_integrity`: Kiểm tra phát hiện đúng khi thiếu file hoặc khi gói hợp lệ.
- `test_attach_model_card`: Kiểm tra sao chép đúng nội dung model card vào `README.md`.
- `test_parse_export_args`: Kiểm tra phân tích cú pháp các cờ CLI.
- `test_upload_to_huggingface_dry_run`: Kiểm tra chế độ chạy thử an toàn.
- `test_upload_to_huggingface_with_mock_api`: Sử dụng mock `HfApi` để kiểm tra việc gọi `create_repo` và `upload_folder` với tham số chính xác.

### 4.2. Kết quả kiểm thử thực tế
```
tests/test_export.py ......                                              [100%]
============================== 6 passed in 0.20s ==============================

Toàn bộ test suite trong dự án:
tests/test_cleanup_deprecated.py ..                                      [  5%]
tests/test_configs.py ..                                                 [ 10%]
tests/test_dataset.py ......                                             [ 27%]
tests/test_export.py ......                                              [ 43%]
tests/test_final_structure.py ...                                        [ 51%]
tests/test_scaffold.py ..                                                [ 56%]
tests/test_script_guards.py .                                            [ 59%]
tests/test_train.py ...............                                      [100%]
============================= 37 passed in 0.51s ==============================
```
Toàn bộ 37 bài test trong toàn bộ repository đều đạt 100% PASS.
