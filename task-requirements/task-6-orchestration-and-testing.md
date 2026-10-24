# Đặc tả nhiệm vụ: Task 6 - Tối ưu pipeline điều phối và bộ test toàn diện

- **Mã nhiệm vụ:** `TASK-06`
- **Tên nhiệm vụ:** Tối ưu hóa pipeline điều phối 3 bước và hoàn thiện toàn diện bộ test
- **Trạng thái:** Chờ thực hiện (Pending)
- **Tài liệu tham chiếu:** [run_all.sh](file:///d:/DoAn/qwen2.5-7b-eagle3.1-speculator/run_all.sh), [modal_pipeline.py](file:///d:/DoAn/qwen2.5-7b-eagle3.1-speculator/modal_pipeline.py), [README.md](file:///d:/DoAn/qwen2.5-7b-eagle3.1-speculator/README.md)

---

## 1. Mục tiêu kỹ thuật
1. **Rút gọn pipeline từ 4 bước xuống 3 bước tinh gọn:**
   - Bước 1: Chuẩn bị dữ liệu (`python data/prepare_hybrid_dataset.py` -> `train.jsonl`, `val.jsonl`, `test.jsonl`).
   - Bước 2: Huấn luyện với SpecForge (`python training/train.py --config configs/train_config.yaml`).
   - Bước 3: Đóng gói và upload (`python export/package_and_upload.py`).
2. **Duy trì tuyệt đối các chốt chặn an toàn (Guards) trong `run_all.sh`:**
   - Disk Overflow Guard: Kiểm tra dung lượng ổ đĩa trống >= 50GB.
   - Cost Ceiling Guard: Giới hạn thời gian chạy tối đa cứng là 6.0 giờ.
   - Auto-stop Instance: Tự động ngắt máy chủ Vast.ai ngay khi kết thúc hoặc gặp lỗi.
3. **Cập nhật Serverless 1-Click trên Modal (`modal_pipeline.py`):**
   - Loại bỏ function extract hidden states ra Modal Volume.
   - Định nghĩa container image cài đặt sẵn SpecForge và SGLang trên GPU NVIDIA A100-80GB.
4. **Cập nhật tài liệu `README.md` và kiểm thử toàn diện:**
   - Cập nhật bài test cấu trúc dự án `tests/test_final_structure.py`.
   - Chạy toàn bộ pytest suite đạt 100% PASS.

---

## 2. Tiêu chuẩn nghiệm thu (Definition of Done - DoD)
- [ ] `run_all.sh` điều phối đúng 3 bước, bảo lưu Disk Guard và Cost Guard.
- [ ] `modal_pipeline.py` loại bỏ trích xuất offline, cấu hình container SpecForge chuẩn.
- [ ] `notebooks/train_on_vastai.ipynb` cập nhật luồng clone/pip install SpecForge và chạy pipeline mới.
- [ ] `README.md` phản ánh trung thực quy trình 3 bước, tài liệu hóa bộ dữ liệu Train/Val/Test.
- [ ] Toàn bộ bộ test `pytest tests/` chạy hoàn hảo 100% không có lỗi.
