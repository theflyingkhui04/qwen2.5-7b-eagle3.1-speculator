# Task 6: Tối ưu pipeline điều phối 3 bước trên Modal.com và hoàn thiện bộ test

- **Dự án:** `qwen2.5-7b-eagle3.1-speculator`
- **Mã nhiệm vụ:** `TASK-06`
- **Mô hình mục tiêu (Target Model):** `Qwen/Qwen2.5-7B-Instruct`
- **Kiến trúc đầu nháp (Draft Head):** EAGLE-3.1 Speculator (1 Decoder Layer, FC Normalization, multi-layer hidden fusion)
- **Framework huấn luyện:** `sgl-project/SpecForge`
- **Hạ tầng huấn luyện chính:** `Modal.com` Serverless Cloud (GPU NVIDIA A100-80GB)
- **Ngày lập:** 25/10/2026
- **Trạng thái:** Đã hoàn thành (Done & Verified)
- **Tài liệu tham chiếu:** [task-requirements/task-6-orchestration-and-testing.md](file:///d:/DoAn/qwen2.5-7b-eagle3.1-speculator/task-requirements/task-6-orchestration-and-testing.md), [modal_pipeline.py](file:///d:/DoAn/qwen2.5-7b-eagle3.1-speculator/modal_pipeline.py), [run_all.sh](file:///d:/DoAn/qwen2.5-7b-eagle3.1-speculator/run_all.sh), [README.md](file:///d:/DoAn/qwen2.5-7b-eagle3.1-speculator/README.md)

---

## 1. Bối cảnh kỹ thuật và mục tiêu nhiệm vụ

Trước đây, hệ thống sử dụng quy trình 4 bước cồng kềnh với bước trích xuất offline thủ công lưu hàng chục gigabytes ra ổ đĩa, gây nghẽn I/O và lãng phí thời gian nạp dữ liệu.

Nhiệm vụ Task 6 hoàn thiện việc tối ưu hóa tầng điều phối (Orchestration Layer), chuyển đổi trọng tâm hạ tầng huấn luyện sang **Modal.com** (Serverless GPU A100-80GB) theo yêu cầu thực tế của dự án:
1. **Tái cấu trúc [modal_pipeline.py](file:///d:/DoAn/qwen2.5-7b-eagle3.1-speculator/modal_pipeline.py):**
   - Loại bỏ hàm trích xuất offline cũ `extract_features`.
   - Xây dựng container image tích hợp `sgl-project/SpecForge` và `sglang[all]`.
   - Gắn Persistent Storage Volume `qwen-eagle3-volume` tại `/vol` để tự động lưu và commit an toàn các checkpoint trọng số.
   - Hàm `train_specforge` chạy trên GPU NVIDIA A100-80GB, tự động cấu hình WandB, nạp dữ liệu ShareGPT JSONL và unroll tự hồi quy 5 bước trực tiếp trong VRAM.
   - Hàm `export_and_upload` chạy trên CPU container, tự động gắn `MODEL_CARD.md` và tải lên Hugging Face Hub.
   - Hỗ trợ chế độ chạy ngầm `--detach` (tắt terminal, đóng nắp laptop).
2. **Duy trì các chốt chặn an toàn trong [run_all.sh](file:///d:/DoAn/qwen2.5-7b-eagle3.1-speculator/run_all.sh):**
   - Rút gọn về 3 bước chuẩn.
   - Bảo lưu Disk Overflow Guard (>= 50GB) và Cost Ceiling Guard (<= 6.0 giờ).
3. **Cập nhật và hoàn thiện toàn bộ test suite:**
   - Cập nhật [tests/test_script_guards.py](file:///d:/DoAn/qwen2.5-7b-eagle3.1-speculator/tests/test_script_guards.py) và [tests/test_final_structure.py](file:///d:/DoAn/qwen2.5-7b-eagle3.1-speculator/tests/test_final_structure.py).
   - Đảm bảo 100% tests trong repo đạt PASS.

---

## 2. Kiến trúc Serverless 1-Click trên Modal.com

### 2.1. Cấu hình Container Image & Volume
```python
volume = modal.Volume.from_name("qwen-eagle3-volume", create_if_missing=True)

app_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .pip_install(
        "torch>=2.4.0",
        "transformers>=4.45.0",
        "accelerate>=0.34.0",
        "safetensors>=0.4.4",
        "datasets>=2.20.0",
        "pyyaml>=6.0.2",
        "tqdm>=4.66.5",
        "huggingface_hub>=0.24.0",
        "wandb>=0.16.0",
        "sglang[all]",
        "git+https://github.com/sgl-project/SpecForge.git",
    )
    .add_local_python_source("training", "data", "export")
    .add_local_dir("configs", remote_path="/root/configs")
    .add_local_dir("data/processed", remote_path="/root/data/processed")
    .add_local_file("MODEL_CARD.md", remote_path="/root/MODEL_CARD.md")
)
```

### 2.2. Lệnh thực thi trên Modal.com
- **Kiểm thử nhanh 1 epoch (Test Run):**
  ```bash
  modal run modal_pipeline.py --step test
  ```
- **Chạy toàn bộ pipeline khép kín từ Train đến Upload Hugging Face (Chạy ngầm tắt máy):**
  ```bash
  modal run --detach modal_pipeline.py --step all
  ```
- **Chỉ chạy huấn luyện:**
  ```bash
  modal run --detach modal_pipeline.py --step train
  ```
- **Chỉ đóng gói và upload checkpoint:**
  ```bash
  modal run modal_pipeline.py --step export
  ```

---

## 3. Nhật ký kiểm thử và nghiệm thu kỹ thuật (Execution Log)

### 3.1. Tổng kết toàn bộ test suite
```
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

Toàn bộ **37 bài test** bao phủ 6 tasks của dự án đều đạt **100% PASS**.
Tất cả 6 tasks từ Task 1 đến Task 6 trong [task-requirements/README.md](file:///d:/DoAn/qwen2.5-7b-eagle3.1-speculator/task-requirements/README.md) đã chính thức chuyển sang trạng thái **DONE**.
