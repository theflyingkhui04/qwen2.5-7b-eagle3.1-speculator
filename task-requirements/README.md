# Bảng thiết lập và theo dõi lộ trình nhiệm vụ (Task Roadmap Setup)

- **Dự án:** `qwen2.5-7b-eagle3.1-speculator`
- **Mục tiêu:** Chuyển đổi toàn bộ hệ thống sang `sgl-project/SpecForge`, ghim cứng chuẩn kiến trúc EAGLE-3.1 cho `Qwen/Qwen2.5-7B-Instruct`.
- **Cập nhật lần cuối:** 24/10/2026

---

## 1. Tổng quan lộ trình 6 nhiệm vụ

Dự án được chuẩn hóa thành đúng **6 nhiệm vụ cốt lõi (Task 1 đến Task 6)**. Mỗi nhiệm vụ có một file đặc tả chi tiết riêng biệt trong thư mục [`task-requirements/`](file:///d:/DoAn/qwen2.5-7b-eagle3.1-speculator/task-requirements/) kèm theo tiêu chuẩn nghiệm thu (DoD - Definition of Done).

```
task-requirements/
├── README.md                                    # File tổng quan theo dõi tiến độ (file này)
├── task-1-dataset-pipeline.md                   # Task 1: Tầng dữ liệu & phân chia Train/Val/Test
├── task-2-cleanup-and-eagle3.1-pinning.md       # Task 2: Dọn dẹp code cũ & ghim chuẩn EAGLE-3.1
├── task-3-dependencies-and-configs.md           # Task 3: Cập nhật thư viện & file cấu hình
├── task-4-specforge-launcher.md                 # Task 4: Xây dựng launcher gọi specforge.train
├── task-5-export-and-packaging.md               # Task 5: Đóng gói checkpoint & upload Hugging Face
└── task-6-orchestration-and-testing.md          # Task 6: Tối ưu pipeline điều phối & bộ test
```

---

## 2. Bảng theo dõi tiến độ thực hiện (Traceability Matrix)

| Nhiệm vụ | Trọng tâm | Trạng thái | File đặc tả chi tiết |
| :--- | :--- | :---: | :--- |
| **Task 1** | Chuẩn hóa tầng dữ liệu sang ShareGPT JSONL, phân chia 40k Train / 3k Val / 2k Test | `DONE` | [task-1-dataset-pipeline.md](file:///d:/DoAn/qwen2.5-7b-eagle3.1-speculator/task-requirements/task-1-dataset-pipeline.md) |
| **Task 2** | Dọn dẹp `extraction/`, `training/model.py`, `training/loss.py` và ghim chuẩn EAGLE-3.1 | `DONE` | [task-2-cleanup-and-eagle3.1-pinning.md](file:///d:/DoAn/qwen2.5-7b-eagle3.1-speculator/task-requirements/task-2-cleanup-and-eagle3.1-pinning.md) |
| **Task 3** | Cập nhật `requirements.txt`, `configs/train_config.yaml`, `configs/eagle3.1_qwen2.5_7b.json` | `DONE` | [task-3-dependencies-and-configs.md](file:///d:/DoAn/qwen2.5-7b-eagle3.1-speculator/task-requirements/task-3-dependencies-and-configs.md) |
| **Task 4** | Viết lại `training/train.py` thành SpecForge Launcher gọi `torchrun -m specforge.train` | `PENDING` | [task-4-specforge-launcher.md](file:///d:/DoAn/qwen2.5-7b-eagle3.1-speculator/task-requirements/task-4-specforge-launcher.md) |
| **Task 5** | Tinh chỉnh `export/package_and_upload.py` để đóng gói checkpoint SpecForge | `PENDING` | [task-5-export-and-packaging.md](file:///d:/DoAn/qwen2.5-7b-eagle3.1-speculator/task-requirements/task-5-export-and-packaging.md) |
| **Task 6** | Cập nhật `run_all.sh`, `modal_pipeline.py`, `notebooks/`, hoàn thiện `tests/` và `README.md` | `PENDING` | [task-6-orchestration-and-testing.md](file:///d:/DoAn/qwen2.5-7b-eagle3.1-speculator/task-requirements/task-6-orchestration-and-testing.md) |

---

## 3. Danh sách kiểm tra chi tiết từng nhiệm vụ (Checklist)

### Task 1: Tầng dữ liệu và phân chia Train / Val / Test
- [x] Thiết kế kiến trúc phân bổ: 40% Tiếng Việt, 35% Hóa đơn/JSON, 25% Tiếng Anh/CoT ([worklogs/task-1.md](file:///d:/DoAn/qwen2.5-7b-eagle3.1-speculator/worklogs/task-1.md)).
- [x] Chuẩn hóa `data/prepare_hybrid_dataset.py` xuất định dạng ShareGPT JSON Lines (`.jsonl`).
- [x] Cập nhật bài test TDD trong `tests/test_dataset.py` (6/6 tests PASS).
- [x] Xây dựng module `build_train_val_test_splits` phân chia tự động 3 tập: `train.jsonl`, `val.jsonl`, `test.jsonl`.
- [x] Đã nghiệm thu và xác nhận 100% ID các tập hoàn toàn phân tách (disjoint), không có rò rỉ dữ liệu.

### Task 2: Dọn dẹp module cũ và ghim chuẩn kiến trúc EAGLE-3.1
- [x] Viết test `tests/test_cleanup_deprecated.py` kiểm tra việc loại bỏ module cũ.
- [x] Xóa toàn bộ thư mục `extraction/` (`extract_hidden_states.py`, `__init__.py`).
- [x] Xóa `training/model.py` và `training/loss.py` tự viết cũ.
- [x] Xóa các bài test cũ: `tests/test_extraction.py`, `tests/test_loss.py`, `tests/test_model.py`.
- [x] Cập nhật `training/__init__.py`.
- [x] Ghim quy tắc kiến trúc EAGLE-3.1 vào `AGENTS.md` và `configs/eagle3.1_qwen2.5_7b.json` (bổ sung `norm_output: true`).

### Task 3: Cập nhật thư viện và file cấu hình
- [x] Cập nhật `requirements.txt` bổ sung `sglang[all]`, `git+https://github.com/sgl-project/SpecForge.git`, `datasets`.
- [x] Chuẩn hóa `configs/train_config.yaml` khớp với SpecForge (`ttt_length: 5`, `tp_size: 1`, `max_seq_len: 2048`).
- [x] Khóa cứng `configs/eagle3.1_qwen2.5_7b.json` (giữ `use_fc_norm: true`, `norm_output: true`, `target_layer: 28`).
- [x] Cập nhật và chạy `tests/test_configs.py` đạt 100% PASS.

### Task 4: Xây dựng SpecForge Launcher
- [ ] Viết lại `training/train.py` thành CLI builder đọc YAML và gọi `torchrun -m specforge.train`.
- [ ] Hỗ trợ đầy đủ các tham số: `--model-path`, `--draft-config`, `--data-path`, `--output-dir`, `--ttt-length`, v.v.
- [ ] Viết `tests/test_train.py` kiểm thử mock `subprocess.run` xác thực lệnh sinh ra.

### Task 5: Tinh chỉnh quy trình đóng gói và upload checkpoint
- [ ] Cập nhật `export/package_and_upload.py` để nạp trực tiếp checkpoint chuẩn từ thư mục output của SpecForge.
- [ ] Tự động kiểm tra tính toàn vẹn của `config.json`, `model.safetensors` và sinh `MODEL_CARD.md`.
- [ ] Hỗ trợ upload lên Hugging Face Hub qua `huggingface_hub.HfApi`.
- [ ] Cập nhật `tests/test_export.py` đạt PASS.

### Task 6: Tối ưu pipeline điều phối và bộ test toàn diện
- [ ] Rút gọn `run_all.sh` từ 4 bước xuống 3 bước, bảo lưu Disk Guard (>= 50GB) và Cost Ceiling Guard (<= 6.0h).
- [ ] Cập nhật `modal_pipeline.py` (cài đặt SpecForge trên A100-80GB, bỏ hàm trích xuất offline, gắn volume).
- [ ] Cập nhật `notebooks/train_on_vastai.ipynb` theo pipeline 3 bước mới.
- [ ] Cập nhật `tests/test_final_structure.py` và chạy toàn bộ test suite.
- [ ] Cập nhật `README.md` chuẩn hóa theo hướng dẫn 1-click SpecForge.
