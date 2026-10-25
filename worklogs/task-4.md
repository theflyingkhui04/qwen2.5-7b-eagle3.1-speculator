# Task 4: Xây dựng SpecForge training launcher và kiểm thử mô phỏng dòng lệnh

- **Dự án:** `qwen2.5-7b-eagle3.1-speculator`
- **Mã nhiệm vụ:** `TASK-04`
- **Mô hình mục tiêu (Target Model):** `Qwen/Qwen2.5-7B-Instruct`
- **Kiến trúc đầu nháp (Draft Head):** EAGLE-3.1 Speculator (1 Decoder Layer, FC Normalization, multi-layer hidden fusion)
- **Framework huấn luyện:** `sgl-project/SpecForge`
- **Ngày lập:** 25/10/2026
- **Trạng thái:** Đã hoàn thành (Done & Verified)
- **Tài liệu tham chiếu:** [task-requirements/task-4-specforge-launcher.md](file:///d:/DoAn/qwen2.5-7b-eagle3.1-speculator/task-requirements/task-4-specforge-launcher.md), [training/train.py](file:///d:/DoAn/qwen2.5-7b-eagle3.1-speculator/training/train.py), [tests/test_train.py](file:///d:/DoAn/qwen2.5-7b-eagle3.1-speculator/tests/test_train.py)

---

## 1. Bối cảnh kỹ thuật và mục tiêu nhiệm vụ

Trong phiên bản ban đầu của dự án, file `training/train.py` tự cài đặt toàn bộ vòng lặp PyTorch thủ công (tự tính forward qua mô hình nháp tự viết, tự tính loss và gradient step). Cách làm này không thể thực hiện cơ chế unroll tự hồi quy đa bước (Test-Time Training - TTT) tối ưu với FlashAttention và KV cache của SpecForge, đồng thời phát sinh lỗi khi nạp module `training.loss` đã được dọn dẹp ở Task 2.

Nhiệm vụ Task 4 chuyển đổi [training/train.py](file:///d:/DoAn/qwen2.5-7b-eagle3.1-speculator/training/train.py) thành một **CLI Training Launcher chuẩn mực**:
1. Đọc và phân tích file cấu hình siêu tham số [configs/train_config.yaml](file:///d:/DoAn/qwen2.5-7b-eagle3.1-speculator/configs/train_config.yaml).
2. Tự động ánh xạ các tham số từ YAML sang cờ dòng lệnh của module chính thức `torchrun -m specforge.train`.
3. Hỗ trợ linh hoạt cả đơn GPU (`--gpus 1`) lẫn đa GPU (`--gpus 2`, `--gpus 4`).
4. Hỗ trợ cờ kiểm thử an toàn `--dry-run` để sinh và kiểm tra cú pháp câu lệnh mà không cần nạp GPU thật hay cài đặt môi trường CUDA nặng nề.
5. Xây dựng bộ test [tests/test_train.py](file:///d:/DoAn/qwen2.5-7b-eagle3.1-speculator/tests/test_train.py) sử dụng mock `subprocess.run` để kiểm chứng toàn bộ quy trình sinh lệnh.

---

## 2. Kiến trúc câu lệnh và ánh xạ tham số (Parameter Mapping)

Launcher chuyển hóa cấu hình YAML thành câu lệnh phân tán chuẩn PyTorch `torchrun`:

```bash
torchrun --nproc_per_node=1 -m specforge.train \
    --model-path "Qwen/Qwen2.5-7B-Instruct" \
    --draft-config "configs/eagle3.1_qwen2.5_7b.json" \
    --data-path "data/processed/train.jsonl" \
    --output-dir "checkpoints/qwen2.5-7b-eagle3.1" \
    --num-epochs 3 \
    --micro-batch-size 2 \
    --grad-accum-steps 4 \
    --lr 0.0001 \
    --max-seq-len 2048 \
    --ttt-length 5 \
    --tp-size 1
```

### Bảng đối chiếu ánh xạ cấu hình:

| Khóa trong `train_config.yaml` | Cờ tương ứng trong SpecForge | Giá trị mặc định | Giải thích vai trò |
| :--- | :--- | :---: | :--- |
| `target_model_path` | `--model-path` | `Qwen/Qwen2.5-7B-Instruct` | Đường dẫn mô hình gốc thẩm định trên Hugging Face. |
| `draft_model_config` | `--draft-config` | `configs/eagle3.1_qwen2.5_7b.json` | Cấu hình kiến trúc mạng nháp EAGLE-3.1. |
| `train_data_path` | `--data-path` | `data/processed/train.jsonl` | Tập dữ liệu ShareGPT JSONL đã chuẩn hóa (40.000 mẫu). |
| `output_dir` | `--output-dir` | `checkpoints/qwen2.5-7b-eagle3.1` | Thư mục lưu trữ checkpoint trọng số và cấu hình xuất xưởng. |
| `num_epochs` | `--num-epochs` | `3` | Số chu kỳ huấn luyện toàn bộ dữ liệu. |
| `batch_size` | `--micro-batch-size` | `2` | Kích thước lô nạp vào VRAM trên mỗi GPU. |
| `gradient_accumulation_steps` | `--grad-accum-steps` | `4` | Số lượt tích lũy gradient trước khi tối ưu trọng số. |
| `learning_rate` | `--lr` | `1.0e-4` | Tốc độ học của bộ tối ưu AdamW. |
| `max_seq_len` | `--max-seq-len` | `2048` | Độ dài chuỗi token tối đa cho mỗi mẫu. |
| `ttt_length` | `--ttt-length` | `5` | Độ sâu unroll tự hồi quy trong quá trình sinh nháp. |
| `tp_size` | `--tp-size` | `1` | Kích thước song song mô hình Tensor Parallelism. |

---

## 3. Thiết kế module và giao diện (Interfaces)

Tệp [training/train.py](file:///d:/DoAn/qwen2.5-7b-eagle3.1-speculator/training/train.py) cung cấp 4 hàm cốt lõi:

1. `parse_training_args(args: Optional[List[str]] = None) -> argparse.Namespace`:
   - Phân tích cú pháp dòng lệnh, hỗ trợ `--config`, `--gpus`, `--dry-run`.
2. `load_training_config(config_path: str) -> Dict[str, Any]`:
   - Đọc và phân giải tệp cấu hình YAML an toàn.
3. `find_best_checkpoint(checkpoint_dir: Any) -> Optional[Path]`:
   - Quét tìm checkpoint tốt nhất (ưu tiên thư mục `best/`, `best_checkpoint/` hoặc metadata `best_metric.json`).
4. `find_latest_checkpoint(checkpoint_dir: Any) -> Optional[Path]`:
   - Quét tìm checkpoint có số step/epoch lớn nhất trong thư mục đầu ra, hỗ trợ cả thư mục con (`checkpoint-*`, `epoch-*`) và file (`.pt`, `.safetensors`).
5. `find_resume_checkpoint(checkpoint_dir: Any, prefer_best: bool = True) -> Optional[Path]`:
   - Ưu tiên chọn checkpoint tốt nhất (best), nếu chưa có sẽ tự động fallback về checkpoint gần nhất (latest).
6. `prune_old_checkpoints(checkpoint_dir: Any, keep_limit: int = 2) -> List[Path]`:
   - Dọn dẹp tự động các checkpoint cũ dư thừa sau khi train, luôn bảo tồn tuyệt đối checkpoint `best` và tối đa `keep_limit` checkpoint mới nhất, giải phóng ổ đĩa và tiết kiệm thời gian tải về.
7. `setup_wandb_metrics(cfg: Dict[str, Any]) -> bool`:
   - Tiền cấu hình tự động toàn bộ trục hoành độ (`step_metric="train/step"`) và bảng tóm tắt giá trị tốt nhất (`summary="min"` cho loss, `summary="max"` cho acceptance rates) lên WandB trước khi chạy.
8. `build_specforge_command(...) -> List[str]`:
   - Chuyển đổi từ điển tham số thành danh sách chuỗi lệnh `torchrun`, xử lý bổ sung cờ `--save-interval`, `--draft-checkpoint-path` (ưu tiên best) và các cờ WandB (`--report-to wandb`, `--wandb-project`, `--wandb-name`).
9. `run_specforge_training(...) -> List[str]`:
   - Điều phối toàn bộ quy trình: nạp cấu hình, kích hoạt tiền cấu hình WandB metrics, dựng lệnh, gọi `subprocess.run(cmd, check=True)` và kích hoạt dọn dẹp checkpoint dư thừa sau khi hoàn thành.

---

## 4. Nhật ký kiểm thử và nghiệm thu kỹ thuật (Execution Log)

### 4.1. Bộ test tự động `tests/test_train.py`
Bộ test bao gồm 15 bài kiểm tra chuyên biệt:
- `test_parse_training_args_defaults`: Kiểm tra giá trị cờ mặc định.
- `test_parse_training_args_custom`: Kiểm tra các cờ tùy biến (`--gpus 4`, `--resume`, `--checkpoint-path`, `--wandb`, `--dry-run`).
- `test_find_latest_checkpoint_empty_or_nonexistent`: Kiểm tra hành vi khi thư mục rỗng hoặc không tồn tại.
- `test_find_latest_checkpoint_subdirectories`: Kiểm tra dò tìm checkpoint dạng thư mục con (`checkpoint-500`, `checkpoint-1500`).
- `test_find_latest_checkpoint_files`: Kiểm tra dò tìm checkpoint dạng file (`checkpoint_epoch_3.pt`).
- `test_find_best_checkpoint_directory_or_metadata`: Kiểm tra phát hiện thư mục `best/` hoặc file `best_metric.json`.
- `test_find_resume_checkpoint_prefers_best`: Kiểm tra logic ưu tiên chọn `best` khi resume thay vì `latest`.
- `test_prune_old_checkpoints`: Kiểm tra cơ chế tự động dọn dẹp xóa các checkpoint cũ và bảo vệ tuyệt đối thư mục `best`.
- `test_build_specforge_command`: Kiểm tra đầy đủ từng cờ của câu lệnh `torchrun` và tham số ánh xạ.
- `test_build_specforge_command_with_wandb_and_save_interval`: Kiểm tra sinh đúng các cờ `--report-to wandb`, `--wandb-project`, `--wandb-name`, `--save-interval`.
- `test_build_specforge_command_with_resume`: Kiểm tra tự động phát hiện và sinh cờ `--draft-checkpoint-path`.
- `test_run_specforge_training_dry_run`: Kiểm tra chế độ chạy thử không kích hoạt tiến trình hệ thống.
- `test_run_specforge_training_executes_subprocess`: Sử dụng mock `unittest.mock.patch("subprocess.run")` để xác nhận `subprocess.run` được gọi đúng 1 lần với tham số chính xác và cờ `check=True`.
- `test_setup_wandb_metrics_disabled`: Kiểm tra trả về False khi WandB bị vô hiệu hóa trong cấu hình.
- `test_setup_wandb_metrics_enabled`: Sử dụng mock `wandb` để kiểm tra toàn bộ việc gọi `define_metric` cho các trục tọa độ `train/*`, `train/loss (min)`, `eval/loss (min)`, `acceptance/alpha_overall (max)`, `alpha_step_1...5 (max)`.

### 4.2. Kết quả kiểm thử thực tế
```
tests/test_train.py ...............                                      [100%]
============================== 15 passed in 0.19s ==============================

Toàn bộ test suite trong dự án:
tests/test_cleanup_deprecated.py ..                                      [  5%]
tests/test_configs.py ..                                                 [ 11%]
tests/test_dataset.py ......                                             [ 29%]
tests/test_export.py ...                                                 [ 38%]
tests/test_final_structure.py ...                                        [ 47%]
tests/test_scaffold.py ..                                                [ 52%]
tests/test_script_guards.py .                                            [ 55%]
tests/test_train.py ...............                                      [100%]
============================= 34 passed in 0.19s ==============================
```
Toàn bộ 34 bài test trong codebase đạt 100% PASS mà không có bất kỳ lỗi hay cảnh báo nào.
