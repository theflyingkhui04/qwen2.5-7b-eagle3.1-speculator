# Đặc tả nhiệm vụ: Task 4 - Xây dựng SpecForge Launcher

- **Mã nhiệm vụ:** `TASK-04`
- **Tên nhiệm vụ:** Xây dựng module launcher khởi chạy huấn luyện SpecForge
- **Trạng thái:** Đã hoàn thành (Done)
- **Tài liệu tham chiếu:** [worklogs/task-4.md](file:///d:/DoAn/qwen2.5-7b-eagle3.1-speculator/worklogs/task-4.md), [training/train.py](file:///d:/DoAn/qwen2.5-7b-eagle3.1-speculator/training/train.py)

---

## 1. Mục tiêu kỹ thuật
1. Viết lại hoàn toàn file `training/train.py` từ việc tự viết vòng lặp PyTorch sang một **CLI Launcher chuẩn chỉ**.
2. Script có nhiệm vụ đọc file cấu hình `configs/train_config.yaml`, chuyển hóa các siêu tham số thành danh sách cờ dòng lệnh và thực thi lệnh huấn luyện qua `torchrun -m specforge.train`.
3. Cho phép chạy linh hoạt cả đơn GPU (`--gpus 1`) lẫn đa GPU (`--gpus 2` hoặc `--gpus 4`).

---

## 2. Kiến trúc lệnh thực thi

Lệnh sinh ra từ launcher sẽ có dạng:
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

---

## 3. Tiêu chuẩn nghiệm thu (Definition of Done - DoD)
- [x] `training/train.py` thực hiện đọc YAML và xây dựng lệnh `torchrun` chính xác 100%.
- [x] Có bài test `tests/test_train.py` sử dụng mock `subprocess.run` để kiểm thử cú pháp câu lệnh (dry-run) mà không cần nạp GPU thật.
- [x] Bài test `tests/test_train.py` đạt 100% PASS.
