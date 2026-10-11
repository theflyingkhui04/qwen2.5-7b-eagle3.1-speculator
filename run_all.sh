#!/usr/bin/env bash
# ==============================================================================
# Script: run_all.sh
# Pipeline Tự Động Hóa 1-Click Huấn Luyện Đầu Nháp EAGLE-3.1 Cho Qwen2.5-7B
# Tích hợp: Disk Guard (>=50GB), Cost Ceiling Guard (<=6.0h), Auto-Stop Vast.ai
# ==============================================================================

set -euo pipefail

# 1. Cấu hình biến môi trường và định danh Vast.ai
CONTAINER_ID="${CONTAINER_ID:-${VAST_CONTAINERLABEL:-""}}"
MAX_RUNTIME_HOURS=6.0
MIN_DISK_FREE_GB=50
LOG_FILE="training_pipeline.log"

echo "=== [EAGLE-3.1 Speculator Pipeline Bắt Đầu] ===" | tee -a "$LOG_FILE"
echo "Thời gian bắt đầu: $(date)" | tee -a "$LOG_FILE"
echo "Container ID Vast.ai: ${CONTAINER_ID:-'(Chạy local/chưa cấu hình)'}" | tee -a "$LOG_FILE"

# 2. Hàm dọn dẹp và tự động tắt máy Vast.ai bảo vệ ngân sách
cleanup_and_stop() {
    local exit_code=$?
    echo "=== [Kết thúc tiến trình với mã thoát: $exit_code] ===" | tee -a "$LOG_FILE"
    echo "Thời gian kết thúc: $(date)" | tee -a "$LOG_FILE"

    if [ -n "$CONTAINER_ID" ]; then
        echo ">>> Kích hoạt tự hủy máy Vast.ai để triệt tiêu chi phí không tải: vastai stop instance $CONTAINER_ID" | tee -a "$LOG_FILE"
        vastai stop instance "$CONTAINER_ID" || true
    else
        echo ">>> Không phát hiện CONTAINER_ID. Bỏ qua lệnh vastai stop instance." | tee -a "$LOG_FILE"
    fi
    exit $exit_code
}

trap cleanup_and_stop EXIT INT TERM

# 3. Disk Overflow Guard: Kiểm tra dung lượng đĩa trống
echo ">>> [Kiểm tra ổ đĩa] Yêu cầu tối thiểu: ${MIN_DISK_FREE_GB} GB..." | tee -a "$LOG_FILE"
FREE_DISK_GB=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')

if [ "$FREE_DISK_GB" -lt "$MIN_DISK_FREE_GB" ]; then
    echo "❌ LỖI: Dung lượng ổ đĩa khả dụng (${FREE_DISK_GB} GB) nhỏ hơn mức an toàn (${MIN_DISK_FREE_GB} GB)!" | tee -a "$LOG_FILE"
    exit 1
fi
echo "✅ Dung lượng đĩa khả dụng: ${FREE_DISK_GB} GB (Đạt chuẩn an toàn)" | tee -a "$LOG_FILE"

# 4. Cost Ceiling Guard: Bộ đếm giờ tự ngắt sau 6.0 giờ (21600 giây)
(
    sleep 21600
    echo "❌ CẢNH BÁO: Quá trần thời gian chạy tối đa 6.0 giờ! Tự động ngắt tiến trình bảo vệ tài khoản..." | tee -a "$LOG_FILE"
    if [ -n "$CONTAINER_ID" ]; then
        vastai stop instance "$CONTAINER_ID"
    fi
    kill -9 $$
) &
TIMER_PID=$!

# ==============================================================================
# 5. Thực thi tuần tự 4 bước kỹ thuật
# ==============================================================================

# Bước 1: Chuẩn bị dữ liệu lai
echo ">>> [Bước 1/4] Chuẩn bị tập dữ liệu lai (ShareGPT 20k + Tiếng Việt 10k)..." | tee -a "$LOG_FILE"
python data/prepare_hybrid_dataset.py \
    --sharegpt-path "data/raw/sharegpt.json" \
    --vi-path "data/raw/vietnamese_invoices.json" \
    --output-path "data/processed/hybrid_dataset.json" \
    --sharegpt-limit 20000 \
    --vi-limit 10000

# Bước 2: Trích xuất vector ẩn tầng 28
echo ">>> [Bước 2/4] Trích xuất Hidden States tầng 28 từ Qwen2.5-7B..." | tee -a "$LOG_FILE"
python extraction/extract_hidden_states.py \
    --model-id "Qwen/Qwen2.5-7B-Instruct" \
    --dataset-path "data/processed/hybrid_dataset.json" \
    --output-dir "data/features" \
    --target-layer 28 \
    --chunk-size 5000 \
    --min-disk-gb "$MIN_DISK_FREE_GB"

# Bước 3: Huấn luyện mạng đầu nháp EAGLE-3.1
echo ">>> [Bước 3/4] Huấn luyện đầu nháp EAGLE-3.1 (20 Epochs, Dual Loss, FC Norm)..." | tee -a "$LOG_FILE"
python training/train.py \
    --config "configs/train_config.yaml" \
    --resume

# Bước 4: Đóng gói checkpoint chuẩn Hugging Face & Upload
echo ">>> [Bước 4/4] Đóng gói chuẩn vLLM float16 và upload lên Hugging Face Hub..." | tee -a "$LOG_FILE"
python export/package_and_upload.py \
    --checkpoint-path "checkpoints/checkpoint_epoch_20.pt" \
    --config-path "configs/eagle3_qwen2.5_7b.json" \
    --output-dir "output/eagle3_qwen2.5_7b" \
    --repo-id "theflyingkhui04/qwen2.5-7b-eagle3.1-speculator" \
    --precision "float16" \
    --upload

# Hủy bộ đếm giờ nếu hoàn thành trước hạn
kill "$TIMER_PID" || true

echo "=== [HOÀN TẤT THÀNH CÔNG TOÀN BỘ PIPELINE HUẤN LUYỆN] ===" | tee -a "$LOG_FILE"
