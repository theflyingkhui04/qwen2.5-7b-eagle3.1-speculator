# Qwen2.5-7B EAGLE-3.1 Speculator

> **Dự án:** Huấn luyện đầu nháp suy đoán (Speculative Draft Head) EAGLE-3.1 cho `Qwen/Qwen2.5-7B-Instruct`.  
> **Tác giả:** Cấn Đức Khôi (`theflyingkhui04`)  
> **Mục tiêu:** Tăng tốc suy luận (Lossless Acceleration) 2.5x – 3.2x chuyên biệt cho Tiếng Việt và Hóa đơn MC-OCR, tương thích vLLM trên GPU Tesla T4 (16GB).

---

## 1. Tổng Quan Kiến Trúc

- **Mô hình mục tiêu (Target Model):** `Qwen/Qwen2.5-7B-Instruct` (7.61B tham số, `hidden_size = 3584`, `num_hidden_layers = 28`).
- **Tầng trích xuất (Feature Extraction Layer):** Tầng thứ 28 (tầng cuối cùng trước LM Head, mang vector biểu diễn giàu ngữ nghĩa nhất).
- **Mô hình đầu nháp (Draft Head):** EAGLE-3.1 (~270M tham số, checkpoint ~540 MB).
  - 1 tầng Transformer Decoder tự hồi quy (28 attention heads, 4 KV heads).
  - Tầng **FC Normalization** (cốt lõi của EAGLE-3.1): chuẩn hóa vector ẩn sau mỗi bước nháp nhằm triệt tiêu hiện tượng Attention Drift khi suy đoán sâu `k >= 3`.
- **Hàm mất mát kết hợp (Dual Loss):**
  ```text
  Total_Loss = Loss_CrossEntropy(token_prediction) + 0.5 * Loss_SmoothL1(predicted_hidden, target_hidden)
  ```

---

## 2. Cấu Trúc Thư Mục Repository

```text
qwen2.5-7b-eagle3.1-speculator/
├── README.md                      # Hướng dẫn chi tiết chạy 1-click trên Vast.ai & Colab
├── requirements.txt               # Dependencies: torch, transformers, accelerate, safetensors...
├── configs/
│   ├── eagle3_qwen2.5_7b.json     # Cấu hình kiến trúc đầu nháp EAGLE-3.1
│   └── train_config.yaml          # Siêu tham số huấn luyện (batch size, lr, epochs, loss weights)
├── data/
│   ├── prepare_hybrid_dataset.py  # Script gộp ShareGPT (20k) + Tiếng Việt / Hóa đơn (10k)
│   └── raw/                       # Thư mục lưu cache dữ liệu thô
├── extraction/
│   └── extract_hidden_states.py   # Forward Qwen2.5-7B trích xuất activations tầng 28 ra ổ đĩa
├── training/
│   ├── model.py                   # Cài đặt mạng EAGLE-3.1 Speculator Head (kèm FC Normalization)
│   ├── loss.py                    # Dual Loss: Cross-Entropy + Smooth L1 Regression
│   └── train.py                   # Vòng lặp huấn luyện chính với PyTorch AdamW & Checkpoint resume
├── export/
│   └── package_and_upload.py      # Đóng gói config.json, weights (float16) và push lên Hugging Face Hub
├── notebooks/
│   └── train_on_vastai.ipynb      # Notebook JupyterLab chạy trực quan từng bước
└── run_all.sh                     # Script Bash chạy ngầm tự động (tmux + auto-stop máy)
```

---

## 3. Quy Trình 4 Bước Huấn Luyện

1. **Bước 1: Chuẩn bị dữ liệu lai (Hybrid Dataset):**
   Gộp 20.000 mẫu ShareGPT (năng lực chỉ dẫn tổng quát) và 10.000 mẫu Tiếng Việt / Hóa đơn MC-OCR (định hướng key-value extraction).
   ```bash
   python data/prepare_hybrid_dataset.py \
       --sharegpt-path "data/raw/sharegpt.json" \
       --vi-path "data/raw/vietnamese_invoices.json" \
       --output-path "data/processed/hybrid_dataset.json"
   ```

2. **Bước 2: Trích xuất Vector Ẩn (Feature Extraction):**
   Forward Qwen2.5-7B (bfloat16) qua 30.000 mẫu, áp dụng Sequence Bucketing (<=512, <=1024, <=2048) và lưu chunks safetensors (mỗi chunk 5.000 mẫu) vào `data/features/`.
   ```bash
   python extraction/extract_hidden_states.py \
       --model-id "Qwen/Qwen2.5-7B-Instruct" \
       --dataset-path "data/processed/hybrid_dataset.json" \
       --output-dir "data/features" \
       --target-layer 28
   ```

3. **Bước 3: Huấn luyện Đầu Nháp EAGLE-3.1:**
   Huấn luyện 20 epochs với AdamW, Cosine Annealing decay, Gradient Accumulation. Tự động lưu và phục hồi checkpoint mỗi 5 epochs.
   ```bash
   python training/train.py \
       --config "configs/train_config.yaml" \
       --resume
   ```

4. **Bước 4: Đóng gói Chuẩn Hugging Face & Upload:**
   Chuyển đổi trọng số sang `float16` (đảm bảo tương thích 100% với Tesla T4) và đẩy lên Hugging Face Hub.
   ```bash
   export HF_TOKEN="hf_your_token_here"
   python export/package_and_upload.py \
       --checkpoint-path "checkpoints/checkpoint_epoch_20.pt" \
       --config-path "configs/eagle3_qwen2.5_7b.json" \
       --output-dir "output/eagle3_qwen2.5_7b" \
       --repo-id "theflyingkhui04/qwen2.5-7b-eagle3.1-speculator" \
       --precision "float16" \
       --upload
   ```

---

## 4. Chạy Tự Động 1-Click Trên Vast.ai (Chi Phí <= 2.50 USD)

Trên instance 1x NVIDIA A100 (80GB VRAM) của Vast.ai:
```bash
git clone https://github.com/theflyingkhui04/qwen2.5-7b-eagle3.1-speculator.git
cd qwen2.5-7b-eagle3.1-speculator
pip install -r requirements.txt

# Thiết lập token và ID container (để tự động tắt máy khi xong)
export HF_TOKEN="hf_..."
export CONTAINER_ID="your_vast_container_id"

# Chạy ngầm trong tmux
tmux new -s train "bash run_all.sh"
```

### Các Chốt Chặn An Toàn (Safeguards):
- **Disk Overflow Guard:** Script kiểm tra dung lượng ổ đĩa trống >= 50 GB trước khi chạy.
- **Cost Ceiling Guard:** Giới hạn thời gian tối đa cứng là **6.0 giờ**. Nếu quá giờ, tiến trình tự ngắt.
- **Auto-Stop Instance:** Dù thành công hay gặp lỗi, script tự động gọi `vastai stop instance $CONTAINER_ID` để ngắt máy ngay lập tức, chống phát sinh tiền không tải.

---

## 5. Triển Khai vLLM Trên GPU Tesla T4 (16GB)

Trên môi trường Google Colab hoặc máy chủ trang bị Tesla T4 (Turing sm_75):

```bash
# Phục vụ vLLM với Speculative Decoding sử dụng base model AWQ (INT4)
vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ \
    --speculative-model theflyingkhui04/qwen2.5-7b-eagle3.1-speculator \
    --num-speculative-tokens 5 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 2048 \
    --dtype float16
```