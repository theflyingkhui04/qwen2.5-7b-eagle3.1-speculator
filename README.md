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
├── README.md                      # Hướng dẫn chi tiết chạy 1-click trên Modal & Vast.ai
├── modal_pipeline.py              # Serverless Pipeline 1-Click trên Modal.com (A100-80GB)
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
└── run_all.sh                     # Script Bash chạy ngầm tự động trên Vast.ai (tmux + auto-stop máy)
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

4. **Bước 4: Đóng gói Chuẩn vLLM & Upload Hugging Face (Export & Package):**
   * **Tại sao cần đóng gói?** File `checkpoint_epoch_20.pt` chứa cả optimizer và scheduler của PyTorch (~1.6 GB) nên vLLM không thể nhận diện trực tiếp. Script đóng gói thực hiện 4 nhiệm vụ cốt lõi:
     1. Lọc sạch optimizer và scheduler, chỉ trích xuất đúng trọng số mô hình (`model_state_dict`).
     2. Ép kiểu trọng số sang `float16` (giảm dung lượng xuống còn ~540 MB, tương thích 100% với kiến trúc Turing trên Tesla T4).
     3. Lưu thành file an toàn `model.safetensors` (loại bỏ nguy cơ pickle injection, tốc độ nạp nhanh gấp 10 lần).
     4. Sinh file `config.json` chuẩn Hugging Face (`architectures: ["EagleQwen2ForCausalLM"]`, `target_layer: 28`).
   * **Cấu trúc thư mục thành phẩm sau khi đóng gói:**
     ```text
     output/eagle3_qwen2.5_7b/
     ├── config.json          # Cấu hình vLLM nhận diện đầu nháp EAGLE-3.1
     └── model.safetensors    # Trọng số float16 (~540 MB)
     ```
   * **Lệnh chạy đóng gói thủ công (Local / Server):**
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
   * **Lệnh chạy đóng gói trực tiếp trên Modal:**
     ```bash
     modal run modal_pipeline.py --step export --repo-id "theflyingkhui04/qwen2.5-7b-eagle3.1-speculator"
     ```

---

## 4. Chạy Tự Động Serverless Trên Modal.com (Khuyến Nghị - GPU A100-80GB)

Quy trình Serverless trên [Modal.com](https://modal.com) là phương pháp tối ưu và an toàn nhất hiện nay:
- **Tự động cấp phát GPU NVIDIA A100-80GB** khi thực thi và **tự động giải phóng ngay lập tức khi hoàn tất** (không lo quên tắt máy gây phát sinh chi phí như thuê Virtual Machine).
- Tích hợp sẵn Persistent Volume (`qwen-eagle3-volume`) lưu trữ features và checkpoints an toàn giữa các lượt chạy.
- Tự động theo dõi metric huấn luyện thời gian thực trên Weights & Biases (W&B) và tự động đẩy trọng số lên Hugging Face Hub.

### 4.1. Cài Đặt Môi Trường & Thiết Lập 1 Lần Đầu

1. **Cài đặt thư viện Modal CLI:**
   ```bash
   pip install modal
   ```

2. **Đăng nhập và liên kết tài khoản Modal:**
   ```bash
   modal setup
   ```
   *(Trình duyệt sẽ tự động mở để xác thực tài khoản. Modal tặng sẵn 30 USD/tháng free credits cho tài khoản đã liên kết phương thức thanh toán).*

3. **Cấu hình Secrets trên Modal (1 lần duy nhất):**
   Tạo 2 secret trên Modal để cấp quyền nạp/đẩy model lên Hugging Face và theo dõi biểu đồ loss trên W&B:
   ```bash
   # Tạo secret Hugging Face (cần Token có quyền Write để đẩy model)
   modal secret create my-huggingface-secret HF_TOKEN="hf_your_token_here"

   # Tạo secret Weights & Biases (để xem log Loss thời gian thực)
   modal secret create wandb-secret WANDB_API_KEY="your_wandb_api_key_here"
   ```
   *(Hoặc có thể tạo trực tiếp trên giao diện web trực quan tại: https://modal.com/secrets)*

4. **Chốt chặn an toàn ví tiền (Khuyến nghị 100%):**
   Vào Modal Dashboard -> **Settings** -> **Billing**: Thiết lập **Monthly Spend Limit = $0**.
   Điều này đảm bảo khi dùng hết 30 USD credit miễn phí, Modal sẽ dừng tác vụ thay vì tự động trừ tiền vào tài khoản ngân hàng / Google Pay.

---

### 4.2. Các Lệnh Chạy 1-Click (1-Click Execution)

Toàn bộ pipeline được điều khiển tự động qua script `modal_pipeline.py`:

#### 1. Kiểm thử nhanh toàn bộ quy trình (Test Run - 50 mẫu, 1 epoch):
Chạy kiểm thử khép kín để xác thực GPU A100, container image, tokenizer và loss function:
```bash
modal run modal_pipeline.py --step test
```

#### 2. Chạy toàn bộ quy trình từ A đến Z (Full Pipeline - Khuyến nghị):
Tự động chạy tuần tự: Trích xuất 30.000 mẫu trên A100-80GB -> Huấn luyện 20 Epochs -> Đóng gói trọng số chuẩn float16 -> Upload trực tiếp lên Hugging Face Hub:
```bash
modal run modal_pipeline.py --step all --epochs 20
```

> **Mẹo chạy ngầm tắt máy đi ngủ (`--detach` hoặc `-d`):**  
> Thêm cờ `--detach` để ứng dụng chạy độc lập trên đám mây Modal:
> ```bash
> modal run --detach modal_pipeline.py --step all --epochs 20
> ```
> Sau khi màn hình hiển thị liên kết App, bạn có thể **tắt terminal, đóng nắp laptop đi ngủ**. Pipeline sẽ tự động hoàn tất, tự commit checkpoint vào Volume, tự đẩy lên Hugging Face và tự động tắt GPU A100.

#### 3. Chạy độc lập từng bước (Tùy chọn linh hoạt):
Nếu muốn chạy riêng lẻ từng giai đoạn:
- **Chỉ trích xuất Vector ẩn tầng 28 (Feature Extraction):**
  ```bash
  modal run modal_pipeline.py --step extract
  ```
- **Chỉ huấn luyện Đầu nháp EAGLE-3.1 (Train Speculator Head):**
  ```bash
  modal run modal_pipeline.py --step train --epochs 20
  ```
- **Chỉ đóng gói và đẩy model lên Hugging Face Hub:**
  ```bash
  modal run modal_pipeline.py --step export --repo-id "theflyingkhui04/qwen2.5-7b-eagle3.1-speculator"
  ```

---

### 4.3. Chuyển Giao Dữ Liệu & Huấn Luyện Nối Tiếp Giữa Các Tài Khoản (Transfer Workspaces)

Vì gói miễn phí cấp **30 USD/tháng cho mỗi Workspace cá nhân**, các thành viên trong nhóm có thể mở rộng quỹ credit (30 USD x số người) bằng cách luân chuyển dữ liệu và huấn luyện nối tiếp theo các bước sau:

#### Bước 1: Tải Checkpoints & Features từ Workspace cũ về máy
Đứng ở tài khoản cũ (khi sắp dùng hết 30 USD credit), chạy lệnh tải dữ liệu từ Modal Volume về máy tính:
```bash
# Tải toàn bộ features và checkpoints đã có về máy
modal volume get qwen-eagle3-volume features ./data/features
modal volume get qwen-eagle3-volume checkpoints ./checkpoints
```

#### Bước 2: Chuyển sang tài khoản Modal mới
Đăng nhập tài khoản Modal mới (của bạn bè hoặc tài khoản thứ 2):
```bash
modal setup
```
*(Hoặc dùng `modal profile activate <tên_profile>` nếu bạn quản lý nhiều tài khoản trên cùng một máy).*

Thiết lập lại Secrets trên tài khoản mới (1 lần duy nhất):
```bash
modal secret create my-huggingface-secret HF_TOKEN="hf_your_token"
modal secret create --force wandb-secret WANDB_API_KEY="your_wandb_key"
```

#### Bước 3: Đẩy dữ liệu vào Volume của tài khoản mới
Nạp dữ liệu đã tải từ bước 1 vào Volume của tài khoản mới:
```bash
modal volume put qwen-eagle3-volume ./data/features features
modal volume put qwen-eagle3-volume ./checkpoints checkpoints
```

#### Bước 4: Tiếp tục huấn luyện nối tiếp
Chạy lệnh huấn luyện như bình thường:
```bash
modal run modal_pipeline.py --step train --epochs 20
```
> **Cơ chế Auto-Resume:** Script `training/train.py` đã tích hợp sẵn tính năng tự động tìm kiếm checkpoint mới nhất trong `/vol/checkpoints` (ví dụ `checkpoint_epoch_10.pt`). Mô hình, trạng thái AdamW optimizer và Cosine Annealing scheduler sẽ được phục hồi 100% để tiếp tục huấn luyện ngay từ epoch 11 mà không tốn công chạy lại từ đầu.

---

## 5. Chạy Tự Động Trên Vast.ai (Giải Pháp Máy Chủ Thuê Riêng)

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

## 6. Triển Khai vLLM Trên GPU Tesla T4 (16GB)

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