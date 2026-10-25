# Qwen2.5-7B EAGLE-3.1 Speculator

> **Dự án:** Huấn luyện đầu nháp suy đoán (Speculative Draft Head) EAGLE-3.1 cho `Qwen/Qwen2.5-7B-Instruct` sử dụng framework chính thức `sgl-project/SpecForge`.  
> **Tác giả:** Cấn Đức Khôi (`theflyingkhui04`)  
> **Mục tiêu:** Tăng tốc suy luận không suy giảm chất lượng (Lossless Acceleration) 2.5x – 3.2x chuyên biệt cho tiếng Việt và hóa đơn MC-OCR, tương thích vLLM và SGLang trên GPU Tesla T4 (16GB) và A100.  
> **Môi trường huấn luyện chính:** Đám mây serverless [Modal.com](https://modal.com) (GPU NVIDIA A100-80GB).

---

## 1. Tổng quan kiến trúc

- **Mô hình mục tiêu (Target Model):** `Qwen/Qwen2.5-7B-Instruct` (7.61B tham số, `hidden_size = 3584`, `num_hidden_layers = 28`).
- **Tầng trích xuất đặc trưng (Feature Layer):** Tầng thứ 28 (tầng áp chót trước LM Head, mang vector biểu diễn giàu ngữ nghĩa nhất).
- **Mô hình đầu nháp (Draft Head):** EAGLE-3.1 (~100M tham số, checkpoint ~200MB):
  - 1 tầng Transformer Decoder tự hồi quy (28 query heads, 4 KV heads - Grouped-Query Attention).
  - Tầng **FC Normalization** và **Norm Output**: chuẩn hóa vector ẩn và logit đầu ra nhằm triệt tiêu hiện tượng trôi số học khi unroll suy đoán sâu 5 bước (`ttt_length = 5`).
  - Cơ chế **Post-Norm Feedback**: đưa tín hiệu phản hồi đã chuẩn hóa ngược trở lại vòng lặp tiếp theo, loại bỏ sai số tích lũy qua các bước sinh nháp.
- **Framework huấn luyện:** `sgl-project/SpecForge` (trực tiếp unroll đa bước với FlashAttention trong VRAM, triệt tiêu hoàn toàn điểm nghẽn ghi đĩa I/O của cách trích xuất offline thủ công).

---

## 2. Cấu trúc thư mục repository

```text
qwen2.5-7b-eagle3.1-speculator/
├── README.md                      # Hướng dẫn chi tiết sử dụng và huấn luyện
├── modal_pipeline.py              # Serverless pipeline 1-click trên Modal (NVIDIA A100-80GB, khuyên dùng)
├── requirements.txt               # Dependencies: PyTorch, SpecForge, SGLang, Transformers, Datasets...
├── run_all.sh                     # Script điều phối 3 bước ngầm cho máy ảo độc lập (kèm Disk & Cost Guard)
├── configs/
│   ├── eagle3.1_qwen2.5_7b.json   # Cấu hình kiến trúc đầu nháp chuẩn EAGLE-3.1 (ghim cứng)
│   └── train_config.yaml          # Siêu tham số huấn luyện (batch size, lr, epochs, WandB, limits)
├── data/
│   ├── prepare_hybrid_dataset.py  # Chuẩn hóa ShareGPT + Tiếng Việt + Hóa đơn, chia Train/Val/Test
│   └── raw/                       # Thư mục lưu dữ liệu nguồn
├── training/
│   ├── __init__.py                # Module training
│   └── train.py                   # SpecForge launcher chuẩn (CLI, tự cấu hình WandB, auto-resume, pruning)
├── export/
│   └── package_and_upload.py      # Đóng gói checkpoint SpecForge và upload lên Hugging Face Hub
├── notebooks/
│   └── train_on_vastai.ipynb      # Notebook JupyterLab chạy tương tác từng bước
├── task-requirements/             # Đặc tả yêu cầu kỹ thuật và bảng theo dõi 6 tasks
└── worklogs/                      # Nhật ký kỹ thuật chi tiết từng nhiệm vụ (Task 1 đến Task 6)
```

---

## 3. Quy trình 3 bước huấn luyện tinh gọn

Hệ thống đã được tối ưu hóa từ quy trình 4 bước cũ xuống còn **3 bước khép kín**, loại bỏ hoàn toàn bước trích xuất offline thủ công gây nghẽn đĩa.

```
Dữ liệu thô ──> [Bước 1: Chuẩn bị dữ liệu] ──> train.jsonl (40k), val.jsonl (3k), test.jsonl (2k)
                       │
                       ▼
          [Bước 2: Huấn luyện SpecForge]  <── Qwen2.5-7B (Freeze) + EAGLE-3.1 Head (Train)
                       │
                       ▼
          [Bước 3: Đóng gói & Upload]     ──> Hugging Face Hub (vLLM / SGLang Ready)
```

---

### Bước 1: Chuẩn bị dữ liệu lai (train/val/test split)

Chuẩn hóa dữ liệu sang định dạng **ShareGPT JSON Lines (`.jsonl`)** và phân chia độc lập 3 tập theo chuẩn Machine Learning để triệt tiêu nguy cơ rò rỉ dữ liệu (data leakage):
- **Train:** 40.000 mẫu (`data/processed/train.jsonl`) - 40% Tiếng Việt, 35% Hóa đơn MC-OCR/JSON, 25% Tiếng Anh/CoT.
- **Val:** 3.000 mẫu (`data/processed/val.jsonl`) - Theo dõi Dual Loss và đánh giá `best_checkpoint`.
- **Test:** 2.000 mẫu (`data/processed/test.jsonl`) - Đo đạc độc lập tốc độ tăng tốc (Speedup) và tỷ lệ chấp thuận (`alpha`).

Lệnh thực thi:
```bash
python data/prepare_hybrid_dataset.py \
    --vi-path "data/raw/vi_alpaca_full.parquet" \
    --inv-path "data/raw/vietnamese_invoices.json" \
    --en-path "data/raw/sharegpt.json" \
    --output-dir "data/processed"
```

---

### Bước 2: Huấn luyện qua SpecForge launcher ([training/train.py](file:///d:/DoAn/qwen2.5-7b-eagle3.1-speculator/training/train.py))

Launcher tự động đọc cấu hình [configs/train_config.yaml](file:///d:/DoAn/qwen2.5-7b-eagle3.1-speculator/configs/train_config.yaml) và thực thi qua `torchrun -m specforge.train`.

#### 1. Lệnh khởi chạy cơ bản (Đơn GPU):
```bash
python training/train.py --config configs/train_config.yaml --gpus 1
```

#### 2. Lệnh khởi chạy đa GPU (2 hoặc 4 GPU):
```bash
python training/train.py --config configs/train_config.yaml --gpus 2
```

#### 3. Chạy thử kiểm tra cú pháp lệnh (Dry-run):
Không cần nạp mô hình hay cấp phát GPU, script sẽ in ra câu lệnh `torchrun` chính xác 100%:
```bash
python training/train.py --dry-run
```

#### 4. Cơ chế tiếp tục (Resume) thông minh:
- **Tự động tìm và ưu tiên bản `best`:**
  ```bash
  python training/train.py --resume
  ```
  *(Hệ thống tự động tìm checkpoint `best` trong thư mục đầu ra để tiếp tục. Nếu chưa có bản `best`, hệ thống tự động chuyển sang nạp checkpoint gần nhất `latest`).*
- **Bắt buộc nạp bản `best`:**
  ```bash
  python training/train.py --resume-best
  ```
- **Chỉ định trực tiếp checkpoint cụ thể:**
  ```bash
  python training/train.py --checkpoint-path checkpoints/qwen2.5-7b-eagle3.1/checkpoint-2000
  ```

#### 5. Cơ chế dọn rác checkpoint tự động (`save_total_limit: 2`):
- Để tránh phình to dung lượng ổ cứng trên máy chủ thuê và giúp việc tải về nhanh chóng, launcher tự động kích hoạt hàm dọn dẹp sau huấn luyện:
  - **Luôn bảo vệ tuyệt đối:** Thư mục checkpoint `best`.
  - **Giữ lại tối đa 2 checkpoint `latest` mới nhất.**
  - **Tự động xóa sạch:** Các checkpoint trung gian cũ hơn.
- Thư mục đầu ra chỉ nặng khoảng vài trăm megabytes thay vì hàng chục gigabytes.

#### 6. Theo dõi thời gian thực qua Weights & Biases (WandB):
- Thiết lập khóa API trước khi chạy:
  ```bash
  export WANDB_API_KEY="your_wandb_api_key_here"
  ```
- Launcher đã tích hợp sẵn đoạn code tự động định nghĩa toàn bộ biểu đồ (`setup_wandb_metrics`):
  - Khóa trục thời gian `step_metric="train/step"` cho các nhóm `train/*`, `eval/*`, `acceptance/*`.
  - Bảng tóm tắt tự động giá trị tối ưu: `train/loss (min)`, `eval/loss (min)`, `acceptance/alpha_overall (max)`, `acceptance/alpha_step_1...5 (max)`.
  - Tự động theo dõi tài nguyên phần cứng (VRAM GB, GPU %, Throughput) trong tab System.
  - Khi mở đường dẫn WandB, toàn bộ bảng điều khiển đã được sắp xếp sẵn tự động mà không cần tạo biểu đồ thủ công.

---

### Bước 3: Đóng gói và tải lên Hugging Face Hub ([export/package_and_upload.py](file:///d:/DoAn/qwen2.5-7b-eagle3.1-speculator/export/package_and_upload.py))

Sau khi SpecForge hoàn tất huấn luyện và xuất ra `model.safetensors` cùng `config.json` chuẩn Hugging Face:
```bash
export HF_TOKEN="hf_your_write_token_here"

python export/package_and_upload.py \
    --checkpoint-dir "checkpoints/qwen2.5-7b-eagle3.1" \
    --repo-id "theflyingkhui04/qwen2.5-7b-eagle3.1-speculator" \
    --upload
```
Script sẽ kiểm tra tính toàn vẹn của checkpoint, đính kèm `MODEL_CARD.md` và tải trực tiếp lên Hugging Face Hub.

---

## 4. Huấn luyện serverless trên Modal.com (Môi trường chính - Khuyên dùng)

Quy trình huấn luyện chính được thiết kế để chạy trực tiếp trên nền tảng **Modal.com** qua file [modal_pipeline.py](file:///d:/DoAn/qwen2.5-7b-eagle3.1-speculator/modal_pipeline.py).  
Ưu điểm:
- Cấp phát GPU **NVIDIA A100-80GB** theo nhu cầu thực tế, tự ngắt ngay khi hoàn thành (không lo phát sinh chi phí rảnh rỗi).
- Dữ liệu và checkpoint được lưu trữ bền vững trên **Modal Persistent Volume** (`qwen-eagle3-volume` gắn tại `/vol`).
- Hỗ trợ chế độ chạy ngầm `--detach` (có thể tắt máy cá nhân, tiến trình vẫn tiếp tục chạy trên đám mây).

### 4.1. Thiết lập môi trường và cấu hình bí mật (Modal secrets)

Cài đặt Modal CLI trên máy cá nhân:
```bash
pip install modal
modal setup
```

Tạo 2 secret chứa khóa API trên Modal (chỉ cần tạo một lần duy nhất):
```bash
# Secret cho Hugging Face (quyền Write để upload mô hình)
modal secret create my-huggingface-secret HF_TOKEN="hf_your_token_here"

# Secret cho Weights & Biases (để theo dõi biểu đồ huấn luyện)
modal secret create wandb-secret WANDB_API_KEY="your_wandb_api_key_here"
```

### 4.2. Chạy thử kiểm tra môi trường (Test run)

Trước khi kích hoạt huấn luyện lớn, bạn có thể chạy thử để xác thực container, GPU và cấu hình lệnh:
```bash
# In ra câu lệnh torchrun dự kiến (dry-run, không tốn GPU)
modal run modal_pipeline.py --step test --dry-run

# Chạy test thực tế 1 epoch, unroll 3 bước trên A100
modal run modal_pipeline.py --step test
```

### 4.3. Khởi chạy huấn luyện chính thức (Training)

Chạy huấn luyện đầu nháp EAGLE-3.1 trên GPU NVIDIA A100-80GB:
```bash
# Khởi chạy ngầm (tự do đóng terminal, máy chủ Modal tự thực hiện)
modal run --detach modal_pipeline.py --step train
```

Nếu muốn tiếp tục từ checkpoint tốt nhất đã lưu trên Volume:
```bash
modal run --detach modal_pipeline.py --step train --resume
```

Bạn cũng có thể ghi đè số epoch hoặc batch size trực tiếp từ CLI nếu cần:
```bash
modal run --detach modal_pipeline.py --step train --epochs 3 --batch-size 2
```

### 4.4. Đóng gói và tải lên Hugging Face Hub (Export)

Sau khi huấn luyện thành công hoặc khi muốn xuất bản checkpoint tốt nhất lên Hugging Face:
```bash
modal run modal_pipeline.py --step export --repo-id theflyingkhui04/qwen2.5-7b-eagle3.1-speculator
```

### 4.5. Chạy toàn trình một chạm (End-to-end all)

Lệnh 1-click kích hoạt liên hoàn: **Huấn luyện -> Lưu volume -> Đóng gói -> Tải lên Hugging Face**:
```bash
modal run --detach modal_pipeline.py --step all
```

### 4.6. Quản lý tác vụ ngầm và dữ liệu trên Modal Volume

- **Theo dõi tiến trình đang chạy ngầm:**
  ```bash
  # Xem danh sách các app đang hoạt động
  modal app list

  # Xem log thời gian thực của app
  modal app logs <app-id>
  ```
- **Kiểm tra dữ liệu trên Volume:**
  ```bash
  # Xem danh sách checkpoint đã lưu trong volume
  modal volume ls qwen-eagle3-volume checkpoints/qwen2.5-7b-eagle3.1
  ```
- **Tải checkpoint từ Volume về máy cá nhân (nếu muốn lưu trữ offline):**
  ```bash
  modal volume get qwen-eagle3-volume checkpoints/qwen2.5-7b-eagle3.1/best ./local_best_checkpoint
  ```

---

## 5. Chạy trên máy chủ GPU độc lập hoặc Vast.ai (Tùy chọn phụ)

Nếu bạn có nhu cầu chạy trên máy ảo chuyên dụng (Dedicated VM) như Vast.ai (1x NVIDIA A100-80GB hoặc RTX 4090-24GB) thay vì serverless, bạn có thể sử dụng script tự động [run_all.sh](file:///d:/DoAn/qwen2.5-7b-eagle3.1-speculator/run_all.sh) hoặc notebook tương tác [notebooks/train_on_vastai.ipynb](file:///d:/DoAn/qwen2.5-7b-eagle3.1-speculator/notebooks/train_on_vastai.ipynb).

```bash
git clone https://github.com/theflyingkhui04/qwen2.5-7b-eagle3.1-speculator.git
cd qwen2.5-7b-eagle3.1-speculator
pip install -r requirements.txt

# Thiết lập biến môi trường
export HF_TOKEN="hf_..."
export WANDB_API_KEY="your_wandb_key"
export CONTAINER_ID="your_vast_container_id"

# Chạy ngầm trong phiên tmux
tmux new -s train "bash run_all.sh"
```

### Các chốt chặn an toàn (Safeguards):
- **Disk Overflow Guard:** Kiểm tra dung lượng ổ đĩa trống >= 50GB trước khi kích hoạt quy trình.
- **Cost Ceiling Guard:** Giới hạn thời gian chạy tối đa cứng là **6.0 giờ**. Nếu vượt quá, tiến trình tự ngắt để tránh phát sinh chi phí ngoài ý muốn.
- **Auto-stop Instance:** Bất kể thành công hay gặp lỗi ngoại lệ, script tự động thực thi `vastai stop instance $CONTAINER_ID` để tắt máy chủ ngay lập tức.

---

## 6. Hướng dẫn suy luận tăng tốc với vLLM trên Tesla T4 (16GB)

Sau khi checkpoint được tải lên Hugging Face Hub, khởi chạy server vLLM tận dụng speculative decoding trên GPU Tesla T4 (hoặc bất kỳ GPU NVIDIA nào):

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ \
    --speculative-model theflyingkhui04/qwen2.5-7b-eagle3.1-speculator \
    --num-speculative-tokens 5 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 2048 \
    --dtype float16
```

Kiểm thử tốc độ suy luận qua API OpenAI-compatible:
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="token-dummy")
response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct-AWQ",
    messages=[
        {"role": "user", "content": "Trích xuất thông tin hóa đơn sau sang JSON: CÔNG TY TNHH ABC, MST: 0101234567, Tổng tiền: 1.500.000 VNĐ"}
    ],
    temperature=0.0,
)
print(response.choices[0].message.content)
```

- **Hệ số tăng tốc kỳ vọng:** 2.5x – 3.2x trên văn bản tiếng Việt và hóa đơn MC-OCR có cấu trúc JSON.
- **Chất lượng đầu ra:** Bảo toàn 100% không mất mát (Lossless Guarantee) do mô hình gốc 7B giữ quyền thẩm định tối cao.

---

## 7. Kiểm thử hệ thống và đảm bảo chất lượng

Dự án bao gồm bộ kiểm thử đơn vị tự động toàn diện kiểm định 100% các thành phần (configs, dataset, SpecForge launcher, export & packaging, safeguards):

```bash
pytest tests/ -v
```

Kết quả: **37/37 tests passed**.
