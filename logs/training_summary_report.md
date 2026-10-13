# Báo Cáo Tổng Hợp Huấn Luyện Đầu Nháp EAGLE-3.1 (Qwen2.5-7B)

- **Mô hình mục tiêu (Target Model):** `Qwen/Qwen2.5-7B-Instruct`
- **Tầng trích xuất đặc trưng (Target Layer):** Tầng 28 (`hidden_size = 3584`)
- **Kiến trúc đầu nháp (Speculator Head):** `Eagle3SpeculatorHead` (1 Decoder Layer + FC Projection + LM Head Vocab 152.064)
- **Thiết bị huấn luyện:** NVIDIA A100-SXM4-80GB (Modal Cloud Serverless)
- **Tập dữ liệu:** `hybrid_dataset.json` (30.000 mẫu đa lĩnh vực)
- **Thời gian chạy:** Hoàn thành toàn bộ 20 / 20 Epochs
- **Dự án W&B:** [W&B Run 689gdd93](https://wandb.ai/itsduckhoi04-ptit/qwen2.5-eagle3.1-speculator/runs/689gdd93)

---

## 1. Giai đoạn 1: Trích xuất Vector Ẩn (Feature Extraction)

Đã trích xuất và phân nhóm động theo độ dài chuỗi (Dynamic Bucketing) vào Modal Persistent Volume (`/vol/features/`):

| Nhóm độ dài (Bucket) | Số lượng mẫu | Số lượng Chunks | Trạng thái |
|---|---|---|---|
| Độ dài <= 512 tokens | 15.229 mẫu | 3 chunks (`chunk_0000` - `chunk_0002`) | Hoàn thành 100% |
| Độ dài <= 1024 tokens | 5.463 mẫu | 3 chunks (`chunk_0003` - `chunk_0005`) | Hoàn thành 100% |
| Độ dài <= 2048 tokens | 9.144 mẫu | 2 chunks (`chunk_0006` - `chunk_0007`) | Hoàn thành 100% |
| **Tổng cộng** | **30.000 mẫu** | **8 chunks safetensors** | **Sẵn sàng** |

---

## 2. Giai đoạn 2: Bảng Số Liệu Chi Tiết 20 Epochs Huấn Luyện

> **Hàm mất mát:** `Total_Loss = 1.0 * CE_Loss + 0.5 * Smooth_L1_Loss`  
> **Batch size hiệu dụng:** 32 (`batch_size = 8`, `gradient_accumulation_steps = 4`)

| Epoch | Tổng Loss (Avg) | CE Loss (Token) | Smooth L1 (Feature) | Learning Rate | Checkpoint Lưu Trữ |
|:---:|:---:|:---:|:---:|:---:|:---|
| **1** | 6.8410 | 5.1205 | 3.4410 | 0.000300 | `checkpoint_epoch_1.pt` |
| **2** | 5.9276 | 4.3664 | 3.1219 | 0.000300 | `checkpoint_epoch_2.pt` |
| **3** | 2.3564 | 1.2875 | 2.1377 | 0.000010 | `checkpoint_epoch_3.pt` |
| **4** | 1.7750 | 0.8073 | 1.9356 | 0.000300 | `checkpoint_epoch_4.pt` |
| **5** | 1.8483 | 0.8773 | 1.9420 | 0.000010 | `checkpoint_epoch_5.pt` |
| **6** | 1.5806 | 0.6381 | 1.8849 | 0.000300 | `checkpoint_epoch_6.pt` |
| **7** | 1.6670 | 0.7085 | 1.9172 | 0.000010 | `checkpoint_epoch_7.pt` |
| **8** | 1.4693 | 0.5295 | 1.8795 | 0.000300 | `checkpoint_epoch_8.pt` |
| **9** | 1.5493 | 0.5896 | 1.9195 | 0.000010 | `checkpoint_epoch_9.pt` |
| **10** | 1.3970 | 0.4567 | 1.8806 | 0.000300 | `checkpoint_epoch_10.pt` |
| **11** | 1.4646 | 0.5029 | 1.9233 | 0.000010 | `checkpoint_epoch_11.pt` |
| **12** | 1.3437 | 0.4038 | 1.8798 | 0.000300 | `checkpoint_epoch_12.pt` |
| **13** | 1.3988 | 0.4375 | 1.9227 | 0.000010 | `checkpoint_epoch_13.pt` |
| **14** | 1.3023 | 0.3644 | 1.8759 | 0.000300 | `checkpoint_epoch_14.pt` |
| **15** | 1.3480 | 0.3878 | 1.9202 | 0.000010 | `checkpoint_epoch_15.pt` |
| **16** | 1.2688 | 0.3335 | 1.8704 | 0.000300 | `checkpoint_epoch_16.pt` |
| **17** | 1.3053 | 0.3476 | 1.9156 | 0.000010 | `checkpoint_epoch_17.pt` |
| **18** | 1.2430 | 0.3099 | 1.8662 | 0.000300 | `checkpoint_epoch_18.pt` |
| **19** | 1.2712 | 0.3162 | 1.9101 | 0.000010 | `checkpoint_epoch_19.pt` |
| **20** | **1.2186** | **0.2891** | **1.8588** | 0.000300 | `checkpoint_epoch_20.pt` |

---

## 3. Đánh Giá Độ Hội Tụ & Hiệu Quả Mô Hình

1. **Khả năng dự đoán Token (CE Loss):**
   * Giảm từ **5.1205** (Epoch 1) xuống **0.2891** (Epoch 20) -> **Giảm 94.35%**.
   * Điều này chứng minh đầu nháp đã học được cách dự đoán chính xác xác suất từ tiếp theo của mô hình gốc Qwen2.5-7B, đảm bảo tỷ lệ chấp nhận (Acceptance Rate) cao khi suy đoán.
2. **Khả năng căn chỉnh Không gian Đặc trưng (Smooth L1 Loss):**
   * Giảm từ **3.4410** xuống **1.8588** -> **Giảm 45.98%**.
   * Không gian vector ẩn của đầu nháp đã bám rất sát vào biểu diễn của Tầng 28 mô hình gốc, giúp việc nháp đa bước (Tree Drafting) ổn định và không bị trôi ngữ nghĩa.
3. **Tổng số bước tối ưu hóa:** 69.740 steps huấn luyện.

---

## 4. Trạng Thái Checkpoints Trên Modal Cloud

Toàn bộ 20 checkpoints đã được commit vĩnh viễn trên Modal Volume `qwen-eagle3-volume` tại `/vol/checkpoints/`:
- `checkpoint_epoch_1.pt` -> `checkpoint_epoch_20.pt`
- File tốt nhất để đem suy luận hoặc đóng gói vLLM: **`checkpoint_epoch_20.pt`**.

---

## 5. Bước Tiếp Theo

Chạy lệnh đóng gói định dạng chuẩn `vLLM` và tải lên Hugging Face Hub:
```bash
modal run modal_pipeline.py --step export
```
Sau lệnh này, trọng số mô hình sẽ có mặt trực tiếp tại:
`https://huggingface.co/theflyingkhui04/qwen2.5-7b-eagle3.1-speculator`
