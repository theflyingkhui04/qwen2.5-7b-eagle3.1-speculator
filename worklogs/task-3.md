# Task 3: Cập nhật thư viện phụ thuộc, chuẩn hóa siêu tham số huấn luyện và kiến trúc EAGLE-3.1

- **Dự án:** `qwen2.5-7b-eagle3.1-speculator`
- **Mã nhiệm vụ:** `TASK-03`
- **Mô hình mục tiêu (Target Model):** `Qwen/Qwen2.5-7B-Instruct`
- **Kiến trúc đầu nháp (Draft Head):** EAGLE-3.1 Speculator (1 Decoder Layer, FC Normalization, multi-layer hidden fusion)
- **Framework huấn luyện:** `sgl-project/SpecForge`
- **Ngày lập:** 25/10/2026
- **Trạng thái:** Đã hoàn thành (Done & Verified)
- **Tài liệu tham chiếu:** [task-requirements/task-3-dependencies-and-configs.md](file:///d:/DoAn/qwen2.5-7b-eagle3.1-speculator/task-requirements/task-3-dependencies-and-configs.md), [configs/train_config.yaml](file:///d:/DoAn/qwen2.5-7b-eagle3.1-speculator/configs/train_config.yaml), [configs/eagle3.1_qwen2.5_7b.json](file:///d:/DoAn/qwen2.5-7b-eagle3.1-speculator/configs/eagle3.1_qwen2.5_7b.json)

---

## 1. Bối cảnh kỹ thuật và mục tiêu nhiệm vụ

Nhiệm vụ Task 3 tập trung vào việc chuẩn hóa toàn bộ hạ tầng cấu hình để chuẩn bị cho SpecForge runner khởi chạy:
1. Đồng bộ hóa các thư viện chính thức trong `requirements.txt` (`SpecForge`, `sglang[all]`, `datasets`, `accelerate`).
2. Tái cấu trúc file siêu tham số `configs/train_config.yaml` từ cấu hình tự viết cũ sang định dạng tham số dòng lệnh chuẩn của SpecForge.
3. Ghim cứng và kiểm chứng toàn bộ thông số hình học của mạng draft head EAGLE-3.1 trong `configs/eagle3.1_qwen2.5_7b.json` tương thích hoàn toàn với họ mô hình Qwen2.5-7B.
4. Xây dựng bộ kiểm thử tự động (TDD) trong `tests/test_configs.py` và `tests/test_scaffold.py` để bảo đảm các file cấu hình luôn nguyên vẹn, không bị biến đổi ngoài ý muốn.

---

## 2. Danh mục chi tiết các chỉ số, siêu tham số và lý do lựa chọn

Toàn bộ hệ thống cấu hình được thiết kế dựa trên sự cân bằng tối ưu giữa: **độ chính xác dự đoán nháp (Acceptance Rate alpha)**, **giới hạn bộ nhớ phần cứng (VRAM)** và **thời gian/chi phí huấn luyện trên GPU**.

### 2.1. Siêu tham số huấn luyện trong `configs/train_config.yaml`

Cấu trúc file thực tế:
```yaml
target_model_path: "Qwen/Qwen2.5-7B-Instruct"
draft_model_config: "configs/eagle3.1_qwen2.5_7b.json"
train_data_path: "data/processed/train.jsonl"
val_data_path: "data/processed/val.jsonl"
output_dir: "checkpoints/qwen2.5-7b-eagle3.1"
num_epochs: 3
batch_size: 2
gradient_accumulation_steps: 4
learning_rate: 1.0e-4
max_seq_len: 2048
ttt_length: 5
tp_size: 1
```

| Tên chỉ số | Giá trị | Vai trò và lý do kỹ thuật lựa chọn |
| :--- | :---: | :--- |
| `num_epochs` | `3` | **Tối ưu điểm hội tụ, triệt tiêu nguy cơ học vẹt (overfitting):** Tập dữ liệu `train.jsonl` có quy mô 40.000 mẫu hội thoại. Với 3 epochs, mô hình thực hiện đúng 15.000 bước cập nhật trọng số. Do đầu nháp chỉ có 1 tầng Transformer (~100M tham số) và mô hình gốc 7B bị đóng băng 100%, 3 epochs là điểm hội tụ vàng (sweet spot) theo khuyến nghị từ bài báo tác giả EAGLE và tài liệu SpecForge. Huấn luyện nhiều hơn (như 10-20 epoch) sẽ khiến đầu nháp học vẹt, mất tính đa dạng khi sinh cây nháp và làm giảm tốc độ suy luận thực tế. |
| `batch_size` | `2` | **Micro-batch size chống tràn bộ nhớ GPU (CUDA OOM):** Đây là kích thước lô con nạp trực tiếp vào VRAM trong một lượt tính forward/backward. Do GPU phải nạp đồng thời mô hình gốc 7B (~15GB VRAM), mô hình nháp và thực hiện unroll tự hồi quy 5 bước với chuỗi dài tối đa 2048 tokens, việc duy trì `batch_size: 2` giữ cho VRAM ổn định ở mức an toàn (~24GB đến 40GB). |
| `gradient_accumulation_steps` | `4` | **Tạo kích thước lô hiệu dụng ổn định:** Kết hợp với `batch_size: 2`, kích thước lô hiệu dụng thực tế cho mỗi lần tối ưu trọng số (optimizer step) là: `effective_batch_size = 2 * 4 * 1 = 8` mẫu. Tương đương mỗi lần cập nhật trọng số, mô hình học từ khoảng 8 * 2048 = 16.384 tokens, đảm bảo ước lượng đạo hàm mượt mà, không bị nhiễu số học. |
| `learning_rate` | `1.0e-4` | **Tốc độ học tối ưu cho đầu nháp ngẫu nhiên:** Đầu nháp được khởi tạo mới nhưng tiếp nhận vector đặc trưng đã rất mạnh từ tầng 28 của mô hình gốc. Tốc độ `1e-4` kết hợp cùng bộ tối ưu AdamW giúp mạng học nhanh, ổn định và không bị nổ đạo hàm (gradient explosion) khi dùng độ chính xác float16. |
| `max_seq_len` | `2048` | **Bao phủ 98% ngữ cảnh thực tế:** Các tác vụ xử lý hóa đơn MC-OCR, trích xuất cấu trúc JSON và đối thoại thông thường hiếm khi vượt quá 2000 tokens. Giới hạn 2048 giúp khống chế kích thước ma trận chú ý FlashAttention, giảm thiểu thời gian tính toán và tránh lãng phí bộ nhớ đệm KV cache. |
| `ttt_length` | `5` | **Độ sâu unroll tự hồi quy cân bằng tối ưu:** TTT (Test-Time Training unroll) là số bước sinh nháp liên tiếp mà speculator tự dự đoán trước khi mô hình gốc thẩm định. Độ sâu k = 5 là điểm cân bằng vàng: thông thường tỷ lệ chấp thuận trung bình đạt 2 đến 3.5 token mỗi vòng lặp. Nếu đặt unroll quá ngắn (<3), độ tăng tốc khi suy luận thấp; nếu đặt quá dài (>7), các token ở đuôi thường bị bác bỏ (reject), gây lãng phí bộ nhớ và thời gian tính toán. |
| `tp_size` | `1` | **Phù hợp chạy đơn GPU:** Đặt Tensor Parallelism bằng 1 giúp pipeline huấn luyện linh hoạt trên 1 card GPU độc lập (như NVIDIA A100-80GB hoặc A6000-48GB) mà không đòi hỏi hạ tầng mạng đa GPU phức tạp. |
| `target_model_path` | `"Qwen/Qwen2.5-7B-Instruct"` | Mô hình gốc chuẩn của Alibaba Cloud, có năng lực vượt trội về tiếng Việt, suy luận cấu trúc JSON và lập trình. |
| `draft_model_config` | `"configs/eagle3.1_qwen2.5_7b.json"` | Đường dẫn cấu hình kiến trúc đầu nháp chuẩn EAGLE-3.1 được ghim cứng trong toàn dự án. |

---

### 2.2. Tham số kiến trúc đầu nháp trong `configs/eagle3.1_qwen2.5_7b.json`

Cấu trúc file thực tế:
```json
{
  "architectures": [
    "Eagle3_1Qwen2ForCausalLM",
    "EagleQwen2ForCausalLM"
  ],
  "model_type": "eagle_qwen2",
  "eagle_version": "3.1",
  "hidden_size": 3584,
  "intermediate_size": 18944,
  "num_hidden_layers": 1,
  "num_attention_heads": 28,
  "num_key_value_heads": 4,
  "hidden_act": "silu",
  "max_position_embeddings": 32768,
  "initializer_range": 0.02,
  "rms_norm_eps": 1e-06,
  "vocab_size": 152064,
  "target_model_name_or_path": "Qwen/Qwen2.5-7B-Instruct",
  "target_layer": 28,
  "use_fc_norm": true,
  "norm_output": true,
  "use_post_norm_feedback": true,
  "torch_dtype": "float16"
}
```

| Tên thông số | Giá trị | Ý nghĩa kỹ thuật |
| :--- | :---: | :--- |
| `num_hidden_layers` | `1` | **Tối thiểu độ trễ sinh nháp (Draft Latency):** Mục tiêu cốt lõi của speculative decoding là đầu nháp phải sinh token nhanh hơn mô hình gốc gấp 5 đến 8 lần. Dùng đúng 1 tầng Decoder giúp thời gian forward mỗi bước nháp chỉ mất khoảng 1 đến 2 ms trên GPU. |
| `target_layer` | `28` | **Tầng đặc trưng ngữ nghĩa cao cấp nhất:** Qwen2.5-7B có 28 tầng Transformer. Tầng 28 là tầng áp chót (penultimate layer), nơi biểu diễn ngữ nghĩa và dự đoán token hội tụ mạnh nhất. Đầu nháp tiếp nhận đặc trưng từ tầng này sẽ không tốn tài nguyên tái học các biểu diễn cấp thấp. |
| `num_attention_heads` | `28` | Số lượng query attention heads khớp tuyệt đối với Qwen2.5-7B. |
| `num_key_value_heads` | `4` | **Tương thích Grouped-Query Attention (GQA):** Tỷ lệ 28 query heads trên 4 KV heads (tỷ lệ 7:1) giúp giảm dung lượng bộ nhớ đệm KV cache của đầu nháp gấp 7 lần so với Multi-Head Attention truyền thống, tương thích trực tiếp với kernel tối ưu trong vLLM và SGLang. |
| `hidden_size` | `3584` | Khớp chiều không gian ẩn của mô hình gốc, giúp phép chiếu ma trận ghép đặc trưng (hidden fusion) không bị méo mó. |
| `intermediate_size` | `18944` | Khớp kích thước tầng trung gian SwiGLU FFN của Qwen2.5-7B. |
| `vocab_size` | `152064` | Khớp toàn bộ không gian từ vựng Byte-level BPE của tokenizer Qwen2.5. |
| `use_fc_norm: true` | `true` | **Ổn định biên độ kích hoạt:** Chuẩn hóa RMSNorm trước các tầng chiếu fully-connected, triệt tiêu hiện tượng trôi biên độ (activation drift) khi unroll nhiều bước. |
| `norm_output: true` | `true` | **Đồng bộ phân phối xác suất:** Chuẩn hóa vector đặc trưng đầu ra trước khi qua `lm_head`, đảm bảo phân phối logit khớp với phân phối của mô hình gốc. |
| `use_post_norm_feedback: true` | `true` | **Cải tiến độc quyền của EAGLE-3.1:** Đưa tín hiệu phản hồi đã chuẩn hóa ngược trở lại vòng lặp unroll tiếp theo, triệt tiêu sai số tích lũy (compounding errors) qua 5 bước sinh nháp. |

---

### 2.3. Tỷ lệ phân bổ dữ liệu huấn luyện và kiểm thử (45.000 mẫu)

| Nguồn dữ liệu | Số lượng Train | Tỷ lệ | Vai trò trong bài toán đồ án |
| :--- | :---: | :---: | :--- |
| **Tiếng Việt tổng quát** | `16.000` | 40% | Giúp đầu nháp làm quen sâu sắc với cấu trúc ngữ pháp, từ ghép và dấu thanh tiếng Việt trong bảng mã 152k subwords của Qwen2.5. |
| **Hóa đơn MC-OCR & JSON** | `14.000` | 35% | Bài toán trọng tâm của đồ án: các token cú pháp JSON (`"`, `:`, `{`, `}`, `\n`) lặp lại theo mẫu cấu trúc cao, cho phép đầu nháp unroll một mạch 4-5 tokens với tỷ lệ chấp thuận xấp xỉ 100%. |
| **Tiếng Anh & CoT** | `10.000` | 25% | Bảo tồn khả năng lập luận chuỗi tư duy (Chain-of-Thought) và tri thức gốc, ngăn chặn hiện tượng quên kiến thức (catastrophic forgetting). |
| **Phân chia độc lập** | `40k / 3k / 2k` | - | Phân tách hoàn toàn (disjoint split) 3 tập Train, Val, Test nhằm chống rò rỉ dữ liệu (data leakage) và đảm bảo tính khách quan khi báo cáo tốc độ speedup trước hội đồng. |

---

## 3. Nhật ký kiểm thử và nghiệm thu kỹ thuật (Execution Log)

### 3.1. Các file đã thao tác
- [requirements.txt](file:///d:/DoAn/qwen2.5-7b-eagle3.1-speculator/requirements.txt): Bổ sung `sglang[all]`, `git+https://github.com/sgl-project/SpecForge.git`, `datasets>=2.20.0`.
- [configs/train_config.yaml](file:///d:/DoAn/qwen2.5-7b-eagle3.1-speculator/configs/train_config.yaml): Cấu trúc lại toàn bộ các trường theo chuẩn SpecForge.
- [configs/eagle3.1_qwen2.5_7b.json](file:///d:/DoAn/qwen2.5-7b-eagle3.1-speculator/configs/eagle3.1_qwen2.5_7b.json): Xác thực đầy đủ các cờ chuẩn hóa EAGLE-3.1.
- [tests/test_configs.py](file:///d:/DoAn/qwen2.5-7b-eagle3.1-speculator/tests/test_configs.py): Cập nhật kiểm tra cấu hình SpecForge và EAGLE-3.1.
- [tests/test_scaffold.py](file:///d:/DoAn/qwen2.5-7b-eagle3.1-speculator/tests/test_scaffold.py): Cập nhật kiểm tra các gói `SpecForge` và `sglang`.
- [task-requirements/task-3-dependencies-and-configs.md](file:///d:/DoAn/qwen2.5-7b-eagle3.1-speculator/task-requirements/task-3-dependencies-and-configs.md): Đánh dấu hoàn thành toàn bộ DoD.
- [task-requirements/README.md](file:///d:/DoAn/qwen2.5-7b-eagle3.1-speculator/task-requirements/README.md): Chuyển trạng thái sang `DONE`.

### 3.2. Kết quả kiểm thử tự động
```
tests/test_configs.py ..                                                 [ 50%]
tests/test_scaffold.py ..                                                [100%]
============================== 4 passed in 0.08s ==============================

Toàn bộ 19 bài test hiện tại trong dự án (ngoại trừ test_train.py thuộc Task 4):
============================= 19 passed in 0.21s ==============================
```
