# Blueprint tập dữ liệu tối ưu cho huấn luyện EAGLE-3.1 (Qwen2.5-7B)

> **Mục tiêu:** Tối đa hóa tỷ lệ chấp thuận token (Acceptance Rate alpha >= 70% - 85%) và hệ số tăng tốc suy luận (Speedup 2.2x - 3.0x) cho mô hình `theflyingkhui04/qwen2.5-7b-eagle3.1-speculator` khi triển khai thực tế trên vLLM / SGLang.

---

## 1. Cơ sở lý thuyết và bản chất speculative decoding

Trong cơ chế Speculative Decoding với EAGLE-3.1:
- Đầu nháp (Draft Head) không tiếp thu tri thức mới, mà thực hiện bài toán **xấp xỉ phân phối xác suất token của mô hình đích**:
  `p_draft(x_t | x_<t) xấp xỉ p_target(x_t | x_<t)`
- Kỳ vọng số lượng token được sinh ra trên mỗi bước xác thực (verification step) tuân theo:
  `E[tokens/step] = (1 - alpha^(k+1)) / (1 - alpha)`
  *(với k là độ sâu suy đoán ttt_length, và alpha là acceptance rate)*.

Nếu `alpha < 50%`, overhead forward của draft head cộng với chi phí rollback sẽ khiến hệ thống chạy chậm hơn cả sinh tuần tự thông thường.

### Thách thức đặc thù của Qwen2.5 và miền ứng dụng:
1. **Byte-level BPE của Qwen2.5:** Các nguyên âm tiếng Việt mang dấu thanh (như "ế", "ặ", "ồ") được phân rã thành chuỗi 2 đến 3 byte tokens. Việc thiếu dữ liệu tiếng Việt đa dạng làm draft head dự đoán trượt byte đuôi, dẫn đến việc target model bác bỏ (reject) toàn bộ nhánh unroll.
2. **Cấu trúc JSON & Hóa đơn MC-OCR:** Trong bài toán trích xuất hóa đơn, mô hình liên tục sinh các token cú pháp lặp lại (`"`, `:`, `{`, `}`, `\n`). Nếu draft head được học phân phối này, nó có thể unroll một mạch 4 đến 5 tokens cú pháp chỉ trong 1 bước với độ chính xác xấp xỉ 100%.
3. **Hiện tượng lệch phân phối (distribution shift):** Bộ dữ liệu ShareGPT nguyên bản phản ánh văn phong của OpenAI GPT-3.5/GPT-4 đời đầu, không trùng khớp với thói quen sinh từ của `Qwen2.5-7B-Instruct`.

---

## 2. Khung tỷ lệ phối trộn dữ liệu khuyến nghị (Production Mix Ratio)

Quy mô mục tiêu: **45.000 cuộc hội thoại đa lượt** (phân chia chuẩn Machine Learning: 40.000 Train, 3.000 Val, 2.000 Test).

| Nhóm dữ liệu | Tỷ lệ mục tiêu | Nguồn dữ liệu khuyến nghị | Vai trò kỹ thuật |
| :--- | :---: | :--- | :--- |
| **Tiếng Việt tổng quát và tri thức** | **35% - 40%** | `Vie-Alpaca`, `vietnamese-bi-llava` (text subset), `ShareGPT-Vi` lọc sạch | Khắc phục hiện tượng trượt subword/byte BPE tiếng Việt trên các dấu thanh và âm tiết ghép. |
| **Hóa đơn MC-OCR và structured output (JSON)** | **30% - 35%** | Hóa đơn MC-OCR (Key-Value extraction), `Code-Feedback`, `Evol-Instruct-Code` | Đẩy vọt Acceptance Rate trên các ký tự cú pháp lập trình, bảng biểu, dấu ngoặc và schema JSON. |
| **Tiếng Anh tổng quát và chuỗi suy luận (CoT)** | **25% - 30%** | `WildChat-1M` (lọc turn chất lượng cao), `LMSYS-Chat-1M` | Duy trì khả năng suy đoán nháp trên văn bản tiếng Anh và các bước lập luận từng bước. |

---

## 3. Quy trình on-policy distillation (khuyên dùng)

Phương pháp tối ưu tuyệt đối để xóa bỏ distribution shift là để **chính `Qwen/Qwen2.5-7B-Instruct` làm Teacher sinh nhãn (completions)** cho toàn bộ tập prompts đầu vào.

```text
[Tập prompts đa dạng] 
       │
       ▼
[vLLM Offline Engine: Qwen/Qwen2.5-7B-Instruct]
  * Chat Template: Qwen2.5 chuẩn
  * Temperature: 0.7, Top-p: 0.9 (hoặc Greedy cho code/JSON)
       │
       ▼
[Cặp hội thoại hoàn chỉnh: Prompt + Qwen2.5 Response]
       │
       ▼
[Lưu định dạng ShareGPT JSON Lines: data/processed/train.jsonl]
       │
       ▼
[SpecForge Unroll Multi-step Autoregressive Training]
```

### Script mẫu sinh dữ liệu on-policy với vLLM offline:

```python
# data/generate_on_policy_distill.py
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import json

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

prompts_list = [
    "Trích xuất thông tin người bán và tổng tiền từ hóa đơn sau: ...",
    "Viết một hàm Python tính diện tích hình tròn có kiểm tra tham số.",
    # Thêm danh sách prompt tiếng Việt, Code, Reasoning
]

# Chuẩn bị prompts theo Chat Template Qwen2.5
formatted_prompts = [
    tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
    for p in prompts_list
]

llm = LLM(model=MODEL_ID, tensor_parallel_size=1, max_model_len=2048)
sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=1024)

outputs = llm.generate(formatted_prompts, sampling_params)

with open("data/processed/train.jsonl", "w", encoding="utf-8") as f:
    for idx, (p, out) in enumerate(zip(prompts_list, outputs)):
        response_text = out.outputs[0].text
        sample = {
            "id": f"distill_{idx}",
            "conversations": [
                {"from": "human", "value": p},
                {"from": "gpt", "value": response_text}
            ]
        }
        f.write(json.dumps(sample, ensure_ascii=False) + "\n")
```

---

## 4. Tóm tắt phân chia tập dữ liệu trong repo

1. **Dataset TRAIN (`data/processed/train.jsonl`):** 40.000 mẫu dùng để huấn luyện unroll tự hồi quy với SpecForge.
2. **Dataset VAL (`data/processed/val.jsonl`):** 3.000 mẫu dùng để theo dõi Dual Loss sau mỗi epoch.
3. **Dataset TEST (`data/processed/test.jsonl`):** 2.000 mẫu (1.000 Hóa đơn MC-OCR mới, 600 Tiếng Việt mới, 400 Code/JSON mới) độc lập hoàn toàn để đo lường speedup và acceptance rate cho bài báo cáo đồ án.
