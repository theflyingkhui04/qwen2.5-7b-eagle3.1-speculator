# Đặc tả nhiệm vụ: Task 1 - Chuẩn hóa tầng dữ liệu và phân chia Train / Val / Test

- **Mã nhiệm vụ:** `TASK-01`
- **Tên nhiệm vụ:** Chuẩn hóa tầng dữ liệu và phân chia Train / Val / Test theo chuẩn Machine Learning
- **Trạng thái:** Đang thực hiện (In Progress)
- **Tài liệu tham chiếu:** [worklogs/task-1.md](file:///d:/DoAn/qwen2.5-7b-eagle3.1-speculator/worklogs/task-1.md), [docs/dataset_blueprint_production.md](file:///d:/DoAn/qwen2.5-7b-eagle3.1-speculator/docs/dataset_blueprint_production.md)

---

## 1. Mục tiêu kỹ thuật
1. Xây dựng pipeline xử lý dữ liệu chuẩn hóa sang định dạng **ShareGPT JSON Lines (`.jsonl`)** mà SpecForge yêu cầu.
2. Thiết kế và phân chia độc lập 3 tập dữ liệu theo chuẩn Machine Learning để chống rò rỉ dữ liệu (data leakage) và đảm bảo tính khách quan trong báo cáo đồ án:
   - **Train:** 40.000 mẫu (~40M đến 60M tokens).
   - **Val:** 3.000 mẫu (đánh giá Dual Loss, early stopping).
   - **Test:** 2.000 mẫu độc lập (đo đạc tốc độ speedup và acceptance rate alpha).
3. Đảm bảo tỷ lệ phối trộn hợp gu với Qwen2.5-7B: 40% Tiếng Việt tổng quát, 35% Hóa đơn MC-OCR & Structured JSON, 25% Tiếng Anh & CoT.

---

## 2. Đầu vào và đầu ra (Inputs & Outputs)

### Đầu vào (Inputs):
- File nguồn thô:
  - `data/raw/sharegpt.json` (hoặc tập dữ liệu ShareGPT / WildChat).
  - Dữ liệu Tiếng Việt: `bkai-foundation-models/vi-alpaca` (kho 52k mẫu) hoặc `vietnamese_invoices.json`.
  - Dữ liệu Hóa đơn MC-OCR thực tế (nếu có).

### Đầu ra (Outputs):
- `data/processed/train.jsonl` (40.000 dòng JSON độc lập).
- `data/processed/val.jsonl` (3.000 dòng JSON độc lập).
- `data/processed/test.jsonl` (2.000 dòng JSON độc lập: 1.000 hóa đơn, 600 tiếng Việt, 400 code/JSON).

---

## 3. Quy chuẩn định dạng dữ liệu (Data Schema)
Mỗi dòng trong file `.jsonl` bắt buộc có cấu trúc:
```json
{
  "id": "sample_id_unique",
  "conversations": [
    {"from": "human", "value": "Câu hỏi hoặc prompt của người dùng..."},
    {"from": "gpt", "value": "Câu trả lời của mô hình..."}
  ]
}
```

---

## 4. Tiêu chuẩn nghiệm thu (Definition of Done - DoD)
- [x] Script `data/prepare_hybrid_dataset.py` xuất đúng định dạng ShareGPT JSON Lines (`.jsonl`).
- [x] Toàn bộ unit tests trong `tests/test_dataset.py` chạy qua 100% (5/5 PASS).
- [ ] Hoàn thành module phân chia 3 tập (`train.jsonl`, `val.jsonl`, `test.jsonl`) đúng tỷ lệ và số lượng.
- [ ] Không có mẫu rỗng, mẫu 1 lượt, hoặc chứa ký tự ngắt dòng dị thường (`\u2028`, `\u2029`).
