# Đặc tả nhiệm vụ: Task 1 - Chuẩn hóa tầng dữ liệu và phân chia Train / Val / Test

- **Mã nhiệm vụ:** `TASK-01`
- **Tên nhiệm vụ:** Chuẩn hóa tầng dữ liệu và phân chia Train / Val / Test theo chuẩn Machine Learning
- **Trạng thái:** Đã hoàn thành (Done)
- **Tài liệu tham chiếu:** [worklogs/task-1.md](file:///d:/DoAn/qwen2.5-7b-eagle3.1-speculator/worklogs/task-1.md), [docs/dataset_blueprint_production.md](file:///d:/DoAn/qwen2.5-7b-eagle3.1-speculator/docs/dataset_blueprint_production.md)

---

## 1. Mục tiêu kỹ thuật
1. Xây dựng pipeline xử lý dữ liệu chuẩn hóa sang định dạng **ShareGPT JSON Lines (`.jsonl`)** mà SpecForge yêu cầu.
2. Thiết kế và phân chia độc lập 3 tập dữ liệu theo chuẩn Machine Learning để chống rò rỉ dữ liệu (data leakage) và đảm bảo tính khách quan trong báo cáo đồ án:
   - **Train:** `data/processed/train.jsonl` (huấn luyện SpecForge).
   - **Val:** `data/processed/val.jsonl` (đánh giá Dual Loss, early stopping).
   - **Test:** `data/processed/test.jsonl` (đo đạc độc lập tốc độ speedup và acceptance rate alpha).
3. Đảm bảo tỷ lệ phối trộn hợp gu với Qwen2.5-7B: 40% Tiếng Việt tổng quát, 35% Hóa đơn MC-OCR & Structured JSON, 25% Tiếng Anh & CoT.

---

## 2. Đầu vào và đầu ra (Inputs & Outputs)

### Đầu vào (Inputs):
- File nguồn thô:
  - `data/raw/sharegpt.json` (tập dữ liệu ShareGPT tiếng Anh).
  - Dữ liệu Tiếng Việt: `bkai-foundation-models/vi-alpaca` hoặc `data/raw/vietnamese_invoices.json`.
  - Dữ liệu Hóa đơn MC-OCR thực tế.

### Đầu ra (Outputs):
- `data/processed/train.jsonl` (JSON Lines chuẩn ShareGPT).
- `data/processed/val.jsonl` (JSON Lines chuẩn ShareGPT).
- `data/processed/test.jsonl` (JSON Lines chuẩn ShareGPT, hoàn toàn phân tách không trùng mẫu).

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
- [x] Toàn bộ unit tests trong `tests/test_dataset.py` chạy qua 100% (6/6 PASS).
- [x] Hoàn thành module phân chia 3 tập (`train.jsonl`, `val.jsonl`, `test.jsonl`) đảm bảo phân tách hoàn toàn (disjoint), triệt tiêu nguy cơ data leakage.
- [x] Đã lọc sạch mẫu rỗng, mẫu 1 lượt, và ký tự ngắt dòng dị thường (`\u2028`, `\u2029`).
