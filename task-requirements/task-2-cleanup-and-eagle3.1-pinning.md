# Đặc tả nhiệm vụ: Task 2 - Dọn dẹp code cũ và ghim chuẩn kiến trúc EAGLE-3.1

- **Mã nhiệm vụ:** `TASK-02`
- **Tên nhiệm vụ:** Dọn dẹp các module cũ không còn sử dụng và ghim cứng chuẩn kiến trúc EAGLE-3.1
- **Trạng thái:** Đã hoàn thành (Done)
- **Tài liệu tham chiếu:** [docs/2026-10-24-qwen2.5-eagle3.1-specforge-refactor-spec.md](file:///d:/DoAn/qwen2.5-7b-eagle3.1-speculator/docs/2026-10-24-qwen2.5-eagle3.1-specforge-refactor-spec.md)

---

## 1. Mục tiêu kỹ thuật
1. Xóa bỏ hoàn toàn code trích xuất activations offline thủ công và các file mô hình/hàm loss tự viết trước đây để loại bỏ rác codebase và giải quyết triệt để vấn đề nghẽn I/O đĩa.
2. Ghim cứng kiến trúc **EAGLE-3.1** trong toàn bộ cấu hình dự án, nghiêm cấm hạ cấp về EAGLE-1 hoặc EAGLE-2.
3. Đảm bảo cấu hình chứa đầy đủ các cờ chuẩn hóa: `use_fc_norm: true`, `norm_output: true`, `use_post_norm_feedback: true`, `target_layer: 28`.

---

## 2. Danh sách file cần thao tác

### Xóa bỏ (Deletions):
- `extraction/extract_hidden_states.py` (Đã xóa)
- `extraction/__init__.py` (Đã xóa)
- `training/model.py` (Đã xóa)
- `training/loss.py` (Đã xóa)
- `configs/eagle3_qwen2.5_7b.json` (Đã xóa)
- `tests/test_extraction.py` (Đã xóa)
- `tests/test_loss.py` (Đã xóa)
- `tests/test_model.py` (Đã xóa)

### Cập nhật & duy trì (Updates & Pinning):
- `training/__init__.py`: Đã làm sạch module exports.
- `configs/eagle3.1_qwen2.5_7b.json`: Duy trì làm file cấu hình kiến trúc duy nhất của draft head, bổ sung `norm_output: true`.
- `AGENTS.md`: Đã ghim quy tắc kiến trúc EAGLE-3.1 vào Workspace Rules.

---

## 3. Tiêu chuẩn nghiệm thu (Definition of Done - DoD)
- [x] Không còn bất kỳ file code trích xuất offline nào trong repo.
- [x] Mạng tự viết thủ công và loss thủ công đã được gỡ bỏ hoàn toàn.
- [x] Có bài test `tests/test_cleanup_deprecated.py` kiểm tra xác nhận tất cả các file cũ đã biến mất (PASS).
- [x] File cấu hình `configs/eagle3.1_qwen2.5_7b.json` giữ nguyên đầy đủ các trường của EAGLE-3.1 (`num_hidden_layers: 1`, `use_fc_norm: true`, `norm_output: true`, `use_post_norm_feedback: true`, `target_layer: 28`).
