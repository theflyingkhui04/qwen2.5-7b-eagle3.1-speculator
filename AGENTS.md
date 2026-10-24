# Workspace Rules

- **Không tự động commit:** Tuyệt đối không tự động chạy lệnh `git commit`. Sau khi hoàn thành code, chỉ cung cấp commit message gợi ý để người dùng tự kiểm tra và commit.
- **Không dùng LaTeX:** Trình bày công thức bằng chữ thường và Unicode trực quan.
- **Quy tắc chính tả tiếng Việt (No English Title Case):**
  - Tuyệt đối không viết hoa chữ cái đầu của tất cả các từ trong câu hoặc tiêu đề theo kiểu Title Case của tiếng Anh (ví dụ sai: "Báo Cáo Kỹ Thuật & Quyết Định", "Hiện Tượng Bất Định Số Học").
  - Tuân thủ nghiêm ngặt quy tắc chính tả tiếng Việt:
    - Chỉ viết hoa chữ cái đầu tiên của câu, đầu đoạn hoặc đầu tiêu đề (ví dụ đúng: "Báo cáo kỹ thuật và quyết định", "Hiện tượng bất định số học").
    - Chỉ viết hoa danh từ riêng (tên người, địa danh, tên riêng của mô hình, công nghệ: Bùi Đức Đại, Cấn Đức Khôi, Tesla T4, PyTorch, CUDA, vLLM, Qwen2.5, NIST).
    - Viết hoa các từ viết tắt chuẩn (PM, DoD, AWQ, VRAM, OOM, JSON, SQL).
  - Khi tạo hoặc chỉnh sửa bất kỳ tài liệu tiếng Việt nào (file Markdown, commit message, báo cáo), luôn áp dụng quy tắc này để câu văn tự nhiên và chuẩn văn phong học thuật tiếng Việt.
- **Ghim kiến trúc EAGLE-3.1:** Chỉ sử dụng cấu hình `configs/eagle3.1_qwen2.5_7b.json`, duy trì bắt buộc `fc_norm: true`, `norm_output: true`, `use_post_norm_feedback: true`, `target_layer: 28`, 1 decoder layer.
