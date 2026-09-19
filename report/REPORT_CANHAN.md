# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

**Họ tên:** Phạm Quân
**Nhóm:** G41
**Ngày:** 19/09/2026

> **Nộp 1 bản / sinh viên.** Phần nhóm (lựa chọn tài liệu, thiết kế chiến lược, bộ câu hỏi đánh giá, demo) nộp chung 1 bản trong `REPORT_NHOM.md`. Chi tiết thang điểm: `docs/SCORING.md`.

**Tổng điểm phần cá nhân: 60** = Khởi động (5) + Hướng tiếp cận (10) + Hoàn thiện code (30) + Dự đoán độ tương tự (5) + Kết quả truy xuất của tôi (10).

---

## 1. Khởi động (Warm-up) — Cá nhân (5 điểm)

### Độ tương tự Cosine (Cosine Similarity) (Bài tập 1.1)

**Độ tương tự cosine cao (High cosine similarity) nghĩa là gì?**
> Hai vector có hướng gần nhau trong không gian embedding, tức là hai văn bản được mô hình biểu diễn với ngữ nghĩa hoặc nội dung gần nhau. Cosine similarity không phụ thuộc nhiều vào độ dài văn bản vì nó tập trung vào góc giữa hai vector.

**Ví dụ có độ tương tự CAO:**
- Câu A: Sinh viên được mượn 5 tài liệu trong 21 ngày.
- Câu B: Sinh viên chính quy được mượn tối đa 5 tài liệu trong 21 ngày.
- Tại sao tương đồng: Hai câu cùng nói về hạn mức và thời hạn mượn của sinh viên.

**Ví dụ có độ tương tự THẤP:**
- Câu A: Giảng viên được gia hạn tài liệu.
- Câu B: Dịch vụ scan tài liệu tính phí theo trang.
- Tại sao khác: Một câu nói về quyền mượn của một nhóm độc giả, câu kia nói về dịch vụ cung cấp thông tin.

**Tại sao độ tương tự cosine (cosine similarity) được ưu tiên hơn khoảng cách Euclid (Euclidean distance) cho text embeddings?**
> Với text embeddings, hướng vector thường thể hiện ngữ nghĩa còn độ dài có thể bị ảnh hưởng bởi độ dài hoặc cách mã hóa văn bản. Cosine similarity so sánh hướng của hai vector và thuận tiện khi các embedding đã được chuẩn hóa.

### Bài toán tính toán Chunking (Bài tập 1.2)

**Tài liệu 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> Bước dịch cửa sổ là `500 - 50 = 450`. Số chunk là `ceil((10000 - 500) / 450) + 1 = ceil(21.111...) + 1 = 23`.
> *Đáp án: 23 chunks.*

**Nếu độ chồng chéo (overlap) tăng lên 100, số lượng chunk thay đổi thế nào? Tại sao muốn độ chồng chéo nhiều hơn?**
> Bước dịch còn `500 - 100 = 400`, nên số chunk là `ceil(9500 / 400) + 1 = 25`. Overlap lớn hơn giúp giữ ngữ cảnh ở ranh giới hai chunk, nhưng làm tăng số chunk, chi phí embedding và lượng thông tin trùng lặp.

---

## 2. Hướng tiếp cận của tôi (My Approach) — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi lập trình (implement) các phần chính trong gói `src`.

### Các hàm chia nhỏ (Chunking Functions)

**`SentenceChunker.chunk`** — hướng tiếp cận:
> Tôi dùng regex `(?<=[.!?])(?:[ \t]+|\n+)` để tách sau dấu chấm, chấm than hoặc chấm hỏi và giữ lại dấu câu trong câu trước. Các câu được strip khoảng trắng rồi gom tối đa `max_sentences_per_chunk` câu; văn bản rỗng hoặc chỉ có khoảng trắng trả về danh sách rỗng. Cách này chưa xử lý hoàn hảo chữ viết tắt và số thập phân.

**`RecursiveChunker.chunk` / `_split`** — hướng tiếp cận:
> Thuật toán thử các separator theo thứ tự ưu tiên từ đoạn lớn (`\n\n`, `\n`) đến nhỏ hơn (`. `, khoảng trắng), rồi đệ quy các mảnh còn dài hơn `chunk_size` với separator tiếp theo. Base case là mảnh đã không vượt kích thước hoặc danh sách separator đã hết; khi hết separator thì cắt theo kích thước cố định. Sau khi tách, các mảnh liền kề được gom lại nếu vẫn vừa giới hạn để tránh chunk vụn.

### Lớp EmbeddingStore

**`add_documents` + `search`** — hướng tiếp cận:
> `add_documents` tạo record gồm id, content, metadata bản sao và embedding, rồi lưu trong danh sách in-memory. `search` embedding câu hỏi, tính cosine similarity với từng record, sắp xếp giảm dần theo score và trả tối đa `top_k` kết quả; embedding thô không được đưa vào output.

**`search_with_filter` + `delete_document`** — hướng tiếp cận:
> `search_with_filter` lọc metadata trước rồi mới chạy cùng hàm similarity search, tránh việc các kết quả không phù hợp chiếm hết `top_k`. `delete_document` giữ lại các record có `metadata['doc_id']` khác id cần xóa và trả về `True` nếu kích thước store giảm.

### Tác tử KnowledgeBaseAgent

**`answer`** — hướng tiếp cận:
> `answer` lấy top-k record từ store, đánh số từng context `[1]`, kèm nguồn và nội dung. Prompt yêu cầu LLM chỉ dùng context, nói rõ khi không đủ thông tin và trích dẫn số context; nếu store không có kết quả thì trả thông báo và không gọi LLM.

---

## 3. Hoàn thiện code (Core Implementation) — Cá nhân (30 điểm)

Vượt qua bộ kiểm thử là điều kiện tính điểm phần này.

### Kết Quả Kiểm Thử (Test Results)

```
============================= 42 passed =============================
```

**Số lượng bài test vượt qua (pass):** 42 / 42

---

## 4. Dự đoán độ tương tự (Similarity Predictions) — Cá nhân (5 điểm)

| Cặp | Câu A | Câu B | Dự đoán | Điểm thực tế | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Sinh viên được mượn 5 tài liệu trong 21 ngày | Sinh viên chính quy được mượn tối đa 5 tài liệu trong 21 ngày | cao | 0.9651 | Có |
| 2 | Giảng viên được gia hạn tài liệu | Độc giả ngoài trường không được gia hạn | thấp | 0.7425 | Không hoàn toàn, cùng ngữ cảnh mượn trả |
| 3 | Tài liệu không được mượn về nhà | Từ điển và bách khoa có đóng dấu không mượn về | cao | 0.7457 | Có |
| 4 | Tra cứu OPAC | Tìm kiếm và gia hạn tài liệu | cao | 0.7712 | Có |
| 5 | Phí scan tài liệu | Phí photocopy tài liệu | cao | 0.9045 | Có, cùng nhóm dịch vụ sao chụp |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn ý nghĩa?**
> Cặp về hạn mức mượn của sinh viên có score cao nhất `0.9651`, cho thấy Gemini nhận diện tốt hai cách diễn đạt tương đương. Cặp về giảng viên và độc giả ngoài trường vẫn đạt `0.7425` vì cùng thuộc ngữ cảnh mượn trả; embedding có thể nắm chủ đề chung nhưng không thay thế metadata khi cần phân biệt đối tượng.

---

## 5. Kết quả truy xuất của tôi (Competition Results) — Cá nhân (10 điểm)

Chạy **5 câu hỏi đánh giá của nhóm** trên mã nguồn cá nhân của bạn trong gói `src`. **5 câu hỏi này phải trùng với các thành viên cùng nhóm** (xem `REPORT_NHOM.md`).

| # | Câu hỏi (Query) | Top-1 Chunk truy xuất được (tóm tắt) | Điểm Score | Có liên quan không? (Relevant) | Câu trả lời của Agent (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Sinh viên chính quy được mượn bao nhiêu tài liệu và trong bao nhiêu ngày? | Quy định mượn trả tài liệu cho sinh viên; 0.8744 | Có | Lọc `audience=student` đưa đúng chính sách lên top-1. |
| 2 | Giảng viên và nghiên cứu sinh được mượn tài liệu trong bao nhiêu ngày? | Quy định mượn trả tài liệu cho giảng viên; 0.8472 | Có | Lọc `audience=faculty` đưa đúng chính sách lên top-1. |
| 3 | Độc giả ngoài ĐHQG-HCM được mượn bao nhiêu tài liệu và có được gia hạn không? | Quy định mượn trả tài liệu cho giảng viên; 0.8508 | Có một phần | Top-1 liên quan đến chính sách mượn; cần đọc thêm tài liệu điều chỉnh để xác nhận đối tượng ngoài trường. |
| 4 | Những tài liệu nào không được mượn về nhà? | Quy định mượn trả tài liệu cho giảng viên; 0.8440 | Có | Top-3 chứa nội dung về tài liệu không được mượn về nhà. |
| 5 | Dịch vụ cung cấp thông tin theo yêu cầu có những loại phí nào? | Dịch vụ cung cấp thông tin theo yêu cầu; 0.8062 | Có | Top-3 đều thuộc tài liệu dịch vụ cung cấp thông tin. |

**Bao nhiêu câu hỏi trả về chunk có liên quan trong top-3?** 5 / 5

**Điều hay nhất tôi học được từ thành viên khác / nhóm khác (qua demo):**
> Tôi học được rằng metadata phải đi cùng nội dung: `audience=student` và `audience=faculty` giúp tách hai chính sách mượn có từ vựng gần nhau nhưng dành cho đối tượng khác nhau. Gemini đưa đúng tài liệu vào top-1 cho các câu hỏi có filter; với câu hỏi độc giả ngoài trường, cần kết hợp nhiều tài liệu chính sách để tránh trả lời thiếu điều kiện.

---

## Tự Đánh Giá (Phần Cá Nhân)

| Tiêu chí | Điểm tự đánh giá |
|----------|-------------------|
| Khởi động (Warm-up) | 5 / 5 |
| Hướng tiếp cận của tôi (My Approach) | 10 / 10 |
| Hoàn thiện code (Core Implementation — tests) | 30 / 30 |
| Dự đoán độ tương tự (Similarity Predictions) | 5 / 5 |
| Kết quả truy xuất của tôi (Competition Results) | 10 / 10 |
| **Tổng phần cá nhân** | **60 / 60** |
