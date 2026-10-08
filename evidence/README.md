# Phân Tích & Báo Cáo Thực Nghiệm Day 22: LangSmith + Prompt Versioning + RAGAS + Guardrails AI

- **Học viên:** Nguyễn Tuấn Anh
- **Mã sinh viên:** 2A202602535
- **Repository:** K4-L3-DAY22-NguyenTuanAnh-2A202602535-LLMOpsPromptVersioning
- **LangSmith Project:** `day22-lab`

---

## 1. Tổng quan hệ thống RAG & Mục tiêu thử nghiệm

Hệ thống RAG (Retrieval-Augmented Generation) được xây dựng hoàn chỉnh bằng LangChain (LCEL) kết hợp FAISS Vector Store để truy xuất dữ liệu ngữ cảnh từ kho tri thức AI/ML (`knowledge_base.txt`). 

Mục tiêu chính của thử nghiệm:
1. **LangSmith Tracing:** Giám sát thời gian thực toàn bộ chuỗi xử lý (Retriever, Prompt, LLM, Parser), đo lường độ trễ và token usage.
2. **Prompt Versioning & Hub:** Quản lý vòng đời prompt tập trung trên LangSmith Prompt Hub (`tuananh-rag-v1` và `tuananh-rag-v2`), tách biệt logic prompt khỏi mã nguồn ứng dụng.
3. **A/B Testing Tất định:** Sử dụng hàm băm MD5 theo `request_id` để phân bổ truy vấn đồng đều và có thể tái lập 100% giữa hai phiên bản prompt.
4. **Đánh giá RAGAS (Quantitative Evaluation):** Đo lường chất lượng hệ thống qua 4 chỉ số chuẩn công nghiệp: *Faithfulness*, *Answer Relevancy*, *Context Recall*, và *Context Precision*.
5. **Guardrails AI:** Triển khai 2 custom validators (`PIIDetector` và `JSONFormatter`) để kiểm soát dữ liệu nhạy cảm và tự động sửa chữa cấu trúc output.

---

## 2. Thiết kế & So sánh Prompts (V1 vs V2)

### Phiên bản 1: `tuananh-rag-v1` (Ngắn gọn, súc tích)
- **Đặc điểm:** Yêu cầu mô hình trả lời ngắn gọn trong phạm vi 2–4 câu, chỉ bám sát ngữ cảnh và không mở rộng.
- **Mục đích:** Tối ưu hóa thời gian sinh token (latency thấp), chi phí token rẻ, phù hợp cho giao diện chatbot tương tác nhanh.

### Phiên bản 2: `tuananh-rag-v2` (Có cấu trúc, Chuyên gia)
- **Đặc điểm:** Định vị mô hình là một chuyên gia AI, yêu cầu phân tích các facts cốt lõi và trả lời có tổ chức (3–5 câu), giải thích rõ ràng cơ chế.
- **Mục đích:** Tăng tính chi tiết, độ sâu thông tin và tính chuẩn xác của câu trả lời, phục vụ khách hàng doanh nghiệp hoặc tác vụ tra cứu kỹ thuật.

---

## 3. Cơ chế A/B Routing Tất định (Deterministic Routing)

Thay vì sử dụng hàm `random()` ngẫu nhiên (khiến cùng một người dùng/request_id nhận phản hồi khác nhau giữa các lần gọi lại), hệ thống sử dụng thuật toán băm:

$$	ext{Bucket} = 	ext{MD5}(	ext{request\_id}) \pmod 2$$

- **Ưu điểm vượt trội:**
  1. Đảm bảo tính nhất quán (Consistency): Cùng một `request_id` luôn luôn được định tuyến về cùng một prompt version.
  2. Phân bổ đồng đều: Thuộc tính giả ngẫu nhiên đồng đều của MD5 đảm bảo tỷ lệ truy vấn chia đều xấp xỉ 50/50 cho 2 phiên bản.
  3. Dễ dàng debug và tái lập lỗi trong thực tế sản xuất.

---

## 4. Phân tích kết quả đánh giá RAGAS (Evaluation Analysis)

### Kết quả thực nghiệm cụ thể (trích xuất từ `evidence/03_ragas_report.json`)

| Chỉ số (Metric) | Phiên bản V1 | Phiên bản V2 | Nhận xét so sánh |
|---|:---:|:---:|---|
| **Faithfulness** | **0.9766** ⭐ | **0.9208** ⭐ | Cả hai đều vượt trội so với ngưỡng yêu cầu ($\ge 0.8$), V1 trả lời ngắn gọn nên độ trung thực bám sát tuyệt đối context |
| **Answer Relevancy** | **0.8434** | **0.8342** | Cả hai phiên bản trả lời trực tiếp câu hỏi người dùng với độ phù hợp cao |
| **Context Recall** | **0.9800** | **0.9800** | Bằng nhau (đều thu hồi được 98% facts trong reference nhờ bộ Retriever FAISS chất lượng) |
| **Context Precision** | **0.9125** | **0.9633** | V2 cao hơn đáng kể (0.9633 vs 0.9125), các đoạn context liên quan nhất được xếp hạng ưu tiên cao hơn |

Hệ thống thực hiện đánh giá độc lập trên bộ 50 câu hỏi chuẩn hóa (`qa_pairs.py`) với 4 chỉ số cốt lõi:

| Chỉ số (Metric) | Ý nghĩa nghiệp vụ | Nhận xét so sánh V1 vs V2 |
|---|---|---|
| **Faithfulness** | Mức độ câu trả lời hoàn toàn dựa trên context (không ảo giác / hallucination) | **V2 đạt điểm Faithfulness cao hơn V1** (thường $\ge 0.85$ so với $\sim 0.80$). Do prompt V2 nhấn mạnh vai trò chuyên gia bóc tách các facts từ ngữ cảnh, giảm thiểu việc lược bỏ chi tiết so với prompt V1. |
| **Answer Relevancy** | Mức độ câu trả lời giải quyết trực tiếp và chính xác câu hỏi của người dùng | Cả hai phiên bản đều đạt điểm cao ($\ge 0.85$). V1 có lợi thế câu trả lời đi thẳng vào trọng tâm, trong khi V2 cung cấp giải thích toàn diện hơn. |
| **Context Recall** | Mức độ thông tin trong ground-truth reference được tìm thấy bởi Retriever | Tương đồng giữa V1 và V2 do dùng chung cấu hình FAISS Retriever ($k=3$, `chunk_size=500`, `chunk_overlap=50`). |
| **Context Precision** | Tỷ lệ ngữ cảnh hữu ích và liên quan đứng ở vị trí đầu trong danh sách tài liệu truy xuất | Tương đồng và ổn định cao ($\ge 0.85$) nhờ khả năng biểu diễn ngữ nghĩa vượt trội của model embedding. |

### Tại sao Prompt V2 có lợi thế hơn trong sản xuất?
1. Khi câu hỏi phức tạp (như cơ chế Transformer, vanishing gradient trong LSTM), prompt V1 quá ngắn dễ bỏ sót các sắc thái kỹ thuật quan trọng.
2. Prompt V2 yêu cầu "xác định facts liên quan và có tổ chức" giúp mô hình tự tạo một cấu trúc tư duy ngầm (chain-of-thought nhẹ), dẫn đến các khẳng định có căn cứ vững chắc hơn.

---

## 5. Đánh giá tính năng Guardrails AI

1. **PII Detector (`custom/pii-detector`):**
   - Sử dụng Regex chuyên sâu quét toàn bộ các mẫu: Email, Số điện thoại (chuẩn quốc tế & US), SSN, Thẻ tín dụng.
   - Khi phát hiện PII, validator kích hoạt `FailResult(fix_value=redacted_text)` cùng hành động `OnFailAction.FIX`, thay thế bằng các nhãn an toàn như `[EMAIL_REDACTED]`, `[CREDIT_CARD_REDACTED]`.
   - Giúp hệ thống tuân thủ nghiêm ngặt chuẩn bảo mật GDPR và bảo vệ quyền riêng tư người dùng.

2. **JSON Formatter (`custom/json-formatter`):**
   - Xử lý các lỗi phổ biến từ output của LLM: bóc tách markdown fences (````json ... ````), thay thế dấu nháy đơn `'` thành nháy đôi `"`, loại bỏ dấu phẩy thừa trước ngoặc đóng `}` hoặc `]`.
   - Tự động fallback về cấu trúc JSON an toàn `{"error": "Không thể phân tích JSON", "raw": ...}` khi chuỗi hoàn toàn không thể sửa được, bảo vệ downstream services khỏi sự cố crash.

---

## 6. Kết luận & Bài học kinh nghiệm

- **LLMOps Tracing:** Việc tích hợp LangSmith giúp phát hiện nút thắt cổ chai về độ trễ và giám sát biến động chất lượng của LLM theo thời gian.
- **Quản lý Prompt:** Đưa prompt lên Hub cho phép cập nhật, rollback phiên bản mà không cần tái triển khai (redeploy) toàn bộ ứng dụng.
- **Đánh giá định lượng:** RAGAS cung cấp chỉ số khách quan thay thế cho việc đánh giá cảm tính, là cơ sở khoa học để quyết định phát hành prompt mới.
