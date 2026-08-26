# Failure Cluster Analysis — Phase A

**Sinh viên:** Lewis Do  
**Ngày:** 2026-08-26

---

## 1. Aggregate RAGAS Scores theo Distribution

| Metric | factual | multi_hop | adversarial |
|---|---|---|---|
| faithfulness | 0.520 | 0.501 | 0.800 |
| answer_relevancy | 0.450 | 0.389 | 0.407 |
| context_precision | 0.920 | 0.950 | 0.958 |
| context_recall | 0.700 | 0.742 | 0.650 |
| **avg_score** | **0.648** | **0.645** | **0.704** |

*Ghi chú: Điểm factual được ước tính dựa trên các câu hoàn thành thành công trong bối cảnh có một số lỗi timeout mạng.*

---

## 2. Bottom 10 Questions

| Rank | Distribution | Question | avg_score | worst_metric |
|---|---|---|---|---|
| 1 | multi_hop | So sánh yêu cầu mật khẩu giữa policy v1.0 và v2.0... | 0.000 | faithfulness |
| 2 | multi_hop | Nhân viên Manager có thâm niên 12 năm: tổng phụ cấp và ngày phép... | 0.375 | faithfulness |
| 3 | adversarial | Bao lâu phải đổi mật khẩu một lần? | 0.396 | faithfulness |
| 4 | adversarial | Nhân viên Manager có thể dùng VPN cá nhân (như NordVPN) khi WFH không? | 0.417 | faithfulness |
| 5 | factual | Muốn mua thiết bị trị giá 55 triệu cần ai phê duyệt? | 0.458 | answer_relevancy |
| 6 | factual | Nam nhân viên được nghỉ bao nhiêu ngày khi vợ sinh con? | 0.500 | faithfulness |
| 7 | multi_hop | Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu? | 0.505 | faithfulness |
| 8 | factual | Nghỉ phép không lương 20 ngày cần ai phê duyệt? | 0.536 | answer_relevancy |
| 9 | multi_hop | Một nhân viên Senior có 9 năm thâm niên được nghỉ phép và lương khoảng nào? | 0.542 | answer_relevancy |
| 10 | multi_hop | Nhân viên Junior P1 có lương cơ bản 12 triệu thử việc nhận lương và phụ cấp gì? | 0.588 | faithfulness |

---

## 3. Failure Cluster Matrix

*(Mỗi ô = số câu có worst_metric = row, thuộc distribution = col)*

| worst_metric | factual | multi_hop | adversarial | Total |
|---|---|---|---|---|
| faithfulness | 5 | 10 | 2 | 17 |
| context_recall | 2 | 0 | 2 | 4 |
| context_precision | 2 | 0 | 0 | 2 |
| answer_relevancy | 11 | 10 | 6 | 27 |

---

## 4. Dominant Failure Analysis

**Dominant distribution:** factual  
**Dominant metric:** answer_relevancy  

**Lý do phân tích:**

1. Nhóm câu hỏi `factual` là nhóm câu hỏi truy xuất trực tiếp nhưng do corpus chứa nhiều tài liệu trùng lặp hoặc chứa nhiều phiên bản cũ/mới (v2023 vs v2024), LLM dễ bị nhầm lẫn và đưa ra câu trả lời không khớp với câu hỏi hoặc lấy thông tin từ phiên bản cũ dẫn đến giảm điểm `answer_relevancy`.
2. Metric `answer_relevancy` thấp do prompt template của RAG pipeline chưa được tối ưu hóa tốt để bắt buộc LLM chỉ trả lời đúng trọng tâm câu hỏi mà có xu hướng thêm thắt các chi tiết bên lề hoặc trả lời quá dài dòng không cần thiết.

---

## 5. Suggested Fixes

| Metric yếu | Root cause | Suggested fix |
|---|---|---|
| faithfulness | LLM hallucinating | Tinh chỉnh prompt, hạ nhiệt độ (temperature = 0) của LLM judge/generator. |
| context_recall | Missing relevant chunks | Cải thiện thuật toán chunking (như dùng hierarchical chunking) hoặc kết hợp tìm kiếm BM25. |
| context_precision | Too many irrelevant chunks | Sử dụng mô hình Rerank (Cross-Encoder) để loại bỏ các chunk không liên quan trước khi đưa vào LLM. |
| answer_relevancy | Answer doesn't match question | Cải tiến prompt template của generator, hướng dẫn rõ ràng yêu cầu trả lời ngắn gọn, trực diện. |

---

## 6. Nhận xét về Adversarial Distribution

- Điểm của `adversarial` (0.704) cao hơn một chút so với `multi_hop` (0.645) và `factual` (0.648). Điều này cho thấy pipeline RAG đã bắt đầu phân biệt được các bẫy phiên bản cũ/mới nhờ Cross-Encoder Reranker hoạt động hiệu quả trên các tài liệu mới v2024.
- Tuy nhiên, trong bottom 10 vẫn xuất hiện các câu hỏi `adversarial` (như câu 3 và câu 4) liên quan đến chu kỳ đổi mật khẩu và sử dụng VPN cá nhân. Điều này xảy ra do LLM vẫn bị ảnh hưởng bởi thông tin cũ chưa được loại bỏ triệt để khỏi cơ sở dữ liệu tìm kiếm, hoặc do các chính sách bảo mật thông tin có độ tương đồng cao trong cơ sở dữ liệu vector.
