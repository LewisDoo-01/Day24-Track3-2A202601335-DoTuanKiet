# LLM Judge Bias Report — Phase B

**Sinh viên:** Lewis Do  
**Ngày:** 2026-08-26  
**Judge model:** gpt-4o-mini

---

## 1. Pairwise Judge Results

| # | Question (tóm tắt) | Winner | Reasoning tóm tắt |
|---|---|---|---|
| 1 | Nhân viên được nghỉ bao nhiêu ngày khi k... | B | Answer B chính xác hơn vì nó làm rõ rằng nhân viên được nghỉ có lương, điều này ... |
| 2 | Muốn mua thiết bị trị giá 55 triệu cần a... | B | Câu trả lời B đưa ra thông tin cụ thể hơn về người có thẩm quyền phê duyệt cho t... |
| 3 | Thưởng Tết tối thiểu cho nhân viên chính... | A | Câu trả lời A rõ ràng nêu rõ điều kiện về thời gian và thưởng, đáp ứng đầy đủ yê... |
| 4 | Một nhân viên Senior có 9 năm thâm niên ... | B | Câu trả lời B đưa ra thông tin đầy đủ hơn về lương của nhân viên Senior, trong k... |
| 5 | Nhân viên được tài trợ khóa học 25 triệu... | A | Câu trả lời A rõ ràng và chính xác hơn, nêu rõ hoàn trả 100% chi phí mà không gâ... |
| 6 | Nhân viên tạm ứng 8 triệu, chưa thanh to... | A | Câu trả lời A cung cấp thông tin rõ ràng về ai phê duyệt và tính toán đúng phí p... |
| 7 | Nhân viên Manager có thâm niên 12 năm: t... | B | Câu trả lời B cung cấp thông tin chính xác về số ngày phép và phụ cấp, trong khi... |
| 8 | Nhân viên được nghỉ bao nhiêu ngày phép ... | A | Câu trả lời A là chính xác hơn theo quy định chung về số ngày phép cho nhân viên... |
| 9 | Nhân viên thử việc có được nghỉ phép năm... | B | Câu trả lời B không chỉ chính xác mà còn cung cấp thêm thông tin về việc nhân vi... |
| 10 | Nhân viên Manager có thể dùng VPN cá nhâ... | B | Answer B cung cấp thông tin rõ ràng về việc sử dụng VPN cá nhân, cho thấy sự đồn... |

---

## 2. Swap-and-Average Results

| # | Pass 1 Winner | Pass 2 Winner | Final | Position Consistent? |
|---|---|---|---|---|
| 1 | B | B | B | True |
| 2 | B | B | B | True |
| 3 | A | A | A | True |
| 4 | B | B | B | True |
| 5 | A | A | A | True |
| 6 | A | A | A | True |
| 7 | B | B | B | True |
| 8 | A | B | tie | False |
| 9 | B | B | B | True |
| 10 | B | B | B | True |

**Position bias rate:** 10%

---

## 3. Cohen's κ Analysis

**Human labels:** `human_labels_10q.json` (10 câu, 5 label=1, 5 label=0)  

| Question ID | Human Label | Judge Label | Agree? |
|---|---|---|---|
| 1 | 1 | 1 | Yes |
| 5 | 0 | 0 | Yes |
| 12 | 1 | 1 | Yes |
| 21 | 1 | 1 | Yes |
| 23 | 1 | 1 | Yes |
| 29 | 0 | 0 | Yes |
| 33 | 1 | 1 | Yes |
| 41 | 0 | 0 | Yes |
| 46 | 1 | 1 | Yes |
| 50 | 0 | 0 | Yes |

**Cohen's κ:** 1.0000  
**Interpretation:** almost perfect

---

## 4. Verbosity Bias

Trong các case có winner rõ ràng (không phải tie):
- A thắng + A dài hơn B: 3 / 9 cases
- B thắng + B dài hơn A: 4 / 9 cases  
- **Verbosity bias rate:** 77%

**Kết luận:** LLM có xu hướng ưa chuộng các câu trả lời dài hơn và chi tiết hơn (verbosity bias). Điều này có thể là một vấn đề vì một câu trả lời dài đôi khi chứa thông tin thừa hoặc thậm chí là hallucination, nhưng vẫn được LLM đánh giá cao hơn câu trả lời ngắn gọn, súc tích và đúng trọng tâm.

---

## 5. Nhận xét chung

- **Độ tin cậy của LLM Judge:** Cohen's κ đạt mức 1.000 (almost perfect), cho thấy LLM và con người có mức độ đồng thuận khá cao.
- **Position bias:** Position bias rate ở mức 10%. Đây là mức chấp nhận được, nhưng phương pháp swap-and-average vẫn rất quan trọng để loại bỏ hoàn toàn position bias.
- **Hiệu quả của Swap-and-Average:** Phương pháp này giúp phát hiện ra các trường hợp LLM không nhất quán khi thay đổi thứ tự A/B, giúp đưa kết quả về 'tie' thay vì chọn sai do thiên vị vị trí.
- **Khuyến nghị trong Production:** Nên sử dụng swap-and-average khi đánh giá chất lượng câu trả lời bằng LLM trong production để giảm bias vị trí. Ngoài ra, cần thiết kế prompt để hạn chế tối đa verbosity bias.