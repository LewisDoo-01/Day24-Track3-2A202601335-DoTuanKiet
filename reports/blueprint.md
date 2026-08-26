# CI/CD Blueprint: RAG Eval + Guardrail Stack

**Sinh viên:** Lewis Do  
**Ngày:** 2026-08-26

---

## Guard Stack Architecture

```
User Input
    │
    ▼ (~14.5ms P95)
[Presidio PII Scan]
    │ block if: VN_CCCD / VN_PHONE / EMAIL detected
    │ action:   return 400 + "PII detected in query"
    ▼ (~9215.0ms P95)
[NeMo Input Rail]
    │ block if: off-topic / jailbreak / prompt injection
    │ action:   return 503 + refuse message
    ▼
[RAG Pipeline (Day 18)]
    │ M1 Chunk → M2 Search → M3 Rerank → GPT-4o-mini
    ▼ (~3.5ms P95)
[NeMo Output Rail]
    │ flag if:  PII in response / sensitive content
    │ action:   replace with safe response
    ▼
User Response
```

---

## Latency Budget

*(Điền từ kết quả Task 12 — measure_p95_latency())*

| Layer | P50 (ms) | P95 (ms) | P99 (ms) | Budget |
|---|---|---|---|---|
| Presidio PII | 14.07 | 14.50 | 14.50 | <10ms |
| NeMo Input Rail | 2800.00 | 9215.00 | 9215.00 | <300ms |
| RAG Pipeline | 1200.00 | 1500.00 | 1800.00 | <2000ms |
| NeMo Output Rail | 2.83 | 3.54 | 3.54 | <300ms |
| **Total Guard** | 2816.41 | **9224.70** | 9224.70 | **<500ms** |

**Budget OK?** [ ] Yes / [x] No  
**Comment:** Thử nghiệm thực tế cho thấy lớp bảo vệ NeMo Guardrails có độ trễ lớn (~9.2 giây P95). Nguyên nhân là do NeMo phải gọi nhiều cuộc gọi API tuần tự tới OpenAI để phân loại ý định (intent) và kiểm tra an toàn. Để triển khai production, chúng ta cần tối ưu hóa bằng cách tự host mô hình ngôn ngữ nhỏ (như Llama Guard) đặt trực tiếp tại server local hoặc cache các check-intent phổ biến để giảm thiểu các cuộc gọi API.

---

## CI/CD Gates (phải pass trước khi merge to main)

```yaml
# .github/workflows/rag_eval.yml
- name: RAGAS Quality Gate
  run: python src/phase_a_ragas.py
  env:
    MIN_FAITHFULNESS: 0.75
    MIN_AVG_SCORE: 0.65

- name: Guardrail Gate
  run: pytest tests/test_phase_c.py -k "test_adversarial_suite_pass_rate"
  # phải ≥ 15/20 (75%)

- name: Latency Gate
  run: python -c "from src.phase_c_guard import measure_p95_latency; ..."
  # P95 total < 500ms
```

---

## Monitoring Dashboard (production)

| Metric | Alert Threshold | Action |
|---|---|---|
| RAGAS faithfulness (daily sample) | < 0.70 | Page on-call |
| Adversarial block rate | < 80% | Review new attack patterns |
| Guard P95 latency | > 600ms | Scale NeMo model |
| PII detected count | spike >10/hour | Security alert |

---

## Kết quả thực tế từ Lab

| | Kết quả |
|---|---|
| RAGAS avg_score (50q) | 0.675 (multi_hop: 0.645, adversarial: 0.704) |
| Worst metric | answer_relevancy |
| Dominant failure distribution | factual |
| Cohen's κ | 1.000 (Almost Perfect) |
| Adversarial pass rate | 20 / 20 (100.0%) |
| Guard P95 latency | 9224.7 ms |

---

## Nhận xét & Cải tiến

1. **Điểm hoạt động tốt:** Hệ thống Presidio PII hoạt động cực kỳ hiệu quả sau khi giới hạn quét các thực thể mục tiêu (CCCD, SĐT, Email), không bị false positive trên tiếng Việt. NeMo Guardrails đạt tỷ lệ chặn tuyệt đối 20/20 (100% pass rate).
2. **Điểm cần cải thiện:** Độ trễ (latency) của NeMo Guardrails khi sử dụng OpenAI model là rất cao. Cần chuyển sang mô hình local hoặc tối ưu hóa luồng gọi bất đồng bộ.
3. **Thay đổi khi deploy thực tế:** Triển khai một mô hình phân loại intent cực kỳ nhẹ (như FastText hoặc mô hình BERT nhỏ được fine-tune) để thay thế nhiệm vụ phân loại của LLM trong NeMo Guardrails, giúp kéo độ trễ từ giây xuống mức mili-giây.
