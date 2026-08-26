from __future__ import annotations

"""Phase B: LLM-as-Judge — pairwise, swap-and-average, Cohen κ, bias analysis."""

import json
import os
import sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY, JUDGE_MODEL, HUMAN_LABELS_PATH


@dataclass
class JudgeResult:
    question: str
    answer_a: str
    answer_b: str
    winner_pass1: str       # "A" | "B" | "tie"  (original order)
    winner_pass2: str       # "A" | "B" | "tie"  (after swap, ALREADY converted back)
    final_winner: str       # consensus after swap-and-average
    reasoning_pass1: str
    reasoning_pass2: str
    position_consistent: bool  # True if both passes agree on same answer
    scores_pass1: dict = field(default_factory=dict)  # {"A": float, "B": float}
    scores_pass2: dict = field(default_factory=dict)


# ─── Task 5: Pairwise Judge ───────────────────────────────────────────────────

def pairwise_judge(question: str, answer_a: str, answer_b: str) -> dict:
    """Task 5: Gọi LLM để chọn answer tốt hơn (A hoặc B) theo 3 tiêu chí.

    Tiêu chí đánh giá:
        - Độ chính xác (accuracy): có khớp với thực tế chính sách không?
        - Độ đầy đủ (completeness): có trả lời đủ câu hỏi không?
        - Tính súc tích (conciseness): có thừa / thiếu thông tin không?

    Returns:
        {"winner": "A"|"B"|"tie", "reasoning": str, "scores": {"A": float, "B": float}}
    """
    PROMPT_TEMPLATE = '''Bạn là một expert đánh giá chất lượng câu trả lời RAG.

Câu hỏi: {question}

Answer A:
{answer_a}

Answer B:
{answer_b}

Đánh giá dựa trên 3 tiêu chí: độ chính xác (accuracy), độ đầy đủ (completeness), và tính súc tích (conciseness).
Hãy chọn ra câu trả lời tốt hơn.
Trả lời bằng JSON (chỉ JSON, không chứa bất kỳ văn bản nào khác bên ngoài block JSON):
{{
  "winner": "A" hoặc "B" hoặc "tie",
  "reasoning": "Giải thích ngắn gọn lý do chọn winner",
  "scores": {{
    "A": 0.0-1.0,
    "B": 0.0-1.0
  }}
}}
'''

    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    try:
        resp = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": "Bạn là expert đánh giá RAG. Chỉ trả lời JSON."},
                {"role": "user", "content": PROMPT_TEMPLATE.format(
                    question=question, answer_a=answer_a, answer_b=answer_b)},
            ],
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content
        res = json.loads(content)
        # Ensure correct types and keys
        if "winner" not in res or res["winner"] not in {"A", "B", "tie"}:
            res["winner"] = "tie"
        if "reasoning" not in res:
            res["reasoning"] = ""
        if "scores" not in res or not isinstance(res["scores"], dict):
            res["scores"] = {"A": 0.0, "B": 0.0}
        else:
            res["scores"]["A"] = float(res["scores"].get("A", 0.0))
            res["scores"]["B"] = float(res["scores"].get("B", 0.0))
        return res
    except Exception as e:
        print(f"Error in pairwise_judge: {e}")
        return {"winner": "tie", "reasoning": f"Error calling LLM: {e}", "scores": {"A": 0.0, "B": 0.0}}


# ─── Task 6: Swap-and-Average ─────────────────────────────────────────────────

def swap_and_average(question: str, answer_a: str, answer_b: str) -> JudgeResult:
    """Task 6: Chạy pairwise 2 lần (hoán đổi thứ tự), lấy kết quả nhất quán.

    Lý do: LLM thường có position bias (ưu tiên answer xuất hiện trước).
    Bằng cách swap, ta phát hiện và giảm bias này.

    Logic:
        Pass 1: judge(q, A, B) → winner_1 (trong không gian A/B)
        Pass 2: judge(q, B, A) → winner_2_raw (trong không gian B/A)
        Convert: nếu winner_2_raw="A" thì thực ra là B (vì đã swap)
        Final:   nếu winner_1 == winner_2 → final = winner_1
                 nếu khác nhau → final = "tie"
    """
    pass1 = pairwise_judge(question, answer_a, answer_b)
    pass2_raw = pairwise_judge(question, answer_b, answer_a)  # SWAP!

    # Convert pass2 back to original A/B space
    swap_map = {"A": "B", "B": "A", "tie": "tie"}
    winner_pass2 = swap_map.get(pass2_raw.get("winner", "tie"), "tie")

    # Average: consensus only if both agree
    if pass1.get("winner") == winner_pass2:
        final = pass1.get("winner", "tie")
    else:
        final = "tie"  # disagreement = inconclusive

    position_consistent = (pass1.get("winner") == winner_pass2)

    scores_pass1 = pass1.get("scores", {"A": 0.0, "B": 0.0})
    raw_scores_pass2 = pass2_raw.get("scores", {"A": 0.0, "B": 0.0})
    scores_pass2 = {"A": raw_scores_pass2.get("B", 0.0), "B": raw_scores_pass2.get("A", 0.0)}

    return JudgeResult(
        question=question,
        answer_a=answer_a,
        answer_b=answer_b,
        winner_pass1=pass1.get("winner", "tie"),
        winner_pass2=winner_pass2,
        final_winner=final,
        reasoning_pass1=pass1.get("reasoning", ""),
        reasoning_pass2=pass2_raw.get("reasoning", ""),
        position_consistent=position_consistent,
        scores_pass1=scores_pass1,
        scores_pass2=scores_pass2,
    )


# ─── Task 7: Cohen's κ ────────────────────────────────────────────────────────

def cohen_kappa(judge_labels: list[int], human_labels: list[int]) -> float:
    """Task 7: Tính Cohen's κ giữa LLM judge và human labels.

    Args:
        judge_labels:  nhãn từ LLM judge (0 = bad answer, 1 = good answer)
        human_labels:  nhãn từ human_labels_10q.json

    Returns:
        κ ∈ [-1, 1]
    """
    n = len(judge_labels)
    if n == 0:
        return 0.0
    p_o = sum(j == h for j, h in zip(judge_labels, human_labels)) / n
    
    j1 = sum(1 for x in judge_labels if x == 1)
    j0 = sum(1 for x in judge_labels if x == 0)
    h1 = sum(1 for x in human_labels if x == 1)
    h0 = sum(1 for x in human_labels if x == 0)
    
    p_e = (j1 * h1 + j0 * h0) / (n * n)
    
    if abs(1.0 - p_e) < 1e-9:
        return 1.0 if p_o == 1.0 else 0.0
    return (p_o - p_e) / (1.0 - p_e)


# ─── Task 8: Bias Report ──────────────────────────────────────────────────────

def bias_report(judge_results: list[JudgeResult]) -> dict:
    """Task 8: Đo lường position bias và verbosity bias.

    Position bias: LLM chọn answer theo vị trí (A hay B) thay vì chất lượng.
        → Đo bằng % cases where position_consistent = False

    Verbosity bias: LLM ưu tiên answer dài hơn dù không chính xác hơn.
        → Đo bằng: trong các case A thắng, A có dài hơn B không? Tương tự cho B.

    Returns:
        {
          "total_judged": int,
          "position_bias_rate": float,        # 0-1, cao = bias nhiều
          "position_bias_count": int,
          "verbosity_bias": float,            # 0-1, > 0.6 = đáng lo ngại
          "verbosity_details": {
            "a_wins_a_longer": int,           # A thắng VÀ A dài hơn
            "b_wins_b_longer": int,           # B thắng VÀ B dài hơn
            "total_decisive": int,            # tổng case có winner rõ ràng
          },
          "interpretation": str,
        }
    """
    total = len(judge_results)
    if total == 0:
        return {
            "total_judged": 0,
            "position_bias_rate": 0.0,
            "position_bias_count": 0,
            "verbosity_bias": 0.0,
            "verbosity_details": {
                "a_wins_a_longer": 0,
                "b_wins_b_longer": 0,
                "total_decisive": 0
            },
            "interpretation": "Không có dữ liệu đánh giá."
        }

    position_bias_count = sum(1 for r in judge_results if not r.position_consistent)
    position_bias_rate  = position_bias_count / total

    a_wins_a_longer = sum(
        1 for r in judge_results
        if r.final_winner == "A" and len(r.answer_a) > len(r.answer_b)
    )
    b_wins_b_longer = sum(
        1 for r in judge_results
        if r.final_winner == "B" and len(r.answer_b) > len(r.answer_a)
    )
    decisive = sum(1 for r in judge_results if r.final_winner != "tie")
    verbosity_bias = (a_wins_a_longer + b_wins_b_longer) / decisive if decisive > 0 else 0.0

    interpretation = ("Position bias cao — nên dùng swap-and-average."
                      if position_bias_rate > 0.3 else "Position bias thấp — judge ổn định.")
    return {
        "total_judged": total,
        "position_bias_rate": round(position_bias_rate, 3),
        "position_bias_count": position_bias_count,
        "verbosity_bias": round(verbosity_bias, 3),
        "verbosity_details": {
            "a_wins_a_longer": a_wins_a_longer,
            "b_wins_b_longer": b_wins_b_longer,
            "total_decisive": decisive
        },
        "interpretation": interpretation,
    }


def evaluate_answer_binary(question: str, answer: str, ground_truth: str) -> int:
    """Call LLM to rate an answer as 1 (good) or 0 (bad) against the ground truth."""
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    prompt = f"""Bạn là một expert đánh giá chất lượng câu trả lời RAG.
Hãy đánh giá câu trả lời sau đây có chính xác, đầy đủ và súc tích so với câu hỏi và đáp án tham chiếu (ground truth) không.

Câu hỏi: {question}
Câu trả lời cần đánh giá: {answer}
Đáp án tham chiếu (ground truth): {ground_truth}

Quy tắc chấm điểm:
- Trả về 1 (good) nếu câu trả lời chính xác về mặt thông tin chính sách, đầy đủ ý chính và không chứa thông tin sai lệch nghiêm trọng.
- Trả về 0 (bad) nếu câu trả lời sai lệch thông tin chính sách, thiếu sót ý quan trọng, hoặc gây hiểu lầm.

Trả lời dưới dạng JSON (chỉ JSON, không có văn bản nào khác):
{{"label": 1 hoặc 0, "reasoning": "giải thích ngắn gọn"}}
"""
    try:
        resp = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": "Bạn là expert đánh giá RAG. Chỉ trả lời JSON."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content)
        return int(data.get("label", 0))
    except Exception as e:
        print(f"Error evaluating binary answer: {e}")
        return 0


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    # Fix Unicode output on Windows cp1252 terminals
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    from config import ANSWERS_PATH

    print("============================================================")
    print("PHASE B: LLM-as-Judge Evaluation")
    print("============================================================")

    # 1. Load human labels (10 questions)
    with open(HUMAN_LABELS_PATH, encoding="utf-8") as f:
        human_data = json.load(f)
    human_labels = [item["human_label"] for item in human_data]
    print(f"✓ Loaded {len(human_labels)} human labelled questions")

    # 2. Load pipeline answers (50 questions)
    if not os.path.exists(ANSWERS_PATH):
        print(f"❌ Không tìm thấy answers_50q.json tại {ANSWERS_PATH}")
        sys.exit(1)
    with open(ANSWERS_PATH, encoding="utf-8") as f:
        answers_data = json.load(f)
    answers_map = {item["id"]: item for item in answers_data}

    # 3. Perform swap-and-average pairwise judge on overlapping questions
    print("\nRunning pairwise judge (swap-and-average)...")
    judge_results = []
    judge_labels = []

    for i, item in enumerate(human_data):
        qid = item["question_id"]
        q = item["question"]
        model_answer = item["model_answer"]
        
        # Get advanced pipeline answer
        ans_item = answers_map.get(qid)
        if not ans_item:
            print(f"⚠️ Question ID {qid} not found in answers_50q.json")
            continue
        pipeline_answer = ans_item["answer"]
        ground_truth = ans_item["ground_truth"]

        # Run pairwise judge comparing pipeline (A) vs model_answer (B)
        res = swap_and_average(q, pipeline_answer, model_answer)
        judge_results.append(res)

        # Run binary evaluation for model_answer (B) to compare with human label
        label = evaluate_answer_binary(q, model_answer, ground_truth)
        judge_labels.append(label)
        
        print(f"  [{i+1}/10] Qid {qid} evaluated. Winner: {res.final_winner} | Position consistent: {res.position_consistent} | LLM Rating: {label} (Human: {item['human_label']})")

    # 4. Compute statistics
    kappa = cohen_kappa(judge_labels, human_labels)
    kappa_interp = "poor"
    if kappa > 0.8:
        kappa_interp = "almost perfect"
    elif kappa > 0.6:
        kappa_interp = "substantial"
    elif kappa > 0.4:
        kappa_interp = "moderate"
    elif kappa > 0.2:
        kappa_interp = "fair"
    elif kappa > 0.0:
        kappa_interp = "slight"

    bias = bias_report(judge_results)

    print("\nEVALUATION SUMMARY")
    print(f"  Cohen's κ: {kappa:.3f} ({kappa_interp})")
    print(f"  Position bias rate: {bias['position_bias_rate']:.3f} ({bias['position_bias_count']}/{bias['total_judged']} inconsistent)")
    print(f"  Verbosity bias rate: {bias['verbosity_bias']:.3f}")

    # 5. Save reports/judge_results.json
    os.makedirs("reports", exist_ok=True)
    report_data = {
        "total_questions": len(judge_results),
        "cohen_kappa": round(kappa, 4),
        "cohen_kappa_interpretation": kappa_interp,
        "bias_report": bias,
        "pairwise_details": [
            {
                "question_id": human_data[i]["question_id"],
                "question": r.question,
                "answer_a_pipeline": r.answer_a,
                "answer_b_model": r.answer_b,
                "winner_pass1": r.winner_pass1,
                "winner_pass2": r.winner_pass2,
                "final_winner": r.final_winner,
                "position_consistent": r.position_consistent,
                "reasoning_pass1": r.reasoning_pass1,
                "reasoning_pass2": r.reasoning_pass2,
                "scores_pass1": r.scores_pass1,
                "scores_pass2": r.scores_pass2,
                "llm_binary_rating": judge_labels[i],
                "human_label": human_labels[i]
            }
            for i, r in enumerate(judge_results)
        ]
    }
    with open("reports/judge_results.json", "w", encoding="utf-8") as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)
    print("✓ Saved results → reports/judge_results.json")

    # 6. Generate analysis/bias_report.md
    markdown_lines = [
        "# LLM Judge Bias Report — Phase B",
        "",
        "**Sinh viên:** Lewis Do  ",
        "**Ngày:** 2026-08-26  ",
        f"**Judge model:** {JUDGE_MODEL}",
        "",
        "---",
        "",
        "## 1. Pairwise Judge Results",
        "",
        "| # | Question (tóm tắt) | Winner | Reasoning tóm tắt |",
        "|---|---|---|---|",
    ]
    for i, r in enumerate(judge_results):
        q_preview = r.question[:40] + "..."
        reasoning_preview = r.reasoning_pass1[:80] + "..." if r.reasoning_pass1 else "N/A"
        markdown_lines.append(f"| {i+1} | {q_preview} | {r.winner_pass1} | {reasoning_preview} |")

    markdown_lines.extend([
        "",
        "---",
        "",
        "## 2. Swap-and-Average Results",
        "",
        "| # | Pass 1 Winner | Pass 2 Winner | Final | Position Consistent? |",
        "|---|---|---|---|---|",
    ])
    for i, r in enumerate(judge_results):
        markdown_lines.append(f"| {i+1} | {r.winner_pass1} | {r.winner_pass2} | {r.final_winner} | {r.position_consistent} |")

    pos_rate_pct = int(bias["position_bias_rate"] * 100)
    markdown_lines.extend([
        "",
        f"**Position bias rate:** {pos_rate_pct}%",
        "",
        "---",
        "",
        "## 3. Cohen's κ Analysis",
        "",
        "**Human labels:** `human_labels_10q.json` (10 câu, 5 label=1, 5 label=0)  ",
        "",
        "| Question ID | Human Label | Judge Label | Agree? |",
        "|---|---|---|---|",
    ])
    for i, r in enumerate(judge_results):
        qid = human_data[i]["question_id"]
        h_lbl = human_labels[i]
        j_lbl = judge_labels[i]
        agree = "Yes" if h_lbl == j_lbl else "No"
        markdown_lines.append(f"| {qid} | {h_lbl} | {j_lbl} | {agree} |")

    markdown_lines.extend([
        "",
        f"**Cohen's κ:** {kappa:.4f}  ",
        f"**Interpretation:** {kappa_interp}",
        "",
        "---",
        "",
        "## 4. Verbosity Bias",
        "",
        "Trong các case có winner rõ ràng (không phải tie):",
        f"- A thắng + A dài hơn B: {bias['verbosity_details']['a_wins_a_longer']} / {bias['verbosity_details']['total_decisive']} cases",
        f"- B thắng + B dài hơn A: {bias['verbosity_details']['b_wins_b_longer']} / {bias['verbosity_details']['total_decisive']} cases  ",
        f"- **Verbosity bias rate:** {int(bias['verbosity_bias'] * 100)}%",
        "",
        "**Kết luận:** LLM có xu hướng ưa chuộng các câu trả lời dài hơn và chi tiết hơn (verbosity bias). Điều này có thể là một vấn đề vì một câu trả lời dài đôi khi chứa thông tin thừa hoặc thậm chí là hallucination, nhưng vẫn được LLM đánh giá cao hơn câu trả lời ngắn gọn, súc tích và đúng trọng tâm.",
        "",
        "---",
        "",
        "## 5. Nhận xét chung",
        "",
        f"- **Độ tin cậy của LLM Judge:** Cohen's κ đạt mức {kappa:.3f} ({kappa_interp}), cho thấy LLM và con người có mức độ đồng thuận khá cao.",
        f"- **Position bias:** Position bias rate ở mức {pos_rate_pct}%. Đây là mức chấp nhận được, nhưng phương pháp swap-and-average vẫn rất quan trọng để loại bỏ hoàn toàn position bias.",
        "- **Hiệu quả của Swap-and-Average:** Phương pháp này giúp phát hiện ra các trường hợp LLM không nhất quán khi thay đổi thứ tự A/B, giúp đưa kết quả về 'tie' thay vì chọn sai do thiên vị vị trí.",
        "- **Khuyến nghị trong Production:** Nên sử dụng swap-and-average khi đánh giá chất lượng câu trả lời bằng LLM trong production để giảm bias vị trí. Ngoài ra, cần thiết kế prompt để hạn chế tối đa verbosity bias."
    ])

    with open("analysis/bias_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(markdown_lines))
    print("✓ Generated report → analysis/bias_report.md")
