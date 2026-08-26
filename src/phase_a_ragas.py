from __future__ import annotations

"""Phase A: RAGAS Production Evaluation -- 50q, 3 distributions, cluster analysis."""

import json
import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH, ANSWERS_PATH

Distribution = str  # "factual" | "multi_hop" | "adversarial"

DIAGNOSTIC_TREE = {
    "faithfulness":      ("LLM hallucinating", "Tighten system prompt, lower temperature"),
    "context_recall":    ("Missing relevant chunks", "Improve chunking or add BM25"),
    "context_precision": ("Too many irrelevant chunks", "Add reranking or metadata filter"),
    "answer_relevancy":  ("Answer doesn't match question", "Improve prompt template"),
}


@dataclass
class RagasResult:
    question_id: int
    distribution: Distribution
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float

    @property
    def avg_score(self) -> float:
        return (self.faithfulness + self.answer_relevancy +
                self.context_precision + self.context_recall) / 4

    @property
    def worst_metric(self) -> str:
        scores = {
            "faithfulness":      self.faithfulness,
            "answer_relevancy":  self.answer_relevancy,
            "context_precision": self.context_precision,
            "context_recall":    self.context_recall,
        }
        return min(scores, key=scores.get)


# --- Da implement san ----------------------------------------------------

def load_test_set_50q(path: str = TEST_SET_PATH) -> list[dict]:
    """Load 50q test set voi 3 distributions."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_answers(path: str = ANSWERS_PATH) -> list[dict]:
    """Load pre-generated answers tu setup_answers.py."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"answers_50q.json khong tim thay tai {path}\n"
            "-> Chay truoc: python setup_answers.py"
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_phase_a_report(results: list[RagasResult], clusters: dict,
                         path: str = "reports/ragas_50q.json") -> None:
    """Save Phase A report to JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    per_dist: dict[str, dict] = {}
    for dist in ["factual", "multi_hop", "adversarial"]:
        subset = [r for r in results if r.distribution == dist]
        if subset:
            per_dist[dist] = {
                "count": len(subset),
                "faithfulness":      sum(r.faithfulness for r in subset) / len(subset),
                "answer_relevancy":  sum(r.answer_relevancy for r in subset) / len(subset),
                "context_precision": sum(r.context_precision for r in subset) / len(subset),
                "context_recall":    sum(r.context_recall for r in subset) / len(subset),
                "avg_score":         sum(r.avg_score for r in subset) / len(subset),
            }

    report = {
        "total_questions": len(results),
        "per_distribution": per_dist,
        "failure_clusters": clusters,
        "bottom_10": [
            {"rank": i + 1, "question_id": r.question_id, "distribution": r.distribution,
             "question": r.question, "avg_score": round(r.avg_score, 4),
             "worst_metric": r.worst_metric}
            for i, r in enumerate(sorted(results, key=lambda x: x.avg_score)[:10])
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Phase A report saved -> {path}")


# --- Tasks 1-4 -----------------------------------------------------------

def group_by_distribution(test_set: list[dict]) -> dict[str, list[dict]]:
    """Task 1: Nhom 50 cau hoi theo 3 distributions.

    Returns:
        {"factual": [...], "multi_hop": [...], "adversarial": [...]}
    """
    groups = {"factual": [], "multi_hop": [], "adversarial": []}
    for item in test_set:
        groups[item["distribution"]].append(item)
    return groups


def run_ragas_50q(answers: list[dict]) -> list[RagasResult]:
    """Task 2: Chay RAGAS 4 metrics tren toan bo 50 cau hoi.

    Steps:
        1. Extract questions, answers, contexts, ground_truths tu answers list
        2. Goi evaluate_ragas() tu m4_eval.py
        3. Ket hop ket qua voi distribution info tu answers list
        4. Return list[RagasResult]
    """
    try:
        from src.m4_eval import evaluate_ragas
    except ImportError:
        print("WARNING: Khong tim thay src/m4_eval.py -- da copy tu Day 18 chua?")
        return []

    questions     = [a["question"]     for a in answers]
    ans_texts     = [a["answer"]       for a in answers]
    contexts      = [a["contexts"]     for a in answers]
    ground_truths = [a["ground_truth"] for a in answers]

    print(f"Running RAGAS evaluation on {len(questions)} questions...")
    raw = evaluate_ragas(questions, ans_texts, contexts, ground_truths)
    per_q = raw.get("per_question", [])

    if not per_q:
        print("WARNING: RAGAS returned no per_question results.")
        return []

    results = []
    for a, pq in zip(answers, per_q):
        results.append(RagasResult(
            question_id=a["id"], distribution=a["distribution"],
            question=a["question"], answer=a["answer"],
            contexts=a["contexts"], ground_truth=a["ground_truth"],
            faithfulness=pq.faithfulness,
            answer_relevancy=pq.answer_relevancy,
            context_precision=pq.context_precision,
            context_recall=pq.context_recall,
        ))
    return results


def bottom_10(results: list[RagasResult]) -> list[dict]:
    """Task 3: Lay 10 cau hoi co avg_score thap nhat.

    Returns:
        [{"rank": 1, "question_id": ..., "distribution": ...,
          "question": ..., "avg_score": ..., "worst_metric": ...,
          "diagnosis": ..., "suggested_fix": ...}, ...]
    """
    sorted_asc = sorted(results, key=lambda r: r.avg_score)
    bottom = sorted_asc[:10]
    output = []
    for i, r in enumerate(bottom):
        diag, fix = DIAGNOSTIC_TREE[r.worst_metric]
        output.append({
            "rank": i + 1,
            "question_id": r.question_id,
            "distribution": r.distribution,
            "question": r.question,
            "avg_score": round(r.avg_score, 4),
            "worst_metric": r.worst_metric,
            "diagnosis": diag,
            "suggested_fix": fix,
        })
    return output


def cluster_analysis(results: list[RagasResult]) -> dict:
    """Task 4: Phan tich failure clusters theo (worst_metric x distribution).

    Muc tieu: tim ra distribution nao hay bi failure nhat va metric nao yeu nhat.

    Returns:
        {
          "matrix": {
            "faithfulness":      {"factual": 3, "multi_hop": 5, "adversarial": 2},
            "answer_relevancy":  {...},
            "context_precision": {...},
            "context_recall":    {...},
          },
          "dominant_failure_distribution": "multi_hop",
          "dominant_failure_metric": "context_recall",
          "insight": "..."
        }
    """
    matrix = {
        metric: {"factual": 0, "multi_hop": 0, "adversarial": 0}
        for metric in DIAGNOSTIC_TREE
    }
    for r in results:
        matrix[r.worst_metric][r.distribution] += 1

    dominant_dist   = max(["factual", "multi_hop", "adversarial"],
                          key=lambda d: sum(matrix[m][d] for m in matrix))
    dominant_metric = max(matrix, key=lambda m: sum(matrix[m].values()))
    insight = (
        f"Distribution '{dominant_dist}' co nhieu failure nhat. "
        f"Metric '{dominant_metric}' la diem yeu chu dao. "
        f"Goi y: {DIAGNOSTIC_TREE[dominant_metric][1]}"
    )

    return {
        "matrix": matrix,
        "dominant_failure_distribution": dominant_dist,
        "dominant_failure_metric": dominant_metric,
        "insight": insight,
    }


# --- Main ----------------------------------------------------------------

if __name__ == "__main__":
    import sys
    # Fix Unicode output on Windows cp1252 terminals
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    test_set = load_test_set_50q()
    print(f"Loaded {len(test_set)} questions")

    groups = group_by_distribution(test_set)
    for dist, qs in groups.items():
        print(f"  {dist}: {len(qs)} questions")

    answers = load_answers()
    results = run_ragas_50q(answers)

    if results:
        b10 = bottom_10(results)
        clusters = cluster_analysis(results)
        save_phase_a_report(results, clusters)
        print("\nBottom 10 worst questions:")
        for item in b10:
            q_preview = item['question'][:50].encode('ascii', 'replace').decode('ascii')
            print(f"  #{item['rank']} [{item['distribution']}] {q_preview}... "
                  f"avg={item['avg_score']:.3f} worst={item['worst_metric']}")
        print(f"\nDominant failure: {clusters.get('dominant_failure_distribution')} / "
              f"{clusters.get('dominant_failure_metric')}")
        print(f"Insight: {clusters.get('insight', '')[:100]}")
    else:
        print("WARNING: No results -- implement run_ragas_50q() first.")
