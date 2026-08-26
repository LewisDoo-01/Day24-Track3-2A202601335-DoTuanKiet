from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    try:
        from ragas import evaluate
        from ragas.metrics._faithfulness import Faithfulness
        from ragas.metrics._answer_relevance import AnswerRelevancy
        from ragas.metrics._context_precision import LLMContextPrecisionWithReference
        from ragas.metrics._context_recall import LLMContextRecall
        from ragas.llms.base import LangchainLLMWrapper
        from ragas.embeddings.base import LangchainEmbeddingsWrapper
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from datasets import Dataset

        lc_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        lc_emb = OpenAIEmbeddings(model="text-embedding-3-small")
        ragas_llm = LangchainLLMWrapper(lc_llm)
        ragas_emb = LangchainEmbeddingsWrapper(lc_emb)

        dataset = Dataset.from_dict({
            "question": questions, "answer": answers,
            "contexts": contexts, "ground_truth": ground_truths,
        })
        metrics = [
            Faithfulness(),
            AnswerRelevancy(),
            LLMContextPrecisionWithReference(),
            LLMContextRecall(),
        ]
        result = evaluate(dataset, metrics=metrics, llm=ragas_llm, embeddings=ragas_emb)
        df = result.to_pandas()

        # column names in ragas 0.4.x may differ from 0.1.x
        col_map = {
            "faithfulness": "faithfulness",
            "answer_relevancy": "answer_relevancy",
            "llm_context_precision_with_reference": "context_precision",
            "llm_context_recall": "context_recall",
        }
        per_question = []
        for i, (_, row) in enumerate(df.iterrows()):
            faith = float(row.get("faithfulness", 0.0))
            ar    = float(row.get("answer_relevancy", 0.0))
            cp    = float(row.get("llm_context_precision_with_reference",
                          row.get("context_precision", 0.0)))
            cr    = float(row.get("llm_context_recall",
                          row.get("context_recall", 0.0)))
            per_question.append(EvalResult(
                question=questions[i], answer=answers[i],
                contexts=contexts[i], ground_truth=ground_truths[i],
                faithfulness=faith, answer_relevancy=ar,
                context_precision=cp, context_recall=cr,
            ))

        def _avg(key):
            vals = [getattr(r, key) for r in per_question]
            return sum(vals) / len(vals) if vals else 0.0

        return {"faithfulness": _avg("faithfulness"), "answer_relevancy": _avg("answer_relevancy"),
                "context_precision": _avg("context_precision"), "context_recall": _avg("context_recall"),
                "per_question": per_question}
    except Exception as e:
        print(f"  [WARNING] RAGAS evaluation failed: {e}")
        return {"faithfulness": 0.0, "answer_relevancy": 0.0,
                "context_precision": 0.0, "context_recall": 0.0, "per_question": []}


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    diagnostic_tree = {
        "faithfulness": ("LLM hallucinating", "Tighten prompt, lower temperature"),
        "context_recall": ("Missing relevant chunks", "Improve chunking or add BM25"),
        "context_precision": ("Too many irrelevant chunks", "Add reranking or metadata filter"),
        "answer_relevancy": ("Answer doesn't match question", "Improve prompt template"),
    }

    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    scored = []
    for r in eval_results:
        values = {m: getattr(r, m) for m in metrics}
        avg = sum(values.values()) / len(values)
        worst_metric = min(values, key=values.get)
        scored.append((avg, r, worst_metric))

    scored.sort(key=lambda x: x[0])

    failures = []
    for avg, r, worst_metric in scored[:bottom_n]:
        diagnosis, suggested_fix = diagnostic_tree[worst_metric]
        failures.append({
            "question": r.question,
            "worst_metric": worst_metric,
            "score": avg,
            "diagnosis": diagnosis,
            "suggested_fix": suggested_fix,
        })
    return failures


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
