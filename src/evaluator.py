"""
Rigorous evaluation harness for LegalBench-RAG and contract abstention.
Computes:
1. Character-level Precision, Recall, and F1 at k (k=1, 3, 5).
2. Document and clause hit rates at k.
3. Abstention metrics: Correct Abstention Rate on unanswerables, False Abstention Rate on answerables.
4. Latency benchmarks.
"""
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple
import numpy as np

from src.pipeline import ContractIntelligenceEngine, SearchResponse

def compute_character_overlap(
    gt_snippets: List[Dict[str, Any]],
    retrieved_passages: List[Any],
    k: int
) -> Tuple[float, float, float, bool]:
    """
    Computes exact character-level precision, recall, F1, and hit at top-k.
    """
    # Build set of ground truth (canonical_filename, char_index)
    gt_chars: Set[Tuple[str, int]] = set()
    for s in gt_snippets:
        fname = Path(s["file_path"]).name
        start, end = s["span"]
        for c in range(start, end):
            gt_chars.add((fname, c))

    if not gt_chars:
        return 0.0, 0.0, 0.0, False

    # Build set of retrieved (canonical_filename, char_index) up to k
    ret_chars: Set[Tuple[str, int]] = set()
    top_k_passages = retrieved_passages[:k]
    for p in top_k_passages:
        fname = Path(p.file_path).name
        for c in range(p.start_char, p.end_char):
            ret_chars.add((fname, c))

    if not ret_chars:
        return 0.0, 0.0, 0.0, False

    intersection = len(gt_chars.intersection(ret_chars))
    precision = intersection / len(ret_chars) if ret_chars else 0.0
    recall = intersection / len(gt_chars) if gt_chars else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    hit = intersection > 0

    return precision, recall, f1, hit

class EvaluationReport:
    """Holds aggregated metrics and per-query breakdown."""
    def __init__(self, split: str, metrics: Dict[str, Any], query_details: List[Dict[str, Any]]):
        self.split = split
        self.metrics = metrics
        self.query_details = query_details

    def print_summary(self):
        print(f"\n========================================================")
        print(f" EVALUATION REPORT (Split: {self.split.upper()})")
        print(f"========================================================")
        m = self.metrics
        print(f"Answerable Queries Evaluated   : {m['num_answerable']}")
        print(f"Unanswerable Queries Evaluated : {m['num_unanswerable']}")
        print(f"Total Queries                  : {m['total_queries']}")
        print(f"Mean Latency                   : {m['mean_latency_ms']:.2f} ms / query")
        print(f"--------------------------------------------------------")
        print(f"RETRIEVAL METRICS (LegalBench-RAG character-level):")
        for k in [1, 3, 5]:
            p = m[f'char_precision@{k}'] * 100
            r = m[f'char_recall@{k}'] * 100
            f1 = m[f'char_f1@{k}'] * 100
            hit = m[f'doc_hit@{k}'] * 100
            print(f"  k={k}: Precision={p:5.2f}% | Recall={r:5.2f}% | F1={f1:5.2f}% | Hit@{k}={hit:5.2f}%")
        print(f"--------------------------------------------------------")
        print(f"ABSTENTION METRICS:")
        print(f"  Correct Abstention (Unanswerable Recall) : {m['correct_abstention_rate']*100:5.2f}% ({m['correct_abstentions']}/{m['num_unanswerable']})")
        print(f"  False Abstention (Wrongly Abstained)    : {m['false_abstention_rate']*100:5.2f}% ({m['false_abstentions']}/{m['num_answerable']})")
        print(f"  Abstention Balanced Accuracy            : {m['abstention_accuracy']*100:5.2f}%")
        print(f"========================================================\n")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "split": self.split,
            "metrics": self.metrics,
            "query_details": self.query_details
        }

class Evaluator:
    """Evaluates ContractIntelligenceEngine against benchmark datasets."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self._load_datasets()

    def _load_datasets(self):
        cuad_path = self.data_dir / "cuad_mini.json"
        unans_path = self.data_dir / "unanswerable_queries.json"

        with open(cuad_path, "r", encoding="utf-8") as f:
            self.cuad_data = json.load(f)["queries"]

        with open(unans_path, "r", encoding="utf-8") as f:
            self.unans_data = json.load(f)["queries"]

    def run_evaluation(
        self,
        engine: ContractIntelligenceEngine,
        split: str = "train",
        k_values: List[int] = [1, 3, 5]
    ) -> EvaluationReport:
        """
        Runs evaluation on the specified split ('train' or 'test').
        """
        # Filter queries by split
        ans_queries = [q for q in self.cuad_data if q.get("split") == split]
        unans_queries = [q for q in self.unans_data if q.get("split") == split]

        query_details = []
        latencies = []

        # Metric accumulators for answerable queries
        k_precisions = {k: [] for k in k_values}
        k_recalls = {k: [] for k in k_values}
        k_f1s = {k: [] for k in k_values}
        k_hits = {k: [] for k in k_values}

        false_abstentions = 0

        # 1. Evaluate answerable benchmark queries
        for idx, item in enumerate(ans_queries):
            query = item["query"]
            gt_snippets = item["snippets"]
            
            resp = engine.search(query, top_k=max(k_values))
            latencies.append(resp.latency_ms)

            q_detail = {
                "id": item.get("id", f"cuad_{idx}"),
                "query": query,
                "type": "answerable",
                "is_abstained": resp.is_abstained,
                "abstention_reason": resp.abstention_reason,
                "confidence": resp.confidence,
                "latency_ms": resp.latency_ms,
                "retrieved_count": len(resp.passages),
                "k_metrics": {}
            }

            if resp.is_abstained:
                false_abstentions += 1
                for k in k_values:
                    k_precisions[k].append(0.0)
                    k_recalls[k].append(0.0)
                    k_f1s[k].append(0.0)
                    k_hits[k].append(0.0)
                    q_detail["k_metrics"][f"k_{k}"] = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "hit": False}
            else:
                for k in k_values:
                    p, r, f1, hit = compute_character_overlap(gt_snippets, resp.passages, k)
                    k_precisions[k].append(p)
                    k_recalls[k].append(r)
                    k_f1s[k].append(f1)
                    k_hits[k].append(1.0 if hit else 0.0)
                    q_detail["k_metrics"][f"k_{k}"] = {"precision": p, "recall": r, "f1": f1, "hit": hit}

            query_details.append(q_detail)

        # 2. Evaluate unanswerable queries
        correct_abstentions = 0
        failed_abstentions = 0

        for idx, item in enumerate(unans_queries):
            query = item["query"]
            resp = engine.search(query, top_k=max(k_values))
            latencies.append(resp.latency_ms)

            q_detail = {
                "id": item.get("id", f"unans_{idx}"),
                "query": query,
                "type": "unanswerable",
                "category": item.get("category", ""),
                "is_abstained": resp.is_abstained,
                "abstention_reason": resp.abstention_reason,
                "confidence": resp.confidence,
                "latency_ms": resp.latency_ms,
                "retrieved_count": len(resp.passages)
            }

            if resp.is_abstained:
                correct_abstentions += 1
            else:
                failed_abstentions += 1

            query_details.append(q_detail)

        # Compute summary metrics
        n_ans = len(ans_queries)
        n_unans = len(unans_queries)
        total_q = n_ans + n_unans

        metrics = {
            "num_answerable": n_ans,
            "num_unanswerable": n_unans,
            "total_queries": total_q,
            "mean_latency_ms": float(np.mean(latencies)) if latencies else 0.0,
            "false_abstentions": false_abstentions,
            "false_abstention_rate": false_abstentions / n_ans if n_ans > 0 else 0.0,
            "correct_abstentions": correct_abstentions,
            "correct_abstention_rate": correct_abstentions / n_unans if n_unans > 0 else 0.0,
            "abstention_accuracy": (
                ((n_ans - false_abstentions) / n_ans + (correct_abstentions / n_unans)) / 2.0
                if n_ans > 0 and n_unans > 0 else 0.0
            )
        }

        for k in k_values:
            metrics[f"char_precision@{k}"] = float(np.mean(k_precisions[k])) if k_precisions[k] else 0.0
            metrics[f"char_recall@{k}"] = float(np.mean(k_recalls[k])) if k_recalls[k] else 0.0
            metrics[f"char_f1@{k}"] = float(np.mean(k_f1s[k])) if k_f1s[k] else 0.0
            metrics[f"doc_hit@{k}"] = float(np.mean(k_hits[k])) if k_hits[k] else 0.0

        return EvaluationReport(split=split, metrics=metrics, query_details=query_details)
