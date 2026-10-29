"""
Failure Diagnostic Engine for Contract Intelligence Assistant.
Categorizes retrieval and abstention errors into structured failure modes:
1. INCORRECT_ABSTENTION (false negative on answerable queries)
2. FAILED_TO_ABSTAIN (false positive on unanswerable queries)
3. WRONG_DOCUMENT (retrieved wrong contract altogether)
4. WRONG_SECTION (retrieved correct contract, but wrong clause/section)
5. PARTIAL_SPAN_COVERAGE (retrieved correct clause, but truncated by chunk boundary)
6. LOW_PRECISION_DILUTION (answer found, but embedded in large irrelevant chunk)
"""
from typing import List, Dict, Any
from pathlib import Path
from src.evaluator import EvaluationReport

class FailureMode:
    INCORRECT_ABSTENTION = "INCORRECT_ABSTENTION"
    FAILED_TO_ABSTAIN = "FAILED_TO_ABSTAIN"
    WRONG_DOCUMENT = "WRONG_DOCUMENT"
    WRONG_SECTION = "WRONG_SECTION"
    PARTIAL_SPAN_COVERAGE = "PARTIAL_SPAN_COVERAGE"
    LOW_PRECISION_DILUTION = "LOW_PRECISION_DILUTION"
    SUCCESS = "SUCCESS"

class FailureAnalyzer:
    """Diagnoses failures on evaluation reports and proposes corrective actions."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        import json
        with open(self.data_dir / "cuad_mini.json", "r", encoding="utf-8") as f:
            self.cuad_map = {q["query"]: q for q in json.load(f)["queries"]}
        with open(self.data_dir / "unanswerable_queries.json", "r", encoding="utf-8") as f:
            self.unans_map = {q["query"]: q for q in json.load(f)["queries"]}

    def diagnose_report(self, report: EvaluationReport) -> Dict[str, Any]:
        """Analyzes all queries in the report and groups errors by cause."""
        categorized_failures: Dict[str, List[Dict[str, Any]]] = {
            FailureMode.INCORRECT_ABSTENTION: [],
            FailureMode.FAILED_TO_ABSTAIN: [],
            FailureMode.WRONG_DOCUMENT: [],
            FailureMode.WRONG_SECTION: [],
            FailureMode.PARTIAL_SPAN_COVERAGE: [],
            FailureMode.LOW_PRECISION_DILUTION: [],
            FailureMode.SUCCESS: []
        }

        for q in report.query_details:
            query_text = q["query"]
            q_type = q["type"]
            is_abstained = q["is_abstained"]

            # 1. Unanswerable query analysis
            if q_type == "unanswerable":
                if is_abstained:
                    categorized_failures[FailureMode.SUCCESS].append({
                        "id": q["id"],
                        "query": query_text,
                        "diagnosis": "Correctly abstained on unanswerable query."
                    })
                else:
                    unans_meta = self.unans_map.get(query_text, {})
                    categorized_failures[FailureMode.FAILED_TO_ABSTAIN].append({
                        "id": q["id"],
                        "query": query_text,
                        "category": unans_meta.get("category", "unknown"),
                        "confidence": q["confidence"],
                        "diagnosis": (
                            f"System retrieved passages with confidence {q['confidence']:.4f} "
                            f"instead of abstaining. Expected abstention because: {unans_meta.get('explanation', '')}"
                        )
                    })
                continue

            # 2. Answerable query analysis
            gt_item = self.cuad_map.get(query_text)
            if not gt_item:
                continue
            gt_snippets = gt_item["snippets"]
            gt_files = [Path(s["file_path"]).name for s in gt_snippets]

            if is_abstained:
                categorized_failures[FailureMode.INCORRECT_ABSTENTION].append({
                    "id": q["id"],
                    "query": query_text,
                    "confidence": q["confidence"],
                    "ground_truth_contract": gt_files[0] if gt_files else "unknown",
                    "ground_truth_answer": gt_snippets[0]["answer"][:120] if gt_snippets else "",
                    "diagnosis": (
                        f"System incorrectly abstained on an answerable query. "
                        f"Confidence score {q['confidence']:.4f} was below abstention threshold. "
                        f"Ground truth clause exists in {gt_files[0]}."
                    )
                })
                continue

            # Check at k=5
            k5 = q["k_metrics"].get("k_5", {})
            recall_5 = k5.get("recall", 0.0)
            precision_5 = k5.get("precision", 0.0)
            hit_5 = k5.get("hit", False)

            if not hit_5:
                # Did we retrieve the right document at all?
                # We can deduce from hit=False that no passage overlapped with ground truth
                categorized_failures[FailureMode.WRONG_SECTION].append({
                    "id": q["id"],
                    "query": query_text,
                    "ground_truth_contract": gt_files[0],
                    "ground_truth_answer": gt_snippets[0]["answer"][:120],
                    "diagnosis": (
                        f"Retrieved passages did not overlap with target clause. "
                        f"The retriever failed to rank the ground truth section in the top 5."
                    )
                })
            elif recall_5 < 0.99:
                categorized_failures[FailureMode.PARTIAL_SPAN_COVERAGE].append({
                    "id": q["id"],
                    "query": query_text,
                    "recall": recall_5,
                    "precision": precision_5,
                    "ground_truth_answer": gt_snippets[0]["answer"][:120],
                    "diagnosis": (
                        f"Retrieved correct clause but character recall was only {recall_5*100:.1f}%. "
                        f"The answer span extends beyond the chunk boundary."
                    )
                })
            elif precision_5 < 0.15:
                categorized_failures[FailureMode.LOW_PRECISION_DILUTION].append({
                    "id": q["id"],
                    "query": query_text,
                    "recall": recall_5,
                    "precision": precision_5,
                    "diagnosis": (
                        f"Full ground truth recalled (100%), but character precision is diluted ({precision_5*100:.1f}%) "
                        f"due to oversized chunk window or irrelevant surrounding text."
                    )
                })
            else:
                categorized_failures[FailureMode.SUCCESS].append({
                    "id": q["id"],
                    "query": query_text,
                    "recall": recall_5,
                    "precision": precision_5,
                    "diagnosis": "High quality retrieval: high recall and good precision."
                })

        summary = {mode: len(failures) for mode, failures in categorized_failures.items()}
        return {
            "summary": summary,
            "failures": categorized_failures
        }

    def generate_proposals(self, diagnosis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generates evidence-backed architectural proposals based on failure diagnosis."""
        counts = diagnosis["summary"]
        proposals = []

        # If many incorrect abstentions, suggest lowering threshold
        if counts.get(FailureMode.INCORRECT_ABSTENTION, 0) > 5:
            proposals.append({
                "action": "lower_abstention_threshold",
                "reason": f"{counts[FailureMode.INCORRECT_ABSTENTION]} queries suffered incorrect abstention."
            })

        # If many failed abstentions (false positives on unanswerables), suggest raising threshold or neural reranker
        if counts.get(FailureMode.FAILED_TO_ABSTAIN, 0) > 5:
            proposals.append({
                "action": "enable_reranker_and_calibrate_threshold",
                "reason": f"{counts[FailureMode.FAILED_TO_ABSTAIN]} unanswerable queries failed to abstain due to false semantic matches."
            })

        # If wrong section or wrong document is high, suggest hybrid retrieval or clause-aware chunking
        if counts.get(FailureMode.WRONG_SECTION, 0) > 10:
            proposals.append({
                "action": "enable_hybrid_retrieval",
                "reason": f"{counts[FailureMode.WRONG_SECTION]} queries missed the target clause with pure lexical or dense search."
            })

        # If partial span coverage is high, suggest clause-aware chunking
        if counts.get(FailureMode.PARTIAL_SPAN_COVERAGE, 0) > 5:
            proposals.append({
                "action": "use_clause_aware_chunking",
                "reason": f"{counts[FailureMode.PARTIAL_SPAN_COVERAGE]} queries had answer spans truncated by arbitrary fixed chunk boundaries."
            })

        # If precision dilution is high, suggest span refinement
        if counts.get(FailureMode.LOW_PRECISION_DILUTION, 0) > 10:
            proposals.append({
                "action": "enable_span_refinement",
                "reason": f"{counts[FailureMode.LOW_PRECISION_DILUTION]} queries had character precision diluted by large chunks."
            })

        return proposals
