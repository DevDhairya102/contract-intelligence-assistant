"""
Abstention Module for Contract Intelligence Assistant.
Decides whether to return ranked passages or abstain with an explicit reason when no passage is sufficiently relevant.
"""
from typing import List, Tuple, Optional
from src.chunker import Chunk

class AbstentionDecision:
    def __init__(self, should_abstain: bool, reason: str, confidence: float):
        self.should_abstain = should_abstain
        self.reason = reason
        self.confidence = confidence

    def __bool__(self):
        return self.should_abstain

class AbstentionModule:
    """Evaluates candidates to decide whether the engine has sufficient confidence to answer."""

    def __init__(
        self,
        mode: str = "score_threshold",
        threshold: float = 0.0,
        margin: float = 0.0
    ):
        self.mode = mode
        self.threshold = threshold
        self.margin = margin

    def evaluate(
        self,
        candidates: List[Tuple[Chunk, float]]
    ) -> AbstentionDecision:
        """Determines if the system should abstain."""
        if not candidates:
            return AbstentionDecision(
                should_abstain=True,
                reason="Abstained: no candidate passages were retrieved.",
                confidence=0.0
            )

        top_chunk, top_score = candidates[0]

        if self.mode == "none":
            return AbstentionDecision(
                should_abstain=False,
                reason="Answering: abstention disabled.",
                confidence=top_score
            )

        if self.mode == "score_threshold":
            if top_score < self.threshold:
                return AbstentionDecision(
                    should_abstain=True,
                    reason=(
                        f"Abstained: top relevance score ({top_score:.4f}) is below "
                        f"the required confidence threshold ({self.threshold:.4f}). "
                        f"The indexed contracts likely do not contain an answering clause."
                    ),
                    confidence=top_score
                )
            return AbstentionDecision(
                should_abstain=False,
                reason=f"Answering: top score ({top_score:.4f}) meets threshold ({self.threshold:.4f}).",
                confidence=top_score
            )

        if self.mode == "margin":
            if top_score < self.threshold:
                return AbstentionDecision(
                    should_abstain=True,
                    reason=f"Abstained: top score ({top_score:.4f}) below threshold ({self.threshold:.4f}).",
                    confidence=top_score
                )
            if len(candidates) > 1 and self.margin > 0:
                second_score = candidates[1][1]
                gap = top_score - second_score
                if gap < self.margin:
                    return AbstentionDecision(
                        should_abstain=True,
                        reason=f"Abstained: confidence gap ({gap:.4f}) between top candidates is below margin ({self.margin:.4f}).",
                        confidence=top_score
                    )
            return AbstentionDecision(
                should_abstain=False,
                reason=f"Answering: score {top_score:.4f} satisfies threshold and margin.",
                confidence=top_score
            )

        if self.mode == "adaptive":
            # Adaptive thresholding: require higher threshold if candidate is extremely short or vague
            effective_threshold = self.threshold
            if len(top_chunk.text) < 150:
                effective_threshold += 0.05
            
            if top_score < effective_threshold:
                return AbstentionDecision(
                    should_abstain=True,
                    reason=f"Abstained: top score ({top_score:.4f}) below adaptive threshold ({effective_threshold:.4f}).",
                    confidence=top_score
                )
            return AbstentionDecision(
                should_abstain=False,
                reason=f"Answering: top score ({top_score:.4f}) satisfies adaptive threshold.",
                confidence=top_score
            )

        return AbstentionDecision(should_abstain=False, reason="Default accept", confidence=top_score)
