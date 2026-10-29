"""
Reranker module using Cross-Encoder neural models.
Applies cross-attention over query-chunk pairs to accurately score legal relevance.
"""
from typing import List, Tuple
import numpy as np
import torch
from sentence_transformers import CrossEncoder

from src.chunker import Chunk

class NeuralReranker:
    """Neural reranker using CrossEncoder for cross-attention scoring."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CrossEncoder(model_name, device=self.device)

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[Chunk, float]],
        top_k: int = 5
    ) -> List[Tuple[Chunk, float]]:
        """Reranks candidates and returns the top_k with calibrated scores."""
        if not candidates:
            return []

        pairs = [(query, chunk.text) for chunk, _ in candidates]
        raw_scores = self.model.predict(pairs, show_progress_bar=False)

        # Convert logits to calibrated probability using sigmoid: 1 / (1 + exp(-x))
        calibrated_scores = 1.0 / (1.0 + np.exp(-raw_scores))

        scored = [(candidates[i][0], float(calibrated_scores[i])) for i in range(len(candidates))]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
