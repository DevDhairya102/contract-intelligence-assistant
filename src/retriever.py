"""
Retrieval implementations: BM25 (Lexical), Dense (Vector), and Hybrid (Lexical + Dense).
Supports score normalization, Reciprocal Rank Fusion (RRF), and vector caching.
"""
import re
import os
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np

import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src.chunker import Chunk
from src.config import PipelineConfig

def legal_tokenize(text: str) -> List[str]:
    """Tokenizes text preserving section numbers, alphanumeric tokens, and common legal terms."""
    # Convert to lowercase and find all word tokens including dotted section numbers
    tokens = re.findall(r"\b[a-zA-Z0-9_\.]+\b", text.lower())
    # Strip trailing periods from tokens
    cleaned = [t.rstrip(".") for t in tokens if len(t.rstrip(".")) > 1]
    return cleaned

class BaseRetriever:
    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[Chunk, float]]:
        raise NotImplementedError

class BM25Retriever(BaseRetriever):
    """Lexical retrieval using BM25Okapi."""
    
    def __init__(self, chunks: List[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.corpus_tokens = [legal_tokenize(c.text) for c in chunks]
        self.bm25 = BM25Okapi(self.corpus_tokens, k1=k1, b=b)

    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[Chunk, float]]:
        q_tokens = legal_tokenize(query)
        if not q_tokens:
            return [(c, 0.0) for c in self.chunks[:top_k]]
        
        scores = self.bm25.get_scores(q_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.chunks[i], float(scores[i])) for i in top_indices]

    def score_all(self, query: str) -> np.ndarray:
        q_tokens = legal_tokenize(query)
        if not q_tokens:
            return np.zeros(len(self.chunks), dtype=np.float32)
        return np.array(self.bm25.get_scores(q_tokens), dtype=np.float32)


class DenseRetriever(BaseRetriever):
    """Dense semantic retrieval using SentenceTransformers and cosine similarity."""
    
    def __init__(
        self,
        chunks: List[Chunk],
        model_name: str = "all-MiniLM-L6-v2",
        cache_dir: Optional[str] = "experiments/cache",
        use_cache: bool = True
    ):
        self.chunks = chunks
        self.model_name = model_name
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.use_cache = use_cache
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)
        self.chunk_embeddings = self._get_or_compute_embeddings()

    def _get_or_compute_embeddings(self) -> np.ndarray:
        """Loads cached embeddings or computes and persists them."""
        cache_file = None
        if self.use_cache and self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            # Create a unique cache key based on chunk count, first/last chunk id and model name
            safe_model = self.model_name.replace("/", "_")
            key = f"emb_{safe_model}_{len(self.chunks)}"
            if self.chunks:
                key += f"_{abs(hash(self.chunks[0].chunk_id + self.chunks[-1].chunk_id)) % 100000}"
            cache_file = self.cache_dir / f"{key}.npz"
            
            if cache_file.exists():
                data = np.load(cache_file)
                if "embeddings" in data and len(data["embeddings"]) == len(self.chunks):
                    return data["embeddings"]

        # Compute embeddings in batches
        texts = [c.text for c in self.chunks]
        embeddings = self.model.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        if cache_file:
            np.savez_compressed(cache_file, embeddings=embeddings)

        return embeddings

    def score_all(self, query: str) -> np.ndarray:
        q_emb = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
        # Cosine similarity since both query and chunk embeddings are L2 normalized
        scores = np.dot(self.chunk_embeddings, q_emb)
        return scores.astype(np.float32)

    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[Chunk, float]]:
        scores = self.score_all(query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.chunks[i], float(scores[i])) for i in top_indices]


class HybridRetriever(BaseRetriever):
    """
    Hybrid retriever combining Lexical (BM25) and Semantic (Dense) signals.
    Supports Convex linear combination (with min-max normalization) and Reciprocal Rank Fusion (RRF).
    """

    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        dense_retriever: DenseRetriever,
        bm25_weight: float = 0.4,
        dense_weight: float = 0.6,
        fusion_method: str = "convex",
        rrf_k: int = 60
    ):
        self.bm25 = bm25_retriever
        self.dense = dense_retriever
        self.chunks = bm25_retriever.chunks
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        self.fusion_method = fusion_method
        self.rrf_k = rrf_k

    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[Chunk, float]]:
        scores = self.score_all(query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.chunks[i], float(scores[i])) for i in top_indices]

    def score_all(self, query: str) -> np.ndarray:
        bm25_scores = self.bm25.score_all(query)
        dense_scores = self.dense.score_all(query)

        if self.fusion_method == "rrf":
            # Reciprocal Rank Fusion
            bm25_ranks = np.argsort(np.argsort(-bm25_scores))
            dense_ranks = np.argsort(np.argsort(-dense_scores))
            
            rrf_scores = (
                self.bm25_weight * (1.0 / (self.rrf_k + bm25_ranks + 1)) +
                self.dense_weight * (1.0 / (self.rrf_k + dense_ranks + 1))
            )
            return rrf_scores.astype(np.float32)

        else: # "convex" combination
            # Min-Max normalize BM25 scores to [0, 1]
            b_min, b_max = bm25_scores.min(), bm25_scores.max()
            if b_max > b_min:
                bm25_norm = (bm25_scores - b_min) / (b_max - b_min)
            else:
                bm25_norm = np.zeros_like(bm25_scores)

            # Dense cosine scores are typically in [-1, 1], clip to [0, 1] for safety
            dense_norm = np.clip((dense_scores + 1.0) / 2.0, 0.0, 1.0)

            combined = (self.bm25_weight * bm25_norm) + (self.dense_weight * dense_norm)
            return combined.astype(np.float32)
