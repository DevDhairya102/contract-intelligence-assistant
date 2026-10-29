"""
Configuration schema for the Contract Intelligence Assistant pipeline.
"""
from typing import Literal, Optional, Dict, Any
from pydantic import BaseModel, Field

class PipelineConfig(BaseModel):
    """Configuration for chunking, indexing, retrieval, reranking, and abstention."""
    
    # Experiment metadata
    experiment_name: str = Field(default="default_pipeline", description="Name/ID of this configuration")
    description: str = Field(default="", description="Detailed notes on what this config tests")
    
    # Document indexing scope
    data_dir: str = Field(default="data", description="Path to data directory containing corpus and benchmarks")
    
    # Chunking strategy
    chunk_strategy: Literal["fixed", "paragraph", "clause_aware"] = Field(
        default="fixed",
        description="Chunking strategy: 'fixed' sliding window, 'paragraph' split on double newlines, 'clause_aware' splits on legal section headers"
    )
    chunk_size: int = Field(default=800, description="Target chunk size in characters")
    chunk_overlap: int = Field(default=150, description="Overlap in characters between consecutive chunks")
    
    # Retrieval strategy
    retriever_type: Literal["bm25", "dense", "hybrid"] = Field(
        default="bm25",
        description="Retriever type: 'bm25', 'dense' (SentenceTransformer), or 'hybrid' (combined)"
    )
    dense_model_name: str = Field(default="all-MiniLM-L6-v2", description="HuggingFace model for dense embeddings")
    bm25_k1: float = Field(default=1.5, description="BM25 k1 parameter")
    bm25_b: float = Field(default=0.75, description="BM25 b parameter")
    
    # Hybrid retrieval weighting (used when retriever_type == 'hybrid')
    bm25_weight: float = Field(default=0.4, description="Weight for BM25 score in hybrid retrieval")
    dense_weight: float = Field(default=0.6, description="Weight for dense similarity in hybrid retrieval")
    hybrid_fusion_method: Literal["convex", "rrf"] = Field(
        default="convex",
        description="Fusion method for hybrid retrieval: 'convex' linear combination or 'rrf' Reciprocal Rank Fusion"
    )
    rrf_k: int = Field(default=60, description="Smoothing parameter k for RRF")
    
    # Reranking strategy
    use_reranker: bool = Field(default=False, description="Whether to apply a cross-encoder neural reranker")
    reranker_model_name: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Cross-encoder model name"
    )
    reranker_top_n: int = Field(default=15, description="Number of candidate chunks passed to the reranker")
    
    # Span refinement / clause trimming
    span_refinement: bool = Field(
        default=False,
        description="Whether to refine chunk boundaries to the most relevant sentence/subclause span"
    )
    
    # Abstention strategy
    abstention_mode: Literal["none", "score_threshold", "margin", "adaptive"] = Field(
        default="score_threshold",
        description="Method to decide whether to abstain: 'none', 'score_threshold', 'margin', or 'adaptive'"
    )
    abstention_threshold: float = Field(
        default=0.0,
        description="Cutoff threshold below which the system abstains and returns no passages"
    )
    abstention_margin: float = Field(
        default=0.0,
        description="Minimum score gap required between rank 1 and rank 2 when using margin abstention"
    )
    
    # Output limits
    top_k: int = Field(default=5, description="Number of passages to return for an answered query")
    
    # Performance & caching
    use_embedding_cache: bool = Field(default=True, description="Cache dense embeddings to disk for faster loading")

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()
    
    @classmethod
    def from_json(cls, json_path: str) -> "PipelineConfig":
        with open(json_path, "r", encoding="utf-8") as f:
            import json
            return cls(**json.load(f))

    def save_json(self, json_path: str):
        with open(json_path, "w", encoding="utf-8") as f:
            import json
            json.dump(self.to_dict(), f, indent=2)
