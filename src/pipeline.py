"""
End-to-End Contract Intelligence Assistant Pipeline.
Connects DocumentLoader, Chunker, Retriever, Reranker, Span Refinement, and Abstention.
"""
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from src.config import PipelineConfig
from src.document_loader import DocumentLoader
from src.chunker import Chunker, Chunk
from src.retriever import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    BaseRetriever,
    legal_tokenize
)
from src.reranker import NeuralReranker
from src.abstention import AbstentionModule, AbstentionDecision

@dataclass
class RetrievedPassage:
    rank: int
    file_path: str
    start_char: int
    end_char: int
    text: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "file_path": self.file_path,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "text": self.text,
            "score": round(self.score, 4),
            "metadata": self.metadata
        }

@dataclass
class SearchResponse:
    query: str
    is_abstained: bool
    abstention_reason: str
    confidence: float
    passages: List[RetrievedPassage]
    latency_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "is_abstained": self.is_abstained,
            "abstention_reason": self.abstention_reason,
            "confidence": round(self.confidence, 4),
            "passages": [p.to_dict() for p in self.passages],
            "latency_ms": round(self.latency_ms, 2)
        }

class ContractIntelligenceEngine:
    """Enterprise Contract Clause Search and Abstention Engine."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.doc_loader = DocumentLoader(data_dir=self.config.data_dir)
        self.chunker = Chunker(
            strategy=self.config.chunk_strategy,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
        self.chunks: List[Chunk] = []
        self.retriever: Optional[BaseRetriever] = None
        self.reranker: Optional[NeuralReranker] = None
        self.abstention = AbstentionModule(
            mode=self.config.abstention_mode,
            threshold=self.config.abstention_threshold,
            margin=self.config.abstention_margin
        )
        self._build_index()

    def _build_index(self):
        """Indexes all contracts according to the active configuration."""
        contracts = self.doc_loader.list_contracts()
        self.chunks = []

        for c_path in contracts:
            doc_text = self.doc_loader.get_document(c_path)
            if doc_text:
                file_chunks = self.chunker.chunk_document(c_path, doc_text)
                self.chunks.extend(file_chunks)

        # Build retrieval index
        if self.config.retriever_type == "bm25":
            self.retriever = BM25Retriever(
                chunks=self.chunks,
                k1=self.config.bm25_k1,
                b=self.config.bm25_b
            )
        elif self.config.retriever_type == "dense":
            self.retriever = DenseRetriever(
                chunks=self.chunks,
                model_name=self.config.dense_model_name,
                use_cache=self.config.use_embedding_cache
            )
        elif self.config.retriever_type == "hybrid":
            bm25 = BM25Retriever(chunks=self.chunks, k1=self.config.bm25_k1, b=self.config.bm25_b)
            dense = DenseRetriever(
                chunks=self.chunks,
                model_name=self.config.dense_model_name,
                use_cache=self.config.use_embedding_cache
            )
            self.retriever = HybridRetriever(
                bm25_retriever=bm25,
                dense_retriever=dense,
                bm25_weight=self.config.bm25_weight,
                dense_weight=self.config.dense_weight,
                fusion_method=self.config.hybrid_fusion_method,
                rrf_k=self.config.rrf_k
            )

        # Build reranker if enabled
        if self.config.use_reranker:
            self.reranker = NeuralReranker(model_name=self.config.reranker_model_name)
        else:
            self.reranker = None

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        contract_filter: Optional[str] = None
    ) -> SearchResponse:
        """
        Executes query retrieval, reranking, span refinement, and abstention.
        """
        t0 = time.time()
        k = top_k or self.config.top_k
        retrieve_k = self.config.reranker_top_n if self.config.use_reranker else max(k, 10)

        # Optional filtering by contract
        if contract_filter:
            canonical_filter = self.doc_loader.get_canonical_path(contract_filter)
        else:
            canonical_filter = None

        # 1. Retrieve candidates
        candidates = self.retriever.retrieve(query, top_k=retrieve_k * 2 if canonical_filter else retrieve_k)
        
        if canonical_filter:
            candidates = [c for c in candidates if c[0].file_path == canonical_filter][:retrieve_k]

        # 2. Neural Reranking (if enabled)
        if self.reranker and candidates:
            candidates = self.reranker.rerank(query, candidates, top_k=k)
        else:
            candidates = candidates[:k]

        # 3. Abstention evaluation
        decision = self.abstention.evaluate(candidates)
        latency_ms = (time.time() - t0) * 1000.0

        if decision.should_abstain:
            return SearchResponse(
                query=query,
                is_abstained=True,
                abstention_reason=decision.reason,
                confidence=decision.confidence,
                passages=[],
                latency_ms=latency_ms
            )

        # 4. Format Passages with optional Span Refinement
        q_tokens = set(legal_tokenize(query))
        passages: List[RetrievedPassage] = []

        for rank, (chunk, score) in enumerate(candidates, start=1):
            start = chunk.start_char
            end = chunk.end_char
            text = chunk.text

            if self.config.span_refinement:
                r_start, r_end = Chunker.refine_span(chunk, q_tokens)
                doc_full = self.doc_loader.get_document(chunk.file_path)
                if doc_full and r_end > r_start:
                    start = r_start
                    end = r_end
                    text = doc_full[start:end]

            passages.append(RetrievedPassage(
                rank=rank,
                file_path=chunk.file_path,
                start_char=start,
                end_char=end,
                text=text,
                score=float(score),
                metadata=chunk.metadata
            ))

        return SearchResponse(
            query=query,
            is_abstained=False,
            abstention_reason=decision.reason,
            confidence=decision.confidence,
            passages=passages,
            latency_ms=latency_ms
        )
