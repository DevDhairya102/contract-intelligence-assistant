import pytest
from src.config import PipelineConfig
from src.pipeline import ContractIntelligenceEngine

def test_pipeline_search_end_to_end():
    config = PipelineConfig(
        chunk_strategy="fixed",
        chunk_size=600,
        retriever_type="bm25",
        abstention_mode="none"
    )
    engine = ContractIntelligenceEngine(config=config)
    
    resp = engine.search("What is the governing law of this agreement?", top_k=3)
    assert resp.is_abstained is False
    assert len(resp.passages) == 3
    assert resp.latency_ms > 0
    
    # Check passage structure
    top_p = resp.passages[0]
    assert top_p.start_char >= 0
    assert top_p.end_char > top_p.start_char
    assert len(top_p.text) > 0
    assert top_p.score >= 0
