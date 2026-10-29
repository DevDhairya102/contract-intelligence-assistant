import pytest
from src.chunker import Chunk
from src.abstention import AbstentionModule

def test_score_threshold_abstention():
    chunk = Chunk("c1", "doc1.txt", 0, 50, "Sample text")
    
    # Threshold 0.5, candidate score 0.3 -> Abstain
    abstain_module = AbstentionModule(mode="score_threshold", threshold=0.5)
    decision = abstain_module.evaluate([(chunk, 0.3)])
    assert decision.should_abstain is True
    assert "below the required confidence threshold" in decision.reason

    # Threshold 0.5, candidate score 0.7 -> Answer
    decision_pass = abstain_module.evaluate([(chunk, 0.7)])
    assert decision_pass.should_abstain is False

def test_empty_candidates_abstention():
    abstain_module = AbstentionModule(mode="score_threshold", threshold=0.1)
    decision = abstain_module.evaluate([])
    assert decision.should_abstain is True
    assert "no candidate passages" in decision.reason
