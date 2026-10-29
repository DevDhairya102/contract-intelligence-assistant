import pytest
from src.evaluator import compute_character_overlap
from src.pipeline import RetrievedPassage

def test_compute_character_overlap_exact_match():
    gt_snippets = [
        {"file_path": "doc1.txt", "span": [100, 200], "answer": "text"}
    ]
    retrieved = [
        RetrievedPassage(rank=1, file_path="doc1.txt", start_char=100, end_char=200, text="text", score=0.9)
    ]
    p, r, f1, hit = compute_character_overlap(gt_snippets, retrieved, k=1)
    assert p == 1.0
    assert r == 1.0
    assert f1 == 1.0
    assert hit is True

def test_compute_character_overlap_partial_and_diluted():
    # Ground truth is 50 chars from 100 to 150
    gt_snippets = [
        {"file_path": "doc1.txt", "span": [100, 150], "answer": "text"}
    ]
    # Retrieved chunk is 100 chars from 100 to 200 (contains all GT + 50 extra chars)
    retrieved = [
        RetrievedPassage(rank=1, file_path="doc1.txt", start_char=100, end_char=200, text="text", score=0.9)
    ]
    p, r, f1, hit = compute_character_overlap(gt_snippets, retrieved, k=1)
    assert r == 1.0        # All 50 GT chars retrieved
    assert p == 0.5        # 50 GT chars / 100 retrieved chars = 0.5
    assert hit is True

def test_compute_character_overlap_wrong_doc():
    gt_snippets = [
        {"file_path": "doc1.txt", "span": [100, 200], "answer": "text"}
    ]
    retrieved = [
        RetrievedPassage(rank=1, file_path="wrong_doc.txt", start_char=100, end_char=200, text="text", score=0.9)
    ]
    p, r, f1, hit = compute_character_overlap(gt_snippets, retrieved, k=1)
    assert p == 0.0
    assert r == 0.0
    assert f1 == 0.0
    assert hit is False
