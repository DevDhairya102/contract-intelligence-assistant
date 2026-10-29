import pytest
from src.chunker import Chunk
from src.retriever import BM25Retriever, legal_tokenize

def test_legal_tokenize():
    tokens = legal_tokenize("Section 4.2(a) provides for 100% royalty in Delaware.")
    assert "section" in tokens
    assert "4.2" in tokens
    assert "royalty" in tokens
    assert "delaware" in tokens

def test_bm25_retrieval():
    chunks = [
        Chunk("c1", "doc1.txt", 0, 50, "The governing law shall be the laws of California."),
        Chunk("c2", "doc2.txt", 0, 50, "The agreement may be terminated by either party upon notice."),
        Chunk("c3", "doc3.txt", 0, 50, "Payment shall be made within thirty days of invoice.")
    ]
    retriever = BM25Retriever(chunks)
    results = retriever.retrieve("Which state is the governing law?", top_k=1)
    
    assert len(results) == 1
    best_chunk, score = results[0]
    assert best_chunk.chunk_id == "c1"
    assert score > 0
