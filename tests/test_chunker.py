import pytest
from src.chunker import Chunker, Chunk

SAMPLE_CONTRACT_TEXT = """1. DEFINITIONS AND INTERPRETATION.
In this Agreement, the following terms shall have the specified meanings unless the context requires otherwise.
"Affiliate" means any entity that directly or indirectly controls.

2. TERM AND TERMINATION.
This Agreement shall commence on January 1, 2020 and continue for a period of three (3) years.
Either party may terminate this Agreement immediately upon written notice if the other party breaches any material term.

3. GOVERNING LAW.
This Agreement shall be construed and enforced in accordance with the laws of Delaware.
"""

def test_fixed_chunking_exact_offsets():
    chunker = Chunker(strategy="fixed", chunk_size=120, chunk_overlap=30)
    chunks = chunker.chunk_document("test_doc.txt", SAMPLE_CONTRACT_TEXT)
    
    assert len(chunks) > 1
    for c in chunks:
        # Every chunk's text must match the slice of original text exactly
        assert SAMPLE_CONTRACT_TEXT[c.start_char:c.end_char] == c.text
        assert c.start_char >= 0
        assert c.end_char <= len(SAMPLE_CONTRACT_TEXT)

def test_clause_aware_chunking():
    chunker = Chunker(strategy="clause_aware", chunk_size=300, min_chunk_size=50)
    chunks = chunker.chunk_document("test_doc.txt", SAMPLE_CONTRACT_TEXT)
    
    assert len(chunks) >= 2
    for c in chunks:
        assert SAMPLE_CONTRACT_TEXT[c.start_char:c.end_char] == c.text

def test_span_refinement():
    chunk = Chunk(
        chunk_id="c1",
        file_path="test.txt",
        start_char=0,
        end_char=len(SAMPLE_CONTRACT_TEXT),
        text=SAMPLE_CONTRACT_TEXT
    )
    query_tokens = {"governing", "law", "delaware"}
    r_start, r_end = Chunker.refine_span(chunk, query_tokens)
    
    # Should identify the governing law clause
    refined_text = SAMPLE_CONTRACT_TEXT[r_start:r_end]
    assert "Delaware" in refined_text
    assert len(refined_text) < len(SAMPLE_CONTRACT_TEXT)
