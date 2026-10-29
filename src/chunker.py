"""
Chunking and text segmentation module for contract documents.
Supports:
1. Fixed-size sliding character windows with whitespace boundary snapping.
2. Paragraph-based segmentation respecting natural formatting.
3. Clause-aware segmentation identifying legal article and section headers.
4. Span refinement to boost Character Precision@k.
"""
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class Chunk:
    chunk_id: str
    file_path: str
    start_char: int
    end_char: int
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "file_path": self.file_path,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "text": self.text,
            "metadata": self.metadata
        }

class Chunker:
    """Creates chunks from documents while tracking character offsets with 100% precision."""

    # Regex for common legal contract clause / section headers
    CLAUSE_PATTERN = re.compile(
        r"(?m)^(?:\s*(?:ARTICLE|Article|SECTION|Section|CLAUSE|Clause)\s+([0-9IVXLCDM]+(?:\.[0-9]+)*)|"
        r"\s*([0-9]{1,2}\.[0-9]{1,2}(?:\.[0-9]+)*)|"
        r"\s*([0-9]{1,2}\.)\s+[A-Z]|"
        r"\s*(\([a-z0-9ivx]+\))\s+[A-Z])"
    )

    def __init__(
        self,
        strategy: str = "fixed",
        chunk_size: int = 800,
        chunk_overlap: int = 150,
        min_chunk_size: int = 100
    ):
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    def chunk_document(self, file_path: str, doc_text: str) -> List[Chunk]:
        """Segments doc_text into Chunks according to the chosen strategy."""
        if not doc_text:
            return []
        
        if self.strategy == "clause_aware":
            chunks = self._chunk_clause_aware(file_path, doc_text)
        elif self.strategy == "paragraph":
            chunks = self._chunk_paragraph(file_path, doc_text)
        else: # "fixed"
            chunks = self._chunk_fixed(file_path, doc_text)

        # Verification check: each chunk.text must strictly equal doc_text[start:end]
        for c in chunks:
            assert doc_text[c.start_char:c.end_char] == c.text, (
                f"Offset corruption in chunk {c.chunk_id}: text mismatch!"
            )
        return chunks

    def _chunk_fixed(self, file_path: str, doc_text: str) -> List[Chunk]:
        """Sliding window chunking with whitespace snapping."""
        chunks = []
        n = len(doc_text)
        start = 0
        chunk_idx = 0

        while start < n:
            target_end = min(start + self.chunk_size, n)
            
            # Snap to nearest newline or whitespace if not at end of document
            end = target_end
            if end < n:
                # Look backwards for a double newline, newline, or whitespace within a 150-char window
                window_lookback = min(150, end - start - self.min_chunk_size)
                if window_lookback > 0:
                    sub = doc_text[end - window_lookback:end]
                    # Try double newline first
                    nl2 = sub.rfind("\n\n")
                    if nl2 != -1:
                        end = (end - window_lookback) + nl2 + 2
                    else:
                        nl1 = sub.rfind("\n")
                        if nl1 != -1:
                            end = (end - window_lookback) + nl1 + 1
                        else:
                            sp = sub.rfind(" ")
                            if sp != -1:
                                end = (end - window_lookback) + sp + 1

            # Ensure progress
            if end <= start:
                end = min(start + self.chunk_size, n)

            text_slice = doc_text[start:end]
            chunk_id = f"{file_path}#chunk_{chunk_idx:04d}"
            chunks.append(Chunk(
                chunk_id=chunk_id,
                file_path=file_path,
                start_char=start,
                end_char=end,
                text=text_slice,
                metadata={"strategy": "fixed", "index": chunk_idx}
            ))
            chunk_idx += 1

            if end >= n:
                break
            
            # Next start position with overlap
            start = max(start + 1, end - self.chunk_overlap)

        return chunks

    def _chunk_paragraph(self, file_path: str, doc_text: str) -> List[Chunk]:
        """Segments text on double-newlines, combining small paragraphs."""
        chunks = []
        n = len(doc_text)
        
        # Find all paragraph start/end boundaries
        para_spans = []
        last_end = 0
        for m in re.finditer(r"\n\s*\n", doc_text):
            p_start = last_end
            p_end = m.start()
            if p_end > p_start:
                para_spans.append((p_start, p_end))
            last_end = m.end()
        if last_end < n:
            para_spans.append((last_end, n))

        if not para_spans:
            return self._chunk_fixed(file_path, doc_text)

        curr_start, curr_end = para_spans[0]
        chunk_idx = 0

        for p_start, p_end in para_spans[1:]:
            potential_len = p_end - curr_start
            if potential_len <= self.chunk_size:
                curr_end = p_end
            else:
                # Flush current combined paragraph chunk
                text_slice = doc_text[curr_start:curr_end]
                chunks.append(Chunk(
                    chunk_id=f"{file_path}#chunk_{chunk_idx:04d}",
                    file_path=file_path,
                    start_char=curr_start,
                    end_char=curr_end,
                    text=text_slice,
                    metadata={"strategy": "paragraph", "index": chunk_idx}
                ))
                chunk_idx += 1
                curr_start, curr_end = p_start, p_end

        # Flush final chunk
        if curr_end > curr_start:
            chunks.append(Chunk(
                chunk_id=f"{file_path}#chunk_{chunk_idx:04d}",
                file_path=file_path,
                start_char=curr_start,
                end_char=curr_end,
                text=doc_text[curr_start:curr_end],
                metadata={"strategy": "paragraph", "index": chunk_idx}
            ))

        return chunks

    def _chunk_clause_aware(self, file_path: str, doc_text: str) -> List[Chunk]:
        """Identifies legal section/clause headers and segments accordingly."""
        matches = list(self.CLAUSE_PATTERN.finditer(doc_text))
        
        # If very few clause headers found, fall back to paragraph chunking
        if len(matches) < 3:
            return self._chunk_paragraph(file_path, doc_text)

        boundaries = [0]
        for m in matches:
            idx = m.start()
            if idx > boundaries[-1] + self.min_chunk_size:
                boundaries.append(idx)
        if boundaries[-1] < len(doc_text):
            boundaries.append(len(doc_text))

        raw_sections = []
        for i in range(len(boundaries) - 1):
            s_start = boundaries[i]
            s_end = boundaries[i + 1]
            raw_sections.append((s_start, s_end))

        chunks = []
        chunk_idx = 0
        curr_start, curr_end = raw_sections[0]

        for s_start, s_end in raw_sections[1:]:
            curr_len = curr_end - curr_start
            next_len = s_end - s_start

            # If section is excessively long, sub-chunk it with fixed sliding window
            if curr_len > self.chunk_size * 1.5:
                sub_text = doc_text[curr_start:curr_end]
                sub_chunks = self._chunk_fixed(file_path, sub_text)
                for sc in sub_chunks:
                    real_start = curr_start + sc.start_char
                    real_end = curr_start + sc.end_char
                    chunks.append(Chunk(
                        chunk_id=f"{file_path}#clause_{chunk_idx:04d}",
                        file_path=file_path,
                        start_char=real_start,
                        end_char=real_end,
                        text=doc_text[real_start:real_end],
                        metadata={"strategy": "clause_aware_sub", "index": chunk_idx}
                    ))
                    chunk_idx += 1
                curr_start, curr_end = s_start, s_end
            elif curr_len + next_len <= self.chunk_size:
                curr_end = s_end
            else:
                chunks.append(Chunk(
                    chunk_id=f"{file_path}#clause_{chunk_idx:04d}",
                    file_path=file_path,
                    start_char=curr_start,
                    end_char=curr_end,
                    text=doc_text[curr_start:curr_end],
                    metadata={"strategy": "clause_aware", "index": chunk_idx}
                ))
                chunk_idx += 1
                curr_start, curr_end = s_start, s_end

        # Final section
        if curr_end > curr_start:
            chunks.append(Chunk(
                chunk_id=f"{file_path}#clause_{chunk_idx:04d}",
                file_path=file_path,
                start_char=curr_start,
                end_char=curr_end,
                text=doc_text[curr_start:curr_end],
                metadata={"strategy": "clause_aware", "index": chunk_idx}
            ))

        return chunks

    @staticmethod
    def refine_span(chunk: Chunk, query_tokens: set) -> tuple[int, int]:
        """
        Refines chunk boundaries to the line/paragraph span that best matches query tokens.
        Uses newline-based segmentation (more reliable than sentence-ending punctuation in
        legal contracts where section headers like '3. GOVERNING LAW.' are single-line).
        Preserves character offsets exactly.
        """
        # Split on newlines - more suitable for contract layout than sentence punctuation
        lines = chunk.text.split("\n")
        # Filter empty lines but keep their positions
        
        # Build line offsets within the chunk
        offset = 0
        line_offsets = []
        for line in lines:
            line_start = offset
            line_end = offset + len(line)
            line_offsets.append((line_start, line_end, line))
            offset = line_end + 1  # +1 for the \n

        if len(line_offsets) <= 1:
            return chunk.start_char, chunk.end_char

        # Find best contiguous 1-4 lines with highest normalised token match
        best_score = -1
        best_span = (0, len(chunk.text))

        for window_size in [1, 2, 3, 4]:
            for i in range(len(line_offsets) - window_size + 1):
                win = line_offsets[i:i+window_size]
                text_win = " ".join([w[2] for w in win]).lower()
                if not text_win.strip():
                    continue
                matched = sum(1 for tok in query_tokens if tok in text_win)
                if matched == 0:
                    continue
                # Primary: maximise distinct query tokens matched.
                # Secondary: prefer shorter windows (small length penalty as tiebreaker).
                score = matched - len(text_win) * 0.0005
                if score > best_score:
                    best_score = score
                    best_span = (win[0][0], win[-1][1])

        if best_score > 0:
            rel_start, rel_end = best_span
            refined_start = chunk.start_char + rel_start
            refined_end = chunk.start_char + min(rel_end, len(chunk.text))
            return refined_start, refined_end

        return chunk.start_char, chunk.end_char
