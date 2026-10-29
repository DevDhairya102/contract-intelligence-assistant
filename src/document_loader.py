"""
Document Loader for LegalBench-RAG CUAD contract corpus.
Handles UTF-8 text mode loading, exact character indexing, and path normalization.
"""
import os
from pathlib import Path
from typing import Dict, List, Optional

class DocumentLoader:
    """Loads and caches contract files for the 29 CUAD mini contracts."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.documents: Dict[str, str] = {}
        self.canonical_paths: Dict[str, str] = {}
        self._load_all_contracts()

    def _load_all_contracts(self):
        """Scans the corpus/cuad directory and loads each contract into memory."""
        candidates = []
        corpus_dir = self.data_dir / "corpus" / "cuad"
        if corpus_dir.exists():
            candidates.extend(list(corpus_dir.glob("*.txt")))
        
        alt_dir = self.data_dir / "cuad"
        if alt_dir.exists():
            for p in alt_dir.glob("*.txt"):
                if p.name not in [c.name for c in candidates]:
                    candidates.append(p)

        for p in candidates:
            # LegalBench-RAG character offsets strictly assume standard Python text mode with UTF-8
            text = p.read_text(encoding="utf-8")
            filename = p.name
            
            # Map various relative path aliases to the content
            aliases = [
                filename,
                f"cuad/{filename}",
                f"corpus/cuad/{filename}",
                str(p.as_posix()),
            ]
            for alias in aliases:
                self.documents[alias] = text
                self.canonical_paths[alias] = f"cuad/{filename}"

    def get_document(self, file_path: str) -> Optional[str]:
        """Returns the full text of a contract by its relative path or filename."""
        if file_path in self.documents:
            return self.documents[file_path]
        
        # Try finding by basename
        base = Path(file_path).name
        if base in self.documents:
            return self.documents[base]
        
        return None

    def get_slice(self, file_path: str, start: int, end: int) -> str:
        """Retrieves exact character slice document[start:end]."""
        doc = self.get_document(file_path)
        if doc is None:
            raise FileNotFoundError(f"Contract not found in loaded corpus: {file_path}")
        return doc[start:end]

    def get_canonical_path(self, file_path: str) -> str:
        """Returns canonical relative path like 'cuad/filename.txt'."""
        if file_path in self.canonical_paths:
            return self.canonical_paths[file_path]
        return f"cuad/{Path(file_path).name}"

    def list_contracts(self) -> List[str]:
        """Returns list of all canonical contract names."""
        unique_canonical = sorted(list(set(self.canonical_paths.values())))
        return unique_canonical

    def total_contracts(self) -> int:
        return len(self.list_contracts())
