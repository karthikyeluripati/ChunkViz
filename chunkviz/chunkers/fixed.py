from typing import List
from ..core.types import Document, Chunk
from ..core.utils import stable_chunk_id

class FixedChunker:
    """Token-approximate fixed-size chunker with overlap (MVP uses word heuristic)."""
    name = "fixed"

    def __init__(self, target_tokens: int = 400, overlap_tokens: int = 80):
        self.n = max(1, target_tokens)
        self.o = max(0, overlap_tokens)

    def chunk(self, doc: Document) -> List[Chunk]:
        words = doc.text.split()
        chunks: List[Chunk] = []
        i = 0
        # Approx: 1 token ~ 1.33 words (heuristic)
        step = max(1, int(self.n * 1.33) - int(self.o * 1.33))
        win = max(1, int(self.n * 1.33))
        while i < len(words):
            window_words = words[i:i+win]
            text = " ".join(window_words)
            start_char = len(" ".join(words[:i])) + (1 if i > 0 else 0)
            end_char = start_char + len(text)
            cid = stable_chunk_id(doc.id, start_char, end_char)
            chunks.append(Chunk(id=cid, doc_id=doc.id, text=text, char_span=(start_char, end_char), tokens=len(window_words)))
            i += step
        return chunks
