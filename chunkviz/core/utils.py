import hashlib
from .types import Chunk

def stable_chunk_id(doc_id: str, start: int, end: int) -> str:
    return hashlib.sha1(f"{doc_id}:{start}:{end}".encode()).hexdigest()[:12]

def tokens_estimate(text: str) -> int:
    # Simple heuristic (replace later with tiktoken)
    return max(1, len(text.split()) // 0.75)
