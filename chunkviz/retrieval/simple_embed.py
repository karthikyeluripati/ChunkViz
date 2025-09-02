from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer

class SimpleEmbedIndex:
    """
    Minimal embedding index (cosine) in memory.
    """
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.ids: List[str] = []
        self.texts: List[str] = []
        self.vecs: np.ndarray | None = None

    def index(self, ids: List[str], texts: List[str]):
        self.ids = ids
        self.texts = texts
        self.vecs = self.model.encode(texts, normalize_embeddings=True)

    def search(self, query: str, k: int = 10):
        if self.vecs is None or not len(self.ids):
            return []
        q = self.model.encode([query], normalize_embeddings=True)[0]
        scores = self.vecs @ q
        order = np.argsort(-scores)[:k]
        return [(self.ids[i], float(scores[i])) for i in order]
