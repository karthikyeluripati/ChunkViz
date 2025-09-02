from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class LexicalIndex:
    """
    TF-IDF cosine as a lightweight lexical retriever (BM25-ish).
    """
    def __init__(self):
        self.vec = None
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.ids: List[str] = []
        self.texts: List[str] = []

    def index(self, ids: List[str], texts: List[str]):
        self.ids = ids
        self.texts = texts
        self.vec = self.vectorizer.fit_transform(texts)

    def search(self, query: str, k: int = 10) -> List[Tuple[str,float]]:
        if self.vec is None or not len(self.ids):
            return []
        qv = self.vectorizer.transform([query])
        scores = (self.vec @ qv.T).toarray().ravel()
        order = np.argsort(-scores)[:k]
        return [(self.ids[i], float(scores[i])) for i in order]
