from typing import List, Tuple

class HybridIndex:
    """
    Alpha-blend of embedding + lexical scores: score = alpha*emb + (1-alpha)*lex
    """
    def __init__(self, embed_index, lexical_index, alpha: float = 0.5):
        self.embed = embed_index
        self.lex = lexical_index
        self.alpha = alpha
        self.ids: List[str] = []

    def index(self, ids: List[str], texts: List[str]):
        self.ids = ids
        self.embed.index(ids, texts)
        self.lex.index(ids, texts)

    def search(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        emb = dict(self.embed.search(query, k=max(k,50)))
        lex = dict(self.lex.search(query, k=max(k,50)))
        # union of ids
        all_ids = set(emb.keys()) | set(lex.keys())
        blended = []
        for cid in all_ids:
            e = emb.get(cid, 0.0); l = lex.get(cid, 0.0)
            blended.append((cid, self.alpha*e + (1-self.alpha)*l))
        blended.sort(key=lambda x: -x[1])
        return blended[:k]
