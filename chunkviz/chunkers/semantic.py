from typing import List
from ..core.types import Document, Chunk
from ..core.utils import stable_chunk_id
import nltk

# Robust downloads for NLTK >= 3.9
def _ensure_nltk():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab")

_ensure_nltk()

from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
import numpy as np


class SemanticChunker:
    name = "semantic"

    def __init__(self, target_tokens: int = 400, overlap_tokens: int = 80,
                 merge_sim_threshold: float = 0.82,
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.n = max(1, target_tokens)
        self.o = max(0, overlap_tokens)
        self.merge_sim = merge_sim_threshold
        self.model = SentenceTransformer(model_name)

    def _cos(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)

    def chunk(self, doc: Document) -> List[Chunk]:
        sents = sent_tokenize(doc.text)
        if not sents:
            return []
        embs = self.model.encode(sents, normalize_embeddings=True)
        chunks: List[Chunk] = []

        # greedy merge sentences until approx token budget (1 token ~ 1.33 words heuristic)
        i = 0
        while i < len(sents):
            cur = [sents[i]]
            cur_emb = embs[i]
            cur_words = len(sents[i].split())
            j = i + 1
            while j < len(sents):
                if cur_words >= int(self.n * 1.33):
                    break
                sim = self._cos(cur_emb, embs[j])
                if sim >= self.merge_sim or cur_words < int(self.n * 0.6 * 1.33):
                    cur.append(sents[j])
                    cur_emb = (cur_emb + embs[j]) / 2.0
                    cur_words += len(sents[j].split())
                    j += 1
                else:
                    break
            text = " ".join(cur)
            start_char = doc.text.find(cur[0])
            end_char = start_char + len(text) if start_char >= 0 else 0
            cid = stable_chunk_id(doc.id, start_char if start_char>=0 else 0, end_char)
            chunks.append(Chunk(
                id=cid, doc_id=doc.id, text=text,
                char_span=(start_char if start_char>=0 else 0, end_char),
                tokens=int(cur_words/1.33)
            ))
            # overlap: back up some sentences
            back = 0
            if self.o > 0:
                # approximate overlap as some fraction of sentences
                back = max(0, int(len(cur) * (self.o / max(1,self.n))))
            i = max(i + len(cur) - back, i + 1)
        return chunks
