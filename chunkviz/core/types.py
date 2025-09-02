from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

BBox = Tuple[float, float, float, float]  # x0, y0, x1, y1 in PDF coords

@dataclass
class Document:
    id: str
    path: str
    text: str
    pages: int = 1
    # blocks: optional layout info (mainly for PDFs)
    # each block: {"page": int, "bbox": (x0,y0,x1,y1), "text": str}
    blocks: Optional[List[Dict[str, Any]]] = None

@dataclass
class Chunk:
    id: str
    doc_id: str
    text: str
    page_span: Tuple[int, int] = (1, 1)
    char_span: Tuple[int, int] = (0, 0)
    tokens: int = 0
    # optional layout bboxes, one per page region covered by this chunk
    bboxes: Optional[List[Dict[str, Any]]] = None
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Query:
    id: str
    text: str
    gold_spans: Optional[List[Dict[str, Any]]] = None
    tags: Optional[List[str]] = None

@dataclass
class Retrieval:
    query_id: str
    chunk_ids: List[str]
    scores: List[float]
    latency_ms: float

@dataclass
class RunConfig:
    name: str
    chunker: Dict[str, Any]
    retriever: Dict[str, Any] | None = None
    reranker: Dict[str, Any] | None = None
    metrics: Dict[str, Any] | None = None
