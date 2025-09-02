from typing import Protocol, Iterable, List
from .types import Document, Chunk, Query, Retrieval

class Chunker(Protocol):
    name: str
    def chunk(self, doc: Document) -> List[Chunk]: ...

class Retriever(Protocol):
    name: str
    def index(self, chunks: Iterable[Chunk]) -> None: ...
    def search(self, query: Query, k: int = 10) -> Retrieval: ...

class Reranker(Protocol):
    name: str
    def rerank(self, query: Query, chunks: List[Chunk]) -> List[Chunk]: ...

class Visualizer(Protocol):
    def overlay(self, doc: Document, chunks: List[Chunk]) -> None: ...
