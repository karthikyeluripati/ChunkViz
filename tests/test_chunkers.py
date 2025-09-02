import os, json
from chunkviz.core.types import Document
from chunkviz.chunkers.fixed import FixedChunker
from chunkviz.chunkers.layout import LayoutChunker
from chunkviz.chunkers.semantic import SemanticChunker

def test_fixed_chunker_smoke():
    doc = Document(id="t1", path="t1.txt", text="Hello world. " * 100, pages=1, blocks=None)
    ch = FixedChunker(target_tokens=20, overlap_tokens=5)
    chunks = ch.chunk(doc)
    assert chunks, "FixedChunker should return chunks"

def test_layout_chunker_smoke():
    blocks = [{"page": 1, "bbox": (0,0,10,10), "text": "This is a line"} for _ in range(5)]
    doc = Document(id="t2", path="t2.pdf", text="\n".join(b["text"] for b in blocks), pages=1, blocks=blocks)
    ch = LayoutChunker(max_lines=3, min_chars=5)
    chunks = ch.chunk(doc)
    assert all(c.text for c in chunks)

def test_semantic_chunker_smoke():
    doc = Document(id="t3", path="t3.txt", text="Sentence one. Sentence two. Sentence three.", pages=1, blocks=None)
    ch = SemanticChunker(target_tokens=30, overlap_tokens=5)
    chunks = ch.chunk(doc)
    assert chunks
