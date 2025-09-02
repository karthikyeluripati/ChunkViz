from typing import List, Dict, Any, Tuple
from ..core.types import Document, Chunk
from ..core.utils import stable_chunk_id
import statistics
import math

class LayoutChunker:
    """
    Column-aware chunker that groups adjacent lines into visual chunks.
    - Filters narrow sidebars (e.g., arXiv stamps)
    - Splits pages into columns (k=1 or k=2) via x0 clustering
    - Packs lines into chunks of up to max_lines
    """
    name = "layout"

    def __init__(self,
                 max_lines: int = 8,
                 min_chars: int = 40,
                 min_width_ratio: float = 0.25,   # drop lines narrower than 25% of page width
                 column_gap_ratio: float = 0.12): # if median x0 gap > 12% page width, treat as 2 columns
        self.max_lines = max(1, max_lines)
        self.min_chars = max(0, min_chars)
        self.min_width_ratio = min_width_ratio
        self.column_gap_ratio = column_gap_ratio

    def _page_blocks(self, blocks: List[Dict[str, Any]], page_w: float) -> List[Dict[str, Any]]:
        """Filter out narrow sidebars / stamps and empty lines."""
        filtered = []
        for b in blocks:
            x0, y0, x1, y1 = b["bbox"]
            width = max(1.0, x1 - x0)
            if not b["text"].strip():
                continue
            if width / page_w < self.min_width_ratio:
                # likely margin note / page label; skip
                continue
            filtered.append(b)
        return filtered

    def _split_columns(self, blks: List[Dict[str, Any]], page_w: float) -> List[List[Dict[str, Any]]]:
        """Heuristic 1- or 2-column split based on x0 gap."""
        if not blks:
            return []
        xs = sorted([b["bbox"][0] for b in blks])
        gaps = [xs[i+1]-xs[i] for i in range(len(xs)-1)]
        if not gaps:
            return [blks]
        median_gap = statistics.median(gaps)
        # If the left edge gap is big relative to page width, assume 2 columns
        if median_gap / page_w > self.column_gap_ratio:
            # split by mid x
            mid_x = statistics.median(xs)
            left  = [b for b in blks if b["bbox"][0] <= mid_x]
            right = [b for b in blks if b["bbox"][0] >  mid_x]
            return [sorted(left, key=lambda x:(x["bbox"][1], x["bbox"][0])),
                    sorted(right,key=lambda x:(x["bbox"][1], x["bbox"][0]))]
        else:
            return [sorted(blks, key=lambda x:(x["bbox"][1], x["bbox"][0]))]

    def _pack_lines(self, lines: List[Dict[str, Any]], doc_text: str, doc_id: str, page_no: int) -> List[Chunk]:
        chunks: List[Chunk] = []
        i = 0
        n = len(lines)
        while i < n:
            group = lines[i:i+self.max_lines]
            text = "\n".join(g["text"] for g in group)
            if len(text) < self.min_chars and (i+self.max_lines) < n:
                # extend a little if too short
                group = lines[i:i+self.max_lines+2]
                text = "\n".join(g["text"] for g in group)
            x0 = min(g["bbox"][0] for g in group); y0 = min(g["bbox"][1] for g in group)
            x1 = max(g["bbox"][2] for g in group); y1 = max(g["bbox"][3] for g in group)

            start_char = doc_text.find(group[0]["text"])
            end_char = start_char + len(text) if start_char >= 0 else 0
            cid = stable_chunk_id(doc_id, start_char if start_char>=0 else 0, end_char)

            chunks.append(Chunk(
                id=cid,
                doc_id=doc_id,
                text=text,
                page_span=(page_no, page_no),
                char_span=(start_char if start_char>=0 else 0, end_char),
                tokens=len(text.split()),
                bboxes=[{"page": page_no, "bbox": (x0,y0,x1,y1)}],
            ))
            i += self.max_lines
        return chunks

    def chunk(self, doc: Document) -> List[Chunk]:
        if not doc.blocks:
            cid = stable_chunk_id(doc.id, 0, len(doc.text))
            return [Chunk(id=cid, doc_id=doc.id, text=doc.text, tokens=len(doc.text.split()))]

        # Group by page and compute approximate page width from bboxes
        by_page: Dict[int, List[Dict[str, Any]]] = {}
        page_width: Dict[int, float] = {}
        for b in doc.blocks:
            p = b["page"]
            by_page.setdefault(p, []).append(b)
            x0, y0, x1, y1 = b["bbox"]
            page_width[p] = max(page_width.get(p, 0.0), x1)  # rough width

        out: List[Chunk] = []
        for pageno, blks in by_page.items():
            w = max(1.0, page_width.get(pageno, 1000.0))
            cleaned = self._page_blocks(blks, page_w=w)
            for column_lines in self._split_columns(cleaned, page_w=w):
                out.extend(self._pack_lines(column_lines, doc.text, doc.id, pageno))
        return out
