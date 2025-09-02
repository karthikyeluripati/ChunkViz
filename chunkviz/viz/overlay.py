from typing import List, Optional, Tuple
from ..core.types import Document, Chunk
from PIL import Image, ImageDraw, ImageFont
import fitz  # PyMuPDF
import io, itertools
import os

def _load_page_image(pdf_path: str, page_no: int, zoom: float = 2.0) -> Image.Image:
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_no - 1)
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGBA")
    doc.close()
    return img

def _scale_bbox(bbox, zoom=2.0):
    x0, y0, x1, y1 = bbox
    return (x0*zoom, y0*zoom, x1*zoom, y1*zoom)

def _palette():
    base = [
        (31,119,180), (255,127,14), (44,160,44),
        (214,39,40),  (148,103,189), (140,86,75),
        (227,119,194),(127,127,127),(188,189,34),
        (23,190,207)
    ]
    for c in itertools.cycle(base):
        yield c

def render_page_with_boxes(doc: Document, chunks: List[Chunk], page_no: int, zoom: float = 2.0) -> Optional[Image.Image]:
    if not doc.path.lower().endswith(".pdf"):
        return None

    base = _load_page_image(doc.path, page_no, zoom=zoom)           # RGBA
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))            # transparent layer
    draw = ImageDraw.Draw(overlay)
    colors = _palette()

    page_chunks = [c for c in chunks if c.bboxes and any(bb.get("page")==page_no for bb in c.bboxes)]
    page_chunks.sort(key=lambda c: min(bb["bbox"][1] for bb in c.bboxes if bb.get("page")==page_no))

    for idx, ch in enumerate(page_chunks, start=1):
        col = next(colors)
        fill = (col[0], col[1], col[2], 70)   # <-- translucent
        line = (col[0], col[1], col[2], 255)

        for bb in ch.bboxes:
            if bb.get("page") != page_no:
                continue
            x0, y0, x1, y1 = _scale_bbox(bb["bbox"], zoom=zoom)
            draw.rectangle([x0, y0, x1, y1], outline=line, width=3, fill=fill)

            # label background + text
            label = f"{idx}"
            tx, ty = x0 + 6, y0 + 4
            draw.rectangle([tx-4, ty-2, tx+26, ty+16], fill=(255, 255, 255, 190))
            draw.text((tx, ty), label, fill=(0, 0, 0, 255), font=ImageFont.load_default())

    # alpha-composite the overlay on top of the base
    return Image.alpha_composite(base, overlay)

def render_text_preview(doc: Document, chunks: List[Chunk]) -> Tuple[str, list]:
    preview = doc.text[:800] + ("..." if len(doc.text) > 800 else "")
    spans = [(c.char_span[0], c.char_span[1]) for c in chunks[:10]]
    return preview, spans

def save_thumbnail(doc: Document, chunks: List[Chunk], path: str, page_no:int=1, zoom:float=2.0):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = render_page_with_boxes(doc, chunks, page_no, zoom=zoom) if doc.path.lower().endswith(".pdf") else None
    if img:
        img.save(path)

