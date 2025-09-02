import streamlit as st, os, json
from chunkviz.core.types import Document, Chunk
from chunkviz.core.runio import run_dir, write_json, read_json
from chunkviz.chunkers.fixed import FixedChunker
from chunkviz.chunkers.semantic import SemanticChunker
from chunkviz.chunkers.layout import LayoutChunker
from chunkviz.viz.overlay import render_page_with_boxes, render_text_preview
from chunkviz.retrieval.simple_embed import SimpleEmbedIndex
from chunkviz.retrieval.lexical import LexicalIndex
from chunkviz.retrieval.hybrid import HybridIndex
import pdfplumber

st.set_page_config(page_title="ChunkViz", layout="wide")
st.title("ChunkViz")

tabs = st.tabs(["Overlay", "Search", "Compare"])

# --- Sidebar controls
uploaded = st.sidebar.file_uploader("Upload a .pdf or .txt", type=["pdf","txt"])
chunker_type = st.sidebar.selectbox("Chunker", ["layout", "semantic", "fixed"])
target_tokens = st.sidebar.slider("Target tokens", 100, 1000, 400, 50)
overlap_tokens = st.sidebar.slider("Overlap tokens", 0, 400, 80, 10)
run_name = st.sidebar.text_input("Run name", "demo_run")

doc = None
chunks = []

if uploaded:
    tmp_path = os.path.join(".", "data", "tmp_upload_" + uploaded.name)
    os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
    with open(tmp_path, "wb") as f: f.write(uploaded.read())

    if uploaded.name.lower().endswith(".txt"):
        text = open(tmp_path, "r", encoding="utf-8", errors="ignore").read()
        doc = Document(id=uploaded.name, path=tmp_path, text=text, pages=1, blocks=None)
    else:
        full_text, blocks, pages = [], [], 0
        with pdfplumber.open(tmp_path) as pdf:
            for pageno, page in enumerate(pdf.pages, start=1):
                words = page.extract_words(x_tolerance=2, y_tolerance=2, keep_blank_chars=False)
                lines = {}
                for w in words:
                    top_key = int(round(w["top"]))
                    lines.setdefault(top_key, []).append(w)
                for _, ws in lines.items():
                    ws = sorted(ws, key=lambda x: x["x0"])
                    t = " ".join(w["text"] for w in ws).strip()
                    if t:
                        x0 = min(w["x0"] for w in ws); y0 = min(w["top"] for w in ws)
                        x1 = max(w["x1"] for w in ws); y1 = max(w["bottom"] for w in ws)
                        blocks.append({"page": pageno, "bbox": (x0,y0,x1,y1), "text": t})
                        full_text.append(t)
                pages = pageno
        doc = Document(id=uploaded.name, path=tmp_path, text="\n".join(full_text), pages=pages or 1, blocks=blocks)

    # choose chunker
    if chunker_type == "layout":
        chunker = LayoutChunker()
    elif chunker_type == "semantic":
        chunker = SemanticChunker(target_tokens=target_tokens, overlap_tokens=overlap_tokens)
    else:
        chunker = FixedChunker(target_tokens=target_tokens, overlap_tokens=overlap_tokens)

    chunks = chunker.chunk(doc)

# --- Tab: Overlay
with tabs[0]:
    if doc:
        st.subheader("Preview")
        if doc.path.lower().endswith(".pdf") and chunker_type == "layout":
            page_no = st.number_input("Page", min_value=1, max_value=max(1, doc.pages), value=1, step=1)
            img = render_page_with_boxes(doc, chunks, page_no, zoom=2.0)
            if img: st.image(img, caption=f"Page {page_no} with chunk overlays", use_column_width=True)
        else:
            preview, spans = render_text_preview(doc, chunks)
            st.code(preview)
            st.write(f"Chunks: {len(chunks)} (showing first 10 spans)")
            st.json([{"char_span": s} for s in spans])

        if st.button("💾 Save run"):
            rd = run_dir(run_name)
            write_json(os.path.join(rd, "chunks.json"), [c.__dict__ if isinstance(c, Chunk) else c for c in chunks])
            st.success(f"Saved to runs/{run_name}")

# --- Tab: Search
with tabs[1]:
    st.subheader("Search demo")
    query = st.text_input("Query", "late payment clause")
    backend = st.selectbox("Backend", ["hybrid", "embed", "lexical"])
    topk = st.slider("k", 1, 20, 10)

    if st.button("Search", type="primary"):
        if not chunks: st.warning("Load a document and create chunks first."); 
        else:
            ids = [c.id if isinstance(c, Chunk) else c["id"] for c in chunks]
            texts = [c.text if isinstance(c, Chunk) else c["text"] for c in chunks]
            if backend == "embed":
                idx = SimpleEmbedIndex(); idx.index(ids, texts); res = idx.search(query, k=topk)
            elif backend == "lexical":
                idx = LexicalIndex(); idx.index(ids, texts); res = idx.search(query, k=topk)
            else:
                idx = HybridIndex(SimpleEmbedIndex(), LexicalIndex(), alpha=0.5)
                idx.index(ids, texts); res = idx.search(query, k=topk)
            st.json([{"id": cid, "score": score} for cid,score in res])

# --- Tab: Compare
# --- Tab: Compare
with tabs[2]:
    import pandas as pd

    st.subheader("Compare runs")

    runs = sorted([p for p in os.listdir("runs") if os.path.isdir(os.path.join("runs", p))]) if os.path.exists("runs") else []
    if not runs:
        st.info("No saved runs yet. Save at least two runs from the Overlay tab.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            r1 = st.selectbox("Run A", runs, index=0, key="runA")
        with c2:
            r2 = st.selectbox("Run B", runs, index=min(1, len(runs)-1), key="runB")

        def load_canary(run_name: str):
            path = os.path.join("runs", run_name, "canary.json")
            return read_json(path) if os.path.exists(path) else []

        def summarize(rows):
            if not rows:
                return {"count": 0, "recall@k": 0.0, "mrr": 0.0, "ndcg": 0.0}
            return {
                "count": len(rows),
                "recall@k": round(sum(r.get("recall@k", 0.0) for r in rows) / len(rows), 4),
                "mrr":      round(sum(r.get("mrr", 0.0)       for r in rows) / len(rows), 4),
                "ndcg":     round(sum(r.get("ndcg", 0.0)      for r in rows) / len(rows), 4),
            }

        canaryA = load_canary(r1)
        canaryB = load_canary(r2)
        sA = summarize(canaryA)
        sB = summarize(canaryB)

        # Summary table
        st.markdown("### Summary")
        df = pd.DataFrame(
            {
                "Run": [r1, r2],
                "#Queries": [sA["count"], sB["count"]],
                "Recall@k": [sA["recall@k"], sB["recall@k"]],
                "MRR": [sA["mrr"], sB["mrr"]],
                "NDCG": [sA["ndcg"], sB["ndcg"]],
            }
        )
        st.dataframe(df, hide_index=True, use_container_width=True)

        # Metric bar chart
        st.markdown("### Metrics bar chart")
        metric_choice = st.selectbox("Metric", ["Recall@k", "MRR", "NDCG"], index=0, key="metric_choice")
        chart_df = pd.DataFrame(
            { "Run": [r1, r2], metric_choice: [sA[metric_choice.lower()], sB[metric_choice.lower()]] }
        ).set_index("Run")
        st.bar_chart(chart_df, use_container_width=True)

        # Optional: show raw JSON (collapsible)
        with st.expander("Show raw canary JSON"):
            colA, colB = st.columns(2)
            with colA:
                st.caption(r1)
                st.json(canaryA)
            with colB:
                st.caption(r2)
                st.json(canaryB)

