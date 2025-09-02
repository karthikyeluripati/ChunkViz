# chunkviz/cli/app.py
from __future__ import annotations

import os
import json
import time
import typer
from typing import Optional, List, Dict, Any

from ..core.types import Document, Chunk
from ..core.runio import run_dir, write_json, read_json
from ..chunkers.fixed import FixedChunker
from ..eval.metrics import recall_at_k, mrr, ndcg_at_k
from ..viz.overlay import save_thumbnail

app = typer.Typer(help="ChunkViz CLI")


# -----------------------------
# Ingest (.txt and .pdf)
# -----------------------------
def _ingest_txt(path: str) -> Document:
    txt = open(path, "r", encoding="utf-8").read()
    return Document(id=os.path.basename(path), path=path, text=txt, pages=1, blocks=None)


def _ingest_pdf(path: str) -> Document:
    import pdfplumber

    full_text: List[str] = []
    blocks: List[Dict[str, Any]] = []
    pages = 1
    with pdfplumber.open(path) as pdf:
        for pageno, page in enumerate(pdf.pages, start=1):
            # words with bbox
            words = page.extract_words(
                x_tolerance=2, y_tolerance=2, keep_blank_chars=False
            )
            # group words into lines by 'top'
            lines_map: Dict[int, List[Dict[str, Any]]] = {}
            for w in words:
                top_key = int(round(w["top"]))
                lines_map.setdefault(top_key, []).append(w)
            # build blocks per line
            for _, ws in lines_map.items():
                ws = sorted(ws, key=lambda x: x["x0"])
                text = " ".join(w["text"] for w in ws).strip()
                if not text:
                    continue
                x0 = min(w["x0"] for w in ws)
                y0 = min(w["top"] for w in ws)
                x1 = max(w["x1"] for w in ws)
                y1 = max(w["bottom"] for w in ws)
                blocks.append({"page": pageno, "bbox": (x0, y0, x1, y1), "text": text})
                full_text.append(text)
            pages = pageno

    return Document(
        id=os.path.basename(path),
        path=path,
        text="\n".join(full_text),
        pages=pages if blocks else 1,
        blocks=blocks or None,
    )


@app.command()
def ingest(input_path: str, out: str = "data/parquet"):
    """
    Ingest .txt or .pdf into data/parquet/docs.json
    """
    os.makedirs(out, exist_ok=True)
    docs: List[Dict[str, Any]] = []

    def _add(p: str):
        if p.lower().endswith(".txt"):
            docs.append(_ingest_txt(p).__dict__)
        elif p.lower().endswith(".pdf"):
            docs.append(_ingest_pdf(p).__dict__)

    if os.path.isdir(input_path):
        for fn in sorted(os.listdir(input_path)):
            _add(os.path.join(input_path, fn))
    else:
        _add(input_path)

    json.dump(
        docs, open(os.path.join(out, "docs.json"), "w", encoding="utf-8"),
        ensure_ascii=False, indent=2
    )
    typer.echo(f"Ingested {len(docs)} docs -> {out}/docs.json")


# -----------------------------
# Chunk (with --run-name)
# -----------------------------
@app.command()
def chunk(
    config: str = "data/configs/demo.yaml",
    out: str = "data/parquet/chunks.json",
    run_name: Optional[str] = None,
):
    """
    Chunk docs.json using fixed/semantic/layout (from YAML config).
    Optionally save to runs/<run_name>/chunks.json and thumbnails.
    """
    import yaml

    cfg = yaml.safe_load(open(config, "r", encoding="utf-8"))
    chunker_cfg = cfg.get("chunker", {})
    ctype = chunker_cfg.get("type", "fixed")

    if ctype == "semantic":
        from ..chunkers.semantic import SemanticChunker

        chunker = SemanticChunker(
            target_tokens=chunker_cfg.get("target_tokens", 400),
            overlap_tokens=chunker_cfg.get("overlap_tokens", 80),
            merge_sim_threshold=chunker_cfg.get("similarity_merge", 0.82),
            model_name=chunker_cfg.get(
                "embedder", "sentence-transformers/all-MiniLM-L6-v2"
            ),
        )
    elif ctype == "layout":
        from ..chunkers.layout import LayoutChunker

        chunker = LayoutChunker(
            max_lines=chunker_cfg.get("max_lines", 8),
            min_chars=chunker_cfg.get("min_chars", 40),
        )
    else:
        chunker = FixedChunker(
            target_tokens=chunker_cfg.get("target_tokens", 400),
            overlap_tokens=chunker_cfg.get("overlap_tokens", 80),
        )

    docs_path = "data/parquet/docs.json"
    if not os.path.exists(docs_path):
        typer.secho("docs.json not found. Run `chunkviz ingest ...` first.", fg="red")
        raise typer.Exit(code=1)

    docs = json.load(open(docs_path, "r", encoding="utf-8"))
    all_chunks: List[Dict[str, Any]] = []

    for d in docs:
        doc = Document(**d)
        t0 = time.time()
        doc_chunks = [c.__dict__ for c in chunker.chunk(doc)]
        dt_ms = (time.time() - t0) * 1000.0
        all_chunks += doc_chunks

        # Optional thumbnail for PDFs
        if run_name and doc.path.lower().endswith(".pdf"):
            thumb_path = os.path.join("runs", run_name, f"{doc.id}_p1.png")
            save_thumbnail(doc, [Chunk(**c) for c in doc_chunks], path=thumb_path, page_no=1)

        typer.echo(f"Chunked {doc.id}: {len(doc_chunks)} chunks in {dt_ms:.1f} ms")

    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(all_chunks, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    typer.echo(f"→ {out} ({len(all_chunks)} chunks)")

    if run_name:
        rd = run_dir(run_name)
        write_json(os.path.join(rd, "chunks.json"), all_chunks)
        with open(os.path.join(rd, "config_path.txt"), "w", encoding="utf-8") as f:
            f.write(config)
        typer.secho(f"Saved run → runs/{run_name}", fg="green")


# -----------------------------
# Index (embed / lexical / hybrid)
# -----------------------------
@app.command()
def index(
    run_name: str,
    backend: str = "hybrid",
    alpha: float = 0.5,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
):
    """
    Build an in-memory index for a saved run.
    (This persists only small meta; Streamlit/CLI rebuilds on demand.)
    """
    run_path = os.path.join("runs", run_name, "chunks.json")
    if not os.path.exists(run_path):
        typer.secho(f"Run not found: {run_path}", fg="red")
        raise typer.Exit(code=1)

    chunks = read_json(run_path)
    ids = [c["id"] for c in chunks]
    texts = [c["text"] for c in chunks]

    if backend == "embed":
        from ..retrieval.simple_embed import SimpleEmbedIndex

        idx = SimpleEmbedIndex(model_name=model_name)
        idx.index(ids, texts)
    elif backend == "lexical":
        from ..retrieval.lexical import LexicalIndex

        idx = LexicalIndex()
        idx.index(ids, texts)
    else:
        from ..retrieval.simple_embed import SimpleEmbedIndex
        from ..retrieval.lexical import LexicalIndex
        from ..retrieval.hybrid import HybridIndex

        idx = HybridIndex(SimpleEmbedIndex(model_name=model_name), LexicalIndex(), alpha=alpha)
        idx.index(ids, texts)

    write_json(os.path.join("runs", run_name, "index_meta.json"),
               {"backend": backend, "alpha": alpha, "model": model_name})
    typer.echo(f"Indexed {len(ids)} chunks with backend={backend}")


# -----------------------------
# Search
# -----------------------------
@app.command()
def search(
    run_name: str,
    query: str,
    k: int = 10,
    backend: str = "hybrid",
    alpha: float = 0.5,
):
    """
    Search a run's chunks using embed/lexical/hybrid.
    """
    run_path = os.path.join("runs", run_name, "chunks.json")
    if not os.path.exists(run_path):
        typer.secho(f"Run not found: {run_path}", fg="red")
        raise typer.Exit(code=1)

    chunks = read_json(run_path)
    id2chunk = {c["id"]: c for c in chunks}
    ids = [c["id"] for c in chunks]
    texts = [c["text"] for c in chunks]

    if backend == "embed":
        from ..retrieval.simple_embed import SimpleEmbedIndex

        idx = SimpleEmbedIndex()
        idx.index(ids, texts)
        results = idx.search(query, k=k)
    elif backend == "lexical":
        from ..retrieval.lexical import LexicalIndex

        idx = LexicalIndex()
        idx.index(ids, texts)
        results = idx.search(query, k=k)
    else:
        from ..retrieval.simple_embed import SimpleEmbedIndex
        from ..retrieval.lexical import LexicalIndex
        from ..retrieval.hybrid import HybridIndex

        idx = HybridIndex(SimpleEmbedIndex(), LexicalIndex(), alpha=alpha)
        idx.index(ids, texts)
        results = idx.search(query, k=k)

    out = [{"id": cid, "score": score, "text": id2chunk[cid]["text"][:200]} for cid, score in results]
    typer.echo(json.dumps(out, ensure_ascii=False, indent=2))


# -----------------------------
# Canary eval
# -----------------------------
@app.command()
def canary(run_name: str, queries: str = "data/queries.jsonl", k: int = 10, backend: str = "hybrid"):
    """
    Run a small canary set against the current run index and write runs/<run>/canary.json.
    queries.jsonl lines should be: {"text": "...", "gold_ids": ["chunk_id1", ...]}
    """
    run_path = os.path.join("runs", run_name, "chunks.json")
    if not os.path.exists(run_path):
        typer.secho(f"Run not found: {run_path}", fg="red")
        raise typer.Exit(code=1)

    chunks = read_json(run_path)
    ids = [c["id"] for c in chunks]
    texts = [c["text"] for c in chunks]

    # build index
    if backend == "embed":
        from ..retrieval.simple_embed import SimpleEmbedIndex
        idx = SimpleEmbedIndex(); idx.index(ids, texts)
        search_fn = lambda q: idx.search(q, k=k)
    elif backend == "lexical":
        from ..retrieval.lexical import LexicalIndex
        idx = LexicalIndex(); idx.index(ids, texts)
        search_fn = lambda q: idx.search(q, k=k)
    else:
        from ..retrieval.simple_embed import SimpleEmbedIndex
        from ..retrieval.lexical import LexicalIndex
        from ..retrieval.hybrid import HybridIndex
        idx = HybridIndex(SimpleEmbedIndex(), LexicalIndex(), alpha=0.5)
        idx.index(ids, texts)
        search_fn = lambda q: idx.search(q, k=k)

    recs = []
    with open(queries, "r", encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            gold = q.get("gold_ids", [])
            ranked = [cid for cid, _ in search_fn(q["text"])]
            recs.append({
                "query": q["text"],
                "recall@k": recall_at_k(gold, ranked, k),
                "mrr": mrr(ranked, gold),
                "ndcg": ndcg_at_k(gold, ranked, k),
            })

    rd = run_dir(run_name)
    write_json(os.path.join(rd, "canary.json"), recs)
    typer.echo(json.dumps(recs, indent=2))


# -----------------------------
# Report (very basic HTML)
# -----------------------------
@app.command()
def report(run: str, compare: Optional[str] = None, out: Optional[str] = None):
    """
    Write an HTML report with summary stats, thumbnails (if present), and optional compare.
    """
    from statistics import mean
    rd = run_dir(run)
    chunks_path = os.path.join(rd, "chunks.json")
    if not os.path.exists(chunks_path):
        typer.secho(f"No chunks at {chunks_path}", fg="red"); raise typer.Exit(code=1)

    chunks = read_json(chunks_path)
    canary_path = os.path.join(rd, "canary.json")
    canary_data = read_json(canary_path) if os.path.exists(canary_path) else []

    # summaries
    def summarize(rows):
        if not rows: return {"count": 0, "recall_k": 0, "mrr": 0, "ndcg": 0}
        return {
            "count": len(rows),
            "recall_k": round(mean(r.get("recall@k", 0.0) for r in rows), 4),
            "mrr": round(mean(r.get("mrr", 0.0) for r in rows), 4),
            "ndcg": round(mean(r.get("ndcg", 0.0) for r in rows), 4),
        }

    s1 = summarize(canary_data)

    cmp_block = ""
    if compare:
        rd2 = run_dir(compare)
        canary2 = os.path.join(rd2, "canary.json")
        data2 = read_json(canary2) if os.path.exists(canary2) else []
        s2 = summarize(data2)
        cmp_block = f"""
        <h2>Compare vs <code>{compare}</code></h2>
        <table class="kpi">
          <tr><th></th><th>{run}</th><th>{compare}</th></tr>
          <tr><td>#Queries</td><td>{s1['count']}</td><td>{s2['count']}</td></tr>
          <tr><td>Avg Recall@k</td><td>{s1['recall_k']}</td><td>{s2['recall_k']}</td></tr>
          <tr><td>Avg MRR</td><td>{s1['mrr']}</td><td>{s2['mrr']}</td></tr>
          <tr><td>Avg NDCG</td><td>{s1['ndcg']}</td><td>{s2['ndcg']}</td></tr>
        </table>
        <details><summary>Raw canary JSON</summary>
          <div class="json"><h4>{run}</h4><pre>{json.dumps(canary_data, indent=2)}</pre>
          <h4>{compare}</h4><pre>{json.dumps(data2, indent=2)}</pre></div>
        </details>
        """

    # simple thumbnail gallery (if you saved any during chunking)
    thumbs = [f for f in os.listdir(rd) if f.lower().endswith(".png")]
    thumbs_html = ""
    if thumbs:
        thumbs_html = "<h2>Thumbnails</h2><div class='thumbs'>" + "".join(
            f"<figure><img src='{t}'/><figcaption>{t}</figcaption></figure>" for t in sorted(thumbs)
        ) + "</div>"

    html = f"""
    <html>
    <head>
      <meta charset="utf-8" />
      <title>ChunkViz Report – {run}</title>
      <style>
        body {{ font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; margin: 24px; }}
        h1, h2 {{ margin-bottom: 8px; }}
        .kpi td, .kpi th {{ padding: 6px 10px; border-bottom: 1px solid #eee; text-align: left; }}
        .json pre {{ background:#fafafa; border:1px solid #eee; padding:12px; border-radius:8px; overflow:auto; }}
        .thumbs {{ display:flex; flex-wrap:wrap; gap:12px; }}
        .thumbs img {{ max-width: 360px; height:auto; border:1px solid #ddd; border-radius:8px; }}
        figure {{ margin:0; }}
        figcaption {{ font-size: 12px; color: #666; margin-top: 4px; }}
        .grid {{ display:grid; grid-template-columns: repeat(2, minmax(240px, 1fr)); gap: 8px; }}
        code {{ background:#f3f3f3; padding:2px 6px; border-radius:4px; }}
      </style>
    </head>
    <body>
      <h1>ChunkViz Report – <code>{run}</code></h1>
      <div class="grid">
        <div><strong>Total Chunks</strong><div>{len(chunks)}</div></div>
        <div><strong># Canary Queries</strong><div>{s1['count']}</div></div>
        <div><strong>Avg Recall@k</strong><div>{s1['recall_k']}</div></div>
        <div><strong>Avg MRR</strong><div>{s1['mrr']}</div></div>
        <div><strong>Avg NDCG</strong><div>{s1['ndcg']}</div></div>
      </div>

      {thumbs_html}

      <h2>Canary (raw)</h2>
      <div class="json"><pre>{json.dumps(canary_data, indent=2)}</pre></div>

      {cmp_block}
    </body>
    </html>
    """

    out_path = out or os.path.join(rd, "report.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    typer.echo(f"Wrote {out_path}")


@app.command("make-canary")
def make_canary(run_name: str, out: str = "data/canary.jsonl", n: int = 5):
    """
    Build a small synthetic canary set from a run's chunks:
    takes the first N chunks, uses their first ~10 words as the query,
    and sets gold_ids to the chunk id.
    """
    import os, json
    from ..core.runio import read_json
    path = os.path.join("runs", run_name, "chunks.json")
    if not os.path.exists(path):
        typer.secho(f"No chunks at {path}. Save a run first.", fg="red")
        raise typer.Exit(code=1)
    chunks = read_json(path)
    n = max(1, min(n, len(chunks)))
    items = []
    for c in chunks[:n]:
        words = c["text"].split()[:10]
        items.append({"text": " ".join(words), "gold_ids": [c["id"]]})
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    typer.echo(f"Wrote {out} ({n} queries)")

@app.command("make-canary-hard")
def make_canary_hard(run_name: str, out: str = "data/canary_hard.jsonl", n: int = 20, drop_ratio: float = 0.25):
    """
    Hard canary: create paraphrase-ish queries by deleting random words and shuffling,
    so lexical match is weaker. Gold ids remain the true chunk ids.
    """
    import os, json, random, re
    from ..core.runio import read_json

    path = os.path.join("runs", run_name, "chunks.json")
    if not os.path.exists(path):
        typer.secho(f"No chunks at {path}. Save a run first.", fg="red")
        raise typer.Exit(code=1)

    chunks = read_json(path)
    n = max(1, min(n, len(chunks)))
    rng = random.Random(1337)

    def fuzz(text: str) -> str:
        words = re.findall(r"\w+(?:-\w+)?", text)  # keep hyphenated tokens
        if not words:
            return text
        # sample a span that’s not the very beginning to avoid exact title match
        start = rng.randrange(0, max(1, len(words)-10))
        span = words[start:start+16]
        # drop some words
        keep = [w for w in span if rng.random() > drop_ratio]
        if len(keep) < 6:
            keep = span[:6]
        # slight shuffle in the middle to weaken order features
        mid = keep[1:-1]
        rng.shuffle(mid)
        q = [keep[0]] + mid + ([keep[-1]] if len(keep) > 1 else [])
        return " ".join(q)

    items = []
    for c in chunks[:n]:
        q = fuzz(c["text"])
        items.append({"text": q, "gold_ids": [c["id"]]})

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    typer.echo(f"Wrote {out} ({n} harder queries)")



def main():
    app()


if __name__ == "__main__":
    main()
