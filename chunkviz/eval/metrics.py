from typing import List, Sequence, Dict, Any
import numpy as np

def recall_at_k(gold_ids: Sequence[str], ranked_ids: Sequence[str], k: int = 10) -> float:
    if not gold_ids: 
        return 0.0
    return len(set(gold_ids) & set(ranked_ids[:k])) / max(1, len(set(gold_ids)))

def mrr(ranked_ids: Sequence[str], gold_ids: Sequence[str]) -> float:
    gold = set(gold_ids)
    for idx, cid in enumerate(ranked_ids, start=1):
        if cid in gold:
            return 1.0 / idx
    return 0.0

def ndcg_at_k(gold_ids: Sequence[str], ranked_ids: Sequence[str], k:int=10) -> float:
    gold = set(gold_ids)
    dcg = 0.0
    for i, cid in enumerate(ranked_ids[:k], start=1):
        rel = 1.0 if cid in gold else 0.0
        if rel:
            dcg += rel / np.log2(i + 1)
    # ideal DCG
    ideal = sum(1.0 / np.log2(i + 1) for i in range(1, min(k, len(gold)) + 1))
    return dcg / ideal if ideal > 0 else 0.0

def cost_tokens(prompt_tokens:int, completion_tokens:int, prices:Dict[str,Dict[str,float]], model:str) -> float:
    p = prices.get(model, {"prompt":0.0, "completion":0.0})
    return prompt_tokens * p["prompt"] + completion_tokens * p["completion"]

def summarize_latencies(latencies_ms: Sequence[float]) -> Dict[str, float]:
    arr = np.array(latencies_ms) if len(latencies_ms) else np.array([0.0])
    return {
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "mean_ms": float(arr.mean())
    }
