"""Retrieval metrics on dense score matrices."""

from __future__ import annotations

import numpy as np


def gt_ranks(scores: np.ndarray, positives: list[set[int]] | np.ndarray) -> np.ndarray:
    """Best (0-based) rank of any positive item for each query row."""
    order = np.argsort(-scores, axis=1)
    ranks = np.empty(len(scores), dtype=np.int64)
    for q in range(len(scores)):
        pos = positives[q]
        pos = pos if isinstance(pos, (set, frozenset)) else {int(pos)}
        hit = np.flatnonzero(np.isin(order[q], list(pos)))
        ranks[q] = hit[0] if len(hit) else scores.shape[1]
    return ranks


def retrieval_metrics(scores: np.ndarray, positives, ks=(1, 5, 10)) -> dict[str, float]:
    r = gt_ranks(scores, positives)
    out = {f"R@{k}": float((r < k).mean() * 100) for k in ks}
    out["RSUM"] = sum(out[f"R@{k}"] for k in ks)
    out["MRR"] = float((1.0 / (r + 1)).mean() * 100)
    out["MedR"] = float(np.median(r) + 1)
    return out
