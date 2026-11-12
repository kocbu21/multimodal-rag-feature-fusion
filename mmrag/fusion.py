"""Visual-textual feature fusion strategies.

Three levels where vision and text can be combined:

1. Document side (early fusion): an image item is represented by
   norm(alpha * v_image + (1 - alpha) * v_caption). alpha=1 -> pure image, alpha=0 -> pure caption.
2. Query side: an image + text query becomes norm(beta * q_image + (1 - beta) * q_text).
3. Score side (late fusion): ranked lists from several indexes are merged with
   weighted normalised scores or Reciprocal Rank Fusion (RRF).

Functions work on dense score matrices (n_queries, n_items) so the evaluation code and
the online retriever share one implementation.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


def l2norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12, None)


def early_fuse(image_vecs: np.ndarray, caption_vecs: np.ndarray, alpha: float) -> np.ndarray:
    return l2norm(alpha * image_vecs + (1.0 - alpha) * caption_vecs)


def fuse_query(image_vec: np.ndarray | None, text_vec: np.ndarray | None, beta: float) -> np.ndarray:
    if image_vec is None:
        return l2norm(text_vec)
    if text_vec is None:
        return l2norm(image_vec)
    return l2norm(beta * image_vec + (1.0 - beta) * text_vec)


def minmax_rows(s: np.ndarray) -> np.ndarray:
    lo, hi = s.min(axis=1, keepdims=True), s.max(axis=1, keepdims=True)
    return (s - lo) / np.clip(hi - lo, 1e-12, None)


def zscore_rows(s: np.ndarray) -> np.ndarray:
    return (s - s.mean(axis=1, keepdims=True)) / np.clip(s.std(axis=1, keepdims=True), 1e-12, None)


NORMALISERS = {"minmax": minmax_rows, "zscore": zscore_rows, "none": lambda s: s}


def weighted_late_fusion(scores: Mapping[str, np.ndarray], weights: Mapping[str, float],
                         norm: str = "zscore") -> np.ndarray:
    """Weighted sum of per-row normalised score matrices. Raw cosine scores of different
    modality pairs live on different scales (the CLIP modality gap), hence the normalisation."""
    f = NORMALISERS[norm]
    total = sum(weights[k] for k in scores)
    return sum(weights[k] / total * f(s) for k, s in scores.items())


def ranks_from_scores(s: np.ndarray) -> np.ndarray:
    """0-based rank of every column within its row (0 = best)."""
    order = np.argsort(-s, axis=1)
    ranks = np.empty_like(order)
    rows = np.arange(s.shape[0])[:, None]
    ranks[rows, order] = np.arange(s.shape[1])[None, :]
    return ranks


def rrf_fusion(scores: Sequence[np.ndarray] | Mapping[str, np.ndarray], k: int = 60,
               weights: Mapping[str, float] | None = None) -> np.ndarray:
    """Reciprocal Rank Fusion on dense score matrices: sum_m w_m / (k + rank_m + 1)."""
    items = scores.items() if isinstance(scores, Mapping) else enumerate(scores)
    out = None
    for key, s in items:
        w = 1.0 if weights is None else weights.get(key, 1.0)
        contrib = w / (k + ranks_from_scores(s) + 1.0)
        out = contrib if out is None else out + contrib
    return out


def fuse_ranked_lists(lists: Mapping[str, list[tuple[int, float]]], method: str = "rrf", k: int = 60,
                      weights: Mapping[str, float] | None = None) -> list[tuple[int, float]]:
    """Fuse top-k lists [(item_row, score), ...] coming from different indexes.

    Used online, where only the top candidates of each index are available.
    """
    weights = weights or {}
    agg: dict[int, float] = {}
    for name, hits in lists.items():
        if not hits:
            continue
        w = weights.get(name, 1.0)
        if method == "rrf":
            for rank, (row, _) in enumerate(hits):
                agg[row] = agg.get(row, 0.0) + w / (k + rank + 1.0)
        elif method == "weighted":
            vals = np.array([s for _, s in hits], dtype=np.float32)
            lo, hi = float(vals.min()), float(vals.max())
            for row, s in hits:
                agg[row] = agg.get(row, 0.0) + w * ((s - lo) / (hi - lo) if hi > lo else 1.0)
        else:
            raise ValueError(f"Unknown late fusion method: {method}")
    return sorted(agg.items(), key=lambda kv: kv[1], reverse=True)
