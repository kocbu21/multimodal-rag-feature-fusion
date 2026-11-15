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


# ---------------------------------------------------------------------------------------------
# v2 additions: multi-vector items, caption-bank hubness normalisation, fast fused scoring
# ---------------------------------------------------------------------------------------------

def early_fused_scores(A: np.ndarray, B: np.ndarray, cos_ic: np.ndarray, alpha: float) -> np.ndarray:
    """<q, norm(alpha*v_I + (1-alpha)*v_C)> computed from A = Q @ I.T, B = Q @ C.T and cos_ic[g] = <v_I,g, v_C,g>,
    without materialising the fused vectors (cheap to sweep alpha on large galleries)."""
    norm = np.sqrt(np.clip(alpha ** 2 + (1 - alpha) ** 2 + 2 * alpha * (1 - alpha) * cos_ic, 1e-12, None))
    return ((alpha * A + (1 - alpha) * B) / norm[None, :]).astype(np.float32)


def max_pool_by_owner(scores: np.ndarray, owner: np.ndarray, n_items: int, fill: float = -1.0) -> np.ndarray:
    """(n_q, n_vectors) scores of multi-vector items -> (n_q, n_items) max over each item's vectors."""
    out = np.full((scores.shape[0], n_items), fill, dtype=np.float32)
    order = np.argsort(owner, kind="stable")
    owner_sorted = owner[order]
    starts = np.flatnonzero(np.r_[True, owner_sorted[1:] != owner_sorted[:-1]])
    out[:, owner_sorted[starts]] = np.maximum.reduceat(scores[:, order], starts, axis=1)
    return out


def bank_log_normaliser(bank_scores: np.ndarray, bank_owner: np.ndarray, beta: float) -> np.ndarray:
    """Caption-bank hubness normaliser (inverted softmax, QB-Norm style, but with a *generated* query bank).

    bank_scores[b, g] is the similarity of bank query b (a generated caption of image bank_owner[b]) to gallery
    item g. For every gallery item g it returns L_g = log mean_{b: owner(b) != g} exp(beta * s(b, g)), i.e. how
    strongly g attracts descriptions of *other* images. Hub images get a large L_g. No test query is used.
    """
    z = beta * bank_scores.astype(np.float64)
    n_items = bank_scores.shape[1]
    own = bank_owner[:, None] == np.arange(n_items)[None, :]
    z = np.where(own, -np.inf, z)
    m = z.max(axis=0)
    lse = m + np.log(np.exp(z - m[None, :]).sum(axis=0))
    count = (~own).sum(axis=0)
    return (lse - np.log(np.clip(count, 1, None))).astype(np.float32)


def apply_hubness(scores: np.ndarray, log_norm: np.ndarray, beta: float, rho: float) -> np.ndarray:
    """s'(q, g) = s(q, g) - rho * L_g / beta. rho=1 is the exact inverted softmax, rho=0 switches it off."""
    if rho == 0:
        return scores
    return scores - (rho / beta) * (log_norm - log_norm.mean())[None, :]


def dual_softmax(scores: np.ndarray, temp: float = 0.01) -> np.ndarray:
    """Dual softmax over the *test queries* (transductive: uses all test queries at once). Reported only as a
    clearly labelled ablation, never in the main comparison."""
    z = scores / temp
    z = z - z.max(axis=0, keepdims=True)
    prior = np.exp(z)
    prior /= prior.sum(axis=0, keepdims=True)
    return scores * prior


def collapse_max(hits: list[tuple[int, float]]) -> list[tuple[int, float]]:
    """Keep the best score per item row in a ranked list that may contain several vectors of one item."""
    best: dict[int, float] = {}
    for row, s in hits:
        if row not in best or s > best[row]:
            best[row] = s
    return sorted(best.items(), key=lambda kv: kv[1], reverse=True)
