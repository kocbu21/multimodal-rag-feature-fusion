"""Multimodal search: text query, image query, or image + text query.

Stage 1: dual-encoder search over one or more indexes, merged by late fusion.
Stage 2 (optional): image candidates in the top-K are rescored with a cross-attention ITM model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from .encoders import DualEncoder
from .fusion import fuse_query, fuse_ranked_lists, zscore_rows
from .vector_store import UnifiedVectorStore

STRATEGIES = {
    "image": ["image"],
    "caption": ["caption"],
    "text": ["text"],
    "fused": ["fused"],
    "late": ["image", "caption", "text"],
    "auto": ["fused", "text"],
    "hybrid": ["fused", "text", "te_caption", "te_text"],
}
TE_INDEXES = {"te_caption", "te_text"}


@dataclass
class Hit:
    item: dict
    score: float
    sources: dict[str, float] = field(default_factory=dict)

    def __repr__(self) -> str:
        label = self.item.get("path") or (self.item.get("text") or "")[:60]
        return f"Hit({self.item['id']}, {self.score:.4f}, {label!r})"


class MultimodalRetriever:
    def __init__(self, store: UnifiedVectorStore, encoder: DualEncoder, cfg: dict,
                 text_embedder=None, reranker=None):
        self.store, self.encoder, self.cfg = store, encoder, cfg
        self.text_embedder, self.reranker = text_embedder, reranker

    def embed_query(self, text: str | None = None, image: Image.Image | str | None = None,
                    beta: float | None = None) -> np.ndarray:
        if text is None and image is None:
            raise ValueError("Provide a text query, an image query, or both.")
        beta = self.cfg["fusion"]["beta"] if beta is None else beta
        q_txt = self.store.center(self.encoder.encode_texts([text]), "text") if text else None
        q_img = self.store.center(self.encoder.encode_images([image]), "image") if image is not None else None
        return fuse_query(q_img, q_txt, beta)

    def _weights(self) -> dict[str, float]:
        w = self.cfg["fusion"].get("text_weight", 0.2)
        return {"fused": 1 - w, "text": 1 - w, "te_caption": w, "te_text": w,
                "image": 1.0, "caption": 1.0}

    def search(self, text: str | None = None, image: Image.Image | str | None = None, k: int | None = None,
               strategy: str = "auto", beta: float | None = None, method: str | None = None,
               rerank: bool | None = None) -> list[Hit]:
        scfg, fcfg = self.cfg["search"], self.cfg["fusion"]
        k = k or scfg["top_k"]
        query = self.embed_query(text, image, beta)
        te_query = self.text_embedder.encode([text]) if (text and self.text_embedder is not None) else None

        names = [n for n in STRATEGIES[strategy] if n in self.store.indexes
                 and (n not in TE_INDEXES or te_query is not None)]
        if not names:
            return []
        if method is None:
            method = "weighted" if strategy == "hybrid" else fcfg["late_method"]

        rr_cfg = self.cfg.get("reranker", {})
        do_rerank = (rr_cfg.get("enabled", False) if rerank is None else rerank) and self.reranker is not None and text
        stage1_k = max(k, rr_cfg.get("top_k", 16)) if do_rerank else k
        cand_k = max(stage1_k, scfg.get("candidate_k", 100)) if len(names) > 1 else stage1_k

        lists: dict[str, list[tuple[int, float]]] = {}
        for name in names:
            q = te_query if name in TE_INDEXES else query
            scores, rows = self.store.indexes[name].search(q, cand_k)
            lists[name] = list(zip(rows[0].tolist(), scores[0].tolist()))

        if len(names) == 1:
            ranked = lists[names[0]]
        else:
            ranked = fuse_ranked_lists(lists, method=method, k=fcfg.get("rrf_k", 60), weights=self._weights())
        ranked = ranked[:stage1_k]

        raw = {n: dict(lst) for n, lst in lists.items()}
        hits = [Hit(item=self.store.items[row], score=float(score),
                    sources={n: raw[n][row] for n in names if row in raw[n]}) for row, score in ranked]
        if do_rerank:
            hits = self._rerank(text, hits, rr_cfg.get("weight", 0.7))
        return hits[:k]

    def _rerank(self, text: str, hits: list[Hit], lam: float) -> list[Hit]:
        """Rescore image hits with ITM; they are re-ordered among the slots image hits already hold."""
        slots = [i for i, h in enumerate(hits) if h.item["modality"] == "image"]
        if len(slots) < 2:
            return hits
        img_hits = [hits[i] for i in slots]
        itm = self.reranker.score_pairs([text] * len(img_hits), [h.item["path"] for h in img_hits])
        first = np.array([h.score for h in img_hits], dtype=np.float32)
        comb = lam * zscore_rows(itm[None])[0] + (1 - lam) * zscore_rows(first[None])[0]
        for h, s, c in zip(img_hits, itm, comb):
            h.sources["itm"] = float(s)
            h.score = float(c)
        out = list(hits)
        for slot, h in zip(slots, sorted(img_hits, key=lambda h: h.score, reverse=True)):
            out[slot] = h
        return out
