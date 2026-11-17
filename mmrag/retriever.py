"""Multimodal search: text query, image query, or image + text query.

Stage 1: dual-encoder search over one or more indexes, merged by late fusion.
Stage 2 (optional): image candidates in the top-K are rescored with a cross-attention ITM model.

v2 ("full" strategy, stores built with build_index.py --v2): extra channels for region crops, object tags and
BLIP-2 ITC, caption-bank hubness normalisation of every channel, and an ITM + ITC combination in stage 2.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from .encoders import DualEncoder
from .fusion import collapse_max, fuse_query, fuse_ranked_lists, zscore_rows
from .vector_store import MULTI_VECTOR, UnifiedVectorStore

STRATEGIES = {
    "image": ["image"],
    "caption": ["caption"],
    "text": ["text"],
    "fused": ["fused"],
    "late": ["image", "caption", "text"],
    "auto": ["fused", "text"],
    "hybrid": ["fused", "text", "te_caption", "te_text"],
    "full": ["fused", "text", "te_caption", "te_text", "te_tags", "region", "itc"],
}
TE_INDEXES = {"te_caption", "te_text", "te_tags"}


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
                 text_embedder=None, reranker=None, extra_rerankers=None, vlm=None):
        self.store, self.encoder, self.cfg = store, encoder, cfg
        self.text_embedder, self.reranker = text_embedder, reranker
        self.extra_rerankers = extra_rerankers or []   # [(ITMReranker, weight)], v3
        self.vlm = vlm                                 # (VLMScorer, top_m, weight), v3

    def embed_query(self, text: str | None = None, image: Image.Image | str | None = None,
                    beta: float | None = None) -> np.ndarray:
        if text is None and image is None:
            raise ValueError("Provide a text query, an image query, or both.")
        beta = self.cfg["fusion"]["beta"] if beta is None else beta
        q_txt = self.store.center(self.encoder.encode_texts([text]), "text") if text else None
        q_img = self.store.center(self.encoder.encode_images([image]), "image") if image is not None else None
        return fuse_query(q_img, q_txt, beta)

    def _weights(self, strategy: str = "hybrid") -> dict[str, float]:
        if strategy == "full" and self.cfg["fusion"].get("full_weights"):
            fw = dict(self.cfg["fusion"]["full_weights"])
            fw.setdefault("text", fw.get("fused", 0.5))
            fw.setdefault("te_text", fw.get("te_caption", 0.2))
            return fw
        w = self.cfg["fusion"].get("text_weight", 0.2)
        return {"fused": 1 - w, "text": 1 - w, "te_caption": w, "te_text": w,
                "image": 1.0, "caption": 1.0}

    def _itc_ok(self) -> bool:
        return self.reranker is not None and getattr(self.reranker, "blip2", False)

    def search(self, text: str | None = None, image: Image.Image | str | None = None, k: int | None = None,
               strategy: str = "auto", beta: float | None = None, method: str | None = None,
               rerank: bool | None = None) -> list[Hit]:
        scfg, fcfg = self.cfg["search"], self.cfg["fusion"]
        k = k or scfg["top_k"]
        query = self.embed_query(text, image, beta)
        te_query = self.text_embedder.encode([text]) if (text and self.text_embedder is not None) else None

        names = [n for n in STRATEGIES[strategy] if n in self.store.indexes
                 and (n not in TE_INDEXES or te_query is not None)
                 and (n != "itc" or (text and self._itc_ok()))]
        if not names:
            return []
        if method is None:
            method = "weighted" if strategy in ("hybrid", "full") else fcfg["late_method"]
        itc_query = self.reranker.itc_text_embeds([text]) if "itc" in names else None
        hub_rho = fcfg.get("hub_rho", 0.0) if self.store.hub_L else 0.0

        rr_cfg = self.cfg.get("reranker", {})
        do_rerank = (rr_cfg.get("enabled", False) if rerank is None else rerank) and self.reranker is not None and text
        stage1_k = max(k, rr_cfg.get("top_k", 16)) if do_rerank else k
        cand_k = max(stage1_k, scfg.get("candidate_k", 100)) if len(names) > 1 else stage1_k

        lists: dict[str, list[tuple[int, float]]] = {}
        for name in names:
            q = te_query if name in TE_INDEXES else (itc_query if name == "itc" else query)
            ix = self.store.indexes[name]
            if name in MULTI_VECTOR:   # several vectors per item: over-fetch, keep each item's best vector
                per_item = max(1, len(ix) // max(1, len(set(ix.item_rows.tolist()))))
                scores, rows = ix.search(q, cand_k * per_item)
                hits = collapse_max(list(zip(rows[0].tolist(), scores[0].tolist())))[:cand_k]
            else:
                scores, rows = ix.search(q, cand_k)
                hits = list(zip(rows[0].tolist(), scores[0].tolist()))
            L = self.store.hub_L.get(name) if hub_rho else None
            if L is not None:   # caption-bank hubness normalisation
                beta = self.store.hub_beta
                hits = sorted(((r, s_ - hub_rho / beta * float(L[r])) for r, s_ in hits), key=lambda h: -h[1])
            lists[name] = hits

        if len(names) == 1:
            ranked = lists[names[0]]
        else:
            ranked = fuse_ranked_lists(lists, method=method, k=fcfg.get("rrf_k", 60), weights=self._weights(strategy))
        ranked = ranked[:stage1_k]

        raw = {n: dict(lst) for n, lst in lists.items()}
        hits = [Hit(item=self.store.items[row], score=float(score),
                    sources={n: raw[n][row] for n in names if row in raw[n]}) for row, score in ranked]
        if do_rerank:
            hits = self._rerank(text, hits, rr_cfg.get("weight", 0.7), rr_cfg.get("itc_weight", 0.0))
        return hits[:k]

    def _rerank(self, text: str, hits: list[Hit], lam: float, lam_itc: float = 0.0) -> list[Hit]:
        """Rescore image hits with ITM; they are re-ordered among the slots image hits already hold."""
        slots = [i for i, h in enumerate(hits) if h.item["modality"] == "image"]
        if len(slots) < 2:
            return hits
        img_hits = [hits[i] for i in slots]
        img_paths = [h.item["path"] for h in img_hits]
        itm = self.reranker.score_pairs([text] * len(img_hits), img_paths)
        first = np.array([h.score for h in img_hits], dtype=np.float32)
        lam_itc = lam_itc if self._itc_ok() else 0.0
        comb = lam * zscore_rows(itm[None])[0] + (1 - lam - lam_itc) * zscore_rows(first[None])[0]
        if lam_itc:   # BLIP-2 ITC similarity of the same pairs (image features are already cached by score_pairs)
            t = self.reranker.itc_text_embeds([text])[0]
            itc = (self.reranker.itc_image_embeds(img_paths) @ t).max(1)
            comb = comb + lam_itc * zscore_rows(itc[None])[0]
            for h, x in zip(img_hits, itc):
                h.sources["itc_rr"] = float(x)
        for rr, g in self.extra_rerankers:   # v3: further cross-encoders, mixed in one at a time
            if g:
                p = rr.score_pairs([text] * len(img_hits), img_paths)
                comb = (1 - g) * zscore_rows(comb[None])[0] + g * zscore_rows(p[None])[0]
                for h, x in zip(img_hits, p):
                    h.sources[f"itm_{rr.blip2 and 'blip2' or 'blip'}_x"] = float(x)
        if self.vlm is not None and self.vlm[2]:   # v3: VLM yes/no verification of the top-M only
            scorer, top_m, gv = self.vlm
            top = np.argsort(-comb)[:top_m]
            v = scorer.score_pairs([text] * len(top), [img_paths[i] for i in top])
            comb = comb.copy()
            comb[top] = 1e3 + (1 - gv) * zscore_rows(comb[top][None])[0] + gv * zscore_rows(v[None])[0]
            for i, x in zip(top, v):
                img_hits[i].sources["vlm"] = float(x)
        for h, s, c in zip(img_hits, itm, comb):
            h.sources["itm"] = float(s)
            h.score = float(c)
        out = list(hits)
        for slot, h in zip(slots, sorted(img_hits, key=lambda h: h.score, reverse=True)):
            out[slot] = h
        return out
