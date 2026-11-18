"""Unified vector database: one FAISS index per vector type, shared item IDs and metadata.

Vector types
------------
image   : image encoder output for image items
caption : text encoder output of the BLIP caption of an image item
text    : text encoder output for text items (documents, notes) and image descriptions
fused   : derived on load, norm(alpha * image + (1 - alpha) * caption)   (early fusion)
te_caption / te_text : optional, captions / texts in a dedicated text-embedding space (BGE),
          searched with the query's text-embedding vector and merged by late fusion
v2 (optional, see build_index.py --v2):
region  : several vectors per image (grid crops + Florence-2 object boxes); an image scores by its best region
itc     : 32 BLIP-2 Q-Former vectors per image (ITC head); searched with the BLIP-2 ITC text embedding
te_tags : Florence-2 object tags of each image in BGE space
bank_*  : generated captions used as a query bank for caption-bank hubness normalisation (not searched)

Raw vectors are saved as .npy so fused variants (different alpha, centering) can be
rebuilt without re-running the encoders.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import faiss
import numpy as np

from .fusion import early_fuse, l2norm

VECTOR_TYPES = ("image", "caption", "text")
MULTI_VECTOR = {"region", "itc"}
BANK_FOR = {"fused": "bank_dual", "region": "bank_dual", "te_caption": "bank_te", "te_tags": "bank_te",
            "itc": "bank_itc"}


@dataclass
class VectorIndex:
    vectors: np.ndarray
    item_rows: np.ndarray
    index: faiss.Index = field(init=False)

    def __post_init__(self):
        self.vectors = np.ascontiguousarray(self.vectors, dtype=np.float32)
        self.index = faiss.IndexFlatIP(self.vectors.shape[1])
        if len(self.vectors):
            self.index.add(self.vectors)

    def __len__(self) -> int:
        return len(self.item_rows)

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Returns (scores, item_rows) of shape (n_queries, k'), k' = min(k, len)."""
        k = min(k, len(self))
        if k == 0:
            n = len(queries)
            return np.zeros((n, 0), np.float32), np.zeros((n, 0), np.int64)
        scores, idx = self.index.search(np.ascontiguousarray(queries, dtype=np.float32), k)
        return scores, self.item_rows[idx]


class UnifiedVectorStore:
    def __init__(self, items: list[dict], vectors: dict[str, np.ndarray], rows: dict[str, np.ndarray],
                 encoder_name: str = ""):
        self.items = items
        self.encoder_name = encoder_name
        self.text_embedder_name: str | None = None
        self.raw = {t: vectors[t] for t in vectors}
        self.rows = {t: rows[t] for t in rows}
        self.means: dict[str, np.ndarray] = {}
        self.indexes: dict[str, VectorIndex] = {}
        self.alpha = 0.5
        self.centered = False
        self.hub_beta = 0.0
        self.hub_L: dict[str, np.ndarray] = {}
        self.rebuild()

    def rebuild(self, alpha: float | None = None, center: bool | None = None, hub_beta: float | None = None) -> None:
        """(Re)build FAISS indexes, optionally with modality centering, a new fusion alpha and hubness normalisers."""
        if alpha is not None:
            self.alpha = alpha
        if center is not None:
            self.centered = center
        if hub_beta is not None:
            self.hub_beta = hub_beta

        vecs = dict(self.raw)
        self.means = {}
        if self.centered:
            if len(vecs.get("image", [])):
                self.means["image"] = vecs["image"].mean(0)
            text_side = [vecs[t] for t in ("caption", "text") if len(vecs.get(t, []))]
            if text_side:
                self.means["text"] = np.concatenate(text_side).mean(0)
            vecs = {t: (self.center(v, "image" if t == "image" else "text") if t in VECTOR_TYPES else v)
                    for t, v in vecs.items()}

        self.indexes = {t: VectorIndex(v, self.rows[t]) for t, v in vecs.items()
                        if len(v) and not t.startswith("bank_")}

        if "image" in self.indexes and "caption" in self.indexes:
            cap_by_row = dict(zip(self.rows["caption"].tolist(), range(len(self.rows["caption"]))))
            keep = [i for i, r in enumerate(self.rows["image"]) if r in cap_by_row]
            img = vecs["image"][keep]
            cap = vecs["caption"][[cap_by_row[self.rows["image"][i]] for i in keep]]
            self.indexes["fused"] = VectorIndex(early_fuse(img, cap, self.alpha), self.rows["image"][keep])
        self.compute_hubness()

    def compute_hubness(self) -> None:
        """Caption-bank hubness normaliser L[row] per searchable index (see fusion.bank_log_normaliser).

        The bank holds generated captions of the gallery images; L is large for hub items that attract the
        captions of many *other* images. The retriever subtracts rho / beta * (L - mean L) from each score."""
        self.hub_L = {}
        if not getattr(self, "hub_beta", 0):
            return
        from .fusion import bank_log_normaliser, max_pool_by_owner

        n = len(self.items)
        for name, bank_type in BANK_FOR.items():
            if name not in self.indexes or bank_type not in self.raw:
                continue
            ix = self.indexes[name]
            bank, bank_rows = self.raw[bank_type], self.rows[bank_type]
            uniq, owner = np.unique(ix.item_rows, return_inverse=True)
            step = max(1, int(2e8 // max(1, len(ix.vectors))))
            S = np.concatenate([max_pool_by_owner(bank[i : i + step] @ ix.vectors.T, owner, len(uniq))
                                for i in range(0, len(bank), step)])
            bank_owner = np.searchsorted(uniq, bank_rows)
            bank_owner = np.where(uniq[np.clip(bank_owner, 0, len(uniq) - 1)] == bank_rows, bank_owner, -1)
            L = bank_log_normaliser(S, bank_owner, self.hub_beta)
            full = np.full(n, float(L.mean()), np.float32)
            full[uniq] = L
            self.hub_L[name] = full - float(L.mean())

    def center(self, x: np.ndarray, modality: str) -> np.ndarray:
        """Subtract the modality mean (no-op unless centering is on) and re-normalise."""
        if not self.centered or modality not in self.means or len(x) == 0:
            return x
        return l2norm(x - self.means[modality])

    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "items.jsonl", "w", encoding="utf-8") as f:
            for it in self.items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        for t in self.raw:
            np.save(d / f"{t}_vectors.npy", self.raw[t])
            np.save(d / f"{t}_rows.npy", self.rows[t])
        (d / "store.json").write_text(json.dumps({"encoder": self.encoder_name,
                                                  "text_embedder": self.text_embedder_name,
                                                  "types": sorted(self.raw)}, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path, alpha: float = 0.5, center: bool = False,
             hub_beta: float = 0.0) -> "UnifiedVectorStore":
        d = Path(directory)
        meta = json.loads((d / "store.json").read_text(encoding="utf-8"))
        with open(d / "items.jsonl", encoding="utf-8") as f:
            items = [json.loads(line) for line in f if line.strip()]
        vectors = {t: np.load(d / f"{t}_vectors.npy") for t in meta["types"]}
        rows = {t: np.load(d / f"{t}_rows.npy") for t in meta["types"]}
        store = cls.__new__(cls)
        store.items, store.encoder_name = items, meta["encoder"]
        store.text_embedder_name = meta.get("text_embedder")
        store.raw, store.rows = vectors, rows
        store.alpha, store.centered = alpha, center
        store.hub_beta, store.hub_L = hub_beta, {}
        store.rebuild()
        return store

    def __repr__(self) -> str:
        sizes = ", ".join(f"{t}={len(ix)}" for t, ix in self.indexes.items())
        return f"UnifiedVectorStore(items={len(self.items)}, {sizes}, encoder={self.encoder_name!r})"
