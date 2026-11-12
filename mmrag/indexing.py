"""Turn a list of items into a UnifiedVectorStore (encode images, captions and texts)."""

from __future__ import annotations

import numpy as np

from .captioner import Captioner
from .encoders import DualEncoder
from .fusion import l2norm
from .vector_store import UnifiedVectorStore


def item_captions(item: dict) -> list[str]:
    """All generated captions of an item: item["captions"] (dict name -> text) or the single item["caption"]."""
    caps = item.get("captions")
    if caps:
        return [c for c in caps.values() if c]
    return [item["caption"]] if item.get("caption") else []


def build_store(items: list[dict], encoder: DualEncoder, captioner: Captioner | None = None,
                text_embedder=None) -> UnifiedVectorStore:
    img_rows = [i for i, it in enumerate(items) if it["modality"] == "image"]

    if captioner is not None and img_rows:
        todo = [i for i in img_rows if not items[i].get("caption")]
        if todo:
            caps = captioner.caption([items[i]["path"] for i in todo], show_progress=True)
            for i, c in zip(todo, caps):
                items[i]["caption"] = c

    vectors, rows = {}, {}
    if img_rows:
        vectors["image"] = encoder.encode_images([items[i]["path"] for i in img_rows], show_progress=True)
        rows["image"] = np.array(img_rows, dtype=np.int64)

    cap_rows = [i for i in img_rows if item_captions(items[i])]
    if cap_rows:
        caps = [item_captions(items[i]) for i in cap_rows]
        flat = [c for cs in caps for c in cs]
        enc = encoder.encode_texts(flat, show_progress=True)
        bounds = np.cumsum([0] + [len(cs) for cs in caps])
        vectors["caption"] = l2norm(np.stack([enc[a:b].mean(0) for a, b in zip(bounds[:-1], bounds[1:])]))
        rows["caption"] = np.array(cap_rows, dtype=np.int64)

    txt_rows = [i for i, it in enumerate(items) if it.get("text")]
    if txt_rows:
        vectors["text"] = encoder.encode_texts([items[i]["text"] for i in txt_rows], show_progress=True)
        rows["text"] = np.array(txt_rows, dtype=np.int64)

    if text_embedder is not None:
        if cap_rows:
            vectors["te_caption"] = text_embedder.encode([" ".join(item_captions(items[i])) for i in cap_rows],
                                                         show_progress=True)
            rows["te_caption"] = rows["caption"]
        if txt_rows:
            vectors["te_text"] = text_embedder.encode([items[i]["text"] for i in txt_rows], show_progress=True)
            rows["te_text"] = rows["text"]

    store = UnifiedVectorStore(items, vectors, rows, encoder_name=encoder.name)
    store.text_embedder_name = text_embedder.name if text_embedder is not None else None
    return store
