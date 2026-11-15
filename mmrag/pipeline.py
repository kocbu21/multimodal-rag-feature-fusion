"""One place to assemble store + encoders + optional text embedder / reranker into a retriever."""

from __future__ import annotations

from .config import resolve
from .encoders import build_encoder
from .retriever import MultimodalRetriever
from .vector_store import UnifiedVectorStore


def load_retriever(store_dir: str, cfg: dict, rerank: bool | None = None) -> MultimodalRetriever:
    store = UnifiedVectorStore.load(resolve(store_dir), alpha=cfg["fusion"]["alpha"],
                                    center=cfg["fusion"]["center_modalities"],
                                    hub_beta=cfg["fusion"].get("hub_beta", 0.0) if cfg["fusion"].get("hub_rho", 0.0) else 0.0)
    cfg["encoder"]["name"] = store.encoder_name
    sleep_s = cfg.get("throttle", {}).get("gpu_sleep_s", 0.0)
    device = cfg["encoder"].get("device", "cuda")

    text_embedder = None
    if store.text_embedder_name:
        from .text_embedder import TextEmbedder
        text_embedder = TextEmbedder(store.text_embedder_name, device, sleep_s=sleep_s)

    reranker = None
    use_rr = cfg.get("reranker", {}).get("enabled", False) if rerank is None else rerank
    if use_rr:
        from .reranker import ITMReranker
        reranker = ITMReranker(cfg["reranker"]["name"], device, sleep_s=sleep_s)
        cfg["reranker"]["enabled"] = True

    return MultimodalRetriever(store, build_encoder(cfg), cfg, text_embedder=text_embedder, reranker=reranker)
