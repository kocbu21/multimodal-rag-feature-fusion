"""Command-line multimodal search (and optional answer generation).

python scripts/search.py --store indexes/raw_clip-vit-base-patch32 --text "a dog on the beach"
python scripts/search.py --store ... --image query.jpg
python scripts/search.py --store ... --image query.jpg --text "at night" --beta 0.6
python scripts/search.py --store ... --text "a dog on the beach" --strategy hybrid --rerank
python scripts/search.py --store ... --text "what is the dog doing?" --answer
"""

import argparse

import _bootstrap

from mmrag.config import load_config
from mmrag.generator import ClaudeGenerator
from mmrag.pipeline import load_retriever
from mmrag.retriever import STRATEGIES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--text")
    ap.add_argument("--image")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--strategy", choices=list(STRATEGIES), default="auto")
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--beta", type=float, default=None)
    ap.add_argument("--center", action="store_true")
    ap.add_argument("--rerank", action="store_true", help="rerank image candidates with BLIP ITM")
    ap.add_argument("--answer", action="store_true", help="generate an answer with Claude")
    args = ap.parse_args()

    over = {"fusion": {}}
    if args.alpha is not None:
        over["fusion"]["alpha"] = args.alpha
    if args.center:
        over["fusion"]["center_modalities"] = True
    cfg = load_config(overrides=over)
    retriever = load_retriever(args.store, cfg, rerank=args.rerank or None)

    hits = retriever.search(text=args.text, image=args.image, k=args.k, strategy=args.strategy, beta=args.beta)
    for rank, h in enumerate(hits, 1):
        it = h.item
        label = it.get("path") or it["text"][:100].replace("\n", " ")
        extra = f"  caption: {it['caption']}" if it.get("caption") else ""
        src = ", ".join(f"{k}={v:.3f}" for k, v in h.sources.items())
        print(f"{rank:2d}. [{it['modality']}] {it['id']}  score={h.score:.4f}  ({src})\n    {label}{extra}")

    if args.answer:
        print("\n--- answer ---")
        print(ClaudeGenerator(cfg["generator"]["model"]).answer(args.text or "Describe these results.", hits))


if __name__ == "__main__":
    main()
