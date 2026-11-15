"""Build the unified vector database.

Examples
--------
python scripts/build_index.py --source raw
python scripts/build_index.py --source flickr1k --encoder google/siglip-base-patch16-224
"""

import argparse
import json
import re

import _bootstrap
from caption_images import save_cache

from mmrag.captioner import build_captioner
from mmrag.config import load_config, resolve
from mmrag.data import load_coco5k, load_flickr1k, load_raw_corpus
from mmrag.encoders import build_encoder
from mmrag.indexing import build_store


def default_store_dir(source: str, encoder: str) -> str:
    return f"indexes/{source}_{re.sub(r'[^a-zA-Z0-9]+', '-', encoder.split('/')[-1])}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["raw", "flickr1k", "coco5k"], default="raw")
    ap.add_argument("--config", default=None)
    ap.add_argument("--encoder", default=None, help="override encoder.name")
    ap.add_argument("--out", default=None, help="output directory (default: indexes/<source>_<encoder>)")
    ap.add_argument("--no-caption", action="store_true")
    ap.add_argument("--captioner", default=None, help="override captioner.name (e.g. a VLM)")
    ap.add_argument("--single-caption", action="store_true", help="ignore captioner.extra (one caption set)")
    ap.add_argument("--caption-batch-size", type=int, default=None, help="override captioner.batch_size")
    ap.add_argument("--text-embedder", action="store_true", help="also store BGE vectors (hybrid strategy)")
    args = ap.parse_args()

    overrides = {}
    if args.encoder:
        overrides["encoder"] = {"name": args.encoder}
    if args.no_caption:
        overrides["captioner"] = {"enabled": False}
    if args.captioner:
        overrides.setdefault("captioner", {})["name"] = args.captioner
    if args.caption_batch_size:
        overrides.setdefault("captioner", {})["batch_size"] = args.caption_batch_size
    if args.single_caption:
        overrides.setdefault("captioner", {})["extra"] = []
    if args.text_embedder:
        overrides["text_embedder"] = {"enabled": True}
    cfg = load_config(args.config, overrides)

    loaders = {"flickr1k": lambda: load_flickr1k()[0], "coco5k": lambda: load_coco5k()[0], "raw": load_raw_corpus}
    items = loaders[args.source]()
    if not items:
        raise SystemExit("No items found. Put images in data/raw/images and texts in data/raw/texts.")
    print(f"{len(items)} items ({sum(i['modality'] == 'image' for i in items)} images)")

    if cfg["captioner"].get("enabled", True):
        names = [cfg["captioner"]["name"]] + [n for n in cfg["captioner"].get("extra") or []
                                               if n != cfg["captioner"]["name"]]
        cache_path = resolve("data/caption_cache.json")
        cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
        images = [it for it in items if it["modality"] == "image"]
        for name in names:
            todo = [it for it in images if f"{name}|{it['path']}" not in cache]
            if todo:
                captioner = build_captioner({**cfg, "captioner": {**cfg["captioner"], "name": name}})
                caps = captioner.caption([it["path"] for it in todo], show_progress=True)
                new = {f"{name}|{it['path']}": c for it, c in zip(todo, caps)}
                del captioner
                save_cache(new)
                cache.update(new)
        for it in images:
            it["caption"] = cache[f"{names[0]}|{it['path']}"]
            if len(names) > 1:
                it["captions"] = {n: cache[f"{n}|{it['path']}"] for n in names}

    encoder = build_encoder(cfg)
    text_embedder = None
    if cfg["text_embedder"].get("enabled", False):
        from mmrag.text_embedder import TextEmbedder
        text_embedder = TextEmbedder(cfg["text_embedder"]["name"], cfg["encoder"].get("device", "cuda"),
                                     sleep_s=cfg["throttle"]["gpu_sleep_s"])
    store = build_store(items, encoder, text_embedder=text_embedder)
    out = resolve(args.out or default_store_dir(args.source, cfg["encoder"]["name"]))
    store.save(out)
    print(store)
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
