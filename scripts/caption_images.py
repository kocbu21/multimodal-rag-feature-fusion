"""Generate captions with any captioner into the shared cache (data/caption_cache.json).

python scripts/caption_images.py --captioner Salesforce/blip-image-captioning-large --device cpu
python scripts/caption_images.py --captioner Qwen/Qwen2.5-VL-3B-Instruct
"""

import argparse
import json

import _bootstrap

from mmrag.captioner import build_captioner
from mmrag.config import load_config, resolve
from mmrag.data import load_coco5k, load_flickr1k, load_raw_corpus

CACHE = "data/caption_cache.json"


def load_cache() -> dict:
    p = resolve(CACHE)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def save_cache(new_entries: dict) -> None:
    """Merge into the file on disk, so several captioning jobs can run side by side."""
    cache = load_cache()
    cache.update(new_entries)
    tmp = resolve(CACHE + ".tmp")
    tmp.write_text(json.dumps(cache, ensure_ascii=False, indent=0), encoding="utf-8")
    tmp.replace(resolve(CACHE))


def captions_for(items: list[dict], captioner_name: str) -> list[str | None]:
    cache = load_cache()
    return [cache.get(f"{captioner_name}|{it['path']}") for it in items]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captioner", required=True)
    ap.add_argument("--source", choices=["flickr1k", "coco5k", "raw"], default="flickr1k")
    ap.add_argument("--device", default=None)
    ap.add_argument("--chunk", type=int, default=40, help="save the cache every N images (resumable)")
    args = ap.parse_args()

    over = {"captioner": {"name": args.captioner, "enabled": True}}
    if args.device:
        over["captioner"]["device"] = args.device
    cfg = load_config(overrides=over)
    loaders = {"flickr1k": lambda: load_flickr1k()[0], "coco5k": lambda: load_coco5k()[0], "raw": load_raw_corpus}
    items = loaders[args.source]()
    paths = [it["path"] for it in items if it["modality"] == "image"]

    cache = load_cache()
    todo = [p for p in paths if f"{args.captioner}|{p}" not in cache]
    print(f"{len(paths) - len(todo)} cached, {len(todo)} to caption with {args.captioner}")
    if not todo:
        return
    captioner = build_captioner(cfg)
    for i in range(0, len(todo), args.chunk):
        chunk = todo[i : i + args.chunk]
        new = {f"{args.captioner}|{p}": c for p, c in zip(chunk, captioner.caption(chunk))}
        save_cache(new)
        c = next(iter(new.values()))
        print(f"  {min(i + args.chunk, len(todo))}/{len(todo)}  e.g. {c!r}", flush=True)


if __name__ == "__main__":
    main()
