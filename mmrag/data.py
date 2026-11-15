"""Datasets and corpus loading.

Sources:
* `flickr1k` - Flickr30k 1K test split (1000 images x 5 captions), the standard
  image-text retrieval benchmark used for evaluation.
* `coco5k`   - MS-COCO 2014 5K test split (5000 images x 5 captions), a second benchmark.
* `raw`      - your own files: data/raw/images/* (an optional same-named .txt next to an
  image is used as its description) and data/raw/texts/*.txt|*.md (chunked by paragraph).
"""

from __future__ import annotations

import ast
import csv
import json
import zipfile
from pathlib import Path

from .config import resolve

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
FLICKR_REPO = "nlphuji/flickr_1k_test_image_text_retrieval"
COCO_REPO = "nlphuji/mscoco_2014_5k_test_image_text_retrieval"


def _prepare_hf_benchmark(repo: str, csv_name: str, zip_name: str, out_dir: str | Path) -> Path:
    from huggingface_hub import hf_hub_download

    out = resolve(out_dir)
    ann_path = out / "annotations.json"
    if ann_path.exists():
        return out
    out.mkdir(parents=True, exist_ok=True)
    csv_path = hf_hub_download(repo, csv_name, repo_type="dataset")
    zip_path = hf_hub_download(repo, zip_name, repo_type="dataset")

    img_dir = out / "images"
    img_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if Path(name).suffix.lower() in IMAGE_EXTS:
                (img_dir / Path(name).name).write_bytes(zf.read(name))

    records = []
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            records.append({"filename": row["filename"], "captions": ast.literal_eval(row["raw"])})
    missing = [r["filename"] for r in records if not (img_dir / r["filename"]).exists()]
    if missing:
        raise RuntimeError(f"{len(missing)} images missing after extraction, e.g. {missing[:3]}")
    ann_path.write_text(json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def prepare_flickr1k(out_dir: str | Path = "data/flickr1k") -> Path:
    """Download the Flickr30k 1K test split and write `annotations.json`."""
    return _prepare_hf_benchmark(FLICKR_REPO, "test_1k_flickr.csv", "images_flickr_1k_test.zip", out_dir)


def prepare_coco5k(out_dir: str | Path = "data/coco5k") -> Path:
    """Download the MS-COCO 2014 5K test split and write `annotations.json`."""
    return _prepare_hf_benchmark(COCO_REPO, "test_5k_mscoco_2014.csv", "images_mscoco_2014_5k_test.zip", out_dir)


def _load_benchmark(root: Path, prefix: str, source: str) -> tuple[list[dict], list[str], list[int]]:
    records = json.loads((root / "annotations.json").read_text(encoding="utf-8"))
    items, queries, query_to_image = [], [], []
    for i, r in enumerate(records):
        items.append({"id": f"{prefix}:{Path(r['filename']).stem}", "modality": "image",
                      "path": str(root / "images" / r["filename"]), "text": None,
                      "meta": {"source": source}})
        for c in r["captions"]:
            queries.append(c)
            query_to_image.append(i)
    return items, queries, query_to_image


def load_flickr1k(root: str | Path = "data/flickr1k") -> tuple[list[dict], list[str], list[int]]:
    """Returns (image items, query captions, image index of each caption)."""
    return _load_benchmark(prepare_flickr1k(root), "flickr", "flickr30k-1k-test")


def load_coco5k(root: str | Path = "data/coco5k") -> tuple[list[dict], list[str], list[int]]:
    """Returns (image items, query captions, image index of each caption)."""
    return _load_benchmark(prepare_coco5k(root), "coco", "mscoco-2014-5k-test")


def _chunk(text: str, max_chars: int = 800) -> list[str]:
    paras = [p.strip() for p in text.replace("\r\n", "\n").split("\n\n") if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        if cur and len(cur) + len(p) + 2 > max_chars:
            chunks.append(cur)
            cur = p
        else:
            cur = f"{cur}\n\n{p}" if cur else p
    if cur:
        chunks.append(cur)
    return chunks


def load_raw_corpus(root: str | Path = "data/raw") -> list[dict]:
    root = resolve(root)
    items: list[dict] = []
    img_dir, txt_dir = root / "images", root / "texts"
    if img_dir.exists():
        for p in sorted(img_dir.rglob("*")):
            if p.suffix.lower() in IMAGE_EXTS:
                side = p.with_suffix(".txt")
                desc = side.read_text(encoding="utf-8").strip() if side.exists() else None
                items.append({"id": f"img:{p.relative_to(img_dir).as_posix()}", "modality": "image",
                              "path": str(p), "text": desc, "meta": {"source": "raw"}})
    if txt_dir.exists():
        for p in sorted(txt_dir.rglob("*")):
            if p.suffix.lower() in {".txt", ".md"}:
                for j, chunk in enumerate(_chunk(p.read_text(encoding="utf-8"))):
                    items.append({"id": f"txt:{p.relative_to(txt_dir).as_posix()}#{j}", "modality": "text",
                                  "path": None, "text": chunk, "meta": {"source": "raw", "file": p.name, "chunk": j}})
    return items
