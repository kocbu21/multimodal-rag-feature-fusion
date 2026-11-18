"""Region-level visual features ("look at the important parts of the image").

A dual encoder squeezes a whole image into one 224-pixel global vector, so small objects, counts and
attributes get lost. We add a set of region vectors per image and score an image by the best-matching
region (max pooling), next to its global vector:

* grid    : 4 overlapping quadrant crops (60% of each side) + a 70% centre crop. Cheap, no extra model.
* od      : salient object boxes from Florence-2 ``<OD>`` (largest boxes first, small boxes dropped,
            10% context margin). The detected labels also become an "object tags" text for the BGE channel.

Boxes are cached in data/region_cache.json under "<method>|<image path>" so detection runs once per image
and is shared by all encoders.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image
from tqdm import tqdm

from .config import resolve

REGION_CACHE = "data/region_cache.json"
FLORENCE_OD = "florence-community/Florence-2-large"
KIND_ID = {"grid": 0, "od": 1}


# ---- cache -----------------------------------------------------------------------------------------------

def load_region_cache() -> dict:
    p = resolve(REGION_CACHE)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def save_region_cache(new_entries: dict) -> None:
    cache = load_region_cache()
    cache.update(new_entries)
    tmp = resolve(REGION_CACHE + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    tmp.replace(resolve(REGION_CACHE))


# ---- boxes -----------------------------------------------------------------------------------------------

def grid_boxes(w: int, h: int, frac: float = 0.6, centre: float = 0.7) -> list[list[float]]:
    """Four overlapping quadrant boxes and one centre box, as [x0, y0, x1, y1] in pixels."""
    bw, bh = w * frac, h * frac
    boxes = [[0, 0, bw, bh], [w - bw, 0, w, bh], [0, h - bh, bw, h], [w - bw, h - bh, w, h]]
    cw, ch = w * centre, h * centre
    boxes.append([(w - cw) / 2, (h - ch) / 2, (w + cw) / 2, (h + ch) / 2])
    return [[float(v) for v in b] for b in boxes]


def select_od_boxes(bboxes: Sequence[Sequence[float]], labels: Sequence[str], w: int, h: int,
                    max_boxes: int = 6, min_area: float = 0.02, max_area: float = 0.9,
                    margin: float = 0.1) -> list[tuple[list[float], str]]:
    """Keep the largest informative detections: drop tiny boxes and boxes that are almost the whole image,
    add a context margin, clip to the image, remove near-duplicates (IoU > 0.8)."""
    cand = []
    for b, lab in zip(bboxes, labels):
        x0, y0, x1, y1 = [float(v) for v in b]
        area = max(0.0, x1 - x0) * max(0.0, y1 - y0) / float(w * h)
        if min_area <= area <= max_area:
            mx, my = (x1 - x0) * margin, (y1 - y0) * margin
            cand.append(([max(0.0, x0 - mx), max(0.0, y0 - my), min(float(w), x1 + mx), min(float(h), y1 + my)],
                         lab.strip(), area))
    cand.sort(key=lambda c: -c[2])
    kept: list[tuple[list[float], str]] = []
    for box, lab, _ in cand:
        if all(_iou(box, k[0]) <= 0.8 for k in kept):
            kept.append((box, lab))
        if len(kept) >= max_boxes:
            break
    return kept


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def tags_text(labels: Sequence[str]) -> str:
    """Unique detected labels in order of box size, as a short text for the text-embedding channel."""
    seen: list[str] = []
    for lab in labels:
        lab = lab.strip().lower()
        if lab and lab not in seen:
            seen.append(lab)
    return ", ".join(seen)


class FlorenceDetector:
    """Florence-2 object detection (``<OD>``) returning boxes in pixel coordinates and labels."""

    def __init__(self, name: str = FLORENCE_OD, device: str = "cuda", batch_size: int = 8, sleep_s: float = 0.0):
        import torch
        from transformers import AutoProcessor, Florence2ForConditionalGeneration

        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.device = device
        self.dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.processor = AutoProcessor.from_pretrained(name)
        self.model = Florence2ForConditionalGeneration.from_pretrained(name, dtype=self.dtype).to(device).eval()
        self.batch_size, self.sleep_s = batch_size, sleep_s

    def detect(self, paths: Sequence[str], show_progress: bool = False) -> list[dict]:
        import torch

        out: list[dict] = []
        steps = range(0, len(paths), self.batch_size)
        if show_progress:
            steps = tqdm(steps, desc="Florence-2 <OD>")
        with torch.inference_mode():
            for i in steps:
                imgs = [Image.open(p).convert("RGB") for p in paths[i : i + self.batch_size]]
                inputs = self.processor(text=["<OD>"] * len(imgs), images=imgs, return_tensors="pt")
                inputs = inputs.to(self.device, self.dtype)
                ids = self.model.generate(**inputs, max_new_tokens=256, num_beams=3, do_sample=False)
                texts = self.processor.batch_decode(ids, skip_special_tokens=False)
                for img, t in zip(imgs, texts):
                    r = self.processor.post_process_generation(t, task="<OD>", image_size=img.size)["<OD>"]
                    out.append({"size": list(img.size), "bboxes": [[float(v) for v in b] for b in r["bboxes"]],
                                "labels": list(r["labels"])})
                if self.sleep_s:
                    time.sleep(self.sleep_s)
        return out


def ensure_od_cache(paths: Sequence[str], device: str = "cuda", sleep_s: float = 0.0, chunk: int = 64) -> dict:
    """Run Florence-2 <OD> on every image that has no cached detection yet (resumable)."""
    cache = load_region_cache()
    todo = [p for p in paths if f"od|{p}" not in cache]
    if todo:
        det = FlorenceDetector(device=device, sleep_s=sleep_s)
        for i in tqdm(range(0, len(todo), chunk), desc="object detection (chunks)"):
            part = todo[i : i + chunk]
            new = {f"od|{p}": r for p, r in zip(part, det.detect(part))}
            save_region_cache(new)
            cache.update(new)
        del det
    return cache


def region_boxes(path: str, kinds: Sequence[str], cache: dict | None = None,
                 max_od: int = 6) -> list[tuple[str, list[float], str]]:
    """All region boxes of one image: [(kind, box, label), ...]."""
    out: list[tuple[str, list[float], str]] = []
    if "grid" in kinds:
        with Image.open(path) as im:
            w, h = im.size
        out += [("grid", b, "") for b in grid_boxes(w, h)]
    if "od" in kinds:
        r = (cache or {}).get(f"od|{path}")
        if r is None:
            raise KeyError(f"no cached <OD> detection for {path}; run ensure_od_cache first")
        w, h = r["size"]
        out += [("od", b, lab) for b, lab in select_od_boxes(r["bboxes"], r["labels"], w, h, max_boxes=max_od)]
    return out


def object_tags(path: str, cache: dict, max_od: int = 6) -> str:
    r = cache.get(f"od|{path}")
    if not r:
        return ""
    w, h = r["size"]
    return tags_text([lab for _, lab in select_od_boxes(r["bboxes"], r["labels"], w, h, max_boxes=max_od)])


def encode_regions(encoder, paths: Sequence[str], kinds: Sequence[str], cache: dict | None = None,
                   batch_images: int = 32, show_progress: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Crop every region of every image and encode the crops with the dual encoder.

    Returns (vectors (n_regions, d), owner (n_regions,) = index into `paths`, kind (n_regions,) 0=grid 1=od)."""
    vecs, owner, kind = [], [], []
    steps = range(0, len(paths), batch_images)
    if show_progress:
        steps = tqdm(steps, desc=f"region crops ({'+'.join(kinds)})")
    for i in steps:
        crops = []
        for j, p in enumerate(paths[i : i + batch_images], start=i):
            boxes = region_boxes(p, kinds, cache)
            if not boxes:
                continue
            with Image.open(p) as im:
                im = im.convert("RGB")
                for k, b, _ in boxes:
                    x0, y0, x1, y1 = b
                    if x1 - x0 < 8 or y1 - y0 < 8:
                        continue
                    crops.append(im.crop((int(x0), int(y0), int(round(x1)), int(round(y1)))))
                    owner.append(j)
                    kind.append(KIND_ID[k])
        if crops:
            vecs.append(encoder.encode_images(crops))
    d = vecs[0].shape[1] if vecs else 0
    return (np.concatenate(vecs) if vecs else np.zeros((0, d), np.float32),
            np.asarray(owner, dtype=np.int64), np.asarray(kind, dtype=np.int8))


def region_path(store_dir: str | Path, kinds: Sequence[str]) -> Path:
    return Path(store_dir) / f"regions_{'+'.join(sorted(kinds))}.npz"
