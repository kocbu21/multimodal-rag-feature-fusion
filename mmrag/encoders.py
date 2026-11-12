"""Image and text encoders that map both modalities into one shared embedding space.

Supports CLIP and SigLIP checkpoints from Hugging Face. All returned vectors are
L2-normalised float32 numpy arrays, so inner product == cosine similarity.
"""

from __future__ import annotations

import time
from typing import Iterable, Sequence

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor


def _pick_device(device: str) -> str:
    if device.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return device


def _to_tensor(out) -> torch.Tensor:
    """get_*_features returns a tensor in most transformers versions, an output object in some."""
    if isinstance(out, torch.Tensor):
        return out
    for attr in ("image_embeds", "text_embeds", "pooler_output"):
        value = getattr(out, attr, None)
        if value is not None:
            return value
    raise TypeError(f"Unexpected encoder output type: {type(out)}")


def _l2norm(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12, None)


def _batches(items: Sequence, size: int) -> Iterable[Sequence]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


class DualEncoder:
    """Wraps a CLIP / SigLIP model with `encode_images` and `encode_texts`."""

    def __init__(self, name: str = "openai/clip-vit-base-patch32", device: str = "cuda", batch_size: int = 16,
                 sleep_s: float = 0.0):
        self.name = name
        self.device = _pick_device(device)
        self.batch_size = batch_size
        self.sleep_s = sleep_s
        self.is_siglip = "siglip" in name.lower()
        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        self.model = AutoModel.from_pretrained(name, dtype=dtype).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(name)
        self.dtype = dtype

    @property
    def dim(self) -> int:
        return int(self.encode_texts(["dimension probe"]).shape[1])

    @torch.inference_mode()
    def encode_images(self, images: Sequence[Image.Image | str], show_progress: bool = False) -> np.ndarray:
        out = []
        it = _batches(list(images), self.batch_size)
        if show_progress:
            it = tqdm(it, total=(len(images) + self.batch_size - 1) // self.batch_size, desc="encode images")
        for batch in it:
            pil = [Image.open(b).convert("RGB") if isinstance(b, str) else b.convert("RGB") for b in batch]
            inputs = self.processor(images=pil, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device, self.dtype)
            feats = _to_tensor(self.model.get_image_features(pixel_values=pixel_values))
            out.append(feats.float().cpu().numpy())
            if self.sleep_s:
                time.sleep(self.sleep_s)
        if not out:
            return np.zeros((0, 0), dtype=np.float32)
        return _l2norm(np.concatenate(out)).astype(np.float32)

    @torch.inference_mode()
    def encode_texts(self, texts: Sequence[str], show_progress: bool = False) -> np.ndarray:
        out = []
        it = _batches(list(texts), self.batch_size)
        if show_progress:
            it = tqdm(it, total=(len(texts) + self.batch_size - 1) // self.batch_size, desc="encode texts")
        for batch in it:
            if self.is_siglip:
                inputs = self.processor(text=list(batch), padding="max_length", max_length=64,
                                        truncation=True, return_tensors="pt")
            else:
                inputs = self.processor(text=list(batch), padding=True, truncation=True, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items() if k in ("input_ids", "attention_mask")}
            feats = _to_tensor(self.model.get_text_features(**inputs))
            out.append(feats.float().cpu().numpy())
            if self.sleep_s:
                time.sleep(self.sleep_s)
        if not out:
            return np.zeros((0, 0), dtype=np.float32)
        return _l2norm(np.concatenate(out)).astype(np.float32)


def build_encoder(cfg: dict) -> DualEncoder:
    enc = cfg["encoder"]
    return DualEncoder(enc["name"], enc.get("device", "cuda"), enc.get("batch_size", 16),
                       cfg.get("throttle", {}).get("gpu_sleep_s", 0.0))
