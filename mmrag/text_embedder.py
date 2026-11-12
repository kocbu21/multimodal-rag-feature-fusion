"""A dedicated text embedding model (e.g. BGE) for text-to-text matching.

CLIP/SigLIP text towers are trained only to match images, so they are weak at comparing two
sentences. Query <-> caption similarity is computed in this model's space instead, and then
fused with the image score at score level.
"""

from __future__ import annotations

import time
from typing import Sequence

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


class TextEmbedder:
    def __init__(self, name: str = "BAAI/bge-small-en-v1.5", device: str = "cuda", batch_size: int = 64,
                 sleep_s: float = 0.0, max_length: int = 256):
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.name, self.device, self.batch_size, self.sleep_s, self.max_length = name, device, batch_size, sleep_s, max_length
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name).to(device).eval()

    @torch.inference_mode()
    def encode(self, texts: Sequence[str], show_progress: bool = False) -> np.ndarray:
        texts = list(texts)
        out = []
        steps = range(0, len(texts), self.batch_size)
        if show_progress:
            steps = tqdm(steps, desc=f"embed ({self.name.split('/')[-1]})")
        for i in steps:
            enc = self.tokenizer(texts[i : i + self.batch_size], padding=True, truncation=True,
                                 max_length=self.max_length, return_tensors="pt").to(self.device)
            cls = self.model(**enc).last_hidden_state[:, 0]
            out.append(torch.nn.functional.normalize(cls, dim=-1).float().cpu().numpy())
            if self.sleep_s and self.device.startswith("cuda"):
                time.sleep(self.sleep_s)
        return np.concatenate(out).astype(np.float32)
