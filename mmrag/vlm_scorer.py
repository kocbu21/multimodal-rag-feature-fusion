"""Generative VLM verification of a few top candidates (VQAScore-style yes/no scoring).

A cross-encoder such as BLIP-2 ITM decides most of the final ranking, and in v2 almost all remaining errors are
its ordering errors among near-duplicate candidates. An instruction-tuned VLM reads the whole caption and looks at
the image with a much larger language model, so it can check counts, attributes and relations. It is expensive,
so it only rescores the top-M candidates of the previous stage (a cascade).

score(q, x) = log P("Yes" | image x, "Does this image match ...q...?") - log P("No" | ...), taken from the logits
of the first generated token; no text is generated.
"""

from __future__ import annotations

import time
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

PROMPT = ('Does this image match the following description? Description: "{q}" '
          "Answer with Yes or No.")


class VLMScorer:
    def __init__(self, name: str = "Qwen/Qwen2.5-VL-7B-Instruct", device: str = "cuda", batch_size: int = 16,
                 max_pixels: int = 448 * 448, sleep_s: float = 0.0):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.device, self.batch_size, self.sleep_s, self.name = device, batch_size, sleep_s, name
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        self.processor = AutoProcessor.from_pretrained(name, size={"shortest_edge": 224 * 224, "longest_edge": max_pixels})
        self.processor.tokenizer.padding_side = "left"
        self.model = AutoModelForImageTextToText.from_pretrained(name, dtype=dtype).to(device).eval()
        tok = self.processor.tokenizer
        self.yes_ids = sorted({i for w in ("Yes", "yes", " Yes", " yes", "YES") for i in tok.encode(w, add_special_tokens=False)[:1]})
        self.no_ids = sorted({i for w in ("No", "no", " No", " no", "NO") for i in tok.encode(w, add_special_tokens=False)[:1]})
        assert not set(self.yes_ids) & set(self.no_ids), "yes/no token ids overlap"

    def _texts(self, queries: Sequence[str]) -> list[str]:
        msgs = [[{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT.format(q=q)}]}]
                for q in queries]
        return [self.processor.apply_chat_template(m, add_generation_prompt=True, tokenize=False) for m in msgs]

    @torch.inference_mode()
    def score_pairs(self, queries: Sequence[str], image_paths: Sequence[str], show_progress: bool = False) -> np.ndarray:
        """Yes-minus-No log-odds for each (query, image) pair."""
        out = []
        steps = range(0, len(queries), self.batch_size)
        if show_progress:
            steps = tqdm(steps, desc=f"VLM ({self.name.split('/')[-1]})")
        cache: dict[str, Image.Image] = {}
        for i in steps:
            qs = list(queries[i : i + self.batch_size])
            ps = list(image_paths[i : i + self.batch_size])
            imgs = []
            for p in ps:
                if p not in cache:
                    if len(cache) > 256:
                        cache.clear()
                    cache[p] = Image.open(p).convert("RGB")
                imgs.append(cache[p])
            inputs = self.processor(text=self._texts(qs), images=imgs, padding=True, return_tensors="pt").to(self.device)
            logits = self.model(**inputs).logits[:, -1, :].float()
            logp = torch.log_softmax(logits, dim=-1)
            yes = torch.logsumexp(logp[:, self.yes_ids], dim=-1)
            no = torch.logsumexp(logp[:, self.no_ids], dim=-1)
            out.append((yes - no).cpu().numpy())
            if self.sleep_s and self.device.startswith("cuda"):
                time.sleep(self.sleep_s)
        return np.concatenate(out).astype(np.float32) if out else np.zeros(0, np.float32)
