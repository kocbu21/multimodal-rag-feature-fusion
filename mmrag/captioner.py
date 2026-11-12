"""Image captioning with BLIP. Captions become the third vector type ("caption vector").

The caption is encoded with the *text* encoder, so an image gets a representation that
lives on the text side of the shared space. Fusing it with the image vector is the core
idea of this project.
"""

from __future__ import annotations

import time
from typing import Sequence

import torch
from PIL import Image
from tqdm import tqdm
from transformers import BlipForConditionalGeneration, BlipProcessor


class Captioner:
    def __init__(self, name: str = "Salesforce/blip-image-captioning-base", device: str = "cuda",
                 batch_size: int = 8, max_new_tokens: int = 30, sleep_s: float = 0.0):
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.device = device
        self.dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.processor = BlipProcessor.from_pretrained(name)
        self.model = BlipForConditionalGeneration.from_pretrained(name, dtype=self.dtype).to(device).eval()
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.sleep_s = sleep_s

    @torch.inference_mode()
    def caption(self, images: Sequence[Image.Image | str], show_progress: bool = False) -> list[str]:
        images = list(images)
        captions: list[str] = []
        steps = range(0, len(images), self.batch_size)
        if show_progress:
            steps = tqdm(steps, desc="caption images")
        for i in steps:
            batch = images[i : i + self.batch_size]
            pil = [Image.open(b).convert("RGB") if isinstance(b, str) else b.convert("RGB") for b in batch]
            inputs = self.processor(images=pil, return_tensors="pt").to(self.device, self.dtype)
            ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, num_beams=3)
            captions.extend(c.strip() for c in self.processor.batch_decode(ids, skip_special_tokens=True))
            if self.sleep_s:
                time.sleep(self.sleep_s)
        return captions


VLM_PROMPT = ("Describe this image in one or two sentences for an image search engine. "
              "Mention the main subjects, what they are doing, notable colors and clothing, and the setting.")


class VLMCaptioner:
    """Detailed captions from an instruction-tuned VLM (e.g. Qwen2.5-VL). Richer than BLIP, but slower."""

    def __init__(self, name: str = "Qwen/Qwen2.5-VL-3B-Instruct", device: str = "cuda", batch_size: int = 4,
                 max_new_tokens: int = 60, sleep_s: float = 0.0, prompt: str = VLM_PROMPT):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.device = device
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        self.processor = AutoProcessor.from_pretrained(name, min_pixels=224 * 224, max_pixels=448 * 448)
        self.processor.tokenizer.padding_side = "left"
        self.model = AutoModelForImageTextToText.from_pretrained(name, dtype=dtype).to(device).eval()
        self.batch_size, self.max_new_tokens, self.sleep_s, self.prompt = batch_size, max_new_tokens, sleep_s, prompt

    @torch.inference_mode()
    def caption(self, images: Sequence[Image.Image | str], show_progress: bool = False) -> list[str]:
        images = list(images)
        out: list[str] = []
        steps = range(0, len(images), self.batch_size)
        if show_progress:
            steps = tqdm(steps, desc="VLM captions")
        for i in steps:
            batch = [Image.open(b).convert("RGB") if isinstance(b, str) else b.convert("RGB")
                     for b in images[i : i + self.batch_size]]
            msgs = [[{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": self.prompt}]}]
                    for _ in batch]
            texts = [self.processor.apply_chat_template(m, add_generation_prompt=True, tokenize=False) for m in msgs]
            inputs = self.processor(text=texts, images=batch, padding=True, return_tensors="pt").to(self.device)
            ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
            ids = ids[:, inputs["input_ids"].shape[1]:]
            out.extend(t.strip().replace("\n", " ") for t in self.processor.batch_decode(ids, skip_special_tokens=True))
            if self.sleep_s:
                time.sleep(self.sleep_s)
        return out


class FlorenceCaptioner:
    """Florence-2 captions. The name may carry the task after '#', e.g.
    'florence-community/Florence-2-large#<MORE_DETAILED_CAPTION>' (default: <DETAILED_CAPTION>)."""

    def __init__(self, name: str = "florence-community/Florence-2-large", device: str = "cuda", batch_size: int = 8,
                 max_new_tokens: int = 128, sleep_s: float = 0.0):
        from transformers import AutoProcessor, Florence2ForConditionalGeneration

        model_id, _, task = name.partition("#")
        self.task = task or "<DETAILED_CAPTION>"
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.device = device
        self.dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = Florence2ForConditionalGeneration.from_pretrained(model_id, dtype=self.dtype).to(device).eval()
        self.batch_size, self.max_new_tokens, self.sleep_s = batch_size, max_new_tokens, sleep_s

    @torch.inference_mode()
    def caption(self, images: Sequence[Image.Image | str], show_progress: bool = False) -> list[str]:
        images = list(images)
        out: list[str] = []
        steps = range(0, len(images), self.batch_size)
        if show_progress:
            steps = tqdm(steps, desc="Florence captions")
        for i in steps:
            batch = [Image.open(b).convert("RGB") if isinstance(b, str) else b.convert("RGB")
                     for b in images[i : i + self.batch_size]]
            inputs = self.processor(text=[self.task] * len(batch), images=batch, return_tensors="pt")
            inputs = inputs.to(self.device, self.dtype)
            ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, num_beams=3, do_sample=False)
            out.extend(" ".join(t.split()) for t in self.processor.batch_decode(ids, skip_special_tokens=True))
            if self.sleep_s:
                time.sleep(self.sleep_s)
        return out


def build_captioner(cfg: dict) -> Captioner | VLMCaptioner | FlorenceCaptioner | None:
    cap = cfg.get("captioner", {})
    if not cap.get("enabled", True):
        return None
    device = cap.get("device") or cfg["encoder"].get("device", "cuda")
    sleep_s = cfg.get("throttle", {}).get("gpu_sleep_s", 0.0)
    if "florence" in cap["name"].lower():
        return FlorenceCaptioner(cap["name"], device, cap.get("batch_size", 8), 128, sleep_s)
    if "blip" not in cap["name"].lower():
        return VLMCaptioner(cap["name"], device, cap.get("batch_size", 4), cap.get("max_new_tokens", 60), sleep_s)
    return Captioner(cap["name"], device, cap.get("batch_size", 8), cap.get("max_new_tokens", 30), sleep_s)
