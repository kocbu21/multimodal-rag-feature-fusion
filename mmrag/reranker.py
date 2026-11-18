"""Second-stage reranking with a cross-attention image-text matching (ITM) model.

The first stage (dual encoder + fusion) is fast but compares two independent vectors.
An ITM model runs the text through a transformer that cross-attends to the image patch
features, i.e. the deepest form of visual-textual feature fusion, and outputs P(match).
It is too slow to run on the whole corpus, so it only rescores the top-K candidates.

Two model families are supported, picked from the checkpoint name:
  BLIP   (e.g. Salesforce/blip-itm-base-coco):   ViT-B image encoder, BERT text encoder with cross-attention.
  BLIP-2 (e.g. Salesforce/blip2-itm-vit-g-coco): frozen ViT-g image encoder, Q-Former whose 32 query tokens
         and the text attend to each other and cross-attend to the image; the ITM logits are averaged over
         the query tokens.

Image features are computed once per image and cached, so each (query, candidate) pair only
costs one pass of the text / Q-Former transformer.

BLIP-2 checkpoints also carry an image-text contrastive (ITC) head: 32 Q-Former query embeddings per image and one
text embedding per query, scored by the max cosine over the 32 queries. `itc_image_embeds` / `itc_text_embeds`
expose it, so BLIP-2 can serve as an extra first-stage channel and as a second score inside the rerank stage.
"""

from __future__ import annotations

import time
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


class ITMReranker:
    def __init__(self, name: str = "Salesforce/blip-itm-base-coco", device: str = "cuda", batch_size: int = 32,
                 sleep_s: float = 0.0):
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.device = device
        self.dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.blip2 = "blip2" in name.lower()
        if self.blip2:
            from transformers import AutoProcessor, Blip2ForImageTextRetrieval
            self.processor = AutoProcessor.from_pretrained(name)
            self.model = Blip2ForImageTextRetrieval.from_pretrained(name, dtype=self.dtype, device_map={"": device}).eval()
        else:
            from transformers import BlipForImageTextRetrieval, BlipProcessor
            self.processor = BlipProcessor.from_pretrained(name)
            self.model = BlipForImageTextRetrieval.from_pretrained(name, dtype=self.dtype).to(device).eval()
        self.tokenizer = self.processor.tokenizer
        self.batch_size, self.sleep_s = batch_size, sleep_s
        self._image_cache: dict[str, torch.Tensor] = {}

    @torch.inference_mode()
    def image_features(self, paths: Sequence[str], show_progress: bool = False) -> None:
        todo = [p for p in dict.fromkeys(paths) if p not in self._image_cache]
        steps = range(0, len(todo), self.batch_size)
        if show_progress and todo:
            steps = tqdm(steps, desc="ITM image features")
        for i in steps:
            batch = todo[i : i + self.batch_size]
            pix = self.processor.image_processor([Image.open(p).convert("RGB") for p in batch], return_tensors="pt")
            feats = self.model.vision_model(pixel_values=pix["pixel_values"].to(self.device, self.dtype))[0]
            for p, f in zip(batch, feats):
                self._image_cache[p] = f.cpu()
            self._sleep()

    @torch.inference_mode()
    def score_pairs(self, texts: Sequence[str], image_paths: Sequence[str], show_progress: bool = False,
                    return_logits: bool = False):
        """P(match) for each (text, image) pair. With return_logits=True also returns the positive-class ITM
        logit (BLIP-2's own retrieval protocol adds this logit to the ITC similarity)."""
        self.image_features(image_paths, show_progress)
        out, out_l = [], []
        steps = range(0, len(texts), self.batch_size)
        if show_progress:
            steps = tqdm(steps, desc="ITM rerank")
        for i in steps:
            t = list(texts[i : i + self.batch_size])
            img = torch.stack([self._image_cache[p] for p in image_paths[i : i + self.batch_size]]).to(self.device)
            enc = self.tokenizer(t, padding=True, truncation=True, max_length=64, return_tensors="pt").to(self.device)
            logits = (self._itm_blip2(enc, img) if self.blip2 else self._itm_blip(enc, img)).float()
            out.append(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
            out_l.append(logits[:, 1].cpu().numpy())
            self._sleep()
        p = np.concatenate(out) if out else np.zeros(0, np.float32)
        if return_logits:
            return p, (np.concatenate(out_l) if out_l else np.zeros(0, np.float32))
        return p

    def clear_cache(self) -> None:
        self._image_cache.clear()
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()

    # ---- BLIP-2 image-text contrastive (ITC) features: a second, BLIP-2-native dual encoder -------------------

    @torch.inference_mode()
    def itc_image_embeds(self, paths: Sequence[str], show_progress: bool = False, keep_vit: bool = False) -> np.ndarray:
        """(n, 32, 256) normalised Q-Former query embeddings of each image (BLIP-2 only).

        Mirrors Blip2ForImageTextRetrieval.forward(use_image_text_matching_head=False). The ViT-g features are
        dropped after use unless keep_vit=True, so a large gallery does not fill host memory."""
        if not self.blip2:
            raise ValueError("ITC features need a BLIP-2 checkpoint")
        m, out = self.model, []
        paths = list(paths)
        steps = range(0, len(paths), self.batch_size)
        if show_progress:
            steps = tqdm(steps, desc="BLIP-2 ITC image")
        for i in steps:
            batch = paths[i : i + self.batch_size]
            cached = all(p in self._image_cache for p in batch)
            if cached:
                img = torch.stack([self._image_cache[p] for p in batch]).to(self.device)
            else:
                pix = self.processor.image_processor([Image.open(p).convert("RGB") for p in batch], return_tensors="pt")
                img = m.vision_model(pixel_values=pix["pixel_values"].to(self.device, self.dtype))[0]
                if keep_vit:
                    for p, f in zip(batch, img):
                        self._image_cache[p] = f.cpu()
            att = torch.ones(img.shape[:-1], dtype=torch.long, device=self.device)
            q = m.query_tokens.expand(img.shape[0], -1, -1)
            hid = m.qformer(query_embeds=q, encoder_hidden_states=img, encoder_attention_mask=att,
                            return_dict=True).last_hidden_state
            hid = hid.to(m.vision_projection.weight.dtype)
            out.append(torch.nn.functional.normalize(m.vision_projection(hid), dim=-1).float().cpu().numpy())
            self._sleep()
        return np.concatenate(out) if out else np.zeros((0, 0, 0), np.float32)

    @torch.inference_mode()
    def itc_text_embeds(self, texts: Sequence[str], show_progress: bool = False) -> np.ndarray:
        """(n, 256) normalised BLIP-2 ITC text embeddings ([CLS] of the Q-Former text pass)."""
        if not self.blip2:
            raise ValueError("ITC features need a BLIP-2 checkpoint")
        m, out = self.model, []
        texts = list(texts)
        steps = range(0, len(texts), 4 * self.batch_size)
        if show_progress:
            steps = tqdm(steps, desc="BLIP-2 ITC text")
        for i in steps:
            enc = self.tokenizer(texts[i : i + 4 * self.batch_size], padding=True, truncation=True, max_length=64,
                                 return_tensors="pt").to(self.device)
            emb = m.embeddings(input_ids=enc["input_ids"])
            hid = m.qformer(query_embeds=emb, query_length=0, attention_mask=enc["attention_mask"],
                            return_dict=True).last_hidden_state
            hid = hid.to(m.text_projection.weight.dtype)
            out.append(torch.nn.functional.normalize(m.text_projection(hid[:, 0, :]), dim=-1).float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 0), np.float32)

    def _itm_blip(self, enc, img):
        att = torch.ones(img.shape[:-1], dtype=torch.long, device=self.device)
        hidden = self.model.text_encoder(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                                         encoder_hidden_states=img, encoder_attention_mask=att,
                                         return_dict=False)[0]
        return self.model.itm_head(hidden[:, 0, :])

    def _itm_blip2(self, enc, img):
        m = self.model
        q = m.query_tokens.expand(img.shape[0], -1, -1)
        n_q = q.shape[1]
        att = torch.cat([torch.ones(q.shape[:-1], dtype=torch.long, device=self.device), enc["attention_mask"]], 1)
        emb = m.embeddings(input_ids=enc["input_ids"], query_embeds=q)
        img_att = torch.ones(img.shape[:-1], dtype=torch.long, device=self.device)
        hidden = m.qformer(query_embeds=emb, query_length=n_q, attention_mask=att, encoder_hidden_states=img,
                           encoder_attention_mask=img_att, return_dict=True).last_hidden_state
        hidden = hidden[:, :n_q, :].to(m.itm_head.weight.dtype)
        return m.itm_head(hidden).mean(dim=1)

    def _sleep(self):
        if self.sleep_s and self.device.startswith("cuda"):
            time.sleep(self.sleep_s)
