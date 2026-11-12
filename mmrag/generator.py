"""Optional answer generation: send the retrieved images and texts to a VLM (Claude) and
answer the question grounded on those sources."""

from __future__ import annotations

import base64
import io

from PIL import Image

from .retriever import Hit

SYSTEM_PROMPT = (
    "You answer questions using only the retrieved sources provided (images and text). "
    "Cite sources as [S1], [S2], ... If the sources do not contain the answer, say so. "
    "Answer in the language of the question."
)


def _image_block(path: str, max_side: int = 768) -> dict:
    img = Image.open(path).convert("RGB")
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return {"type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg",
                       "data": base64.standard_b64encode(buf.getvalue()).decode("utf-8")}}


def build_content(question: str, hits: list[Hit]) -> list[dict]:
    content: list[dict] = []
    for i, hit in enumerate(hits, 1):
        it = hit.item
        header = f"[S{i}] id={it['id']} modality={it['modality']}"
        if it["modality"] == "image":
            content.append({"type": "text", "text": header + (f"\ncaption: {it['caption']}" if it.get("caption") else "")
                            + (f"\ndescription: {it['text']}" if it.get("text") else "")})
            content.append(_image_block(it["path"]))
        else:
            content.append({"type": "text", "text": f"{header}\n{it['text']}"})
    content.append({"type": "text", "text": f"Question: {question}"})
    return content


class ClaudeGenerator:
    def __init__(self, model: str = "claude-opus-5-5"):
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = model

    def answer(self, question: str, hits: list[Hit]) -> str:
        response = self.client.beta.messages.create(
            model=self.model,
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            output_config={"effort": "medium"},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=[{"role": "user", "content": build_content(question, hits)}],
        )
        if response.stop_reason == "refusal":
            return "The model declined to answer this question."
        return "".join(b.text for b in response.content if b.type == "text")


def build_generator(cfg: dict):
    g = cfg.get("generator", {})
    if g.get("backend", "none") == "claude":
        return ClaudeGenerator(g.get("model", "claude-opus-5-5"))
    return None
