"""Gradio demo: search the unified vector DB with text, an image, or both.

python app.py --store indexes/raw_clip-vit-base-patch32
"""

import argparse

import gradio as gr

from mmrag.config import load_config
from mmrag.generator import ClaudeGenerator
from mmrag.pipeline import load_retriever
from mmrag.retriever import STRATEGIES


def make_app(store_dir: str, rerank: bool = False) -> gr.Blocks:
    cfg = load_config()
    retriever = load_retriever(store_dir, cfg, rerank=rerank or None)
    store = retriever.store
    state = {"alpha": store.alpha, "center": store.centered}

    def run(text, image, strategy, k, alpha, beta, center, rerank, answer):
        if (alpha, center) != (state["alpha"], state["center"]):
            store.rebuild(alpha=alpha, center=center)
            state.update(alpha=alpha, center=center)
        text = (text or "").strip() or None
        if text is None and image is None:
            return [], "Enter a text query, upload an image, or both.", ""
        hits = retriever.search(text=text, image=image, k=int(k), strategy=strategy, beta=beta, rerank=rerank)
        gallery, lines = [], []
        for i, h in enumerate(hits, 1):
            it = h.item
            src = ", ".join(f"{n}={v:.3f}" for n, v in h.sources.items())
            if it["modality"] == "image":
                gallery.append((it["path"], f"{i}. {it.get('caption') or it['id']}"))
                lines.append(f"**{i}. [image]** `{it['id']}` score={h.score:.4f} ({src})  \ncaption: {it.get('caption')}")
            else:
                lines.append(f"**{i}. [text]** `{it['id']}` score={h.score:.4f} ({src})  \n{it['text'][:300]}")
        ans = ""
        if answer:
            try:
                ans = ClaudeGenerator(cfg["generator"]["model"]).answer(text or "Describe these results.", hits)
            except Exception as e:
                ans = f"Answer generation failed: {e}"
        return gallery, "\n\n".join(lines), ans

    with gr.Blocks(title="Multimodal RAG") as demo:
        gr.Markdown(f"## Multimodal RAG - visual-textual feature fusion\n`{store!r}`")
        with gr.Row():
            with gr.Column(scale=1):
                text = gr.Textbox(label="Text query")
                image = gr.Image(label="Image query", type="pil")
                strategy = gr.Dropdown(list(STRATEGIES), value="auto", label="Strategy")
                k = gr.Slider(1, 30, value=8, step=1, label="top-k")
                alpha = gr.Slider(0, 1, value=store.alpha, step=0.05, label="alpha (doc fusion: 1=image, 0=caption)")
                beta = gr.Slider(0, 1, value=cfg["fusion"]["beta"], step=0.05, label="beta (query fusion: 1=image, 0=text)")
                center = gr.Checkbox(value=store.centered, label="Modality centering")
                rerank = gr.Checkbox(value=retriever.reranker is not None, interactive=retriever.reranker is not None,
                                     label="ITM reranking (start the app with --rerank)")
                answer = gr.Checkbox(value=False, label="Generate answer with Claude")
                btn = gr.Button("Search", variant="primary")
            with gr.Column(scale=2):
                gallery = gr.Gallery(label="Image results", columns=4, height=420)
                results = gr.Markdown()
                answer_box = gr.Markdown()
        btn.click(run, [text, image, strategy, k, alpha, beta, center, rerank, answer], [gallery, results, answer_box])
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--rerank", action="store_true")
    a = ap.parse_args()
    make_app(a.store, a.rerank).launch(server_port=a.port)
