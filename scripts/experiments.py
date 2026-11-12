"""Improvement experiments on Flickr30k-1K, text -> image.

Stages compared (all weights picked with 2-fold CV over images, as in evaluate.py):

  image                    dual-encoder baseline
  early[c]                 early fusion with caption set c (BLIP-base, BLIP-large, VLM, ...)
  early[multi]             early fusion with the mean of all caption sets' vectors
  te-caption[c]            query vs caption in a dedicated text-embedding space (BGE) only
  hybrid[c]                late fusion: early-fused dual-encoder score + BGE caption score  (alpha, w)
  rerank-ITM / -BLIP2      BLIP / BLIP-2 ITM cross-attention rescoring of the best first stage's top-K (lambda)
  rerank-ITM+BLIP2         both cross-encoders and the first-stage score combined (weights on the simplex)

python scripts/experiments.py --store indexes/flickr1k_clip-vit-base-patch32
"""

import argparse
import itertools
import json
import re

import _bootstrap
import numpy as np
import pandas as pd
import torch

from mmrag.config import load_config, resolve
from mmrag.data import load_flickr1k
from mmrag.fusion import early_fuse, l2norm, weighted_late_fusion, zscore_rows
from mmrag.metrics import retrieval_metrics
from mmrag.vector_store import UnifiedVectorStore

FLORENCE = "florence-community/Florence-2-large"
CAPTIONERS = ["Salesforce/blip-image-captioning-base", "Salesforce/blip-image-captioning-large",
              f"{FLORENCE}#<DETAILED_CAPTION>", f"{FLORENCE}#<MORE_DETAILED_CAPTION>",
              "Qwen/Qwen2.5-VL-3B-Instruct"]
SHORT = {"Salesforce/blip-image-captioning-base": "blip-b", "Salesforce/blip-image-captioning-large": "blip-l",
         f"{FLORENCE}#<DETAILED_CAPTION>": "flor-d", f"{FLORENCE}#<MORE_DETAILED_CAPTION>": "flor-md",
         "Qwen/Qwen2.5-VL-3B-Instruct": "qwen-vl", "concat": "concat"}
ALPHAS = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0]
WEIGHTS = [round(float(x), 2) for x in np.linspace(0, 1, 11)]


def slug(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", name.split("/")[-1])


def auto_device(min_free_gb: float = 3.0) -> str:
    """Use the GPU only if it has room, so other jobs on it are not disturbed."""
    if not torch.cuda.is_available():
        return "cpu"
    free, _ = torch.cuda.mem_get_info()
    return "cuda" if free / 1e9 >= min_free_gb else "cpu"


def cv_pick(score_fn, grid, positives, fold, metric="RSUM"):
    """Memory-light 2-fold CV: evaluate each grid point on both folds, then rebuild only the winners."""
    pos = np.asarray(positives)
    res = {g: [None, None] for g in grid}
    for g in grid:
        S = score_fn(g)
        for f in (0, 1):
            m = fold == f
            res[g][f] = retrieval_metrics(S[m], pos[m])[metric]
    picked = [max(grid, key=lambda g: res[g][f]) for f in (0, 1)]
    out = None
    for f, g in enumerate(picked):
        S = score_fn(g)
        out = np.empty_like(S) if out is None else out
        out[fold != f] = S[fold != f]
    return out, picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--text-embedder", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--reranker", default="Salesforce/blip-itm-base-coco",
                    help="one or more ITM checkpoints, comma-separated")
    ap.add_argument("--rerank-k", type=int, default=16)
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    device = auto_device() if args.device == "auto" else args.device
    cfg = load_config()
    sleep_s = cfg["throttle"]["gpu_sleep_s"]
    print(f"device: {device}")

    store_dir = resolve(args.store)
    store = UnifiedVectorStore.load(store_dir)
    _, queries, q2img = load_flickr1k()
    q2img = np.asarray(q2img)
    paths = [store.items[r]["path"] for r in store.rows["image"]]
    n_img = len(paths)
    fold = (np.random.default_rng(0).permutation(n_img) % 2)[q2img]
    I, Q = store.raw["image"], np.load(store_dir / "eval_query_vectors.npy")

    cache_path = resolve("data/caption_cache.json")
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    avail = [c for c in CAPTIONERS if all(f"{c}|{p}" in cache for p in paths)]
    cap_text = {c: [cache[f"{c}|{p}"] for p in paths] for c in avail}
    if len(avail) > 1:
        cap_text["concat"] = [" ".join(parts) for parts in zip(*(cap_text[c] for c in avail))]
    print("caption sets:", [SHORT[c] for c in cap_text])

    encoder = None
    dual_caps = {}
    for c in avail:
        p = store_dir / f"capvec_{slug(c)}.npy"
        if not p.exists():
            if encoder is None:
                from mmrag.encoders import build_encoder
                encoder = build_encoder(load_config(overrides={"encoder": {"name": store.encoder_name, "device": device}}))
            np.save(p, encoder.encode_texts(cap_text[c]))
        dual_caps[c] = np.load(p)
    del encoder

    te_dir = resolve("indexes/text_emb")
    te_dir.mkdir(parents=True, exist_ok=True)
    te = None

    def te_vecs(tag, texts):
        nonlocal te
        p = te_dir / f"{slug(args.text_embedder)}_{tag}.npy"
        if not p.exists():
            if te is None:
                from mmrag.text_embedder import TextEmbedder
                te = TextEmbedder(args.text_embedder, device, sleep_s=sleep_s)
            np.save(p, te.encode(texts, show_progress=True))
        return np.load(p)

    Qt = te_vecs("flickr1k_queries", queries)
    def cap_tag(c):
        return "concat_" + "+".join(SHORT[a] for a in avail) if c == "concat" else slug(c)

    Ct = {c: te_vecs(f"flickr1k_caps_{cap_tag(c)}", cap_text[c]) for c in cap_text}

    rows = []

    def add(method, S, extra=""):
        r = {"method": method, **retrieval_metrics(S, q2img), "params": extra}
        rows.append(r)
        print(f"{method:24s} R@1={r['R@1']:.2f} R@5={r['R@5']:.2f} R@10={r['R@10']:.2f} RSUM={r['RSUM']:.2f}  {extra}",
              flush=True)
        return r

    S_img = Q @ I.T
    add("image", S_img)

    early_caps = dict(dual_caps)
    if len(dual_caps) > 1:
        early_caps["multi"] = l2norm(np.mean([dual_caps[c] for c in avail], axis=0))
    for c, C in early_caps.items():
        S, pk = cv_pick(lambda a: Q @ early_fuse(I, C, a).T, ALPHAS, q2img, fold)
        add(f"early[{SHORT.get(c, c)}]", S, f"alpha={pk}")

    best = {"RSUM": -1}
    for c, Ct_c in Ct.items():
        S_te = Qt @ Ct_c.T
        add(f"te-caption[{SHORT[c]}]", S_te)
        C = dual_caps.get(c, early_caps.get("multi", next(iter(dual_caps.values()))))
        grid = list(itertools.product(ALPHAS, WEIGHTS))
        fn = lambda g: weighted_late_fusion({"v": Q @ early_fuse(I, C, g[0]).T, "t": S_te},
                                            {"v": 1 - g[1] + 1e-9, "t": g[1] + 1e-9})
        S, pk = cv_pick(fn, grid, q2img, fold)
        r = add(f"hybrid[{SHORT[c]}]", S, f"(alpha,w_text)={pk}")
        if r["RSUM"] > best["RSUM"]:
            best = {"RSUM": r["RSUM"], "S": S, "name": r["method"]}

    if not args.no_rerank:
        K = args.rerank_k
        first = best["S"]
        topk = np.argpartition(-first, K, axis=1)[:, :K]
        recall_k = float(np.mean([q2img[q] in topk[q] for q in range(len(q2img))]) * 100)
        print(f"rerank input: {best['name']}, recall@{K} (upper bound) = {recall_k:.2f}")

        def itm_scores(model):
            path = store_dir / f"itm_{slug(model)}.npz"
            known = {}
            if path.exists():
                z = np.load(path)
                known = dict(zip(zip(z["q"].tolist(), z["i"].tolist()), z["s"].tolist()))
            need = [(q, int(i)) for q in range(len(q2img)) for i in topk[q] if (q, int(i)) not in known]
            if need:
                from mmrag.reranker import ITMReranker
                rr = ITMReranker(model, device, sleep_s=sleep_s)
                rr.image_features(sorted({paths[i] for _, i in need}), show_progress=True)
                chunk = 8000
                for c in range(0, len(need), chunk):
                    part = need[c : c + chunk]
                    print(f"ITM pairs {c + len(part)}/{len(need)}", flush=True)
                    scores = rr.score_pairs([queries[q] for q, _ in part], [paths[i] for _, i in part])
                    known.update(zip(part, scores.tolist()))
                    keys = list(known)
                    np.savez(path, q=np.array([k[0] for k in keys]), i=np.array([k[1] for k in keys]),
                             s=np.array([known[k] for k in keys], dtype=np.float32))
                del rr
                torch.cuda.empty_cache()
            return zscore_rows(np.array([[known[(q, int(i))] for i in topk[q]] for q in range(len(q2img))],
                                        dtype=np.float32))

        first_k = zscore_rows(np.take_along_axis(first, topk, axis=1))

        def placed(comb):
            S = first - 1e6
            np.put_along_axis(S, topk, comb, axis=1)
            return S

        tags = {}
        for model in args.reranker.split(","):
            tag = "BLIP2" if "blip2" in model.lower() else "ITM"
            tags[tag] = itm_scores(model)
            S, pk = cv_pick(lambda lam: placed(lam * tags[tag] + (1 - lam) * first_k), WEIGHTS, q2img, fold)
            add(f"rerank-{tag}(top{K})", S, f"lambda={pk}, first={best['name']}")
            add(f"{tag}-only(top{K})", placed(tags[tag]), "lambda=1 (not CV)")
        if len(tags) == 2:
            a, b = tags.values()
            grid = [(x, y) for x in WEIGHTS for y in WEIGHTS if x + y <= 1 + 1e-9]
            S, pk = cv_pick(lambda g: placed(g[0] * a + g[1] * b + (1 - g[0] - g[1]) * first_k), grid, q2img, fold)
            add(f"rerank-{'+'.join(tags)}(top{K})", S, f"(w_{'_'.join(tags)})={pk}, first={best['name']}")

    out = resolve("results/tables")
    out.mkdir(parents=True, exist_ok=True)
    name = store_dir.name
    df = pd.DataFrame(rows)
    df.to_csv(out / f"{name}_improvements.csv", index=False)
    (out / f"{name}_improvements.md").write_text(df.to_markdown(index=False, floatfmt=".2f"), encoding="utf-8")
    print(f"wrote {out / (name + '_improvements.md')}")


if __name__ == "__main__":
    import os
    import sys

    main()
    sys.stdout.flush()
    os._exit(0)
