"""Evaluate cross-modal retrieval on Flickr30k 1K test with and without visual-textual fusion.

Text -> Image (5000 caption queries, 1000 images):
  image        query vs. image vectors                     (standard CLIP/SigLIP baseline)
  caption      query vs. BLIP-caption vectors              (text-only proxy)
  early(a)     query vs. norm(a*image + (1-a)*caption)     (document-side early fusion)
  late-z(w)    w*z(image scores) + (1-w)*z(caption scores) (score-level late fusion)
  rrf          reciprocal rank fusion of image and caption rankings
Each method is also run with modality centering ("+ctr").

Image -> Text (1000 image queries, 5000 captions):
  image        image vector vs. caption-text vectors
  early(a)     norm(a*image + (1-a)*own BLIP caption) as query

alpha / w are picked with 2-fold cross-validation over images, so reported numbers are
not tuned on the queries they are measured on. The full sweep is saved for plotting.

Usage: python scripts/evaluate.py --store indexes/flickr1k_clip-vit-base-patch32
"""

import argparse
import json
from pathlib import Path

import _bootstrap
import numpy as np
import pandas as pd

from mmrag.config import load_config, resolve
from mmrag.data import load_flickr1k
from mmrag.encoders import build_encoder
from mmrag.fusion import early_fuse, l2norm, rrf_fusion, weighted_late_fusion
from mmrag.metrics import retrieval_metrics
from mmrag.vector_store import UnifiedVectorStore

GRID = [round(x, 2) for x in list(np.arange(0, 0.8, 0.1)) + list(np.arange(0.8, 1.0001, 0.02))]


def cv_pick(score_fn, grid, positives, query_fold, metric="RSUM"):
    """2-fold CV: choose the grid value on one fold of queries, report it on the other fold.

    query_fold assigns each query to fold 0/1 by its *image*, so no image is in both folds.
    """
    positives = list(positives)
    sweep = {g: score_fn(g) for g in grid}
    picked, out = [], None
    for f in (0, 1):
        tune = query_fold == f
        tune_pos = [positives[i] for i in np.flatnonzero(tune)]
        best = max(grid, key=lambda g: retrieval_metrics(sweep[g][tune], tune_pos)[metric])
        picked.append(best)
        if out is None:
            out = np.empty_like(sweep[best])
        out[~tune] = sweep[best][~tune]
    return out, picked, sweep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    store_dir = resolve(args.store)
    store = UnifiedVectorStore.load(store_dir)
    _, queries, q2img = load_flickr1k()
    q2img = np.asarray(q2img)
    n_img = len(store.rows["image"])
    assert (store.rows["image"] == np.arange(n_img)).all() and (store.rows["caption"] == np.arange(n_img)).all()

    qpath = store_dir / "eval_query_vectors.npy"
    if qpath.exists():
        Q = np.load(qpath)
    else:
        cfg = load_config(overrides={"encoder": {"name": store.encoder_name}})
        Q = build_encoder(cfg).encode_texts(queries, show_progress=True)
        np.save(qpath, Q)

    I_raw, C_raw = store.raw["image"], store.raw["caption"]
    rows, sweeps = [], {}
    image_fold = np.random.default_rng(0).permutation(n_img) % 2
    t2i_fold, i2t_fold = image_fold[q2img], image_fold

    for center in (False, True):
        tag = "+ctr" if center else ""
        if center:
            mu_i = I_raw.mean(0)
            mu_t = np.concatenate([C_raw, Q]).mean(0)
            I, C, Qc = l2norm(I_raw - mu_i), l2norm(C_raw - mu_t), l2norm(Q - mu_t)
        else:
            I, C, Qc = I_raw, C_raw, Q

        S_img, S_cap = Qc @ I.T, Qc @ C.T

        def add(task, method, S, positives, extra=None):
            r = {"task": task, "method": method + tag, **retrieval_metrics(S, positives)}
            if extra:
                r.update(extra)
            rows.append(r)
            print(f"{task:4s} {method + tag:22s} " + " ".join(f"{k}={r[k]:.2f}" for k in ("R@1", "R@5", "R@10", "RSUM")))

        add("t2i", "image", S_img, q2img)
        add("t2i", "caption", S_cap, q2img)
        S, picked, sw = cv_pick(lambda a: Qc @ early_fuse(I, C, a).T, GRID, q2img, t2i_fold)
        add("t2i", "early-fusion", S, q2img, {"param": str([float(p) for p in picked])})
        sweeps[f"t2i early{tag}"] = {a: retrieval_metrics(s, q2img) for a, s in sw.items()}
        S, picked, sw = cv_pick(lambda w: weighted_late_fusion({"i": S_img, "c": S_cap}, {"i": w, "c": 1 - w + 1e-9}),
                                GRID, q2img, t2i_fold)
        add("t2i", "late-fusion-z", S, q2img, {"param": str([float(p) for p in picked])})
        sweeps[f"t2i late{tag}"] = {w: retrieval_metrics(s, q2img) for w, s in sw.items()}
        add("t2i", "rrf", rrf_fusion([S_img, S_cap], k=60), q2img)

        img_pos = [set(np.flatnonzero(q2img == i).tolist()) for i in range(n_img)]
        add("i2t", "image", I @ Qc.T, img_pos)
        S, picked, sw = cv_pick(lambda a: early_fuse(I, C, a) @ Qc.T, GRID, img_pos, i2t_fold)
        add("i2t", "early-fusion", S, img_pos, {"param": str([float(p) for p in picked])})
        sweeps[f"i2t early{tag}"] = {a: retrieval_metrics(s, img_pos) for a, s in sw.items()}

    out = resolve(args.out)
    tables, jsons, figs = out / "tables", out / "json", out / "figures"
    for d in (tables, jsons, figs):
        d.mkdir(parents=True, exist_ok=True)
    name = Path(store_dir).name
    df = pd.DataFrame(rows)
    df.to_csv(tables / f"{name}_metrics.csv", index=False)
    (jsons / f"{name}_sweeps.json").write_text(json.dumps(sweeps, indent=1), encoding="utf-8")
    with open(tables / f"{name}_metrics.md", "w", encoding="utf-8") as f:
        f.write(df.to_markdown(index=False, floatfmt=".2f"))
    print(f"\nwrote {tables / (name + '_metrics.csv')}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, task in zip(axes, ("t2i", "i2t")):
            for key, sweep in sweeps.items():
                if key.startswith(task + " early"):
                    xs = sorted(sweep)
                    ax.plot(xs, [sweep[x]["R@1"] for x in xs], marker="o", label=key.split(" ", 1)[1])
            ax.set_title(f"{task.upper()} R@1 vs alpha  ({name})")
            ax.set_xlabel("alpha  (0 = caption only, 1 = image only)")
            ax.set_ylabel("R@1 (%)")
            ax.grid(alpha=0.3)
            ax.legend()
        fig.tight_layout()
        fig.savefig(figs / f"{name}_alpha_sweep.png", dpi=150)
    except Exception as e:
        print(f"plot skipped: {e}")


if __name__ == "__main__":
    main()
