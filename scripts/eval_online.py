"""End-to-end check of the live search path on Flickr30k-1K.

evaluate.py / experiments.py score dense matrices offline. This script instead sends queries one
by one through MultimodalRetriever.search (FAISS top-K lists, online late fusion, ITM reranking),
exactly as the CLI and the Gradio app do, and reports R@1/5/10 and MRR@10.

python scripts/eval_online.py --store indexes/flickr1k_multi_clip-vit-base-patch32 --n 1000
"""

import argparse
import json
import time

import _bootstrap
import numpy as np
from tqdm import tqdm

from mmrag.config import load_config, resolve
from mmrag.data import load_coco5k, load_flickr1k
from mmrag.pipeline import load_retriever


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--config", default=None, help="extra YAML merged over configs/default.yaml")
    ap.add_argument("--n", type=int, default=1000, help="number of queries (random sample, fixed seed); 0 = all 5000")
    ap.add_argument("--strategies", default="image,fused,hybrid", help="first-stage strategies to measure")
    ap.add_argument("--sleep", type=float, default=None, help="override throttle.gpu_sleep_s")
    ap.add_argument("--reranker", default=None, help="override reranker.name (e.g. Salesforce/blip2-itm-vit-g-coco)")
    ap.add_argument("--weight", type=float, default=None, help="override reranker.weight (lambda)")
    ap.add_argument("--dataset", choices=["flickr1k", "coco5k"], default="flickr1k")
    ap.add_argument("--rerank-strategy", default="hybrid", help="first stage under the reranker ('full' for v2 stores)")
    ap.add_argument("--rerank-k", type=int, default=None, help="override reranker.top_k")
    ap.add_argument("--itc-weight", type=float, default=None, help="override reranker.itc_weight (BLIP-2 ITC in stage 2)")
    ap.add_argument("--suffix", default="", help="appended to the output file name, e.g. _blip2")
    args = ap.parse_args()

    over = {}
    if args.sleep is not None:
        over["throttle"] = {"gpu_sleep_s": args.sleep}
    rr = {k: v for k, v in (("name", args.reranker), ("weight", args.weight), ("top_k", args.rerank_k),
                            ("itc_weight", args.itc_weight)) if v is not None}
    if rr:
        over["reranker"] = rr
    cfg = load_config(args.config, overrides=over or None)
    retriever = load_retriever(args.store, cfg, rerank=True)
    store = retriever.store
    print(store)

    items, queries, q2i = {"flickr1k": load_flickr1k, "coco5k": load_coco5k}[args.dataset]()
    row_of = {it["id"]: r for r, it in enumerate(store.items)}
    gt = [row_of[items[i]["id"]] for i in q2i]
    qidx = np.arange(len(queries))
    if args.n and args.n < len(queries):
        qidx = np.sort(np.random.default_rng(0).choice(len(queries), args.n, replace=False))

    runs = [(s, False) for s in args.strategies.split(",") if s] + [(args.rerank_strategy, True)]
    results = {}
    for strategy, rerank in runs:
        name = f"{strategy}+rerank" if rerank else strategy
        ranks, t0 = [], time.time()
        for q in tqdm(qidx, desc=name):
            hits = retriever.search(text=queries[q], k=10, strategy=strategy, rerank=rerank)
            rows = [row_of[h.item["id"]] for h in hits]
            ranks.append(rows.index(gt[q]) if gt[q] in rows else 10)
        r = np.array(ranks)
        results[name] = {"R@1": (r < 1).mean() * 100, "R@5": (r < 5).mean() * 100, "R@10": (r < 10).mean() * 100,
                         "MRR@10": np.where(r < 10, 1.0 / (r + 1), 0).mean() * 100,
                         "ms/query": (time.time() - t0) / len(qidx) * 1000}
        print(name, {k: round(v, 2) for k, v in results[name].items()})

    tag = resolve(args.store).name
    lines = [f"Online search on `{tag}`, {len(qidx)} text queries (seed 0), "
             f"alpha={store.alpha}, text_weight={cfg['fusion']['text_weight']}, "
             f"rerank {cfg['reranker']['name']} top_k={cfg['reranker']['top_k']} weight={cfg['reranker']['weight']} "
             f"itc_weight={cfg['reranker'].get('itc_weight', 0.0)}, hub_rho={cfg['fusion'].get('hub_rho', 0.0)}, "
             f"gpu_sleep_s={cfg['throttle']['gpu_sleep_s']}", "",
             "| strategy | R@1 | R@5 | R@10 | MRR@10 | ms/query |", "|---|---:|---:|---:|---:|---:|"]
    for name, m in results.items():
        lines.append(f"| {name} | {m['R@1']:.2f} | {m['R@5']:.2f} | {m['R@10']:.2f} | {m['MRR@10']:.2f} | {m['ms/query']:.0f} |")
    out = resolve(f"results/tables/{tag}_online{args.suffix}.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    json_dir = out.parent.parent / "json"
    json_dir.mkdir(parents=True, exist_ok=True)
    (json_dir / out.with_suffix(".json").name).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\n".join(lines))
    print(f"saved {out}")


if __name__ == "__main__":
    import os
    import sys
    main()
    sys.stdout.flush()
    os._exit(0)
