"""Fail fast before a long GPU run: load each new v3 model once and score two pairs.

python scripts/smoke_v3_models.py --encoder google/siglip2-so400m-patch14-384 \
    --extra Salesforce/blip-itm-large-coco --vlm Qwen/Qwen2.5-VL-7B-Instruct
"""

import argparse

import _bootstrap
import numpy as np
import torch
from PIL import Image

from mmrag.config import resolve
from mmrag.data import load_flickr1k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--extra", default="")
    ap.add_argument("--vlm", default="")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    items, queries, q2i = load_flickr1k()
    paths = [items[q2i[0]]["path"], items[q2i[5]]["path"]]
    qs = [queries[0], queries[5]]

    from mmrag.encoders import DualEncoder
    enc = DualEncoder(args.encoder, device, batch_size=2)
    t, v = enc.encode_texts(qs), enc.encode_images([Image.open(p) for p in paths])
    S = t @ v.T
    print(f"{args.encoder}: dim {t.shape[1]}, similarity matrix {np.round(S, 3).tolist()}", flush=True)
    assert S[0, 0] > S[0, 1] and S[1, 1] > S[1, 0], "dual encoder does not match its own pairs"
    del enc

    for m in [x for x in args.extra.split(",") if x]:
        from mmrag.reranker import ITMReranker
        r = ITMReranker(m, device)
        p = r.score_pairs([qs[0], qs[0]], paths)
        print(f"{m}: P(match) right {p[0]:.3f} wrong {p[1]:.3f}", flush=True)
        assert p[0] > p[1], f"{m} prefers the wrong image"
        del r

    if args.vlm:
        from mmrag.vlm_scorer import VLMScorer
        s = VLMScorer(args.vlm, device, batch_size=2)
        y = s.score_pairs([qs[0], qs[0]], paths)
        print(f"{args.vlm}: yes-no log-odds right {y[0]:.2f} wrong {y[1]:.2f}", flush=True)
        assert y[0] > y[1], f"{args.vlm} prefers the wrong image"
    torch.cuda.empty_cache()
    print("SMOKE-OK")


if __name__ == "__main__":
    main()
