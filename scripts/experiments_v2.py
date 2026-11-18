"""v2 experiments: region features, object tags, BLIP-2 ITC channel, caption-bank hubness normalisation,
multi-encoder ensembling and an ITM+ITC reranker with a cross-validated candidate pool. Text -> image.

Everything is training-free and every weight is chosen with the same 2-fold cross-validation over images as
in v1 (scripts/experiments.py): a weight is picked on the queries of one image fold and applied to the other.
Stages are added greedily; each new weight grid contains 0, so CV can switch a component off.

Stage-1 chain per dual encoder E (all channels z-normalised per query):
  image            <E_T(q), v_I>                                                (baseline)
  hybrid           (1-w) z(<E_T(q), v_F(alpha)>) + w z(<B(q), B(captions)>)    (v1 best first stage)
  +tags            + Florence-2 <OD> object tags in BGE space                    (weight g_tag)
  +regions         + max over region crops (grid + detected objects) of <E_T(q), E_I(crop)>   (g_reg)
  +itc             + BLIP-2 ITC: max over 32 Q-Former queries of <t_q, z_k>     (g_itc)
  +hub             caption-bank hubness normalisation of every channel           (beta, rho)
ensemble           mean of z(stage-1) over all encoders (no extra weight)
Stage 2: top-K (K in --rerank-ks, CV-picked) rescored with lambda_itm z(p_ITM) + lambda_itc z(ITC) + rest z(stage1)

Reproduced baselines in the same code: dual encoders alone, BLIP-2 ITC alone, and BLIP-2's own retrieval
protocol (ITC top-k -> ITM logit + ITC similarity, k=--native-k), plus the v1 system (hybrid + BLIP-2 top-16).

Caption-bank hubness normalisation (CB-Norm): hub images are close to many queries and steal rank 1. QB-Norm
corrects this with a bank of *training* queries; we use the *generated* captions of the gallery images
(BLIP-base and BLIP-large) as the bank, so no test query and no training data is touched:
  s'(q, g) = s(q, g) - rho / beta * L_g,   L_g = log mean_{b: img(b) != g} exp(beta * s(b, g)).

python scripts/experiments_v2.py --dataset flickr1k
python scripts/experiments_v2.py --dataset coco5k --rerank-ks 16,32 --native-k 0
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import _bootstrap
import numpy as np
import torch

from mmrag.config import load_config, resolve
from mmrag.data import load_coco5k, load_flickr1k
from mmrag.fusion import (apply_hubness, bank_log_normaliser, dual_softmax, early_fused_scores, l2norm,
                          max_pool_by_owner, zscore_rows)
from mmrag.vector_store import UnifiedVectorStore

FLORENCE = "florence-community/Florence-2-large"
CAPS4 = ["Salesforce/blip-image-captioning-base", "Salesforce/blip-image-captioning-large",
         f"{FLORENCE}#<DETAILED_CAPTION>", f"{FLORENCE}#<MORE_DETAILED_CAPTION>"]
BANK_CAPS = ["Salesforce/blip-image-captioning-base", "Salesforce/blip-image-captioning-large"]
SHORT = {CAPS4[0]: "blip-b", CAPS4[1]: "blip-l", CAPS4[2]: "flor-d", CAPS4[3]: "flor-md"}
ENCODERS = ["openai/clip-vit-base-patch32", "openai/clip-vit-large-patch14", "google/siglip-base-patch16-224"]
SHORT_ENC = {"openai/clip-vit-base-patch32": "CLIP-B/32", "openai/clip-vit-large-patch14": "CLIP-L/14",
             "google/siglip-base-patch16-224": "SigLIP-B", "google/siglip2-so400m-patch14-384": "SigLIP2-SO400M"}
ALPHAS = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0]
WEIGHTS = [round(float(x), 2) for x in np.linspace(0, 1, 11)]
HUB_GRID = [(b, r) for b in (10.0, 20.0, 50.0) for r in (0.0, 0.25, 0.5, 0.75, 1.0)]
LOADERS = {"flickr1k": load_flickr1k, "coco5k": load_coco5k}


def slug(name: str) -> str:
    import re
    return re.sub(r"[^a-zA-Z0-9]+", "-", name.split("/")[-1])


def auto_device(min_free_gb: float = 3.0) -> str:
    if not torch.cuda.is_available():
        return "cpu"
    free, _ = torch.cuda.mem_get_info()
    return "cuda" if free / 1e9 >= min_free_gb else "cpu"


# ---- fast metrics (one positive per query) -----------------------------------------------------------------

def gt_rank(S: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """0-based rank of the ground-truth column in each row (ties count against the ground truth)."""
    g = S[np.arange(len(S)), gt]
    return (S > g[:, None]).sum(1) + (S == g[:, None]).sum(1) - 1


def metrics_from_ranks(r: np.ndarray) -> dict[str, float]:
    out = {f"R@{k}": float((r < k).mean() * 100) for k in (1, 5, 10)}
    out["RSUM"] = out["R@1"] + out["R@5"] + out["R@10"]
    out["MRR"] = float((1.0 / (r + 1)).mean() * 100)
    return out


def paired_bootstrap(r_a: np.ndarray, r_b: np.ndarray, n: int = 2000, seed: int = 0) -> dict[str, float]:
    """Delta R@1 (b - a) with a 95% bootstrap CI over queries and a two-sided p-value (sign of the delta)."""
    rng = np.random.default_rng(seed)
    a, b = (r_a < 1).astype(np.float64), (r_b < 1).astype(np.float64)
    d = b - a
    idx = rng.integers(0, len(d), size=(n, len(d)))
    boots = d[idx].mean(1) * 100
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    return {"delta_R@1": float(d.mean() * 100), "ci_lo": float(np.percentile(boots, 2.5)),
            "ci_hi": float(np.percentile(boots, 97.5)), "p": float(min(1.0, p))}


# ---- cross-validation over image folds ---------------------------------------------------------------------

class CV:
    """Greedy, stage-wise 2-fold CV. params[f] holds the weights picked on fold f; they score fold 1-f."""

    def __init__(self, fold: np.ndarray, gt: np.ndarray, metric: str = "RSUM"):
        self.fold, self.gt, self.metric = fold, gt, metric
        self.params = [{}, {}]

    def pick(self, key, grid, score_fn):
        """score_fn(params) -> (n_q, n_img). Picks `key` on each fold, stores it, returns the CV-composed matrix."""
        best = [None, None]
        for f in (0, 1):
            m = self.fold == f
            res = {}
            for g in grid:
                S = score_fn({**self.params[f], key: g}, m)
                res[g] = metrics_from_ranks(gt_rank(S, self.gt[m]))[self.metric]
            best[f] = max(grid, key=lambda g: res[g])
            self.params[f][key] = best[f]
        return self.compose(score_fn), best

    def compose(self, score_fn) -> np.ndarray:
        out = None
        for f in (0, 1):
            other = self.fold != f
            S = score_fn(self.params[f], other)
            if out is None:
                out = np.empty((len(self.fold), S.shape[1]), np.float32)
            out[other] = S
        return out


# ---- main --------------------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(LOADERS), default="flickr1k")
    ap.add_argument("--config", default=None, help="extra YAML merged over configs/default.yaml")
    ap.add_argument("--encoders", default=",".join(ENCODERS))
    ap.add_argument("--store-pattern", default="indexes/{dataset}_multi_{enc}",
                    help="multi-caption store per encoder (built by build_index.py --text-embedder)")
    ap.add_argument("--text-embedder", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--reranker", default="Salesforce/blip2-itm-vit-g-coco")
    ap.add_argument("--rerank-ks", default="16,32,64")
    ap.add_argument("--native-k", type=int, default=128, help="BLIP-2 protocol baseline: ITC top-k -> ITM (0 = skip)")
    ap.add_argument("--regions", default="grid,od", help="region kinds, or '' to skip")
    ap.add_argument("--no-tags", action="store_true")
    ap.add_argument("--no-itc", action="store_true")
    ap.add_argument("--no-hub", action="store_true")
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--no-region-ablation", action="store_true", help="skip grid-only / od-only rows (saves RAM)")
    ap.add_argument("--max-queries", type=int, default=0, help="random query subset (seed 0); 0 = all")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--sleep", type=float, default=None, help="override throttle.gpu_sleep_s (default: config)")
    ap.add_argument("--tag", default="", help="suffix for the output files")
    ap.add_argument("--itm-batch", type=int, default=32, help="ITM / ITC batch size (raise on large GPUs)")
    # v3 cascade on the ensemble chain (all optional)
    ap.add_argument("--extra-rerankers", default="", help="extra ITM cross-encoders (comma-separated), e.g. "
                    "Salesforce/blip-itm-large-coco; added greedily to the v2 stage-2 score with CV weights")
    ap.add_argument("--ft-rerankers", default="", help="cross-encoders fine-tuned on the target benchmark's train split "
                    "(Flickr30k only); reported as a separate FT protocol, e.g. Salesforce/blip-itm-large-flickr")
    ap.add_argument("--vlm", default="", help="generative VLM for yes/no verification of the top-M, e.g. "
                    "Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--vlm-m", default="4,8", help="candidate counts M for VLM verification (CV-selected)")
    ap.add_argument("--vlm-batch", type=int, default=16)
    args = ap.parse_args()

    t_start = time.time()
    device = auto_device() if args.device == "auto" else args.device
    cfg = load_config(args.config)
    sleep_s = cfg["throttle"]["gpu_sleep_s"] if args.sleep is None else args.sleep
    encoders = [e for e in args.encoders.split(",") if e]
    region_kinds = [k for k in args.regions.split(",") if k]
    ks = sorted(int(k) for k in args.rerank_ks.split(",") if k)
    print(f"device={device} sleep={sleep_s} encoders={encoders} regions={region_kinds} K={ks}", flush=True)

    items, queries_all, q2img_all = LOADERS[args.dataset]()
    paths = [it["path"] for it in items]
    n_img = len(paths)
    q2img_all = np.asarray(q2img_all)
    qsel = np.arange(len(queries_all))
    if args.max_queries and args.max_queries < len(qsel):
        qsel = np.sort(np.random.default_rng(0).choice(len(qsel), args.max_queries, replace=False))
    queries = [queries_all[i] for i in qsel]
    gt = q2img_all[qsel]
    fold = (np.random.default_rng(0).permutation(n_img) % 2)[gt]
    n_q = len(queries)
    print(f"{args.dataset}: {n_img} images, {n_q} queries", flush=True)

    cache_dir = resolve(f"indexes/{args.dataset}_v2")
    cache_dir.mkdir(parents=True, exist_ok=True)
    te_dir = resolve("indexes/text_emb")
    te_dir.mkdir(parents=True, exist_ok=True)

    cap_cache = json.loads(resolve("data/caption_cache.json").read_text(encoding="utf-8"))
    missing = [c for c in CAPS4 if not all(f"{c}|{p}" in cap_cache for p in paths)]
    if missing:
        raise SystemExit(f"captions missing in data/caption_cache.json for {missing}; run caption_images.py first")
    concat = [" ".join(cap_cache[f"{c}|{p}"] for c in CAPS4) for p in paths]
    bank_text = [cap_cache[f"{c}|{p}"] for c in BANK_CAPS for p in paths]
    bank_owner = np.tile(np.arange(n_img), len(BANK_CAPS))

    od_cache = {}
    if "od" in region_kinds or not args.no_tags:
        from mmrag.regions import ensure_od_cache
        od_cache = ensure_od_cache(paths, device=device, sleep_s=sleep_s)
    if not args.no_tags:
        from mmrag.regions import object_tags
        tags = [object_tags(p, od_cache) or "an image" for p in paths]

    # ---- BGE text channel -----------------------------------------------------------------------------
    te_model = None

    def te_vecs(name, texts):
        nonlocal te_model
        p = te_dir / f"{slug(args.text_embedder)}_{name}.npy"
        if not p.exists():
            if te_model is None:
                from mmrag.text_embedder import TextEmbedder
                te_model = TextEmbedder(args.text_embedder, device, sleep_s=sleep_s)
            np.save(p, te_model.encode(texts, show_progress=True))
        return np.load(p)

    caps_tag = "concat_" + "+".join(SHORT[c] for c in CAPS4)
    Qt_all = te_vecs(f"{args.dataset}_queries", queries_all)
    Qt = Qt_all[qsel]
    Ct = te_vecs(f"{args.dataset}_caps_{caps_tag}", concat)
    Bt = te_vecs(f"{args.dataset}_bank_{'+'.join(SHORT[c] for c in BANK_CAPS)}", bank_text)
    Gt = te_vecs(f"{args.dataset}_odtags", tags) if not args.no_tags else None
    del te_model
    te_model = None

    # channel matrices: eval queries and bank queries
    ch_eval, ch_bank = {}, {}
    ch_eval["te"], ch_bank["te"] = Qt @ Ct.T, Bt @ Ct.T
    if Gt is not None:
        ch_eval["tags"], ch_bank["tags"] = Qt @ Gt.T, Bt @ Gt.T

    # ---- BLIP-2 ITC channel ---------------------------------------------------------------------------
    rr = None

    def reranker():
        nonlocal rr
        if rr is None:
            from mmrag.reranker import ITMReranker
            rr = ITMReranker(args.reranker, device, batch_size=args.itm_batch, sleep_s=sleep_s)
        return rr

    def itc_matrix(T, Z, chunk=512):
        """max over the 32 query embeddings: (n_t, 256) x (n_img, 32, 256) -> (n_t, n_img)."""
        dev = "cuda" if device.startswith("cuda") else "cpu"
        Zt = torch.from_numpy(Z.astype(np.float16 if dev == "cuda" else np.float32)).to(dev)
        flat = Zt.reshape(-1, Zt.shape[-1])
        out = np.empty((len(T), Z.shape[0]), np.float32)
        for i in range(0, len(T), chunk):
            t = torch.from_numpy(T[i : i + chunk]).to(dev, flat.dtype)
            out[i : i + chunk] = (t @ flat.T).reshape(len(t), Z.shape[0], Z.shape[1]).amax(-1).float().cpu().numpy()
        del Zt, flat
        return out

    use_itc = not args.no_itc and "blip2" in args.reranker.lower()
    if use_itc or args.native_k:
        p_img = cache_dir / f"itc_image_{slug(args.reranker)}.npy"
        p_txt = cache_dir / f"itc_queries_{slug(args.reranker)}.npy"
        p_bank = cache_dir / f"itc_bank_{slug(args.reranker)}.npy"
        if not p_img.exists():
            np.save(p_img, reranker().itc_image_embeds(paths, show_progress=True).astype(np.float16))
        if not p_txt.exists():
            np.save(p_txt, reranker().itc_text_embeds(queries_all, show_progress=True))
        if not p_bank.exists():
            np.save(p_bank, reranker().itc_text_embeds(bank_text, show_progress=True))
        Z = np.load(p_img).astype(np.float32)
        X_eval = itc_matrix(np.load(p_txt)[qsel], Z)
        X_bank = itc_matrix(np.load(p_bank), Z)
        del Z
        if use_itc:
            ch_eval["itc"], ch_bank["itc"] = X_eval, X_bank

    # ---- per-encoder dual-encoder channels ------------------------------------------------------------
    per_enc = {}
    for enc_name in encoders:
        store_dir = resolve(args.store_pattern.format(dataset=args.dataset, enc=slug(enc_name)))
        if not (store_dir / "store.json").exists():
            raise SystemExit(f"missing store {store_dir}; build it with scripts/build_index.py --source "
                             f"{args.dataset} --encoder {enc_name} --text-embedder --out {store_dir}")
        store = UnifiedVectorStore.load(store_dir)
        ids = [store.items[r]["id"] for r in store.rows["image"]]
        assert ids == [it["id"] for it in items], f"{store_dir}: image order differs from the loader"
        assert (store.rows["caption"] == store.rows["image"]).all()
        I, C = store.raw["image"], store.raw["caption"]

        encoder = None

        def get_encoder():
            nonlocal encoder
            if encoder is None:
                from mmrag.encoders import build_encoder
                encoder = build_encoder(load_config(args.config, overrides={"encoder": {"name": store.encoder_name, "device": device},
                                                               "throttle": {"gpu_sleep_s": sleep_s}}))
            return encoder

        def cached(name, fn):
            p = store_dir / name
            if not p.exists():
                np.save(p, fn())
            return np.load(p)

        Q_all = cached(f"v2_queries_{args.dataset}.npy", lambda: get_encoder().encode_texts(queries_all, show_progress=True))
        Q = Q_all[qsel]
        Bq = cached("v2_bank_queries.npy", lambda: get_encoder().encode_texts(bank_text, show_progress=True))
        e = {"A": Q @ I.T, "B": Q @ C.T, "bA": Bq @ I.T, "bB": Bq @ C.T, "cos": (I * C).sum(1)}
        if region_kinds:
            from mmrag.regions import encode_regions, region_path
            rp = region_path(store_dir, region_kinds)
            if not rp.exists():
                v, own, kind = encode_regions(get_encoder(), paths, region_kinds, od_cache)
                np.savez(rp, vectors=v.astype(np.float16), owner=own, kind=kind)
            z = np.load(rp)
            Rv, own, kind = z["vectors"].astype(np.float32), z["owner"], z["kind"]

            def pooled(T, mask):
                out = np.full((len(T), n_img), -1.0, np.float32)
                for i in range(0, len(T), 2048):
                    out[i : i + 2048] = max_pool_by_owner(T[i : i + 2048] @ Rv[mask].T, own[mask], n_img)
                return out

            all_mask = np.ones(len(own), bool)
            e["R"], e["bR"] = pooled(Q, all_mask), pooled(Bq, all_mask)
            if len(region_kinds) > 1 and not args.no_region_ablation:
                for kname, kid in (("grid", 0), ("od", 1)):
                    if kname in region_kinds:
                        e[f"R_{kname}"] = pooled(Q, kind == kid)
            print(f"{SHORT_ENC.get(enc_name, enc_name)}: {len(own)} region vectors "
                  f"({len(own) / n_img:.1f} per image)", flush=True)
        del encoder
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        per_enc[enc_name] = e

    # ---- scoring --------------------------------------------------------------------------------------
    L_cache: dict = {}

    def hub_L(key, bank_fn, beta):
        k = (key, beta)
        if k not in L_cache:
            L_cache[k] = bank_log_normaliser(bank_fn(), bank_owner, beta)
        return L_cache[k]

    def stage1(enc_name, P, rows, upto="all", region_key="R"):
        """Score eval queries `rows` with params P. upto: hybrid|tags|regions|itc|all."""
        e = per_enc[enc_name]
        alpha, w = P["alpha"], P["w"]
        beta, rho = P.get("hub", (20.0, 0.0))

        def adj(S, key, bank_fn):
            if rho == 0:
                return S
            return apply_hubness(S, hub_L((enc_name,) + key, bank_fn, beta), beta, rho)

        F = early_fused_scores(e["A"][rows], e["B"][rows], e["cos"], alpha)
        F = adj(F, ("F", alpha), lambda: early_fused_scores(e["bA"], e["bB"], e["cos"], alpha))
        T = adj(ch_eval["te"][rows], ("te",), lambda: ch_bank["te"])
        s = (1 - w) * zscore_rows(F) + w * zscore_rows(T)
        order = ["tags", "regions", "itc"]
        stop = len(order) if upto == "all" else (order.index(upto) + 1 if upto in order else 0)
        for name in order[:stop]:
            g = P.get(name, 0.0)
            if g == 0:
                continue
            if name == "tags" and "tags" in ch_eval:
                X = adj(ch_eval["tags"][rows], ("tags",), lambda: ch_bank["tags"])
            elif name == "regions" and region_key in e:
                X = adj(e[region_key][rows], (region_key,), lambda: e["bR"]) if region_key == "R" else e[region_key][rows]
            elif name == "itc" and "itc" in ch_eval:
                X = adj(ch_eval["itc"][rows], ("itc",), lambda: ch_bank["itc"])
            else:
                continue
            s = (1 - g) * zscore_rows(s) + g * zscore_rows(X)
        return s.astype(np.float32)

    rows_out, ranks_store, picks_out = [], {}, {}
    rerank_results: dict = {}
    v3_info: dict = {}

    def add(method, S=None, ranks=None, extra="", group=""):
        r = gt_rank(S, gt) if ranks is None else ranks
        m = metrics_from_ranks(r)
        rows_out.append({"group": group, "method": method, **m, "params": extra})
        ranks_store[method] = r
        print(f"{method:44s} R@1={m['R@1']:6.2f} R@5={m['R@5']:6.2f} R@10={m['R@10']:6.2f} RSUM={m['RSUM']:7.2f}  {extra}",
              flush=True)
        return m

    all_rows = np.ones(n_q, bool)
    finals, v1_firsts = {}, {}
    for enc_name in encoders:
        E = SHORT_ENC.get(enc_name, slug(enc_name))
        e = per_enc[enc_name]
        grp = f"stage1 {E}"
        add(f"{E} image", e["A"], group=grp)
        cv = CV(fold, gt)
        S, pk = cv.pick("alpha", ALPHAS, lambda P, m: early_fused_scores(e["A"][m], e["B"][m], e["cos"], P["alpha"]))
        add(f"{E} early[multi]", S, extra=f"alpha={pk}", group=grp)
        cv = CV(fold, gt)
        S, pk = cv.pick("aw", list(itertools.product(ALPHAS, WEIGHTS)),
                        lambda P, m: stage1(enc_name, {"alpha": P["aw"][0], "w": P["aw"][1]}, m, upto="hybrid"))
        for f in (0, 1):
            cv.params[f]["alpha"], cv.params[f]["w"] = cv.params[f].pop("aw")
        add(f"{E} hybrid (v1)", S, extra=f"(alpha,w)={pk}", group=grp)
        v1_firsts[E] = S

        # ablations on top of hybrid (not chained)
        for key in [k for k in ("R_grid", "R_od") if k in e]:
            cva = CV(fold, gt)
            cva.params = [dict(p) for p in cv.params]
            S_a, pk_a = cva.pick("regions", WEIGHTS, lambda P, m, key=key: stage1(enc_name, P, m, "regions", key))
            add(f"{E} hybrid +regions[{key[2:]}] (ablation)", S_a, extra=f"g={pk_a}", group=grp)
        if "itc" in ch_eval:
            cva = CV(fold, gt)
            cva.params = [dict(p) for p in cv.params]
            S_a, pk_a = cva.pick("itc", WEIGHTS, lambda P, m: stage1(enc_name, {**P, "tags": 0, "regions": 0}, m, "itc"))
            add(f"{E} hybrid +itc only (ablation)", S_a, extra=f"g={pk_a}", group=grp)

        # main chain
        for stage, on in (("tags", "tags" in ch_eval), ("regions", "R" in e), ("itc", "itc" in ch_eval)):
            if not on:
                continue
            S, pk = cv.pick(stage, WEIGHTS, lambda P, m, st=stage: stage1(enc_name, P, m, st))
            add(f"{E} +{stage}", S, extra=f"g={pk}", group=grp)
        if not args.no_hub:
            S, pk = cv.pick("hub", HUB_GRID, lambda P, m: stage1(enc_name, P, m))
            add(f"{E} +hub (CB-Norm)", S, extra=f"(beta,rho)={pk}", group=grp)
        finals[E] = S
        picks_out[E] = cv.params

    if len(finals) > 1:
        S_ens = np.mean([zscore_rows(S) for S in finals.values()], axis=0).astype(np.float32)
        add(f"ensemble({'+'.join(finals)})", S_ens, extra="equal weights, no tuning", group="stage1 ensemble")
        finals["ensemble"] = S_ens
    best_first_key = "ensemble" if "ensemble" in finals else next(iter(finals))
    add(f"{best_first_key} + dual-softmax [TRANSDUCTIVE, not comparable]",
        dual_softmax(zscore_rows(finals[best_first_key]), temp=1.0), extra="uses all test queries", group="ablation")

    # reference channels alone
    add("BGE caption concat only", ch_eval["te"], group="reference")
    if "tags" in ch_eval:
        add("BGE object tags only", ch_eval["tags"], group="reference")
    if args.native_k or use_itc:
        add("BLIP-2 ITC only (full gallery)", X_eval, group="reference")

    # ---- stage 2: ITM rerank --------------------------------------------------------------------------
    if not args.no_rerank:
        Kmax = max(ks)
        chains = {f"{k} final": S for k, S in finals.items()}
        chains.update({f"{E} hybrid (v1)": S for E, S in v1_firsts.items()})
        topk = {name: np.argpartition(-S, Kmax, axis=1)[:, :Kmax] for name, S in chains.items()}
        for name, S in chains.items():   # sort each top-K by first-stage score so prefixes are top-k for k < Kmax
            sc = np.take_along_axis(S, topk[name], 1)
            topk[name] = np.take_along_axis(topk[name], np.argsort(-sc, 1), 1)
        native = None
        if args.native_k:
            native = np.argpartition(-X_eval, args.native_k, axis=1)[:, :args.native_k]
            native = np.take_along_axis(native, np.argsort(-np.take_along_axis(X_eval, native, 1), 1), 1)

        itm_path = cache_dir / f"itm_{slug(args.reranker)}.npz"
        known_p, known_l = {}, {}
        if itm_path.exists():
            z = np.load(itm_path)
            keys = list(zip(z["q"].tolist(), z["i"].tolist()))
            known_p, known_l = dict(zip(keys, z["p"].tolist())), dict(zip(keys, z["l"].tolist()))

        def save_itm():
            keys = list(known_p)
            np.savez(itm_path, q=np.array([k[0] for k in keys], np.int64), i=np.array([k[1] for k in keys], np.int64),
                     p=np.array([known_p[k] for k in keys], np.float32), l=np.array([known_l[k] for k in keys], np.float32))

        need = set()
        for tk in list(topk.values()) + ([native] if native is not None else []):
            for qi in range(n_q):
                q0 = int(qsel[qi])
                need.update((q0, int(i)) for i in tk[qi] if (q0, int(i)) not in known_p)
        print(f"ITM pairs to score: {len(need)} (cached {len(known_p)})", flush=True)
        if need:
            by_img: dict[int, list[int]] = {}
            for q0, i in need:
                by_img.setdefault(i, []).append(q0)
            imgs = sorted(by_img)
            r = reranker()
            done, t0 = 0, time.time()
            for c in range(0, len(imgs), 128):
                chunk = imgs[c : c + 128]
                pairs = [(q0, i) for i in chunk for q0 in by_img[i]]
                r.image_features([paths[i] for i in chunk])
                p, l = r.score_pairs([queries_all[q0] for q0, _ in pairs], [paths[i] for _, i in pairs], return_logits=True)
                known_p.update(zip(pairs, p.tolist()))
                known_l.update(zip(pairs, l.tolist()))
                r.clear_cache()
                done += len(pairs)
                print(f"  ITM {done}/{len(need)} pairs, {done / (time.time() - t0):.0f} pairs/s", flush=True)
                if c // 128 % 10 == 9:
                    save_itm()
            save_itm()

        def itm_of(tk, which="p"):
            d = known_p if which == "p" else known_l
            return np.array([[d[(int(qsel[qi]), int(i))] for i in tk[qi]] for qi in range(n_q)], np.float32)

        def base_rank(S):
            return gt_rank(S, gt)

        simplex = [(a, b) for a in WEIGHTS for b in WEIGHTS if a + b <= 1 + 1e-9]
        for name, S in chains.items():
            grp = f"stage2 {name}"
            tkK = topk[name]
            base = base_rank(S)
            P_itm = zscore_rows(itm_of(tkK))
            X_k = zscore_rows(np.take_along_axis(X_eval, tkK, 1)) if (use_itc or args.native_k) else None
            F_k = zscore_rows(np.take_along_axis(S, tkK, 1))
            for K in ks:
                rec = float(np.mean([gt[q] in tkK[q, :K] for q in range(n_q)]) * 100)
                print(f"{name}: recall@{K} = {rec:.2f}", flush=True)
                rows_out.append({"group": grp, "method": f"{name} recall@{K} (bound)", "R@1": np.nan, "R@5": np.nan,
                                 "R@10": rec, "RSUM": np.nan, "MRR": np.nan, "params": "upper bound of stage 2"})

            def comb_for(P, m, use_x):
                K = P["K"]
                a, b = P["lam"] if use_x else (P["lam"], 0.0)
                pz = zscore_rows(P_itm[m][:, :K])
                fz = zscore_rows(F_k[m][:, :K])
                c = a * pz + (1 - a - b) * fz
                if use_x and X_k is not None:
                    c = c + b * zscore_rows(X_k[m][:, :K])
                return c

            for use_x in ([False, True] if X_k is not None else [False]):
                grid = [(K, lam) for K in ks for lam in (simplex if use_x else WEIGHTS)]
                res_by_fold = []
                for f in (0, 1):
                    m = fold == f
                    sc = {}
                    for K, lam in grid:
                        c = comb_for({"K": K, "lam": lam}, m, use_x)
                        r_ = rerank_ranks_sub(c, tkK[m][:, :K], gt[m], base[m])
                        sc[(K, lam)] = metrics_from_ranks(r_)["RSUM"]
                    res_by_fold.append(max(grid, key=lambda g: sc[g]))
                ranks = np.empty(n_q, np.int64)
                for f in (0, 1):
                    other = fold != f
                    K, lam = res_by_fold[f]
                    c = comb_for({"K": K, "lam": lam}, other, use_x)
                    ranks[other] = rerank_ranks_sub(c, tkK[other][:, :K], gt[other], base[other])
                label = "ITM+ITC" if use_x else "ITM"
                key = f"{name} -> rerank {label}"
                add(key, ranks=ranks, extra=f"(K,lambda) per fold={res_by_fold}", group=grp)
                rerank_results[key] = res_by_fold
            # v1 protocol for exact reproduction: K=16, ITM only
            if name.endswith("hybrid (v1)") and 16 in ks:
                cv16 = []
                for f in (0, 1):
                    m = fold == f
                    cv16.append(max(WEIGHTS, key=lambda lam: metrics_from_ranks(rerank_ranks_sub(
                        comb_for({"K": 16, "lam": lam}, m, False), tkK[m][:, :16], gt[m], base[m]))["RSUM"]))
                ranks = np.empty(n_q, np.int64)
                for f in (0, 1):
                    other = fold != f
                    ranks[other] = rerank_ranks_sub(comb_for({"K": 16, "lam": cv16[f]}, other, False),
                                                    tkK[other][:, :16], gt[other], base[other])
                add(f"{name} -> rerank ITM top-16 (v1 protocol)", ranks=ranks, extra=f"lambda={cv16}", group=grp)

        if native is not None:
            l_n = itm_of(native, "l")
            x_n = np.take_along_axis(X_eval, native, 1)
            comb = l_n + x_n
            base = gt_rank(X_eval, gt)
            add(f"BLIP-2 protocol (ITC top-{args.native_k} -> ITM logit + ITC)",
                ranks=rerank_ranks_sub(comb, native, gt, base), extra="reproduced, no tuning", group="reference")

        # ---- v3 cascade: extra cross-encoders and VLM verification on the ensemble chain -------------
        extra = [m for m in args.extra_rerankers.split(",") if m]
        ft = [m for m in args.ft_rerankers.split(",") if m] if args.dataset == "flickr1k" else []
        if extra or ft or args.vlm:
            v3_cascade(args, chains, finals, topk, rerank_results, itm_of, X_eval, extra, ft, qsel, queries_all,
                       paths, gt, fold, n_q, cache_dir, device, sleep_s, add, ranks_store, v3_info)

    # ---- significance tests -------------------------------------------------------------------------
    sig = {}
    ours = [k for k in ranks_store if k.startswith("ensemble final -> rerank ITM+ITC")] or \
           [k for k in ranks_store if k.endswith("final -> rerank ITM+ITC")] or \
           [k for k in ranks_store if k.endswith("final -> rerank ITM")]
    if ours:   # fixed a priori: the full system (ensemble if available), not the best row on the test set
        for ref in [k for k in ranks_store if "(v1 protocol)" in k or k.startswith("BLIP-2 protocol")]:
            sig[f"{ours[0]} vs {ref}"] = paired_bootstrap(ranks_store[ref], ranks_store[ours[0]])
            print(f"significance: {ours[0]} vs {ref}: {sig[f'{ours[0]} vs {ref}']}", flush=True)
    for v3_key in [k for k in ranks_store if k.startswith("v3 ")]:   # v3 final rows vs v2 and the BLIP-2 protocol
        for ref in [k for k in ranks_store if k == (ours[0] if ours else None) or k.startswith("BLIP-2 protocol")]:
            sig[f"{v3_key} vs {ref}"] = paired_bootstrap(ranks_store[ref], ranks_store[v3_key])
            print(f"significance: {v3_key} vs {ref}: {sig[f'{v3_key} vs {ref}']}", flush=True)

    # ---- outputs ------------------------------------------------------------------------------------
    import pandas as pd

    out_t, out_j = resolve("results/tables"), resolve("results/json")
    out_t.mkdir(parents=True, exist_ok=True)
    out_j.mkdir(parents=True, exist_ok=True)
    name = f"{args.dataset}_v2{args.tag}"
    df = pd.DataFrame(rows_out)
    df.to_csv(out_t / f"{name}.csv", index=False)
    header = (f"v2 experiments on {args.dataset}: {n_img} images, {n_q} text queries; encoders {encoders}; "
              f"reranker {args.reranker}; K grid {ks}; regions {region_kinds}; 2-fold CV over images (seed 0).\n\n")
    (out_t / f"{name}.md").write_text(header + df.to_markdown(index=False, floatfmt=".2f") + "\n", encoding="utf-8")

    def flat_weights(p):
        w, gt_, gr, gi = p.get("w", 0), p.get("tags", 0), p.get("regions", 0), p.get("itc", 0)
        return {"fused": (1 - w) * (1 - gt_) * (1 - gr) * (1 - gi), "te_caption": w * (1 - gt_) * (1 - gr) * (1 - gi),
                "te_tags": gt_ * (1 - gr) * (1 - gi), "region": gr * (1 - gi), "itc": gi}

    summary = {"dataset": args.dataset, "n_images": n_img, "n_queries": n_q, "encoders": encoders,
               "encoder_short": {SHORT_ENC.get(e, slug(e)): e for e in encoders},
               "reranker": args.reranker, "rerank_ks": ks, "regions": region_kinds,
               "metrics": {r["method"]: {k: r[k] for k in ("R@1", "R@5", "R@10", "RSUM", "MRR")} for r in rows_out},
               "picked": {E: [{k: (list(v) if isinstance(v, tuple) else v) for k, v in p.items()} for p in ps]
                          for E, ps in picks_out.items()},
               "live_flat_weights_suggestion": {E: [flat_weights(p) for p in ps] for E, ps in picks_out.items()},
               "rerank_picks": {k: [[K, list(lam) if isinstance(lam, tuple) else lam] for K, lam in v]
                                for k, v in rerank_results.items()},
               "significance": sig, "v3": v3_info, "runtime_min": (time.time() - t_start) / 60}
    (out_j / f"{name}.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print(f"wrote {out_t / (name + '.md')} and {out_j / (name + '.json')}  ({summary['runtime_min']:.1f} min)")


def _pair_cache(path):
    if path.exists():
        z = np.load(path)
        return dict(zip(zip(z["q"].tolist(), z["i"].tolist()), z["s"].tolist()))
    return {}


def _save_pairs(path, known):
    keys = list(known)
    np.savez(path, q=np.array([k[0] for k in keys], np.int64), i=np.array([k[1] for k in keys], np.int64),
             s=np.array([known[k] for k in keys], np.float32))


def score_cross_encoder(model, pairs, queries_all, paths, cache_dir, device, sleep_s, batch):
    """P(match) of an ITM cross-encoder (BLIP or BLIP-2) for (query index, image index) pairs, cached on disk."""
    path = cache_dir / f"itmx_{slug(model)}.npz"
    known = _pair_cache(path)
    need = sorted({p for p in pairs if p not in known}, key=lambda p: p[1])
    print(f"{model}: {len(need)} pairs to score (cached {len(known)})", flush=True)
    if need:
        from mmrag.reranker import ITMReranker
        r = ITMReranker(model, device, batch_size=batch, sleep_s=sleep_s)
        by_img: dict[int, list[int]] = {}
        for q0, i in need:
            by_img.setdefault(i, []).append(q0)
        imgs, t0, done = sorted(by_img), time.time(), 0
        for c in range(0, len(imgs), 128):
            chunk = imgs[c : c + 128]
            part = [(q0, i) for i in chunk for q0 in by_img[i]]
            r.image_features([paths[i] for i in chunk])
            known.update(zip(part, r.score_pairs([queries_all[q] for q, _ in part], [paths[i] for _, i in part]).tolist()))
            r.clear_cache()
            done += len(part)
            print(f"  {model.split('/')[-1]} {done}/{len(need)} pairs, {done / (time.time() - t0):.0f} pairs/s", flush=True)
        _save_pairs(path, known)
        del r
        torch.cuda.empty_cache()
    return known


def score_vlm(model, pairs, queries_all, paths, cache_dir, device, sleep_s, batch):
    """Yes/no log-odds of a generative VLM for (query index, image index) pairs, cached on disk."""
    path = cache_dir / f"vlm_{slug(model)}.npz"
    known = _pair_cache(path)
    need = sorted({p for p in pairs if p not in known}, key=lambda p: p[1])
    print(f"{model}: {len(need)} pairs to score (cached {len(known)})", flush=True)
    if need:
        from mmrag.vlm_scorer import VLMScorer
        v = VLMScorer(model, device, batch_size=batch, sleep_s=sleep_s)
        t0 = time.time()
        for c in range(0, len(need), 2048):
            part = need[c : c + 2048]
            known.update(zip(part, v.score_pairs([queries_all[q] for q, _ in part], [paths[i] for _, i in part]).tolist()))
            print(f"  VLM {c + len(part)}/{len(need)} pairs, {(c + len(part)) / (time.time() - t0):.1f} pairs/s", flush=True)
            if c // 2048 % 5 == 4:
                _save_pairs(path, known)
        _save_pairs(path, known)
        del v
        torch.cuda.empty_cache()
    return known


def v3_cascade(args, chains, finals, topk, rerank_results, itm_of, X_eval, extra, ft, qsel, queries_all, paths, gt,
               fold, n_q, cache_dir, device, sleep_s, add, ranks_store, v3_info):
    """v2 stage 2 -> + extra cross-encoders (greedy CV) -> [+ FT cross-encoder] -> VLM yes/no on the top-M (CV M, weight).

    Runs on the ensemble chain, fixed in advance. Every step starts from the per-fold picks of the previous step,
    exactly like the stage-1 chain, and every weight grid contains 0."""
    cname = "ensemble final" if "ensemble final" in chains else f"{next(iter(finals))} final"
    key2 = next(k for k in (f"{cname} -> rerank ITM+ITC", f"{cname} -> rerank ITM") if k in rerank_results)
    cand = topk[cname]
    Kmax = cand.shape[1]
    base1 = gt_rank(chains[cname], gt)
    Pi = itm_of(cand)
    F = np.take_along_axis(chains[cname], cand, 1)
    X = np.take_along_axis(X_eval, cand, 1)
    q0 = qsel[:, None].repeat(Kmax, 1)
    pairs_all = list(zip(q0.ravel().tolist(), cand.ravel().tolist()))
    feats = {}
    for model in extra + ft:
        known = score_cross_encoder(model, pairs_all, queries_all, paths, cache_dir, device, sleep_s, args.itm_batch)
        feats[model] = np.array([known[p] for p in pairs_all], np.float32).reshape(n_q, Kmax)
    V = np.full((n_q, Kmax), np.nan, np.float32)

    params = []
    for K, lam in rerank_results[key2]:
        params.append({"K": int(K), "lam": tuple(lam) if isinstance(lam, (tuple, list)) else (float(lam), 0.0)})

    def pre_vlm(P, rows):
        K = P["K"]
        a, b = P["lam"]
        c = a * zscore_rows(Pi[rows, :K]) + (1 - a - b) * zscore_rows(F[rows, :K]) + b * zscore_rows(X[rows, :K])
        for j, model in enumerate(extra + ft):
            g = P.get(f"x{j}", 0.0)
            if g:
                c = (1 - g) * zscore_rows(c) + g * zscore_rows(feats[model][rows, :K])
        return c

    def scores(P, rows):
        c = pre_vlm(P, rows)
        gv, M = P.get("gv", 0.0), P.get("M", 0)
        if gv and M:
            order = np.argsort(-c, axis=1)[:, :M]
            vals = np.take_along_axis(c, order, 1)
            vv = np.take_along_axis(V[rows, : P["K"]], order, 1)
            assert not np.isnan(vv).any(), "VLM score missing for a top-M candidate"
            c = c.copy()
            np.put_along_axis(c, order, 1e3 + (1 - gv) * zscore_rows(vals) + gv * zscore_rows(vv), axis=1)
        return c

    def ranks_for(P, rows):
        return rerank_ranks_sub(scores(P, rows), cand[rows, : P["K"]], gt[rows], base1[rows])

    def pick(key, grid):
        for f in (0, 1):
            m = fold == f
            params[f][key] = max(grid, key=lambda g: metrics_from_ranks(ranks_for({**params[f], key: g}, m))["RSUM"])
        out = np.empty(n_q, np.int64)
        for f in (0, 1):
            other = fold != f
            out[other] = ranks_for(params[f], other)
        return out, [params[f][key] for f in (0, 1)]

    def compose():
        out = np.empty(n_q, np.int64)
        for f in (0, 1):
            other = fold != f
            out[other] = ranks_for(params[f], other)
        return out

    grp = f"v3 cascade on {cname}"
    add(f"{cname} -> v2 stage 2 (recomputed)", ranks=compose(), extra=f"picks={rerank_results[key2]}", group=grp)

    def vlm_step(label, models_on):
        if not args.vlm:
            return None
        Ms = sorted(int(m) for m in args.vlm_m.split(",") if m)
        need = set()
        for f in (0, 1):   # top-Mmax under each fold's settings, for all queries (tuning and held-out rows)
            P = {k: v for k, v in params[f].items() if k not in ("gv", "M")}
            order = np.argsort(-pre_vlm(P, np.ones(n_q, bool)), axis=1)[:, : max(Ms)]
            pos = order
            need.update(zip(np.repeat(qsel, pos.shape[1]).tolist(), np.take_along_axis(cand, pos, 1).ravel().tolist()))
        known = score_vlm(args.vlm, need, queries_all, paths, cache_dir, device, sleep_s, args.vlm_batch)
        qidx = {q: r for r, q in enumerate(qsel.tolist())}
        for (q, i), v in known.items():
            r = qidx.get(q)
            if r is not None:
                hit = np.flatnonzero(cand[r] == i)
                if len(hit):
                    V[r, hit[0]] = v
        grid = [(M, g) for M in Ms for g in WEIGHTS]
        for f in (0, 1):
            m = fold == f
            best = max(grid, key=lambda mg: metrics_from_ranks(ranks_for({**params[f], "M": mg[0], "gv": mg[1]}, m))["RSUM"])
            params[f]["M"], params[f]["gv"] = best
        r = compose()
        add(label, ranks=r, extra=f"(M, w_vlm) per fold={[(params[f]['M'], params[f]['gv']) for f in (0, 1)]}", group=grp)
        return r

    for j, model in enumerate(extra):
        r, pk = pick(f"x{j}", WEIGHTS)
        add(f"{cname} -> + {model.split('/')[-1]}", ranks=r, extra=f"w={pk}", group=grp)
    final_zs = vlm_step(f"v3 (zero-shot protocol): + VLM {args.vlm.split('/')[-1]}", extra) if args.vlm else None
    if final_zs is None and extra:
        ranks_store["v3 (zero-shot protocol)"] = compose()
        add("v3 (zero-shot protocol)", ranks=ranks_store["v3 (zero-shot protocol)"], group=grp)
    v3_info["zero_shot"] = [dict(p) for p in params]
    if ft:
        for f in (0, 1):
            params[f].pop("gv", None)
            params[f].pop("M", None)
        for j, model in enumerate(ft, start=len(extra)):
            r, pk = pick(f"x{j}", WEIGHTS)
            add(f"{cname} -> + {model.split('/')[-1]} [FT on {args.dataset}]", ranks=r, extra=f"w={pk}", group=grp)
        if args.vlm:
            vlm_step(f"v3 (FT protocol): + VLM {args.vlm.split('/')[-1]}", extra + ft)
        else:
            add("v3 (FT protocol)", ranks=compose(), group=grp)
        v3_info["ft"] = [dict(p) for p in params]
    v3_info["chain"] = cname
    v3_info["models"] = {"extra": extra, "ft": ft, "vlm": args.vlm}


def rerank_ranks_sub(comb: np.ndarray, tk: np.ndarray, gt: np.ndarray, base: np.ndarray) -> np.ndarray:
    """Rank of gt after reordering the top-k candidates `tk` by `comb`; queries whose gt is outside keep `base`.
    `base` must be the gt rank under the first-stage score, which is >= k whenever gt is outside the top-k."""
    inside = tk == gt[:, None]
    has = inside.any(1)
    gpos = inside.argmax(1)
    cg = comb[np.arange(len(comb)), gpos]
    # ties keep the first-stage order (tk is sorted by first-stage score), never in favour of the ground truth
    earlier = np.arange(tk.shape[1])[None, :] < gpos[:, None]
    r_in = (comb > cg[:, None]).sum(1) + ((comb == cg[:, None]) & earlier).sum(1)
    return np.where(has, r_in, np.maximum(base, tk.shape[1]))


if __name__ == "__main__":
    import os
    import sys

    main()
    sys.stdout.flush()
    os._exit(0)
