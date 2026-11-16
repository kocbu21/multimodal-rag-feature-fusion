# Results - Flickr30k 1K test (1000 images, 5000 captions)

Image captions: BLIP-base (`Salesforce/blip-image-captioning-base`). Fusion weights (alpha / w) are
chosen with 2-fold cross-validation over images, so no query is scored with a weight tuned on it.

## Headline: Text -> Image R@1, image-only baseline vs. early fusion (same centering setting)

| encoder | image only | early fusion | gain | image only +ctr | early fusion +ctr | gain |
|---|---|---|---|---|---|---|
| clip-vit-base-patch32 | 58.76 | 60.98 | +2.22 | 60.14 | 60.96 | +0.82 |
| clip-vit-large-patch14 | 64.58 | 65.88 | +1.30 | 66.78 | 67.16 | +0.38 |
| siglip-base-patch16-224 | 74.58 | 75.10 | +0.52 | 72.18 | 72.38 | +0.20 |

## Text -> Image (all methods)

| encoder                 | method           |   R@1 |   R@5 |   R@10 |   RSUM |   MRR |
|:------------------------|:-----------------|------:|------:|-------:|-------:|------:|
| clip-vit-base-patch32   | image            | 58.76 | 83.44 |  90.08 | 232.28 | 69.69 |
| clip-vit-base-patch32   | caption          | 24.64 | 43.30 |  50.90 | 118.84 | 33.64 |
| clip-vit-base-patch32   | early-fusion     | 60.98 | 84.52 |  90.62 | 236.12 | 71.56 |
| clip-vit-base-patch32   | late-fusion-z    | 60.26 | 84.16 |  90.42 | 234.84 | 70.94 |
| clip-vit-base-patch32   | rrf              | 39.92 | 61.98 |  71.78 | 173.68 | 50.63 |
| clip-vit-base-patch32   | image+ctr        | 60.14 | 84.04 |  90.58 | 234.76 | 70.87 |
| clip-vit-base-patch32   | early-fusion+ctr | 60.96 | 84.86 |  91.16 | 236.98 | 71.61 |
| clip-vit-large-patch14  | image            | 64.58 | 87.06 |  92.16 | 243.80 | 74.54 |
| clip-vit-large-patch14  | caption          | 24.68 | 43.02 |  51.14 | 118.84 | 33.47 |
| clip-vit-large-patch14  | early-fusion     | 65.88 | 88.70 |  93.20 | 247.78 | 75.88 |
| clip-vit-large-patch14  | late-fusion-z    | 65.48 | 88.36 |  92.98 | 246.82 | 75.54 |
| clip-vit-large-patch14  | rrf              | 40.18 | 62.42 |  71.68 | 174.28 | 51.13 |
| clip-vit-large-patch14  | image+ctr        | 66.78 | 87.66 |  93.24 | 247.68 | 76.06 |
| clip-vit-large-patch14  | early-fusion+ctr | 67.16 | 88.40 |  93.48 | 249.04 | 76.60 |
| siglip-base-patch16-224 | image            | 74.58 | 92.30 |  95.64 | 262.52 | 82.40 |
| siglip-base-patch16-224 | caption          | 21.94 | 39.42 |  46.76 | 108.12 | 30.41 |
| siglip-base-patch16-224 | early-fusion     | 75.10 | 92.18 |  95.64 | 262.92 | 82.67 |
| siglip-base-patch16-224 | late-fusion-z    | 75.24 | 92.22 |  95.64 | 263.10 | 82.76 |
| siglip-base-patch16-224 | rrf              | 38.98 | 60.08 |  69.04 | 168.10 | 49.62 |
| siglip-base-patch16-224 | image+ctr        | 72.18 | 90.48 |  94.06 | 256.72 | 80.32 |
| siglip-base-patch16-224 | early-fusion+ctr | 72.38 | 90.62 |  94.16 | 257.16 | 80.47 |

## Image -> Text

| encoder                 | method           |   R@1 |   R@5 |   R@10 |   RSUM |   MRR |
|:------------------------|:-----------------|------:|------:|-------:|-------:|------:|
| clip-vit-base-patch32   | image            | 79.30 | 95.00 |  98.10 | 272.40 | 85.99 |
| clip-vit-base-patch32   | early-fusion     | 79.10 | 95.10 |  98.00 | 272.20 | 85.99 |
| clip-vit-base-patch32   | image+ctr        | 74.00 | 92.50 |  96.50 | 263.00 | 82.07 |
| clip-vit-base-patch32   | early-fusion+ctr | 74.10 | 93.00 |  96.30 | 263.40 | 82.42 |
| clip-vit-large-patch14  | image            | 85.50 | 97.30 |  99.30 | 282.10 | 90.54 |
| clip-vit-large-patch14  | early-fusion     | 85.60 | 97.10 |  99.30 | 282.00 | 90.54 |
| clip-vit-large-patch14  | image+ctr        | 81.10 | 95.90 |  98.50 | 275.50 | 87.40 |
| clip-vit-large-patch14  | early-fusion+ctr | 80.20 | 96.00 |  98.40 | 274.60 | 86.86 |
| siglip-base-patch16-224 | image            | 89.20 | 98.00 |  99.30 | 286.50 | 93.13 |
| siglip-base-patch16-224 | early-fusion     | 89.20 | 98.00 |  99.30 | 286.50 | 93.13 |
| siglip-base-patch16-224 | image+ctr        | 82.00 | 95.60 |  98.10 | 275.70 | 88.19 |
| siglip-base-patch16-224 | early-fusion+ctr | 82.00 | 95.60 |  98.10 | 275.70 | 88.19 |

## Observations

- Early fusion of image + caption vectors improves text->image retrieval for every encoder; the
  gain is largest for the weakest encoder (CLIP B/32) and smallest for SigLIP.
- The best alpha is about 0.9 (mostly image, a little caption). Captions alone are weak
  (R@1 about 22-25%) because BLIP-base captions are short and generic.
- Equal-weight RRF hurts badly: it gives the weak caption ranking as much say as the image ranking.
  Score-level fusion only works with a weight (late-fusion-z), and then matches early fusion.
- Modality centering helps CLIP text->image a little but hurts image->text and SigLIP.
- Image->text gains nothing: the query image's own caption adds little beyond the image vector.

Per-encoder alpha sweeps: `results/figures/*_alpha_sweep.png`, raw numbers: `results/tables/*_metrics.csv`, `results/json/*_sweeps.json`.


---

# Improvements (part 2) - Text -> Image R@1

Captioners: BLIP-base (blip-b), BLIP-large (blip-l), Florence-2-large `<MORE_DETAILED_CAPTION>` (flor-md),
`<DETAILED_CAPTION>` (flor-d). Text-embedding channel: `BAAI/bge-base-en-v1.5`.
Reranker: `Salesforce/blip-itm-base-coco` (trained on COCO, not on Flickr). All weights are 2-fold CV.

## Headline

| encoder | image only | best early fusion | best hybrid | + ITM rerank |
|---|---|---|---|---|
| clip-vit-base-patch32 | 58.76 | 69.54 | 72.78 | 84.92 |
| clip-vit-large-patch14 | 64.58 | 72.96 | 75.64 | 85.92 |
| siglip-base-patch16-224 | 74.58 | 77.92 | 78.78 | 85.50 |

## R@1 by method

| method | clip-vit-base-patch32 | clip-vit-large-patch14 | siglip-base-patch16-224 |
|---|---|---|---|
| image | 58.76 | 64.58 | 74.58 |
| early[blip-b] | 61.16 | 65.88 | 74.64 |
| early[blip-l] | 64.24 | 67.62 | 75.40 |
| early[flor-md] | 63.54 | 71.32 | 76.44 |
| early[multi] | 69.54 | 72.96 | 77.92 |
| te-caption[flor-md] | 60.82 | 60.82 | 60.82 |
| te-caption[concat] | 63.18 | 63.18 | 63.18 |
| hybrid[flor-md] | 70.84 | 74.88 | 78.48 |
| hybrid[concat] | 72.78 | 75.64 | 78.78 |
| rerank-ITM(top16) | 84.92 | 85.92 | 85.50 |

- `early[multi]`: early fusion with the mean of all four caption sets' vectors.
- `te-caption[x]`: query vs captions in the BGE text space only (no image vector at all).
- `hybrid[x]`: score-level fusion of the early-fused dual-encoder score and the BGE caption score.
- `rerank-ITM`: BLIP ITM (cross-attention) rescoring of the best hybrid's top-16 candidates.

## Observations (part 2)

- **Caption quality is the main lever.** Going from BLIP-base to BLIP-large lifts CLIP B/32 early fusion
  from 61.2 to 64.2 R@1. Averaging four caption sets (`early[multi]`) reaches 69.5 (+10.8 over image only).
- **Use a real text model for text-text matching.** In BGE space, detailed Florence captions alone reach
  60.8 R@1 (same for every row: this channel doesn't use the image encoder), more than CLIP B/32's image
  vectors (58.8). CLIP's text tower gets 24-26 from
  the same kind of captions. Long captions also lose information in CLIP (77-token limit) but not in BGE.
- **Hybrid fusion is complementary.** Image vector + caption vector + BGE caption score: 72.8 (CLIP B/32),
  improving every encoder, including SigLIP, where single-caption early fusion barely helped.
- **Cross-attention reranking gives the largest jump** (to about 85-86 R@1 for all encoders). It works
  because the fused first stage puts the right image in the top 16 for about 97% of queries. After reranking,
  the encoder choice hardly matters; the first stage mostly needs high recall@16.
- Costs: captions are computed once per image at indexing time; BGE adds one small text encoding per query;
  reranking adds 16 ITM text-transformer passes per query (image features are cached).

# Part 3: end-to-end check of the live search path

`scripts/eval_online.py` sends queries one at a time through `MultimodalRetriever.search`, the same code
the CLI and the Gradio app run: FAISS top-K lists per index, online late fusion over the top-100
candidates, then ITM reranking of the top-16. Stores are multi-caption (`indexes/flickr1k_multi_*`:
4 caption sets per image, averaged caption vector, BGE over the joined captions), with the config defaults
alpha=0.85, text_weight=0.3, rerank weight=0.8. Evaluated on 1000 random text queries (seed 0) out of 5000,
so a single number has a standard error of about 1-1.5 points.

Text -> Image R@1 (online):

| encoder | image | fused (multi-caption) | hybrid | hybrid + rerank | offline best (part 2) |
|---|---:|---:|---:|---:|---:|
| CLIP B/32 | 55.3 | 64.4 | 72.2 | **84.6** | 84.92 |
| CLIP L/14 | 65.1 | 72.9 | 76.4 | **86.0** | 85.92 |
| SigLIP base | 73.8 | 76.8 | 80.4 | **85.4** | 85.50 |

- The live pipeline reproduces the offline results within sampling noise, so online fusion over
  top-100 lists (instead of full score matrices) and one shared set of weights lose essentially nothing.
- Latency with quiet mode on (`gpu_sleep_s=0.2` per batch, so most of this is sleep): ~430 ms/query
  for the first stage, ~730 ms/query with reranking. Per-file results: `results/tables/flickr1k_multi_*_online.md`.

# Part 4: BLIP-2 reranker and comparison with published results

`scripts/pipelines/run_blip2.sh` reranks the same hybrid[concat] top-16 with BLIP-2 ITM
(`Salesforce/blip2-itm-vit-g-coco`, frozen ViT-g + Q-Former, fine-tuned on COCO, not on Flickr).
The cached-feature scoring path was checked against `Blip2ForImageTextRetrieval.forward` (same probabilities).
Weights are again 2-fold CV. ITM scores are cached in `indexes/flickr1k_*/itm_blip2-itm-vit-g-coco.npz`.

Text -> Image on Flickr30k 1K (5000 queries):

| encoder | hybrid | + BLIP ITM | + BLIP-2 ITM | + BLIP + BLIP-2 | recall@16 |
|---|---:|---:|---:|---:|---:|
| CLIP B/32 | 72.78 | 84.92 | **88.58** | 88.24 | 97.36 |
| CLIP L/14 | 75.64 | 85.92 | 88.36 | **88.68** | 98.06 |
| SigLIP base | 78.78 | 85.50 | 88.28 | **88.36** | 98.02 |

- BLIP-2 adds +2.4 to +3.7 R@1 over BLIP; all three encoders end at 88.3-88.6. CV picks lambda 0.9-1.0 (BLIP-2 alone is almost enough).
- Combining BLIP and BLIP-2 gives no further gain (within noise).
- When the correct image is in the top-16, BLIP-2 ranks it first about 90-91% of the time; the rest of
  the errors come from the first stage (recall@16).
- Cost: 2.8 GB VRAM (fp16); 80K pairs took ~25 min in quiet mode, ~0.13 s/query without the pauses.

Published Flickr30k 1K text->image R@1 (verified from the papers' tables): CLIP ViT-L/14@336 zero-shot 68.7,
ALIGN zero-shot 75.7, ALBEF-14M (COCO-finetuned, zero-shot) 82.8, BLIP ViT-L (COCO, zero-shot) 86.7,
BLIP-2 ViT-g (COCO, zero-shot) 89.7, ALIGN fine-tuned 84.9, BLIP ViT-L fine-tuned 87.6, BEiT-3 fine-tuned 90.3.
Our training-free system (88.6) lands in the COCO-finetuned zero-shot group, 1.1 below BLIP-2. This is not
state of the art. The report table (`report/report.tex`, Table IV) lists the protocols.

# Part 5: live search with the BLIP-2 reranker

Same live-path protocol as Part 3 (`scripts/eval_online.py`, 1000 text queries, seed 0, multi-caption stores,
quiet mode `gpu_sleep_s=0.2`), but the top-16 is reranked with BLIP-2 ITM at lambda=0.9
(`--reranker Salesforce/blip2-itm-vit-g-coco --weight 0.9`; lambda taken from the 0.9-1.0 CV range of Part 4).
BLIP-2 is now the default reranker in `configs/default.yaml`.

Text -> Image (online, hybrid + rerank):

| encoder | BLIP R@1 | **BLIP-2 R@1** | BLIP-2 R@5 | BLIP-2 R@10 | offline BLIP-2 (part 4) | ms/query BLIP -> BLIP-2 |
|---|---:|---:|---:|---:|---:|---:|
| CLIP B/32 | 84.6 | **88.7** | 96.9 | 97.5 | 88.58 | 737 -> 774 |
| CLIP L/14 | 86.0 | **88.9** | 97.4 | 97.9 | 88.36 | 732 -> 773 |
| SigLIP base | 85.4 | **89.0** | 97.0 | 97.8 | 88.28 | 733 -> 765 |

- The live path reproduces the offline BLIP-2 numbers within sampling noise (SE ~1 point at n=1000).
- BLIP-2 adds +2.9 to +4.1 R@1 over BLIP in the live system for only ~35-40 ms more per query, because
  quiet-mode sleep dominates the latency. Cost: ~4.7 GB weights, 2.8 GB VRAM.
- Per-file results: `results/tables/flickr1k_multi_*_online_blip2.md`.

# Part 6: MS-COCO 5K (second benchmark)

Live path (`scripts/eval_online.py --dataset coco5k`), SigLIP base, multi-caption store built from the same four
caption sets, 1000 text queries (seed 0) out of 25,010, hybrid + BLIP-2 rerank (top-16, lambda=0.9), alpha=0.85,
text_weight=0.3, quiet mode `gpu_sleep_s=0.2`. All weights were taken from the Flickr30k runs; nothing was tuned on COCO.
Pipeline: `bash scripts/pipelines/run_coco5k.sh`.

| strategy | R@1 | R@5 | R@10 | MRR@10 | ms/query |
|---|---:|---:|---:|---:|---:|
| image | 46.70 | 72.40 | 80.60 | 57.43 | 449 |
| fused | 50.70 | 75.00 | 81.40 | 61.22 | 432 |
| hybrid | 52.20 | 76.20 | 83.80 | 62.74 | 434 |
| hybrid + BLIP-2 rerank | **67.30** | 84.20 | 87.80 | 74.52 | 1006 |

- Same stage ordering as Flickr30k (image < fused < hybrid < rerank), but a much lower level (SigLIP Flickr live: 73.8 / 76.8 / 80.4 / 89.0).
- Fusion gains over image-only: +4.0 (fused), +5.5 (hybrid); the reranker adds +15.1 over hybrid.
- Reranked R@10 = 87.8 implies first-stage recall@16 >= 87.8%. R@1 stays at least 20 points below that bound, so ordering inside the top-16 is the bottleneck.
- Caveats: one encoder, n=1000 (SE ~1.5 points), BLIP rerank and CLIP encoders not run on COCO; BLIP-2 ITM was fine-tuned on COCO, so this is in-domain for it.
- Per-file results: `results/tables/coco5k_multi_siglip-base-patch16-224_online_blip2.md`.

# Part 7: v2 results

One run on UHeM Altay (1x A100 80 GB, 3 h 8 min including caption generation), captions regenerated with the same
four captioners. Components (all training-free, each added greedily with 2-fold CV over images, every grid contains
"off"): object tags (BGE), region features (5 grid crops + Florence-2 `<OD>` boxes, max pooling), BLIP-2 ITC as a
first-stage channel, caption-bank hubness normalisation (CB-Norm), equal-weight ensemble of the three encoders,
BLIP-2 ITM(+ITC) rerank over a CV-selected pool K. Full tables: `results/tables/flickr1k_v2.md`,
`results/tables/coco5k_v2.md`; picks: `results/json/*_v2.json`; live: `results/tables/flickr1k_multi_*_online_v2.md`.

Sanity checks (v1 recomputed in this run): hybrid 72.58 / 75.60 / 78.94 (v1: 72.78 / 75.64 / 78.78); hybrid + BLIP-2
top-16 88.38 / 88.36 / 88.30 (v1: 88.58 / 88.36 / 88.28); BLIP-2 protocol re-run 89.44 (paper: 89.7).

Text -> Image R@1, stage-1 chain (CLIP B/32 / CLIP L/14 / SigLIP):

| stage | Flickr30k 1K | COCO 5K |
|---|---|---|
| hybrid (v1) | 72.58 / 75.60 / 78.94 | 45.93 / 48.49 / 52.66 |
| + object tags | off / off / off | off / off / off |
| + regions | 73.84 / 76.56 / 80.58 | 46.19 / 48.94 / 53.74 |
| + BLIP-2 ITC | 87.00 / 86.96 / 87.92 | 63.85 / 63.93 / 64.28 |
| + CB-Norm | 87.00 / 86.88 / 87.92 | 64.32 / 64.26 / 64.50 |
| ensemble (stage 1) | 87.74 | 64.67 |
| recall@16, v1 -> v2 | 97.6-98.0 -> 99.4-99.5 | 85.7-89.0 -> 94.6-94.9 |

Final (ensemble + BLIP-2 ITM, CV K; CV gave the ITC term weight 0 in stage 2 because ITC is already in stage 1):

| | R@1 | R@5 | R@10 | dR@1 vs v1 (95% CI) | dR@1 vs BLIP-2 protocol re-run |
|---|---:|---:|---:|---|---|
| Flickr30k 1K, K=64 | 89.66 | 98.42 | 99.26 | +1.28 [0.72, 1.88] (B/32), +1.30, +1.36 | +0.22 [-0.38, 0.82], p=0.48 (not significant) |
| COCO 5K, K=32 | 68.21 | 87.80 | 92.81 | +2.40 [2.09, 2.72], +1.89 [1.58, 2.20], +1.01 [0.70, 1.31] | not re-run (published BLIP-2 COCO-FT: 68.3 / 87.7 / 92.6) |

- The BLIP-2 ITC channel carries most of the gain; it raises candidate recall so that almost all remaining errors
  are ordering errors of the cross-encoder (Flickr recall@64 99.86 vs R@1 89.66; COCO recall@32 97.68 vs R@1 68.21).
- Grid crops help a little (+0.3 to +1.6); detected-object crops and object tags were switched off everywhere
  (tags alone: 5.5 R@1, the `<OD>` vocabulary is too coarse).
- CB-Norm: neutral on Flickr30k (rho=0 in 5 of 6 folds), positive on COCO (on in all 6 folds, +0.22 to +0.47).
- Single-encoder v2 systems are within noise of the ensemble (Flickr 89.76 / 89.94 / 89.82, COCO 68.26 / 68.10 / 68.27).
- Dual softmax over test queries (transductive, not part of the system) hurt with the fixed temperature (81.96 / 58.33).
- Live path (1000 queries, CV-derived settings): 89.1 / 88.9 / 89.7 R@1 with reranking, 86.1 / 85.9 / 85.8 for stage 1;
  22 ms per query for stage 1 and 89-128 ms with reranking on an A100.
