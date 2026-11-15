Online search on `coco5k_multi_siglip-base-patch16-224`, 1000 text queries (seed 0), alpha=0.85, text_weight=0.3, rerank Salesforce/blip2-itm-vit-g-coco top_k=16 weight=0.9, gpu_sleep_s=0.2

| strategy | R@1 | R@5 | R@10 | MRR@10 | ms/query |
|---|---:|---:|---:|---:|---:|
| image | 46.70 | 72.40 | 80.60 | 57.43 | 449 |
| fused | 50.70 | 75.00 | 81.40 | 61.22 | 432 |
| hybrid | 52.20 | 76.20 | 83.80 | 62.74 | 434 |
| hybrid+rerank | 67.30 | 84.20 | 87.80 | 74.52 | 1006 |
