Online search on `flickr1k_multi_clip-vit-base-patch32`, 1000 text queries (seed 0), alpha=0.85, text_weight=0.3, rerank top_k=16 weight=0.8, gpu_sleep_s=0.2

| strategy | R@1 | R@5 | R@10 | MRR@10 | ms/query |
|---|---:|---:|---:|---:|---:|
| image | 55.30 | 82.90 | 90.30 | 66.87 | 435 |
| fused | 64.40 | 89.40 | 94.10 | 74.87 | 431 |
| hybrid | 72.20 | 92.90 | 95.90 | 80.90 | 431 |
| hybrid+rerank | 84.60 | 95.90 | 97.60 | 89.59 | 737 |
