Online search on `flickr1k_multi_siglip-base-patch16-224`, 1000 text queries (seed 0), alpha=0.85, text_weight=0.3, rerank top_k=16 weight=0.8, gpu_sleep_s=0.2

| strategy | R@1 | R@5 | R@10 | MRR@10 | ms/query |
|---|---:|---:|---:|---:|---:|
| image | 73.80 | 92.50 | 95.40 | 81.66 | 423 |
| fused | 76.80 | 93.30 | 96.70 | 84.06 | 429 |
| hybrid | 80.40 | 94.20 | 97.10 | 86.66 | 430 |
| hybrid+rerank | 85.40 | 96.20 | 97.80 | 90.08 | 733 |
