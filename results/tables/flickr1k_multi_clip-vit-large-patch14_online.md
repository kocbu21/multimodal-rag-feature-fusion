Online search on `flickr1k_multi_clip-vit-large-patch14`, 1000 text queries (seed 0), alpha=0.85, text_weight=0.3, rerank top_k=16 weight=0.8, gpu_sleep_s=0.2

| strategy | R@1 | R@5 | R@10 | MRR@10 | ms/query |
|---|---:|---:|---:|---:|---:|
| image | 65.10 | 87.30 | 92.70 | 74.87 | 451 |
| fused | 72.90 | 92.00 | 95.50 | 81.23 | 433 |
| hybrid | 76.40 | 94.60 | 97.10 | 84.06 | 430 |
| hybrid+rerank | 86.00 | 96.40 | 97.80 | 90.59 | 732 |
