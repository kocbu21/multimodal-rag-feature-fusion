Online search on `flickr1k_multi_clip-vit-large-patch14`, 1000 text queries (seed 0), alpha=0.825, text_weight=0.3, rerank Salesforce/blip2-itm-vit-g-coco top_k=16 weight=0.6 itc_weight=0.0, hub_rho=0.125, gpu_sleep_s=0.0

| strategy | R@1 | R@5 | R@10 | MRR@10 | ms/query |
|---|---:|---:|---:|---:|---:|
| image | 65.00 | 87.20 | 92.70 | 74.80 | 11 |
| hybrid | 77.50 | 94.60 | 97.20 | 84.83 | 11 |
| full | 85.90 | 97.70 | 99.00 | 90.98 | 23 |
| full+rerank | 88.90 | 98.20 | 99.00 | 93.12 | 89 |
