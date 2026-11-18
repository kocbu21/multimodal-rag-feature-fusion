Online search on `flickr1k_multi_siglip-base-patch16-224`, 1000 text queries (seed 0), alpha=0.875, text_weight=0.3, rerank Salesforce/blip2-itm-vit-g-coco top_k=32 weight=0.85 itc_weight=0.0, hub_rho=0.0, gpu_sleep_s=0.0

| strategy | R@1 | R@5 | R@10 | MRR@10 | ms/query |
|---|---:|---:|---:|---:|---:|
| image | 73.80 | 92.50 | 95.30 | 81.65 | 11 |
| hybrid | 80.80 | 94.40 | 97.00 | 86.78 | 10 |
| full | 85.80 | 96.90 | 98.60 | 90.79 | 22 |
| full+rerank | 89.70 | 98.00 | 98.90 | 93.49 | 124 |
