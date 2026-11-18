Online search on `flickr1k_multi_siglip2-so400m-patch14-384`, 1000 text queries (seed 0), alpha=0.95, text_weight=0.3, rerank Salesforce/blip2-itm-vit-g-coco top_k=64 weight=0.35 itc_weight=0.0, hub_rho=0.125, gpu_sleep_s=0.0

| strategy | R@1 | R@5 | R@10 | MRR@10 | ms/query |
|---|---:|---:|---:|---:|---:|
| hybrid | 82.60 | 95.60 | 98.30 | 88.52 | 13 |
| full | 86.70 | 97.70 | 98.80 | 91.54 | 26 |
| full+rerank | 90.40 | 98.30 | 99.20 | 94.03 | 761 |
