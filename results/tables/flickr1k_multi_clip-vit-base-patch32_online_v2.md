Online search on `flickr1k_multi_clip-vit-base-patch32`, 1000 text queries (seed 0), alpha=0.8500000000000001, text_weight=0.3, rerank Salesforce/blip2-itm-vit-g-coco top_k=32 weight=0.6 itc_weight=0.0, hub_rho=0.0, gpu_sleep_s=0.0

| strategy | R@1 | R@5 | R@10 | MRR@10 | ms/query |
|---|---:|---:|---:|---:|---:|
| image | 55.30 | 82.90 | 90.20 | 66.87 | 11 |
| hybrid | 71.80 | 92.80 | 95.90 | 80.64 | 11 |
| full | 86.10 | 97.30 | 98.90 | 91.04 | 22 |
| full+rerank | 89.10 | 98.20 | 99.10 | 93.30 | 128 |
