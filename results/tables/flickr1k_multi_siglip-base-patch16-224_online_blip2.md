Online search on `flickr1k_multi_siglip-base-patch16-224`, 1000 text queries (seed 0), alpha=0.85, text_weight=0.3, rerank Salesforce/blip2-itm-vit-g-coco top_k=16 weight=0.9, gpu_sleep_s=0.2

| strategy | R@1 | R@5 | R@10 | MRR@10 | ms/query |
|---|---:|---:|---:|---:|---:|
| hybrid+rerank | 89.00 | 97.00 | 97.80 | 92.77 | 765 |
