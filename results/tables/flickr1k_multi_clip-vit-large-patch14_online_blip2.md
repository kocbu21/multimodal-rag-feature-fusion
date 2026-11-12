Online search on `flickr1k_multi_clip-vit-large-patch14`, 1000 text queries (seed 0), alpha=0.85, text_weight=0.3, rerank Salesforce/blip2-itm-vit-g-coco top_k=16 weight=0.9, gpu_sleep_s=0.2

| strategy | R@1 | R@5 | R@10 | MRR@10 | ms/query |
|---|---:|---:|---:|---:|---:|
| hybrid+rerank | 88.90 | 97.40 | 97.90 | 92.72 | 773 |
