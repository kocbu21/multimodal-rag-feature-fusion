v2 experiments on flickr1k: 1000 images, 5000 text queries; encoders ['openai/clip-vit-base-patch32', 'openai/clip-vit-large-patch14', 'google/siglip-base-patch16-224']; reranker Salesforce/blip2-itm-vit-g-coco; K grid [16, 32, 64]; regions ['grid', 'od']; 2-fold CV over images (seed 0).

| group                        | method                                                   |    R@1 |    R@5 |   R@10 |   RSUM |    MRR | params                                                   |
|:-----------------------------|:---------------------------------------------------------|-------:|-------:|-------:|-------:|-------:|:---------------------------------------------------------|
| stage1 CLIP-B/32             | CLIP-B/32 image                                          |  58.72 |  83.40 |  90.06 | 232.18 |  69.69 |                                                          |
| stage1 CLIP-B/32             | CLIP-B/32 early[multi]                                   |  69.54 |  89.58 |  94.02 | 253.14 |  78.39 | alpha=[0.7, 0.7]                                         |
| stage1 CLIP-B/32             | CLIP-B/32 hybrid (v1)                                    |  72.58 |  92.22 |  95.90 | 260.70 |  81.24 | (alpha,w)=[(0.8, 0.4), (0.9, 0.5)]                       |
| stage1 CLIP-B/32             | CLIP-B/32 hybrid +regions[grid] (ablation)               |  73.42 |  92.90 |  96.46 | 262.78 |  81.97 | g=[0.2, 0.2]                                             |
| stage1 CLIP-B/32             | CLIP-B/32 hybrid +regions[od] (ablation)                 |  72.58 |  92.22 |  95.90 | 260.70 |  81.24 | g=[0.0, 0.0]                                             |
| stage1 CLIP-B/32             | CLIP-B/32 hybrid +itc only (ablation)                    |  86.80 |  97.48 |  99.02 | 283.30 |  91.58 | g=[0.6, 0.6]                                             |
| stage1 CLIP-B/32             | CLIP-B/32 +tags                                          |  72.58 |  92.22 |  95.90 | 260.70 |  81.24 | g=[0.0, 0.0]                                             |
| stage1 CLIP-B/32             | CLIP-B/32 +regions                                       |  73.84 |  92.90 |  96.42 | 263.16 |  82.23 | g=[0.2, 0.2]                                             |
| stage1 CLIP-B/32             | CLIP-B/32 +itc                                           |  87.00 |  97.60 |  98.98 | 283.58 |  91.72 | g=[0.5, 0.6]                                             |
| stage1 CLIP-B/32             | CLIP-B/32 +hub (CB-Norm)                                 |  87.00 |  97.60 |  98.98 | 283.58 |  91.72 | (beta,rho)=[(10.0, 0.0), (10.0, 0.0)]                    |
| stage1 CLIP-L/14             | CLIP-L/14 image                                          |  64.58 |  87.06 |  92.20 | 243.84 |  74.54 |                                                          |
| stage1 CLIP-L/14             | CLIP-L/14 early[multi]                                   |  72.94 |  92.14 |  95.62 | 260.70 |  81.42 | alpha=[0.7, 0.7]                                         |
| stage1 CLIP-L/14             | CLIP-L/14 hybrid (v1)                                    |  75.60 |  93.70 |  96.78 | 266.08 |  83.54 | (alpha,w)=[(0.8, 0.3), (0.85, 0.4)]                      |
| stage1 CLIP-L/14             | CLIP-L/14 hybrid +regions[grid] (ablation)               |  76.30 |  94.62 |  97.12 | 268.04 |  84.23 | g=[0.2, 0.3]                                             |
| stage1 CLIP-L/14             | CLIP-L/14 hybrid +regions[od] (ablation)                 |  75.60 |  93.70 |  96.78 | 266.08 |  83.54 | g=[0.0, 0.0]                                             |
| stage1 CLIP-L/14             | CLIP-L/14 hybrid +itc only (ablation)                    |  86.86 |  97.66 |  99.00 | 283.52 |  91.63 | g=[0.5, 0.6]                                             |
| stage1 CLIP-L/14             | CLIP-L/14 +tags                                          |  75.60 |  93.70 |  96.78 | 266.08 |  83.54 | g=[0.0, 0.0]                                             |
| stage1 CLIP-L/14             | CLIP-L/14 +regions                                       |  76.56 |  94.64 |  97.12 | 268.32 |  84.46 | g=[0.3, 0.3]                                             |
| stage1 CLIP-L/14             | CLIP-L/14 +itc                                           |  86.96 |  97.82 |  99.06 | 283.84 |  91.76 | g=[0.6, 0.5]                                             |
| stage1 CLIP-L/14             | CLIP-L/14 +hub (CB-Norm)                                 |  86.88 |  97.84 |  99.06 | 283.78 |  91.74 | (beta,rho)=[(50.0, 0.25), (10.0, 0.0)]                   |
| stage1 SigLIP-B              | SigLIP-B image                                           |  74.66 |  92.32 |  95.60 | 262.58 |  82.44 |                                                          |
| stage1 SigLIP-B              | SigLIP-B early[multi]                                    |  78.02 |  93.62 |  96.66 | 268.30 |  84.95 | alpha=[0.8, 0.8]                                         |
| stage1 SigLIP-B              | SigLIP-B hybrid (v1)                                     |  78.94 |  94.30 |  97.04 | 270.28 |  85.82 | (alpha,w)=[(0.85, 0.2), (0.9, 0.3)]                      |
| stage1 SigLIP-B              | SigLIP-B hybrid +regions[grid] (ablation)                |  80.60 |  95.56 |  97.64 | 273.80 |  87.15 | g=[0.4, 0.4]                                             |
| stage1 SigLIP-B              | SigLIP-B hybrid +regions[od] (ablation)                  |  78.94 |  94.30 |  97.04 | 270.28 |  85.82 | g=[0.0, 0.0]                                             |
| stage1 SigLIP-B              | SigLIP-B hybrid +itc only (ablation)                     |  87.10 |  97.60 |  98.94 | 283.64 |  91.76 | g=[0.5, 0.5]                                             |
| stage1 SigLIP-B              | SigLIP-B +tags                                           |  78.94 |  94.30 |  97.04 | 270.28 |  85.82 | g=[0.0, 0.0]                                             |
| stage1 SigLIP-B              | SigLIP-B +regions                                        |  80.58 |  95.58 |  97.70 | 273.86 |  87.23 | g=[0.4, 0.4]                                             |
| stage1 SigLIP-B              | SigLIP-B +itc                                            |  87.92 |  97.74 |  99.04 | 284.70 |  92.28 | g=[0.5, 0.5]                                             |
| stage1 SigLIP-B              | SigLIP-B +hub (CB-Norm)                                  |  87.92 |  97.74 |  99.04 | 284.70 |  92.28 | (beta,rho)=[(10.0, 0.0), (10.0, 0.0)]                    |
| stage1 ensemble              | ensemble(CLIP-B/32+CLIP-L/14+SigLIP-B)                   |  87.74 |  97.74 |  99.12 | 284.60 |  92.20 | equal weights, no tuning                                 |
| ablation                     | ensemble + dual-softmax [TRANSDUCTIVE, not comparable]   |  81.96 |  95.48 |  97.76 | 275.20 |  87.91 | uses all test queries                                    |
| reference                    | BGE caption concat only                                  |  63.22 |  85.78 |  91.20 | 240.20 |  73.30 |                                                          |
| reference                    | BGE object tags only                                     |   5.48 |  15.86 |  23.26 |  44.60 |  11.28 |                                                          |
| reference                    | BLIP-2 ITC only (full gallery)                           |  85.44 |  97.16 |  98.80 | 281.40 |  90.73 |                                                          |
| stage2 CLIP-B/32 final       | CLIP-B/32 final recall@16 (bound)                        | nan    | nan    |  99.46 | nan    | nan    | upper bound of stage 2                                   |
| stage2 CLIP-B/32 final       | CLIP-B/32 final recall@32 (bound)                        | nan    | nan    |  99.68 | nan    | nan    | upper bound of stage 2                                   |
| stage2 CLIP-B/32 final       | CLIP-B/32 final recall@64 (bound)                        | nan    | nan    |  99.82 | nan    | nan    | upper bound of stage 2                                   |
| stage2 CLIP-B/32 final       | CLIP-B/32 final -> rerank ITM                            |  89.76 |  98.28 |  99.20 | 287.24 |  93.57 | (K,lambda) per fold=[(32, 0.5), (32, 0.7)]               |
| stage2 CLIP-B/32 final       | CLIP-B/32 final -> rerank ITM+ITC                        |  89.76 |  98.28 |  99.20 | 287.24 |  93.57 | (K,lambda) per fold=[(32, (0.5, 0.0)), (32, (0.7, 0.0))] |
| stage2 CLIP-L/14 final       | CLIP-L/14 final recall@16 (bound)                        | nan    | nan    |  99.52 | nan    | nan    | upper bound of stage 2                                   |
| stage2 CLIP-L/14 final       | CLIP-L/14 final recall@32 (bound)                        | nan    | nan    |  99.72 | nan    | nan    | upper bound of stage 2                                   |
| stage2 CLIP-L/14 final       | CLIP-L/14 final recall@64 (bound)                        | nan    | nan    |  99.88 | nan    | nan    | upper bound of stage 2                                   |
| stage2 CLIP-L/14 final       | CLIP-L/14 final -> rerank ITM                            |  89.94 |  98.26 |  99.24 | 287.44 |  93.68 | (K,lambda) per fold=[(16, 0.6), (16, 0.6)]               |
| stage2 CLIP-L/14 final       | CLIP-L/14 final -> rerank ITM+ITC                        |  89.94 |  98.26 |  99.24 | 287.44 |  93.68 | (K,lambda) per fold=[(16, (0.6, 0.0)), (16, (0.6, 0.0))] |
| stage2 SigLIP-B final        | SigLIP-B final recall@16 (bound)                         | nan    | nan    |  99.44 | nan    | nan    | upper bound of stage 2                                   |
| stage2 SigLIP-B final        | SigLIP-B final recall@32 (bound)                         | nan    | nan    |  99.70 | nan    | nan    | upper bound of stage 2                                   |
| stage2 SigLIP-B final        | SigLIP-B final recall@64 (bound)                         | nan    | nan    |  99.88 | nan    | nan    | upper bound of stage 2                                   |
| stage2 SigLIP-B final        | SigLIP-B final -> rerank ITM                             |  89.82 |  98.24 |  98.98 | 287.04 |  93.60 | (K,lambda) per fold=[(16, 0.8), (32, 0.9)]               |
| stage2 SigLIP-B final        | SigLIP-B final -> rerank ITM+ITC                         |  89.82 |  98.24 |  98.98 | 287.04 |  93.60 | (K,lambda) per fold=[(16, (0.8, 0.0)), (32, (0.9, 0.0))] |
| stage2 ensemble final        | ensemble final recall@16 (bound)                         | nan    | nan    |  99.50 | nan    | nan    | upper bound of stage 2                                   |
| stage2 ensemble final        | ensemble final recall@32 (bound)                         | nan    | nan    |  99.72 | nan    | nan    | upper bound of stage 2                                   |
| stage2 ensemble final        | ensemble final recall@64 (bound)                         | nan    | nan    |  99.86 | nan    | nan    | upper bound of stage 2                                   |
| stage2 ensemble final        | ensemble final -> rerank ITM                             |  89.66 |  98.42 |  99.26 | 287.34 |  93.57 | (K,lambda) per fold=[(64, 0.4), (64, 0.3)]               |
| stage2 ensemble final        | ensemble final -> rerank ITM+ITC                         |  89.66 |  98.42 |  99.26 | 287.34 |  93.57 | (K,lambda) per fold=[(64, (0.4, 0.0)), (64, (0.3, 0.0))] |
| stage2 CLIP-B/32 hybrid (v1) | CLIP-B/32 hybrid (v1) recall@16 (bound)                  | nan    | nan    |  97.64 | nan    | nan    | upper bound of stage 2                                   |
| stage2 CLIP-B/32 hybrid (v1) | CLIP-B/32 hybrid (v1) recall@32 (bound)                  | nan    | nan    |  98.82 | nan    | nan    | upper bound of stage 2                                   |
| stage2 CLIP-B/32 hybrid (v1) | CLIP-B/32 hybrid (v1) recall@64 (bound)                  | nan    | nan    |  99.40 | nan    | nan    | upper bound of stage 2                                   |
| stage2 CLIP-B/32 hybrid (v1) | CLIP-B/32 hybrid (v1) -> rerank ITM                      |  89.00 |  98.00 |  98.74 | 285.74 |  93.03 | (K,lambda) per fold=[(64, 1.0), (64, 0.9)]               |
| stage2 CLIP-B/32 hybrid (v1) | CLIP-B/32 hybrid (v1) -> rerank ITM+ITC                  |  89.50 |  98.04 |  98.98 | 286.52 |  93.31 | (K,lambda) per fold=[(64, (0.3, 0.5)), (64, (0.4, 0.4))] |
| stage2 CLIP-B/32 hybrid (v1) | CLIP-B/32 hybrid (v1) -> rerank ITM top-16 (v1 protocol) |  88.38 |  96.92 |  97.54 | 282.84 |  92.28 | lambda=[1.0, 0.9]                                        |
| stage2 CLIP-L/14 hybrid (v1) | CLIP-L/14 hybrid (v1) recall@16 (bound)                  | nan    | nan    |  98.04 | nan    | nan    | upper bound of stage 2                                   |
| stage2 CLIP-L/14 hybrid (v1) | CLIP-L/14 hybrid (v1) recall@32 (bound)                  | nan    | nan    |  99.08 | nan    | nan    | upper bound of stage 2                                   |
| stage2 CLIP-L/14 hybrid (v1) | CLIP-L/14 hybrid (v1) recall@64 (bound)                  | nan    | nan    |  99.60 | nan    | nan    | upper bound of stage 2                                   |
| stage2 CLIP-L/14 hybrid (v1) | CLIP-L/14 hybrid (v1) -> rerank ITM                      |  88.86 |  98.24 |  99.04 | 286.14 |  93.02 | (K,lambda) per fold=[(64, 1.0), (64, 0.8)]               |
| stage2 CLIP-L/14 hybrid (v1) | CLIP-L/14 hybrid (v1) -> rerank ITM+ITC                  |  89.24 |  98.24 |  99.12 | 286.60 |  93.27 | (K,lambda) per fold=[(64, (0.4, 0.5)), (64, (0.3, 0.5))] |
| stage2 CLIP-L/14 hybrid (v1) | CLIP-L/14 hybrid (v1) -> rerank ITM top-16 (v1 protocol) |  88.36 |  97.38 |  97.94 | 283.68 |  92.37 | lambda=[1.0, 0.9]                                        |
| stage2 SigLIP-B hybrid (v1)  | SigLIP-B hybrid (v1) recall@16 (bound)                   | nan    | nan    |  97.98 | nan    | nan    | upper bound of stage 2                                   |
| stage2 SigLIP-B hybrid (v1)  | SigLIP-B hybrid (v1) recall@32 (bound)                   | nan    | nan    |  99.02 | nan    | nan    | upper bound of stage 2                                   |
| stage2 SigLIP-B hybrid (v1)  | SigLIP-B hybrid (v1) recall@64 (bound)                   | nan    | nan    |  99.50 | nan    | nan    | upper bound of stage 2                                   |
| stage2 SigLIP-B hybrid (v1)  | SigLIP-B hybrid (v1) -> rerank ITM                       |  88.82 |  98.02 |  98.84 | 285.68 |  92.94 | (K,lambda) per fold=[(64, 1.0), (64, 0.8)]               |
| stage2 SigLIP-B hybrid (v1)  | SigLIP-B hybrid (v1) -> rerank ITM+ITC                   |  89.22 |  98.18 |  99.02 | 286.42 |  93.23 | (K,lambda) per fold=[(64, (0.5, 0.4)), (64, (0.4, 0.4))] |
| stage2 SigLIP-B hybrid (v1)  | SigLIP-B hybrid (v1) -> rerank ITM top-16 (v1 protocol)  |  88.30 |  97.26 |  97.84 | 283.40 |  92.36 | lambda=[1.0, 0.9]                                        |
| reference                    | BLIP-2 protocol (ITC top-128 -> ITM logit + ITC)         |  89.44 |  98.08 |  98.92 | 286.44 |  93.31 | reproduced, no tuning                                    |
