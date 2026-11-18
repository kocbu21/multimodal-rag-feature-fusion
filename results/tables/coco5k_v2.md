v2 experiments on coco5k: 5000 images, 25010 text queries; encoders ['openai/clip-vit-base-patch32', 'openai/clip-vit-large-patch14', 'google/siglip-base-patch16-224']; reranker Salesforce/blip2-itm-vit-g-coco; K grid [16, 32]; regions ['grid', 'od']; 2-fold CV over images (seed 0).

| group                        | method                                                   |    R@1 |    R@5 |   R@10 |   RSUM |    MRR | params                                                   |
|:-----------------------------|:---------------------------------------------------------|-------:|-------:|-------:|-------:|-------:|:---------------------------------------------------------|
| stage1 CLIP-B/32             | CLIP-B/32 image                                          |  30.51 |  55.97 |  66.83 | 153.31 |  42.62 |                                                          |
| stage1 CLIP-B/32             | CLIP-B/32 early[multi]                                   |  40.65 |  66.19 |  75.87 | 182.71 |  52.49 | alpha=[0.6, 0.7]                                         |
| stage1 CLIP-B/32             | CLIP-B/32 hybrid (v1)                                    |  45.93 |  71.57 |  80.23 | 197.72 |  57.60 | (alpha,w)=[(0.8, 0.5), (0.85, 0.5)]                      |
| stage1 CLIP-B/32             | CLIP-B/32 hybrid +itc only (ablation)                    |  63.76 |  85.93 |  91.71 | 241.40 |  73.58 | g=[0.6, 0.5]                                             |
| stage1 CLIP-B/32             | CLIP-B/32 +tags                                          |  45.93 |  71.57 |  80.23 | 197.72 |  57.60 | g=[0.0, 0.0]                                             |
| stage1 CLIP-B/32             | CLIP-B/32 +regions                                       |  46.19 |  71.87 |  80.48 | 198.54 |  57.88 | g=[0.2, 0.2]                                             |
| stage1 CLIP-B/32             | CLIP-B/32 +itc                                           |  63.85 |  85.90 |  91.71 | 241.46 |  73.64 | g=[0.6, 0.6]                                             |
| stage1 CLIP-B/32             | CLIP-B/32 +hub (CB-Norm)                                 |  64.32 |  86.08 |  91.77 | 242.16 |  74.01 | (beta,rho)=[(50.0, 0.25), (50.0, 0.25)]                  |
| stage1 CLIP-L/14             | CLIP-L/14 image                                          |  36.52 |  61.14 |  71.06 | 168.71 |  48.15 |                                                          |
| stage1 CLIP-L/14             | CLIP-L/14 early[multi]                                   |  45.92 |  70.50 |  79.41 | 195.83 |  57.30 | alpha=[0.7, 0.7]                                         |
| stage1 CLIP-L/14             | CLIP-L/14 hybrid (v1)                                    |  48.49 |  73.27 |  81.46 | 203.22 |  59.80 | (alpha,w)=[(0.8, 0.4), (0.8, 0.4)]                       |
| stage1 CLIP-L/14             | CLIP-L/14 hybrid +itc only (ablation)                    |  64.01 |  85.82 |  91.68 | 241.52 |  73.72 | g=[0.6, 0.5]                                             |
| stage1 CLIP-L/14             | CLIP-L/14 +tags                                          |  48.49 |  73.27 |  81.46 | 203.22 |  59.80 | g=[0.0, 0.0]                                             |
| stage1 CLIP-L/14             | CLIP-L/14 +regions                                       |  48.94 |  73.33 |  81.65 | 203.91 |  60.16 | g=[0.2, 0.1]                                             |
| stage1 CLIP-L/14             | CLIP-L/14 +itc                                           |  63.93 |  85.87 |  91.66 | 241.46 |  73.69 | g=[0.7, 0.6]                                             |
| stage1 CLIP-L/14             | CLIP-L/14 +hub (CB-Norm)                                 |  64.26 |  86.06 |  91.72 | 242.05 |  73.96 | (beta,rho)=[(20.0, 1.0), (20.0, 0.5)]                    |
| stage1 SigLIP-B              | SigLIP-B image                                           |  47.82 |  72.46 |  80.98 | 201.26 |  59.09 |                                                          |
| stage1 SigLIP-B              | SigLIP-B early[multi]                                    |  51.49 |  75.49 |  83.18 | 210.16 |  62.36 | alpha=[0.8, 0.8]                                         |
| stage1 SigLIP-B              | SigLIP-B hybrid (v1)                                     |  52.66 |  77.01 |  84.59 | 214.25 |  63.66 | (alpha,w)=[(0.85, 0.3), (0.9, 0.3)]                      |
| stage1 SigLIP-B              | SigLIP-B hybrid +itc only (ablation)                     |  64.10 |  86.11 |  91.93 | 242.14 |  73.85 | g=[0.5, 0.5]                                             |
| stage1 SigLIP-B              | SigLIP-B +tags                                           |  52.66 |  77.01 |  84.59 | 214.25 |  63.66 | g=[0.0, 0.0]                                             |
| stage1 SigLIP-B              | SigLIP-B +regions                                        |  53.74 |  77.80 |  85.31 | 216.85 |  64.64 | g=[0.3, 0.3]                                             |
| stage1 SigLIP-B              | SigLIP-B +itc                                            |  64.28 |  86.12 |  91.96 | 242.36 |  73.97 | g=[0.4, 0.4]                                             |
| stage1 SigLIP-B              | SigLIP-B +hub (CB-Norm)                                  |  64.50 |  86.24 |  92.02 | 242.77 |  74.18 | (beta,rho)=[(50.0, 0.25), (50.0, 0.25)]                  |
| stage1 ensemble              | ensemble(CLIP-B/32+CLIP-L/14+SigLIP-B)                   |  64.67 |  86.25 |  91.80 | 242.73 |  74.27 | equal weights, no tuning                                 |
| ablation                     | ensemble + dual-softmax [TRANSDUCTIVE, not comparable]   |  58.33 |  82.41 |  89.07 | 229.80 |  69.16 | uses all test queries                                    |
| reference                    | BGE caption concat only                                  |  39.55 |  65.27 |  74.98 | 179.80 |  51.49 |                                                          |
| reference                    | BGE object tags only                                     |   5.57 |  13.43 |  20.00 |  38.99 |  10.83 |                                                          |
| reference                    | BLIP-2 ITC only (full gallery)                           |  62.97 |  85.70 |  91.50 | 240.18 |  73.07 |                                                          |
| stage2 CLIP-B/32 final       | CLIP-B/32 final recall@16 (bound)                        | nan    | nan    |  94.67 | nan    | nan    | upper bound of stage 2                                   |
| stage2 CLIP-B/32 final       | CLIP-B/32 final recall@32 (bound)                        | nan    | nan    |  97.67 | nan    | nan    | upper bound of stage 2                                   |
| stage2 CLIP-B/32 final       | CLIP-B/32 final -> rerank ITM                            |  68.26 |  87.76 |  92.80 | 248.83 |  77.00 | (K,lambda) per fold=[(32, 0.6), (32, 0.8)]               |
| stage2 CLIP-B/32 final       | CLIP-B/32 final -> rerank ITM+ITC                        |  68.26 |  87.76 |  92.80 | 248.83 |  77.00 | (K,lambda) per fold=[(32, (0.6, 0.0)), (32, (0.8, 0.0))] |
| stage2 CLIP-L/14 final       | CLIP-L/14 final recall@16 (bound)                        | nan    | nan    |  94.57 | nan    | nan    | upper bound of stage 2                                   |
| stage2 CLIP-L/14 final       | CLIP-L/14 final recall@32 (bound)                        | nan    | nan    |  97.61 | nan    | nan    | upper bound of stage 2                                   |
| stage2 CLIP-L/14 final       | CLIP-L/14 final -> rerank ITM                            |  68.15 |  87.78 |  92.78 | 248.71 |  76.95 | (K,lambda) per fold=[(32, 0.6), (32, 0.8)]               |
| stage2 CLIP-L/14 final       | CLIP-L/14 final -> rerank ITM+ITC                        |  68.10 |  87.77 |  92.76 | 248.63 |  76.92 | (K,lambda) per fold=[(32, (0.6, 0.1)), (32, (0.8, 0.0))] |
| stage2 SigLIP-B final        | SigLIP-B final recall@16 (bound)                         | nan    | nan    |  94.86 | nan    | nan    | upper bound of stage 2                                   |
| stage2 SigLIP-B final        | SigLIP-B final recall@32 (bound)                         | nan    | nan    |  97.67 | nan    | nan    | upper bound of stage 2                                   |
| stage2 SigLIP-B final        | SigLIP-B final -> rerank ITM                             |  68.27 |  87.83 |  92.89 | 248.99 |  77.06 | (K,lambda) per fold=[(32, 0.7), (32, 0.8)]               |
| stage2 SigLIP-B final        | SigLIP-B final -> rerank ITM+ITC                         |  68.27 |  87.83 |  92.89 | 248.99 |  77.06 | (K,lambda) per fold=[(32, (0.7, 0.0)), (32, (0.8, 0.0))] |
| stage2 ensemble final        | ensemble final recall@16 (bound)                         | nan    | nan    |  94.75 | nan    | nan    | upper bound of stage 2                                   |
| stage2 ensemble final        | ensemble final recall@32 (bound)                         | nan    | nan    |  97.68 | nan    | nan    | upper bound of stage 2                                   |
| stage2 ensemble final        | ensemble final -> rerank ITM                             |  68.21 |  87.80 |  92.81 | 248.82 |  76.99 | (K,lambda) per fold=[(32, 0.6), (32, 0.8)]               |
| stage2 ensemble final        | ensemble final -> rerank ITM+ITC                         |  68.21 |  87.80 |  92.81 | 248.82 |  76.99 | (K,lambda) per fold=[(32, (0.6, 0.0)), (32, (0.8, 0.0))] |
| stage2 CLIP-B/32 hybrid (v1) | CLIP-B/32 hybrid (v1) recall@16 (bound)                  | nan    | nan    |  85.67 | nan    | nan    | upper bound of stage 2                                   |
| stage2 CLIP-B/32 hybrid (v1) | CLIP-B/32 hybrid (v1) recall@32 (bound)                  | nan    | nan    |  92.20 | nan    | nan    | upper bound of stage 2                                   |
| stage2 CLIP-B/32 hybrid (v1) | CLIP-B/32 hybrid (v1) -> rerank ITM                      |  67.11 |  85.33 |  89.32 | 241.76 |  75.24 | (K,lambda) per fold=[(32, 1.0), (32, 1.0)]               |
| stage2 CLIP-B/32 hybrid (v1) | CLIP-B/32 hybrid (v1) -> rerank ITM+ITC                  |  66.97 |  85.45 |  89.41 | 241.83 |  75.22 | (K,lambda) per fold=[(32, (0.5, 0.4)), (32, (0.7, 0.2))] |
| stage2 CLIP-B/32 hybrid (v1) | CLIP-B/32 hybrid (v1) -> rerank ITM top-16 (v1 protocol) |  65.81 |  82.27 |  84.99 | 233.06 |  73.30 | lambda=[1.0, 1.0]                                        |
| stage2 CLIP-L/14 hybrid (v1) | CLIP-L/14 hybrid (v1) recall@16 (bound)                  | nan    | nan    |  86.45 | nan    | nan    | upper bound of stage 2                                   |
| stage2 CLIP-L/14 hybrid (v1) | CLIP-L/14 hybrid (v1) recall@32 (bound)                  | nan    | nan    |  92.82 | nan    | nan    | upper bound of stage 2                                   |
| stage2 CLIP-L/14 hybrid (v1) | CLIP-L/14 hybrid (v1) -> rerank ITM                      |  67.35 |  85.69 |  89.76 | 242.79 |  75.56 | (K,lambda) per fold=[(32, 1.0), (32, 0.9)]               |
| stage2 CLIP-L/14 hybrid (v1) | CLIP-L/14 hybrid (v1) -> rerank ITM+ITC                  |  67.46 |  85.75 |  89.84 | 243.05 |  75.65 | (K,lambda) per fold=[(32, (0.5, 0.4)), (32, (0.8, 0.1))] |
| stage2 CLIP-L/14 hybrid (v1) | CLIP-L/14 hybrid (v1) -> rerank ITM top-16 (v1 protocol) |  66.32 |  82.78 |  85.75 | 234.86 |  73.84 | lambda=[1.0, 1.0]                                        |
| stage2 SigLIP-B hybrid (v1)  | SigLIP-B hybrid (v1) recall@16 (bound)                   | nan    | nan    |  88.96 | nan    | nan    | upper bound of stage 2                                   |
| stage2 SigLIP-B hybrid (v1)  | SigLIP-B hybrid (v1) recall@32 (bound)                   | nan    | nan    |  94.14 | nan    | nan    | upper bound of stage 2                                   |
| stage2 SigLIP-B hybrid (v1)  | SigLIP-B hybrid (v1) -> rerank ITM                       |  67.66 |  86.51 |  90.86 | 245.02 |  76.09 | (K,lambda) per fold=[(32, 1.0), (32, 0.9)]               |
| stage2 SigLIP-B hybrid (v1)  | SigLIP-B hybrid (v1) -> rerank ITM+ITC                   |  67.63 |  86.61 |  91.07 | 245.31 |  76.16 | (K,lambda) per fold=[(32, (0.5, 0.4)), (32, (0.7, 0.2))] |
| stage2 SigLIP-B hybrid (v1)  | SigLIP-B hybrid (v1) -> rerank ITM top-16 (v1 protocol)  |  67.20 |  84.77 |  88.06 | 240.03 |  75.13 | lambda=[1.0, 0.9]                                        |
