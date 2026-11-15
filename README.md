# Enhancing Cross-Modal Retrieval in Multimodal RAG through Visual-Textual Feature Fusion

Görsel ve metinleri aynı embedding uzayına taşıyıp tek bir vektör veritabanında arayan,
istenirse bulunan kaynaklara dayanarak cevap üreten bir multimodal RAG sistemi. Projenin
araştırma kısmı: görsel vektörlerle (CLIP/SigLIP) metin tarafı vektörlerini (BLIP / Florence-2
caption'ları) birleştirmenin (**feature fusion**) cross-modal retrieval başarısını ne kadar artırdığını ölçmek.

## Klasör yapısı

```
.
├── app.py                  Gradio web demosu
├── requirements.txt
├── configs/                Pipeline ayarları (YAML)
├── mmrag/                  Kütüphane kodu (encoder, füzyon, vector store, retriever, reranker, generator)
├── scripts/                Python betikleri (index kurma, değerlendirme, deneyler, figürler, CLI arama)
│   └── pipelines/          Uçtan uca bash koşum zincirleri (run_*.sh)
├── data/                   Veri (Flickr30k 1K indirilir; kendi verin data/raw/ altına) — git'e girmez
├── indexes/                Üretilen FAISS store'ları ve ITM skor cache'leri — git'e girmez
├── results/
│   ├── SUMMARY.md          Bütün sonuçların özeti
│   ├── tables/             .md / .csv sonuç tabloları
│   ├── json/               Ham sonuçlar (.json)
│   └── figures/            Deney grafikleri (.png)
├── report/                 IEEE raporu (report.tex, report.pdf, figures/)
├── docs/
│   └── assets/             Proje plan görseli
└── logs/                   Koşum logları — git'e girmez
```

## Mimari (plan → kod)

| Plan kutusu | Kod |
|---|---|
| Görseller / Metinler | `mmrag/data.py` (kendi verin: `data/raw/`, benchmark: Flickr30k 1K test) |
| Image Encoder / Text Encoder (CLIP / SigLIP) | `mmrag/encoders.py` → `DualEncoder` |
| Caption vector | `mmrag/captioner.py` (BLIP, Florence-2, Qwen2.5-VL) |
| Metin embedding kanalı (BGE) | `mmrag/text_embedder.py` |
| 2. aşama: cross-attention reranking (BLIP-2 / BLIP ITM) | `mmrag/reranker.py` |
| Unified Vector Database (FAISS + ID + metadata) | `mmrag/vector_store.py` → `UnifiedVectorStore` |
| Multimodal Search (metin / görsel / görsel+metin) | `mmrag/retriever.py` → `MultimodalRetriever` |
| Opsiyonel LLM/VLM cevap üretimi | `mmrag/generator.py` (Claude) |
| Füzyon yöntemleri | `mmrag/fusion.py` |
| Değerlendirme (R@K, MRR) | `mmrag/metrics.py`, `scripts/evaluate.py` |

## Füzyon yöntemleri

1. **Early fusion (doküman tarafı):** `v = norm(α·v_image + (1-α)·v_caption)` — her görsel,
   kendi görsel vektörü ile BLIP caption'ının metin vektörünün karışımıyla temsil edilir.
2. **Query fusion (sorgu tarafı):** görsel + metin sorgusu `q = norm(β·q_image + (1-β)·q_text)`.
3. **Late fusion (skor tarafı):** indexlerden gelen sıralamalar z-score ağırlıklı toplam veya RRF ile birleştirilir.
4. **Modality centering:** CLIP'teki "modality gap"i azaltmak için her modalitenin ortalaması çıkarılır.
5. **Hibrit füzyon:** sorgu-caption benzerliği ayrıca gerçek bir metin modelinde (BGE) hesaplanır ve
   görsel taraf skoruyla birleştirilir (`hybrid` stratejisi).
6. **Cross-attention reranking:** ilk aşamanın top-16 görseli BLIP-2 ITM (varsayılan; daha hafif seçenek BLIP ITM) ile yeniden puanlanır. Metin,
   görselin patch özelliklerine cross-attention ile bakar; füzyonun en derin hali.

## Kurulum

```bash
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python -m pip install -r requirements.txt
```

## Kullanım

```bash
# 1) Benchmark index'i (Flickr30k 1K test indirilir, BLIP caption'ları üretilir)
.venv\Scripts\python scripts/build_index.py --source flickr1k
.venv\Scripts\python scripts/build_index.py --source flickr1k --encoder google/siglip-base-patch16-224

# 2) Değerlendirme -> results/tables/*.csv|*.md, results/json/, results/figures/*_alpha_sweep.png
.venv\Scripts\python scripts/evaluate.py --store indexes/flickr1k_clip-vit-base-patch32

# 3) İyileştirme deneyleri (BLIP-large / Florence caption'ları, BGE hibrit, ITM rerank)
bash scripts/pipelines/run_improvements.sh

# 4) Kendi verin: görselleri data/raw/images, metinleri data/raw/texts içine koy
#    (varsayılan: Florence caption + BGE vektörleri, aramada ITM reranking)
.venv\Scripts\python scripts/build_index.py --source raw --text-embedder
.venv\Scripts\python scripts/search.py --store indexes/raw_clip-vit-base-patch32 --text "a dog on the beach" --strategy hybrid
.venv\Scripts\python app.py --store indexes/raw_clip-vit-base-patch32 --rerank     # Gradio arayüzü
```

Cevap üretimi için `ANTHROPIC_API_KEY` tanımlı olmalı (`--answer` bayrağı veya arayüzdeki kutucuk).

**Sessiz mod:** `configs/default.yaml` içindeki `throttle.gpu_sleep_s` her GPU batch'inden sonra
mola verir (varsayılan 0.5 sn). Tam hız için `0` yapın.

## Sonuçlar (Flickr30k 1K test, Text → Image R@1)

| Encoder | Sadece görsel | Early fusion (BLIP-base) | Early fusion (4 caption seti) | Hibrit (+BGE) | + ITM rerank |
|---|---|---|---|---|---|
| CLIP ViT-B/32 | 58.76 | 61.16 | 69.54 | 72.78 | **84.92** |
| CLIP ViT-L/14 | 64.58 | 65.88 | 72.96 | 75.64 | **85.92** |
| SigLIP base | 74.58 | 74.64 | 77.92 | 78.78 | **85.50** |

Tüm füzyon ağırlıkları 2-fold çapraz doğrulamayla seçildi. Detaylar ve gözlemler: `results/SUMMARY.md`.
Canlı arama yolu (CLI / Gradio ile aynı kod) uçtan uca da ölçüldü (`scripts/eval_online.py`, 1000 sorgu):
hybrid + rerank R@1 = 84.6 (CLIP B/32), 86.0 (CLIP L/14), 85.4 (SigLIP), yani offline sonuçlarla aynı.
Varsayılan store artık görsel başına 4 caption seti tutuyor (`captioner.extra`, tek set için `--single-caption`).

Daha güçlü reranker: BLIP-2 ITM (`Salesforce/blip2-itm-vit-g-coco`, `scripts/pipelines/run_blip2.sh`) aynı top-16'yı
yeniden sıralayınca R@1 = **88.58** (CLIP B/32), **88.36** (CLIP L/14) ve **88.28** (SigLIP) oluyor.
BLIP-2 artık varsayılan reranker (`configs/default.yaml`, λ=0.9). Canlı yolda (1000 sorgu) hybrid + BLIP-2 rerank
R@1 = 88.7 / 88.9 / 89.0; sorgu başına ~0.77 s (BLIP ile ~0.73 s, sessiz mod).
Yayınlanmış sonuçlarla karşılaştırma (BLIP-2 zero-shot 89.7, BEiT-3 fine-tuned 90.3 vb.) raporda Tablo IV'te.

İkinci benchmark: MS-COCO 5K (`scripts/pipelines/run_coco5k.sh`, SigLIP, 1000 sorgu, ağırlıklar Flickr'dan alındı). T→I R@1: sadece görsel 46.7, early fusion 50.7, hybrid 52.2, hybrid + BLIP-2 rerank **67.3**. Aşama sıralaması Flickr'daki gibi, ama 5 kat büyük ve kalabalık galeride düzey çok daha düşük; kazancın büyük kısmı yine reranker'dan geliyor. Ayrıntı: `results/SUMMARY.md` Part 6 ve rapor.

Tekrar üretmek için: `bash scripts/pipelines/run_all.sh`, `bash scripts/pipelines/run_improvements.sh`, `bash scripts/pipelines/run_online.sh`,
`bash scripts/pipelines/run_blip2.sh`, ardından (isteğe bağlı) `bash scripts/pipelines/run_coco5k.sh`.

## Rapor

`report/report.pdf` (IEEE konferans formatı, İngilizce, 7 sayfa). Kaynak: `report/report.tex`. Grafikler
`python scripts/make_figures.py` ile `report/figures/` altına üretilir. PDF'i yeniden derlemek için `report/`
klasörünü Overleaf'e yükleyebilir ya da yerel bir LaTeX derleyicisi (TeX Live, MiKTeX, Tectonic) kullanabilirsin.
