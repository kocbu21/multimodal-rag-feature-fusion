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

## v2: geliştirmeler ve sonuçlar

v1 analizinin bulduğu iki darboğaza (ilk aşamanın kaçırdığı adaylar ve top-16 içindeki birbirine çok benzeyen
sahneler) yönelik, eğitim gerektirmeyen bileşenler. Hepsi çapraz doğrulamayla seçiliyor; CV bir bileşeni
yararsız bulursa ağırlığı 0 olur. Bütün sayılar UHeM Altay'da tek bir A100 koşusundan (3 saat 8 dk) gelir.

**Text → Image, final sistem (encoder ensemble + BLIP-2 ITM, CV ile seçilen aday havuzu):**

| Veri | R@1 | R@5 | R@10 | v1'e göre ΔR@1 [%95 CI] | Referans |
|---|---:|---:|---:|---|---|
| Flickr30k 1K (5000 sorgu) | **89.66** | **98.42** | **99.26** | +1.28 [0.72, 1.88] | BLIP-2 protokolü (bizim kodda) 89.44 / 98.08 / 98.92, fark +0.22 [−0.38, 0.82] anlamlı değil; makalede 89.7 / 98.1 / 98.9 |
| COCO 5K (25.010 sorgu) | **68.21** | **87.80** | **92.81** | +1.01 … +2.40 (hepsi anlamlı) | BLIP-2 ViT-g COCO fine-tuned (makalede) 68.3 / 87.7 / 92.6 |

Bileşen bazında: BLIP-2 ITC kanalı ilk aşamayı +8 ile +18 R@1 yükseltiyor (en büyük katkı), ızgara bölge
kırpımları +0.3 ile +1.6, CB-Norm COCO'da +0.2 ile +0.5 (Flickr'da etkisiz), nesne etiketleri ve tespit edilen
nesne kırpımları CV tarafından kapatıldı. Ayrıntı: `results/tables/{flickr1k,coco5k}_v2.md`, makalede Bölüm V-E
ve COCO bölümü.

| Bileşen | Kod | Ne yapar |
|---|---|---|
| Bölge özellikleri | `mmrag/regions.py` | 5 ızgara kırpımı + Florence-2 `<OD>` nesne kutuları; görsel en iyi eşleşen bölgesiyle puanlanır (max pooling) |
| Nesne etiketleri | `mmrag/regions.py` | Tespit edilen etiketler ("dog, frisbee") BGE uzayında ayrı bir kanal |
| BLIP-2 ITC kanalı | `mmrag/reranker.py` | Reranker checkpoint'inin kontrastif başı (32 Q-Former vektörü) ilk aşamada ek kanal |
| CB-Norm | `mmrag/fusion.py` | Caption-bank hubness normalizasyonu: galerinin kendi üretilmiş caption'ları sorgu bankası; eğitim verisi ya da test sorgusu kullanmaz |
| Encoder ensemble | `scripts/experiments_v2.py` | CLIP B/32 + CLIP L/14 + SigLIP ilk aşama skorlarının eşit ağırlıklı ortalaması |
| ITM+ITC rerank | `mmrag/retriever.py` | Aday havuzu K ∈ {16, 32, 64} ve ağırlıklar CV ile seçilir |

Yeniden üretilen referanslar (aynı kodda): BLIP-2'nin kendi protokolü (ITC top-128 → ITM logit + ITC) ve v1 sistemi.
Anlamlılık için eşleştirilmiş bootstrap güven aralığı (2000 örnekleme) kullanılıyor.

```bash
python tests/test_v2.py                     # GPU gerektirmeyen birim testleri
bash scripts/pipelines/run_v2.sh            # Flickr30k (offline CV + canlı yol) ve COCO 5K, kaldığı yerden devam eder
sbatch scripts/pipelines/v2_slurm.sbatch     # SLURM kümesinde (bölüm/hesap ayarlarını düzenle)
```

Çıktılar: `results/tables/{flickr1k,coco5k}_v2.md`, `results/json/*_v2.json` (CV seçimleri dahil),
`configs/v2_<veri>_<encoder>.yaml` (canlı yol ayarları), `report/figures/v2_stages_*.pdf`.
Canlı arama: `python app.py --store indexes/flickr1k_multi_siglip-base-patch16-224 --rerank` ve "full" stratejisi
(store'a önce `python scripts/build_index.py --upgrade <store>` ile v2 vektörleri eklenmeli). GPU koşusu için
adım adım talimatlar: `docs/GPU_RUNBOOK.md`.

## Rapor

`report/report.pdf` (IEEE konferans formatı, İngilizce, 11 sayfa, v1 + v2). Kaynak: `report/report.tex`. Grafikler
`python scripts/make_figures.py` ile `report/figures/` altına üretilir. PDF'i yeniden derlemek için `report/`
klasörünü Overleaf'e yükleyebilir ya da yerel bir LaTeX derleyicisi (TeX Live, MiKTeX, Tectonic) kullanabilirsin.
