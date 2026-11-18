---
title: "Multimodal RAG v2: GPU benchmark kılavuzu"
subtitle: "Dal: feature/sota-v2 · Repo: kocbu21/multimodal-rag-feature-fusion"
---

# 1. Özet

`feature/sota-v2` dalında v1 analizinin bulduğu iki darboğaza yönelik, **hiçbir model eğitmeyen** beş yeni bileşen
ve bunları ölçen deney hattı var. Kod CPU'da test edildi, ama **benchmark sayıları henüz yok**. Makaledeki (`report/report.tex`)
bütün yeni sayılar kırmızı `\TBD{}` yer tutucusu. Bu kılavuzdaki adımlar GPU'da koşturulup bu yer tutucular
gerçek sonuçlarla doldurulacak.

| Bileşen | Neden | Kod |
|---|---|---|
| Bölge özellikleri (5 ızgara kırpımı + Florence-2 `<OD>` nesne kutuları, max pooling) | Tek global vektör küçük nesneleri, sayıları, nitelikleri kaçırıyor | `mmrag/regions.py` |
| Nesne etiketleri ("dog, frisbee") BGE kanalı | Caption'ların atladığı nesneleri adlandırır | `mmrag/regions.py` |
| BLIP-2 ITC kanalı (32 Q-Former vektörü, max) | İlk aşamanın recall@K'sını, yani reranker'ın tavanını yükseltmek | `mmrag/reranker.py` |
| **CB-Norm**: caption-bank hubness normalizasyonu | "Hub" görseller birçok sorguda 1. sırayı çalıyor. Banka = galerinin kendi üretilmiş caption'ları, test sorgusu ya da eğitim verisi kullanmaz (yeni katkı) | `mmrag/fusion.py` |
| 3 encoder'ın eşit ağırlıklı ensemble'ı | Farklı encoder'ların hataları farklı | `scripts/experiments_v2.py` |
| ITM+ITC rerank, aday havuzu K ∈ {16, 32, 64} CV ile | v1'de K=16 sabitti | `mmrag/retriever.py` |

Referans olarak aynı kodda yeniden koşturulanlar: v1 sistemi (hybrid + BLIP-2 top-16) ve **BLIP-2'nin kendi
retrieval protokolü** (ITC top-128 → ITM logit + ITC). Bütün ağırlıklar v1'deki gibi görsellere göre 2-fold
çapraz doğrulamayla (CV) seçilir. Final sistemin farkı için eşleştirilmiş bootstrap güven aralığı (%95, 2000
örnekleme) raporlanır.

CPU'da yapılan doğrulamalar:

- `tests/test_v2.py`: 7 birim testi. Füzyon matematiği, hızlı sıralama ve rerank sıralarının kaba kuvvet
  sonucuyla birebir aynı olması, CV'nin ayar fold'unu raporlamada kullanmaması kontrol ediliyor.
- Önbellekli BLIP-2 ITM/ITC yolu, `Blip2ForImageTextRetrieval.forward` ile birebir aynı skoru veriyor.
- Sentetik 24 görsellik veride build → upgrade → experiments_v2 → canlı değerlendirme zinciri baştan sona çalıştı.

# 2. Hazırlık

```bash
git fetch origin
git switch feature/sota-v2
.venv/Scripts/python -m pip install -r requirements.txt     # Linux: .venv/bin/python
.venv/Scripts/python tests/test_v2.py                        # "7 tests passed" görülmeli
```

Bu makinede zaten olması gerekenler (v1 koşularından): `data/caption_cache.json` (Flickr ve COCO için 4 caption
seti), `indexes/flickr1k_multi_*` (3 encoder), `indexes/coco5k_multi_siglip-base-patch16-224`. Eksik olanları
pipeline kendisi üretir, ama caption üretimi uzun sürer.

# 3. Çalıştırma

```bash
bash scripts/pipelines/run_v2.sh
```

Her adım önbelleğe yazar. Hata olursa düzeltip aynı komutu tekrar çalıştırmak yeterli, kaldığı yerden devam eder.
Log: `logs/run_v2_*.log`.

| Ortam değişkeni | Varsayılan | Ne zaman değiştirilir |
|---|---|---|
| `SLEEP` | `0` | GPU'yu başka işlerle paylaşıyorsan `0.2` (sessiz mod) |
| `COCO` | `1` | Yalnızca Flickr için `0` |
| `COCO_MAX_Q` | `0` (hepsi, 25.010) | RAM yetmezse `5000` (makalede belirt) |
| `COCO_NATIVE_K` | `0` | Zaman varsa `128` (COCO'da BLIP-2 protokolü, ~3,2M çift, uzun sürer) |
| `ENCODERS` | 3 encoder | Daha hızlı deneme için tek encoder |
| `PY` | otomatik | Python yolu farklıysa |

Kaba süre tahmini (RTX 5070, `SLEEP=0`):

- Flickr: Florence-2 `<OD>` ~10 dk, bölge/ITC vektörleri ~10 dk, ITM çiftleri (~0,5–1M) 30–90 dk, canlı değerlendirme ~30 dk.
- COCO: CLIP store'ları ~15 dk, `<OD>` ~45 dk, ITM (~1,2M çift) 1–2,5 saat.
- RAM: Flickr birkaç GB. COCO tam sorgu kümesiyle ~16–24 GB. Gerekirse `COCO_MAX_Q=5000` kullan.

# 4. Sağlık kontrolleri (makaleye geçmeden önce)

`results/tables/flickr1k_v2.md` içinde:

1. `… hybrid (v1)` satırları v1 ile ±0,2 içinde olmalı: 72.78 / 75.64 / 78.78 (CLIP-B/32, CLIP-L/14, SigLIP-B).
2. `… hybrid (v1) -> rerank ITM top-16 (v1 protocol)` ≈ 88.58 / 88.36 / 88.28 olmalı (±0,3).
3. `BLIP-2 protocol (ITC top-128 -> ITM logit + ITC)` yayınlanan 89.7'ye yakın çıkmalı (±1). Çok düşükse (<87),
   görüntü çözünürlüğünü ve processor'ı kontrol et, ITM/ITC çağrılarını incele.
4. `recall@K` satırları K büyüdükçe artmalı. Rerank R@1 değeri recall@K'yı geçemez.

Bir kontrol tutmazsa **sayıları makaleye yazma**. Önce nedenini bul.

# 5. Makaleyi güncelleme

1. `report/report.tex` içindeki **bütün** `\TBD{…}` değerlerini şu dosyalardan doldur:
   `results/tables/flickr1k_v2.md`, `results/tables/coco5k_v2.md`, `results/tables/flickr1k_multi_*_online_v2.md`.
   Satır adları tablodaki yöntem adlarıyla eşleşiyor.
2. `% TODO(GPU run)` yorumlarındaki soruları cevaplayan metni yaz. CV'nin seçtiği ağırlıklar
   `results/json/flickr1k_v2.json` → `"picked"` içinde.
3. Sayıların desteklemediği hiçbir iddiayı yazma:
   - "BLIP-2'yi (89.7) geçtik" ancak v2 R@1'i bunu sağlıyorsa yazılabilir. Yeniden koşturulan protokole karşı
     güven aralığının alt sınırı 0'ın üstündeyse "anlamlı" denebilir.
   - Bir bileşen yardım etmediyse (CV ağırlığı 0) bunu açıkça yaz. Negatif sonuç da sonuçtur.
   - Dual softmax transdüktiftir. Yalnızca etiketli ablation olarak kalır, ana karşılaştırmaya girmez.
   - Protokol notlarını koru: reranker COCO üzerinde fine-tune edilmiş, BEiT-3 Flickr üzerinde fine-tune edilmiş.
4. Figürü aç (`fig:v2stages` bloğunun yorumunu kaldır). `python scripts/make_figures.py` bu figürü üretir.
5. Özeti (abstract), katkılar ve sonuç bölümündeki sayıları güncelle. README'nin v2 bölümünü ve
   `results/SUMMARY.md` Part 7'yi sonuç tablolarıyla tamamla.
6. PDF'i derle: `cd report && latexmk -pdf report.tex` ya da Overleaf. `grep -n TBD report/report.tex` hiçbir
   sonuç döndürmemeli, ancak `\newcommand{\TBD}` tanım satırı kalabilir.

# 6. Commit ve push

```bash
git add -A results report README.md configs/v2_*.yaml
git commit -m "v2 benchmark results and final report"
git push origin feature/sota-v2
```

`indexes/`, `data/` ve `logs/` git'e girmez (.gitignore). PR'ı ekip birlikte karar verince aç.
