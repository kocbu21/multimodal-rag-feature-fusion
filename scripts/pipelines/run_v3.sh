#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

if [ -z "${PY:-}" ]; then
  if [ -x .venv/Scripts/python.exe ]; then PY=.venv/Scripts/python.exe
  elif [ -x .venv/bin/python ]; then PY=.venv/bin/python
  else PY=python; fi
fi
export HF_HUB_DISABLE_XET=1
ENCODERS=${ENCODERS:-"openai/clip-vit-base-patch32 openai/clip-vit-large-patch14 google/siglip-base-patch16-224 google/siglip2-so400m-patch14-384"}
LIVE_ENC=${LIVE_ENC:-google/siglip2-so400m-patch14-384}
SLEEP=${SLEEP:-0}
COCO=${COCO:-1}
COCO_KS=${COCO_KS:-16,32}
COCO_VLM_M=${COCO_VLM_M:-4}
VLM=${VLM:-Qwen/Qwen2.5-VL-7B-Instruct}
EXTRA=${EXTRA:-Salesforce/blip-itm-large-coco}
FT=${FT:-Salesforce/blip-itm-large-flickr}
CFG=${CFG:-}
CAPTION_BATCH=${CAPTION_BATCH:-8}
ITM_BATCH=${ITM_BATCH:-32}
VLM_BATCH=${VLM_BATCH:-16}
LIVE_N=${LIVE_N:-1000}
mkdir -p logs
LOG=logs/run_v3_$(date +%Y%m%d_%H%M%S).log
exec > >(tee -a "$LOG") 2>&1
step() { echo; echo "== $(date +%T) $*"; }
slug() { echo "${1##*/}" | sed 's/[^a-zA-Z0-9]\+/-/g'; }
ENC_CSV=$(echo $ENCODERS | tr ' ' ',')

step "python: $PY"; $PY -c "import torch, transformers; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'transformers', transformers.__version__)"
step "unit tests"; $PY tests/test_v2.py
step "smoke test of the new models"; $PY scripts/smoke_v3_models.py --encoder "$LIVE_ENC" --extra "$EXTRA" --vlm "$VLM"

for DS in flickr1k $([ "$COCO" = "1" ] && echo coco5k); do
  for ENC in $ENCODERS; do
    OUT=indexes/${DS}_multi_$(slug "$ENC")
    if [ ! -f "$OUT/store.json" ]; then
      step "build $OUT"
      $PY scripts/build_index.py ${CFG:+--config "$CFG"} --source "$DS" --encoder "$ENC" --text-embedder --out "$OUT" \
          --caption-batch-size "$CAPTION_BATCH"
    fi
  done

  if [ "$DS" = flickr1k ]; then
    step "experiments v3 $DS (offline, 5000 queries, 2-fold CV)"
    $PY scripts/experiments_v2.py ${CFG:+--config "$CFG"} --dataset "$DS" --encoders "$ENC_CSV" --rerank-ks 16,32,64 \
        --native-k 128 --sleep "$SLEEP" --itm-batch "$ITM_BATCH" --tag _v3 \
        --extra-rerankers "$EXTRA" --ft-rerankers "$FT" --vlm "$VLM" --vlm-m 4,8 --vlm-batch "$VLM_BATCH"
  else
    step "experiments v3 $DS (offline, all queries, 2-fold CV)"
    $PY scripts/experiments_v2.py ${CFG:+--config "$CFG"} --dataset "$DS" --encoders "$ENC_CSV" --rerank-ks "$COCO_KS" \
        --native-k 0 --sleep "$SLEEP" --itm-batch "$ITM_BATCH" --tag _v3 --no-region-ablation \
        --extra-rerankers "$EXTRA" --vlm "$VLM" --vlm-m "$COCO_VLM_M" --vlm-batch "$VLM_BATCH"
  fi
  $PY scripts/v2_live_config.py --dataset "$DS" --tag _v3

  if [ "$DS" = flickr1k ]; then
    OUT=indexes/${DS}_multi_$(slug "$LIVE_ENC")
    grep -q '"region"' "$OUT/store.json" || { step "add v2 vectors to $OUT"; $PY scripts/build_index.py ${CFG:+--config "$CFG"} --upgrade "$OUT"; }
    step "live eval $OUT (v3 stage-2 cascade)"
    $PY scripts/eval_online.py --store "$OUT" --config "configs/v3_${DS}_$(slug "$LIVE_ENC").yaml" --n "$LIVE_N" \
        --sleep "$SLEEP" --strategies hybrid,full --rerank-strategy full --suffix _v3
  fi
done

step "figures"; $PY scripts/make_figures.py
step "V3-DONE. Results: results/tables/*_v2_v3.md, results/json/*_v2_v3.json. Log: $LOG"
