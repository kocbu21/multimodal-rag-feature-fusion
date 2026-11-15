#!/usr/bin/env bash
# v2 benchmark pipeline: everything needed for the v2 tables of the report, resumable (every step caches).
#
#   bash scripts/pipelines/run_v2.sh                 # Flickr30k 1K (offline CV + live), then MS-COCO 5K offline
#   COCO=0 bash scripts/pipelines/run_v2.sh          # Flickr30k only
#   SLEEP=0.2 bash scripts/pipelines/run_v2.sh       # quiet mode (pause after each GPU batch); default 0 = full speed
#
# Environment knobs: PY (python executable), ENCODERS, SLEEP, COCO, COCO_KS, COCO_NATIVE_K, COCO_MAX_Q, LIVE_N.
set -euo pipefail
cd "$(dirname "$0")/../.."

if [ -z "${PY:-}" ]; then
  if [ -x .venv/Scripts/python.exe ]; then PY=.venv/Scripts/python.exe
  elif [ -x .venv/bin/python ]; then PY=.venv/bin/python
  else PY=python; fi
fi
export HF_HUB_DISABLE_XET=1
ENCODERS=${ENCODERS:-"openai/clip-vit-base-patch32 openai/clip-vit-large-patch14 google/siglip-base-patch16-224"}
SLEEP=${SLEEP:-0}
COCO=${COCO:-1}
COCO_KS=${COCO_KS:-16,32}
COCO_NATIVE_K=${COCO_NATIVE_K:-0}
COCO_MAX_Q=${COCO_MAX_Q:-0}
LIVE_N=${LIVE_N:-1000}
mkdir -p logs
LOG=logs/run_v2_$(date +%Y%m%d_%H%M%S).log
exec > >(tee -a "$LOG") 2>&1
step() { echo; echo "== $(date +%T) $*"; }
slug() { echo "${1##*/}" | sed 's/[^a-zA-Z0-9]\+/-/g'; }
ENC_CSV=$(echo $ENCODERS | tr ' ' ',')

step "python: $PY"; $PY -c "import torch, transformers; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'transformers', transformers.__version__)"
step "unit tests"; $PY tests/test_v2.py

for DS in flickr1k $([ "$COCO" = "1" ] && echo coco5k); do
  for ENC in $ENCODERS; do
    OUT=indexes/${DS}_multi_$(slug "$ENC")
    if [ ! -f "$OUT/store.json" ]; then
      step "build $OUT (captions are cached in data/caption_cache.json and shared)"
      $PY scripts/build_index.py --source "$DS" --encoder "$ENC" --text-embedder --out "$OUT"
    fi
  done

  if [ "$DS" = flickr1k ]; then
    step "experiments_v2 $DS (offline, 5000 queries, 2-fold CV)"
    $PY scripts/experiments_v2.py --dataset "$DS" --encoders "$ENC_CSV" --rerank-ks 16,32,64 --native-k 128 --sleep "$SLEEP"
  else
    step "experiments_v2 $DS (offline, 2-fold CV)"
    $PY scripts/experiments_v2.py --dataset "$DS" --encoders "$ENC_CSV" --rerank-ks "$COCO_KS" \
        --native-k "$COCO_NATIVE_K" --max-queries "$COCO_MAX_Q" --sleep "$SLEEP" --no-region-ablation
  fi
  $PY scripts/v2_live_config.py --dataset "$DS"

  if [ "$DS" = flickr1k ]; then
    for ENC in $ENCODERS; do
      OUT=indexes/${DS}_multi_$(slug "$ENC")
      if ! grep -q '"region"' "$OUT/store.json"; then
        step "add v2 vectors to $OUT"
        $PY scripts/build_index.py --upgrade "$OUT"
      fi
      step "live eval $OUT with configs/v2_${DS}_$(slug "$ENC").yaml"
      $PY scripts/eval_online.py --store "$OUT" --config "configs/v2_${DS}_$(slug "$ENC").yaml" --n "$LIVE_N" \
          --sleep "$SLEEP" --strategies image,hybrid,full --rerank-strategy full --suffix _v2
    done
  fi
done

step "figures"; $PY scripts/make_figures.py
step "V2-DONE. Results: results/tables/*_v2*.md, results/json/*_v2*.json, report/figures/v2_stages_*.pdf. Log: $LOG"
