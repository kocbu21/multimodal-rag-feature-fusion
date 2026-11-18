#!/bin/bash
# SLURM wrapper for run_v2.sh (used for the reported v2 run on one A100 80 GB; 3 h 8 min end to end).
# Set the partition/account for your cluster; MMRAG_BASE holds env/ (python venv), hf/ (HF cache) and repo/.
# Models and data can be fetched beforehand on a login node with internet; the job then runs offline.
#SBATCH -p gpu
#SBATCH -n 16
#SBATCH --gres=gpu:1
#SBATCH -t 1-00:00:00
#SBATCH -J mmrag-v2
#SBATCH -o mmrag-v2_%j.out
B=${MMRAG_BASE:-$HOME/mmrag-v2}
export HF_HOME=$B/hf HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} TRANSFORMERS_OFFLINE=${HF_HUB_OFFLINE:-1} HF_HUB_DISABLE_XET=1
export TMPDIR=$B/scratch MPLCONFIGDIR=$B/scratch/mpl PY=$B/env/bin/python TOKENIZERS_PARALLELISM=false
mkdir -p $TMPDIR $MPLCONFIGDIR
cd $B/repo
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
printf "throttle:\n  gpu_sleep_s: 0\n" > $B/fast.yaml
CFG=$B/fast.yaml CAPTION_BATCH=32 ITM_BATCH=128 SLEEP=0 COCO=${COCO:-1} bash scripts/pipelines/run_v2.sh
