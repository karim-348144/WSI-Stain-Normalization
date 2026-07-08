#!/bin/bash
#SBATCH --job-name=summarize_color_stats_reinhard
#SBATCH --partition=all_usr_prod
#SBATCH --account=ai4bio2025
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=20
#SBATCH --output=/homes/wdkarim/color_statistics/logs/%x_%j.log
#SBATCH --error=/homes/wdkarim/color_statistics/logs/%x_%j.err

set -euo pipefail

mkdir -p /homes/wdkarim/color_statistics/logs

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONNOUSERSITE=1

source /work/ai4bio2025/stain_norm/code/.venv/bin/activate

echo "JOB_ID=${SLURM_JOB_ID}"
echo "HOSTNAME=$(hostname)"
echo "START_TIME=$(date)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "CUDA check:"

date
nvidia-smi || true

python /homes/wdkarim/code/color_statistics/summarize_color_stats.py

date
echo "DONE"