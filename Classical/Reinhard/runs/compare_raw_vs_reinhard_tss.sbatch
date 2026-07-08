#!/bin/bash
#SBATCH --job-name=compare_raw_vs_reinhard
#SBATCH --partition=all_usr_prod
#SBATCH --account=ai4bio2025
#SBATCH --time=04:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=/homes/wdkarim/logs/%x_%j.log
#SBATCH --error=/homes/wdkarim/logs/%x_%j.err

set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source /work/ai4bio2025/stain_norm/code/.venv/bin/activate

echo "JOB_ID=${SLURM_JOB_ID}"
echo "HOSTNAME=$(hostname)"
echo "START_TIME=$(date)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "CUDA check:"

date
nvidia-smi || true

python /homes/wdkarim/code/compare_raw_vs_reinhard_tss.py

date
echo "DONE"