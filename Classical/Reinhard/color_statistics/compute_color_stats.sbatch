#!/bin/bash
#SBATCH --job-name=color_stats_reinhard
#SBATCH --partition=all_usr_prod
#SBATCH --account=ai4bio2025
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=20
#SBATCH --output=/homes/wdkarim/color_statistics/logs/%x_%j.log
#SBATCH --error=/homes/wdkarim/color_statistics/logs/%x_%j.err

set -euo pipefail

mkdir -p /homes/wdkarim/color_statistics/logs
mkdir -p /homes/wdkarim/color_statistics

export PYTHONNOUSERSITE=1
source /work/ai4bio2025/stain_norm/code/.venv/bin/activate

echo "JOB_ID=${SLURM_JOB_ID}"
echo "HOSTNAME=$(hostname)"
echo "START_TIME=$(date)"

date
python /homes/wdkarim/code/color_statistics/compute_color_stats.py
date

echo "DONE"