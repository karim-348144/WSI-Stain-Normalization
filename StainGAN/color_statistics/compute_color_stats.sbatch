#!/bin/bash
#SBATCH --job-name=color_stats_staingan
#SBATCH --partition=all_usr_prod
#SBATCH --account=ai4bio2025
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=20
#SBATCH --output=/work/h2020deciderficarra_shared/gcorso/wdkarim/stain_norm/StainGAN/logs/%x_%j.log
#SBATCH --error=/work/h2020deciderficarra_shared/gcorso/wdkarim/stain_norm/StainGAN/logs/%x_%j.err

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

python /homes/wdkarim/stain_norm/code/staingan/color_statistics/compute_color_stats.py

date
echo "DONE"