#!/bin/bash
#SBATCH --job-name=tsne_tss_mac
#SBATCH --partition=all_usr_prod
#SBATCH --account=ai4bio2025
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=/homes/smdaniyal/stain_norm/logs/%x_%j.out
#SBATCH --error=/homes/smdaniyal/stain_norm/logs/%x_%j.err

set -euo pipefail

mkdir -p /homes/smdaniyal/stain_norm/logs
source /work/ai4bio2025/stain_norm/code/.venv/bin/activate

python /homes/smdaniyal/stain_norm/code/feature_statistics/tsne_by_tss.py