#!/bin/bash
#SBATCH --job-name=compute_feature_batch_effects
#SBATCH --partition=all_usr_prod
#SBATCH --account=ai4bio2025
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=20
#SBATCH --output=/homes/wdkarim/logs/%x_%j.out
#SBATCH --error=/homes/wdkarim/logs/%x_%j.err

set -euo pipefail

source /work/ai4bio2025/stain_norm/code/.venv/bin/activate

python /homes/wdkarim/code/feature_statistics/compute_feature_batch_effects.py

