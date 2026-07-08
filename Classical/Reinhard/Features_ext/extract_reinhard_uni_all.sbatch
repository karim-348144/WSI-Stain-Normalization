#!/bin/bash
#SBATCH --job-name=extract_reinhard_uni_all
#SBATCH --partition=all_usr_prod
#SBATCH --account=ai4bio2025
#SBATCH --time=24:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=/work/ai4bio2025/stain_norm/logs/%x_%j.out
#SBATCH --error=/work/ai4bio2025/stain_norm/logs/%x_%j.err

set -euo pipefail

CODE_DIR="/work/ai4bio2025/stain_norm/code"
SCRIPT="${CODE_DIR}/feature_ext/extract_uni_raw.py"
VENV_ACT="${CODE_DIR}/.venv/bin/activate"

IN_ROOT="/work/ai4bio2025/wdkarim/stain_norm/norm_patches/reinhard"
OUT_ROOT="/homes/wdkarim/features/reinhard_uni_all"
MANIFEST="/work/ai4bio2025/stain_norm/manifests/brca_matching_slide_manifest.csv"

mkdir -p /work/ai4bio2025/stain_norm/logs
source "${VENV_ACT}"

echo "HOST=$(hostname)"
date
nvidia-smi || true

python "${SCRIPT}" \
  --in_root "${IN_ROOT}" \
  --out_root "${OUT_ROOT}" \
  --manifest_csv "${MANIFEST}" \
  --recursive \
  --batch_size 128 \
  --num_workers 4 \
  --device cuda

date
echo "DONE"
