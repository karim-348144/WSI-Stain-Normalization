#!/bin/bash
#SBATCH --job-name=transmil_raw_tss_all
#SBATCH --partition=boost_usr_prod
#SBATCH --account=ai4bio2025
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=/homes/smdaniyal/stain_norm/logs/%x_%j.out
#SBATCH --error=/homes/smdaniyal/stain_norm/logs/%x_%j.err

set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CODE_DIR="/homes/smdaniyal/stain_norm/code/"
SCRIPT="${CODE_DIR}/train_survival.py"
VENV_ACT="/work/ai4bio2025/stain_norm/code/.venv/bin/activate"

export PYTHONPATH="${CODE_DIR}:${PYTHONPATH:-}"

FEAT_ROOT="/work/ai4bio2025/stain_norm/features/raw_uni"
RUNS_ROOT="/homes/smdaniyal/stain_norm/runs"

SPLIT_DIRS=(
  "/homes/smdaniyal/stain_norm/splits/brca_tss_holdout_A2_BH"
  "/homes/smdaniyal/stain_norm/splits/brca_tss_holdout_A2_E2"
  "/homes/smdaniyal/stain_norm/splits/brca_tss_holdout_BH_A2"
  "/homes/smdaniyal/stain_norm/splits/brca_tss_holdout_BH_E2"
  "/homes/smdaniyal/stain_norm/splits/brca_tss_holdout_E2_A2"
  "/homes/smdaniyal/stain_norm/splits/brca_tss_holdout_E2_BH"
)

mkdir -p /homes/smdaniyal/stain_norm/logs
mkdir -p "${RUNS_ROOT}"

source "${VENV_ACT}"

echo "HOST=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
date
nvidia-smi || true

for SPLIT_DIR in "${SPLIT_DIRS[@]}"; do
  SPLIT_NAME="$(basename "${SPLIT_DIR}")"
  HOLDOUT_TAG="${SPLIT_NAME#brca_}"
  OUT_DIR="${RUNS_ROOT}/${HOLDOUT_TAG}/transmil_raw"

  mkdir -p "${OUT_DIR}"

  echo "========================================"
  echo "MODEL=transmil"
  echo "SPLIT_DIR=${SPLIT_DIR}"
  echo "OUT_DIR=${OUT_DIR}"
  date

  python "${SCRIPT}" \
    --model_name transmil \
    --feat_root "${FEAT_ROOT}" \
    --split_dir "${SPLIT_DIR}" \
    --out_dir "${OUT_DIR}" \
    --lr 1e-5 \
    --epochs 100 \
    --patience 12 \
    --hidden_dim 256 \
    --dropout 0.10 \
    --max_patches 256 \
    --seed 42 \
    --device cuda \
    --grad_clip 1.0

  echo "FINISHED ${SPLIT_NAME}"
  date
done

echo "ALL TRANSMIL HOLDOUT RUNS DONE"
date
