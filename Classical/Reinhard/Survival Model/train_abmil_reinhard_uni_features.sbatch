#!/bin/bash
#SBATCH --job-name=train_abmil_reinhard
#SBATCH --partition=boost_usr_prod
#SBATCH --account=ai4bio2025
#SBATCH --time=10:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=/homes/wdkarim/logs/%x_%j.log
#SBATCH --error=/homes/wdkarim/logs/%x_%j.err

set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /work/ai4bio2025/stain_norm/code
source .venv/bin/activate

echo "JOB_ID=${SLURM_JOB_ID}"
echo "HOSTNAME=$(hostname)"
echo "START_TIME=$(date)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "CUDA check:"


CODE_DIR="/homes/wdkarim/code/ABMIL_TransMIL"
export PYTHONPATH="${CODE_DIR}:${PYTHONPATH:-}"
RUNS_ROOT="/homes/wdkarim/runs/"

SPLIT_DIRS=(
  "/homes/wdkarim/splits/brca_tss_holdout_A2_BH"
  "/homes/wdkarim/splits/brca_tss_holdout_A2_E2"
  "/homes/wdkarim/splits/brca_tss_holdout_BH_E2"
  "/homes/wdkarim/splits/brca_tss_holdout_E2_A2"
  "/homes/wdkarim/splits/brca_tss_holdout_E2_BH"
  "/homes/wdkarim/splits/brca_tss_holdout_BH_A2"
)

mkdir -p "${RUNS_ROOT}"


date
nvidia-smi || true

for SPLIT_DIR in "${SPLIT_DIRS[@]}"; do
  SPLIT_NAME="$(basename "${SPLIT_DIR}")"
  HOLDOUT_TAG="${SPLIT_NAME#brca_}"
  OUT_DIR="${RUNS_ROOT}/${HOLDOUT_TAG}/abmil_reinhard"

  mkdir -p "${OUT_DIR}"

  echo "========================================"
  echo "MODEL=abmil"
  echo "SPLIT_DIR=${SPLIT_DIR}"
  echo "OUT_DIR=${OUT_DIR}"
  date

  python /homes/wdkarim/code/ABMIL_TransMIL/train_survival.py \
    --model_name abmil \
    --feat_root /homes/wdkarim/features/reinhard_uni_all \
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

echo "ALL ABMIL HOLDOUT RUNS DONE"
date

