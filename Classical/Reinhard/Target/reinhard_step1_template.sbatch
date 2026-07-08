#!/bin/bash
#SBATCH --job-name=reinhard_tpl
#SBATCH --partition=all_usr_prod
#SBATCH --account=ai4bio2025
#SBATCH --time=02:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --output=/work/ai4bio2025/stain_norm/logs/reinhard_tpl_%j.out
#SBATCH --error=/work/ai4bio2025/stain_norm/logs/reinhard_tpl_%j.err

set -euo pipefail

PROJ="/work/ai4bio2025/stain_norm"
CODE_DIR="${PROJ}/code"
VENV_ACT="${CODE_DIR}/.venv/bin/activate"
SCRIPT="${CODE_DIR}/choose_reinhard_refs.py"

IN_ROOT="/work/h2020deciderficarra_shared/TCGA/BRCA/images"
OUT_JSON="${PROJ}/targets/reinhard_template_agg.json"

mkdir -p "${PROJ}/logs" "${PROJ}/targets"
source "${VENV_ACT}"

python "${SCRIPT}" \
  --in_root "${IN_ROOT}" \
  --n_slides 100 \
  --n_patches 1024 \
  --k_refs 20 \
  --out_json "${OUT_JSON}"

echo "DONE. Template written to: ${OUT_JSON}"

