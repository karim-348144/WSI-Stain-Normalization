#!/bin/bash
#SBATCH --job-name=normalize_macenko_256
#SBATCH --partition=all_usr_prod
#SBATCH --account=ai4bio2025
#SBATCH --time=24:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=8
#SBATCH --array=1-1121%20
#SBATCH --output=/homes/smdaniyal/stain_norm/logs/normalize_macenko_all_%A_%a.out
#SBATCH --error=/homes/smdaniyal/stain_norm/logs/normalize_macenko_all_%A_%a.err

set -euo pipefail

PROJ="/homes/smdaniyal/stain_norm"
CODE_DIR="${PROJ}/code"
VENV_ACT="/work/ai4bio2025/stain_norm/code/.venv/bin/activate"
SCRIPT="/homes/smdaniyal/stain_norm/code/classic_stain_norm.py"

SRC_ROOT="/work/h2020deciderficarra_shared/TCGA/BRCA/images"
OUT_ROOT="/homes/smdaniyal/stain_norm/norm_patches/macenko"
TARGET_IMG="/work/ai4bio2025/stain_norm/targets/target_macenko.jpg"

mkdir -p "${PROJ}/logs" "${OUT_ROOT}"
source "${VENV_ACT}"


if [[ ! -d "${SRC_ROOT}" ]]; then
  echo "ERROR: SRC_ROOT not found: ${SRC_ROOT}"
  exit 2
fi
if [[ ! -f "${TARGET_IMG}" ]]; then
  echo "ERROR: TARGET_IMG not found: ${TARGET_IMG}"
  exit 3
fi

mapfile -t SLIDES < <(find "${SRC_ROOT}" -mindepth 1 -maxdepth 1 -type d -printf "%f\n" | sort)
N_SLIDES="${#SLIDES[@]}"

idx=$((SLURM_ARRAY_TASK_ID - 1))
if (( idx < 0 || idx >= N_SLIDES )); then
  echo "ERROR: SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} out of range (1..${N_SLIDES})"
  exit 4
fi

slide="${SLIDES[$idx]}"

echo "TASK=${SLURM_ARRAY_TASK_ID}/${N_SLIDES} SLIDE=${slide}"
echo "SRC_SLIDE_DIR=${SRC_ROOT}/${slide}"
echo "OUT_SLIDE_DIR=${OUT_ROOT}/${slide}"
date

mkdir -p "${OUT_ROOT}/${slide}"

in_total="$(find "${SRC_ROOT}/${slide}" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.tif' -o -iname '*.tiff' \) | wc -l)"
echo "IN_PATCHES_TOTAL=${in_total} (will process first 256 sorted)"

out_before="$(find "${OUT_ROOT}/${slide}" -type f -iname '*.jpg' 2>/dev/null | wc -l || true)"
echo "OUT_PATCHES_BEFORE=${out_before}"

if [[ "${out_before}" -ge 256 ]]; then
  echo "SKIP (already has >=256 outputs): ${slide}"
  exit 0
fi

TMPROOT="${SLURM_TMPDIR:-/tmp}/inroot_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${TMPROOT}"
ln -sfn "${SRC_ROOT}/${slide}" "${TMPROOT}/${slide}"

python "${SCRIPT}" \
  --method macenko \
  --in_root "${TMPROOT}" \
  --out_root "${OUT_ROOT}" \
  --target_img "${TARGET_IMG}" \
  --jpg_quality 95 \
  --recursive \
  --max_slides 1 \
  --max_patches 256 \
  --quiet

out_after="$(find "${OUT_ROOT}/${slide}" -type f -iname '*.jpg' 2>/dev/null | wc -l || true)"
echo "OUT_PATCHES_AFTER=${out_after}"
date
echo "DONE ${slide}"

