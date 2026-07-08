#!/bin/bash
#SBATCH --job-name=reinhard_250
#SBATCH --partition=all_usr_prod
#SBATCH --account=ai4bio2025
#SBATCH --time=12:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --array=1-5%5
#SBATCH --output=/work/ai4bio2025/stain_norm/logs/reinhard_%A_%a.out
#SBATCH --error=/work/ai4bio2025/stain_norm/logs/reinhard_%A_%a.err

set -euo pipefail

PROJ="/work/ai4bio2025/stain_norm"
CODE_DIR="${PROJ}/code"
VENV_ACT="${CODE_DIR}/.venv/bin/activate"
SCRIPT="${CODE_DIR}/classic_stain_norm.py"

SRC_ROOT="${PROJ}/raw_patches"
OUT_ROOT="${PROJ}/norm_patches/reinhard"
TARGET_IMG="${PROJ}/targets/target_reinhard.jpg"

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

# Safer slide selection: array index -> sorted list of directories
mapfile -t SLIDES < <(find "${SRC_ROOT}" -mindepth 1 -maxdepth 1 -type d -printf "%f\n" | sort)

idx=$((SLURM_ARRAY_TASK_ID - 1))
if (( idx < 0 || idx >= ${#SLIDES[@]} )); then
  echo "ERROR: SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} out of range (1..${#SLIDES[@]})"
  exit 4
fi
slide="${SLIDES[$idx]}"

echo "TASK=${SLURM_ARRAY_TASK_ID} SLIDE=${slide}"
echo "SRC_SLIDE_DIR=${SRC_ROOT}/${slide}"
echo "OUT_SLIDE_DIR=${OUT_ROOT}/${slide}"
date

mkdir -p "${OUT_ROOT}/${slide}"

in_n="$(find "${SRC_ROOT}/${slide}" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l)"
out_n="$(find "${OUT_ROOT}/${slide}" -type f -iname '*.jpg' 2>/dev/null | wc -l || true)"
echo "IN_PATCHES=${in_n} OUT_PATCHES(before)=${out_n}"

TMPROOT="${SLURM_TMPDIR:-/tmp}/inroot_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${TMPROOT}"

# Re-create link safely if it exists (common when re-queuing)
ln -sfn "${SRC_ROOT}/${slide}" "${TMPROOT}/${slide}"

python "${SCRIPT}" \
  --method reinhard \
  --in_root "${TMPROOT}" \
  --out_root "${OUT_ROOT}" \
  --target_img "${TARGET_IMG}" \
  --jpg_quality 95 \
  --recursive \
  --quiet

out_n2="$(find "${OUT_ROOT}/${slide}" -type f -iname '*.jpg' 2>/dev/null | wc -l || true)"
echo "OUT_PATCHES(after)=${out_n2}"
date
echo "DONE ${slide}"
