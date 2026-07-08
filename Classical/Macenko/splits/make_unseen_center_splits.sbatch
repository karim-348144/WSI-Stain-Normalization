#!/bin/bash
#SBATCH --job-name=make_unseen_center_splits
#SBATCH --partition=all_usr_prod
#SBATCH --account=ai4bio2025
#SBATCH --time=24:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=8
#SBATCH --array=1-20%20
#SBATCH --output=/homes/smdaniyal/stain_norm/logs/make_unseen_center_splits_%A_%a.out
#SBATCH --error=/homes/smdaniyal/stain_norm/logs/make_unseen_center_splits_%A_%a.err

set -euo pipefail
VENV_ACT="/work/ai4bio2025/stain_norm/code/.venv/bin/activate"

source "${VENV_ACT}"

echo "===== A2_E2 ====="
python "/homes/smdaniyal/stain_norm/code/splits/brca_tss_holdout_A2_E2.py" \

echo "===== A2_BH ====="
python "/homes/smdaniyal/stain_norm/code/splits/brca_tss_holdout_A2_BH.py" \

echo "===== BH_A2 ====="
python "/homes/smdaniyal/stain_norm/code/splits/brca_tss_holdout_BH_A2.py" \

echo "===== BH_E2 ====="
python "/homes/smdaniyal/stain_norm/code/splits/brca_tss_holdout_BH_E2.py" \

echo "===== E2_A2 ====="
python "/homes/smdaniyal/stain_norm/code/splits/brca_tss_holdout_E2_A2.py" \

echo "===== E2_BH ====="
python "/homes/smdaniyal/stain_norm/code/splits/brca_tss_holdout_E2_BH.py" \

date
echo "DONE"