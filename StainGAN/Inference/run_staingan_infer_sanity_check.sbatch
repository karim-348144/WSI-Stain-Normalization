#!/bin/bash
#SBATCH --job-name=staingan_sanity
#SBATCH --partition=all_usr_prod
#SBATCH --account=ai4bio2025
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=/work/h2020deciderficarra_shared/gcorso/wdkarim/stain_norm/StainGAN/logs/%x_%j.out
#SBATCH --error=/work/h2020deciderficarra_shared/gcorso/wdkarim/stain_norm/StainGAN/logs/%x_%j.err

set -euo pipefail

echo "JOB_ID=${SLURM_JOB_ID}"
echo "HOSTNAME=$(hostname)"
echo "START_TIME=$(date)"

mkdir -p /work/h2020deciderficarra_shared/gcorso/wdkarim/stain_norm/StainGAN/logs
mkdir -p /work/h2020deciderficarra_shared/gcorso/wdkarim/stain_norm/StainGAN/norm_patches_sanity_check

cd /work/ai4bio2025/stain_norm/code
source .venv/bin/activate

export PYTHONPATH=/homes/wdkarim/stain_norm/StainGAN:${PYTHONPATH:-}

cd /homes/wdkarim/stain_norm/code/staingan

echo "Python: $(which python)"
echo "PYTHONPATH=${PYTHONPATH}"
echo "CUDA check:"
nvidia-smi || true

python run_staingan_infer.py \
  --in_root /work/h2020deciderficarra_shared/TCGA/BRCA/images \
  --out_root /work/h2020deciderficarra_shared/gcorso/wdkarim/stain_norm/StainGAN/norm_patches_sanity_check/ \
  --checkpoints_dir /work/h2020deciderficarra_shared/gcorso/wdkarim/stain_norm/StainGAN/checkpoints \
  --name stainGAN_full \
  --epoch 67 \
  --manifest_csv /homes/wdkarim/manifests/brca_matching_slide_tss_manifest.csv \
  --target_tss BH \
  --slide_id TCGA-BH-A0AU-01Z-00-DX1.E852F218-B15C-4CF8-8FD0-F284043834C4 \
  --max_patches 25

echo "END_TIME=$(date)"
echo "DONE"