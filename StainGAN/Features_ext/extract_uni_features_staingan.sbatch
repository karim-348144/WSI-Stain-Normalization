#!/bin/bash
#SBATCH --job-name=features_staingan
#SBATCH --partition=boost_usr_prod
#SBATCH --account=ai4bio2025
#SBATCH --time=24:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=/work/h2020deciderficarra_shared/gcorso/wdkarim/stain_norm/StainGAN/logs/%x_%j.log
#SBATCH --error=/work/h2020deciderficarra_shared/gcorso/wdkarim/stain_norm/StainGAN/logs/%x_%j.err

set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source /work/ai4bio2025/stain_norm/code/.venv/bin/activate

echo "JOB_ID=${SLURM_JOB_ID}"
echo "HOSTNAME=$(hostname)"
echo "START_TIME=$(date)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "CUDA check:"

date
nvidia-smi || true

python /homes/wdkarim/stain_norm/code/staingan/extract_uni_features_staingan.py \
  --in_root /work/h2020deciderficarra_shared/gcorso/wdkarim/stain_norm/StainGAN/norm_patches \
  --out_root /work/h2020deciderficarra_shared/gcorso/wdkarim/stain_norm/StainGAN/features \
  --manifest_csv /work/ai4bio2025/stain_norm/manifests/brca_matching_slide_manifest.csv \
  --recursive \
  --batch_size 128 \
  --num_workers 8 \
  --device cuda

date
echo "DONE"