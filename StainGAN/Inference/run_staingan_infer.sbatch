#!/bin/bash
#SBATCH --job-name=staingan_full_infer
#SBATCH --partition=boost_usr_prod
#SBATCH --account=ai4bio2025
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=/work/h2020deciderficarra_shared/gcorso/wdkarim/stain_norm/StainGAN/logs/%x_%j.log
#SBATCH --error=/work/h2020deciderficarra_shared/gcorso/wdkarim/stain_norm/StainGAN/logs/%x_%j.err

set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "JOB_ID=${SLURM_JOB_ID}"
echo "HOSTNAME=$(hostname)"
echo "START_TIME=$(date)"

mkdir -p /work/h2020deciderficarra_shared/gcorso/wdkarim/stain_norm/StainGAN/logs
mkdir -p /work/h2020deciderficarra_shared/gcorso/wdkarim/stain_norm/StainGAN/norm_patches

source /work/ai4bio2025/stain_norm/code/.venv/bin/activate

export PYTHONPATH=/homes/wdkarim/stain_norm/StainGAN:${PYTHONPATH:-}

cd /homes/wdkarim/stain_norm/code/staingan

echo "Python: $(which python)"
echo "PYTHONPATH=${PYTHONPATH}"
echo "CUDA check:"
nvidia-smi || true

python run_staingan_infer.py \
  --in_root /work/h2020deciderficarra_shared/TCGA/BRCA/images \
  --out_root /work/h2020deciderficarra_shared/gcorso/wdkarim/stain_norm/StainGAN/norm_patches/ \
  --checkpoints_dir /work/h2020deciderficarra_shared/gcorso/wdkarim/stain_norm/StainGAN/checkpoints \
  --name stainGAN_full \
  --epoch 67 \
  --manifest_csv /homes/wdkarim/manifests/brca_matching_slide_tss_manifest.csv \
  --target_tss BH \
  --max_patches 256

echo "END_TIME=$(date)"
echo "DONE"