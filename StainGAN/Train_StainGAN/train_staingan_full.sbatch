#!/bin/bash
#SBATCH --job-name=stainGAN_full
#SBATCH --partition=boost_usr_prod
#SBATCH --account=ai4bio2025
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=/work/h2020deciderficarra_shared/gcorso/wdkarim/stain_norm/StainGAN/logs/stainGAN_full_%j.log
#SBATCH --error=/work/h2020deciderficarra_shared/gcorso/wdkarim/stain_norm/StainGAN/logs/stainGAN_full_%j.err

set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source /work/ai4bio2025/stain_norm/code/.venv/bin/activate

DATAROOT=/homes/wdkarim/stain_norm/staingan_data/BH_vs_nonBH_256
CHECKPOINTS=/work/h2020deciderficarra_shared/gcorso/wdkarim/stain_norm/StainGAN/checkpoints


# --- Run ---
cd /homes/wdkarim/stain_norm/StainGAN

python train.py \
  --dataroot "$DATAROOT" \
  --name stainGAN_full \
  --model cycle_gan \
  --dataset_mode unaligned \
  --which_direction AtoB \
  --loadSize 286 \
  --fineSize 256 \
  --input_nc 3 \
  --output_nc 3 \
  --ngf 64 \
  --ndf 64 \
  --lambda_A 10 \
  --lambda_B 10 \
  --pool_size 50 \
  --batchSize 4 \
  --niter 51 \
  --niter_decay 51 \
  --display_id -1 \
  --print_freq 200 \
  --save_epoch_freq 1 \
  --save_latest_freq 5000 \
  --checkpoints_dir "$CHECKPOINTS" \
  --no_dropout \
  --nThreads 8 \
  --continue_train \
  --epoch_count 68