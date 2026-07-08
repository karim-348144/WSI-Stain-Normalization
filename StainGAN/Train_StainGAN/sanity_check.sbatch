#!/bin/bash
#SBATCH --job-name=stainnorm_sanity
#SBATCH --partition=boost_usr_prod
#SBATCH --account=ai4bio2025
#SBATCH --time=00:30:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --output=/homes/wdkarim/stain_norm/logs/sanity_%j.log
#SBATCH --error=/homes/wdkarim/stain_norm/logs/sanity_%j.err

set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source /work/ai4bio2025/stain_norm/code/.venv/bin/activate

# --- Create tiny subsets (500 lines each) ---
DATAROOT=/work/ai4bio2025/stain_norm/staingan_data/BH_vs_nonBH
SANITY_DIR=/homes/wdkarim/stain_norm/sanity_data

mkdir -p $SANITY_DIR
head -n 500 $DATAROOT/trainA.txt > $SANITY_DIR/trainA.txt
head -n 500 $DATAROOT/trainB.txt > $SANITY_DIR/trainB.txt

# --- Run 2 epochs only ---
cd /homes/wdkarim/stain_norm/StainGAN

python train.py \
  --dataroot $SANITY_DIR \
  --name sanity_BH \
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
  --no_lsgan \
  --pool_size 50 \
  --batchSize 16 \
  --niter 2 \
  --niter_decay 2 \
  --display_id -1 \
  --print_freq 50 \
  --save_epoch_freq 1 \
  --save_latest_freq 9999999 \
  --checkpoints_dir /homes/wdkarim/stain_norm/sanity_checkpoints \
  --no_dropout \
  --nThreads 8