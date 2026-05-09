#!/usr/bin/env bash
set -e

python3 training.py \
  --pixel_step 8 \
  --epochs 20 \
  --num_cams 4 \
  --feature_dim 32 \
  --batch_size 4 \
  --num_workers 0 \
  --base_ch_feat 16 \
  --base_ch_map 16 \
  --feat_lr 1e-4 \
  --projection_lr 1e-4 \
  --mapping_lr 1e-4 \
  --min_lr 1e-5 \
  --weight_decay 1e-4 \
  --ignore_index 255 \
  --log_dir runs/train \
  --data_dir /Users/meshaza/Desktop/projects/static_obstacle_map \
  --mixed_precision
