#!/usr/bin/env bash
set -e

python3 training.py \
  --pixel_step 1 \
  --epochs 6 \
  --num_cams 4 \
  --feature_dim 16 \
  --batch_size 4 \
  --num_workers 0 \
  --base_ch_feat 16 \
  --base_ch_map 16 \
  --depth_scale 0.5 \
  --ground_z 0.0 \
  --default_depth 30.0 \
  --feat_lr 1e-4 \
  --projection_lr 1e-4 \
  --mapping_lr 1e-4 \
  --min_lr 1e-5 \
  --weight_decay 1e-4 \
  --ignore_index 255 \
  --log_dir runs/train \
  --data_dir /Users/meshaza/Desktop/projects/static_obstacle_map \
  --test_data_dir /Users/meshaza/Desktop/projects/static_obstacle_map/autonomy_yandex_dataset_test \
  --submission_dir submission \
  --best_checkpoint runs/train/checkpoints/best.pt \
  --pred_threshold 0.5 \
  --no_feature_model \
  --no_depth \
  --mixed_precision
