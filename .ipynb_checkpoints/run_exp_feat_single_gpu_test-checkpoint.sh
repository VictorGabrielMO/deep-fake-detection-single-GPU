#!/bin/bash

# Define the arguments for your test script
GPUs="$1"
DATA_TYPE="ArtiFact"  # Wang_CVPR20 or Ojha_CVPR23
MODEL_NAME="RN50_mod" # # RN50_mod, RN50, clip_vitl14, clip_rn50
MASK_TYPE="spectral" # spectral, pixel, patch or nomask
BAND="all" # all, low, mid, high
RATIO=15
BATCH_SIZE=8

# Set the CUDA_VISIBLE_DEVICES environment variable to use GPUs
export CUDA_VISIBLE_DEVICES=$GPUs

echo "Using GPUs with IDs: $GPUs"

# Run the test command
python exp_feat_single_gpu.py \
  --data_type $DATA_TYPE \
  --pretrained \
  --model_name $MODEL_NAME \
  --mask_type $MASK_TYPE \
  --band $BAND \
  --ratio 15 \
  --batch_size $BATCH_SIZE \
  --save_dir ./features/ \
  --checkpoint_path /storage/datasets/gabriela.barreto/rn50_modft_spectralmask.pth

  
