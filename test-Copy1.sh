#!/bin/bash

#SBATCH --job-name=freq_mask         # Nome do trabalho
#SBATCH --qos=high                     # Qualidade do serviço
#SBATCH --gres=gpu:2                   # Número de GPUs

#SBATCH --output=saida_%j.out

# Define the arguments for your test script
GPUs="$1"
NUM_GPU=$(echo $GPUs | awk -F, '{print NF}')
DATA_TYPE="ArtiFact"  # Wang_CVPR20 or Ojha_CVPR23
MODEL_NAME="RN50_mod" # # RN50_mod, RN50, clip_vitl14, clip_rn50
MASK_TYPE="spectral" # spectral, pixel, patch or nomask
BAND="all" # all, low, mid, high
RATIO=15
BATCH_SIZE=64

# Set the CUDA_VISIBLE_DEVICES environment variable to use GPUs
export CUDA_VISIBLE_DEVICES=$GPUs

echo "Using $NUM_GPU GPUs with IDs: $GPUs"

# Run the test command
srun hostname &
srun python -m torch.distributed.launch --nproc_per_node=$NUM_GPU test.py \
  -- \
  --data_type $DATA_TYPE \
  --pretrained \
  --model_name $MODEL_NAME \
  --mask_type $MASK_TYPE \
  --band $BAND \
  --ratio 15 \
  --batch_size $BATCH_SIZE \
  --local_rank 0\
  --results_path results/resultsRN50_mod.json \
  # --other_model

  
