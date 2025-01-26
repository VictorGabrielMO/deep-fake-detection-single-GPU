#!/bin/bash

#SBATCH --job-name=feat_extr        # Nome do trabalho
#SBATCH --qos=high                     # Qualidade do serviço
#SBATCH --gres=gpu:1                   # Número de GPUs

#SBATCH --output=saida_%j.out
#SBATCH --error=erro_%j.err         # Error file

# Load Conda
export PATH="/opt/anaconda3/bin:$PATH"
source /opt/anaconda3/etc/profile.d/conda.sh

# Path to your environment.yml file
ENV_YML_PATH="/home/victor.oliveira/deep-fake-detection-single-GPU/environment.yml"

# Create the conda environment using the environment.yml file
echo "Creating conda environment from $ENV_YML_PATH..."
conda env create -f $ENV_YML_PATH

# Activate the new environment
# Note: Conda might not activate environments immediately after creation in batch jobs
conda activate $(basename $ENV_YML_PATH .yml)

# Check if the environment is activated
echo "Activated environments:"
conda info --envs

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
srun python exp_feat_single_gpu.py \
  --data_type $DATA_TYPE \
  --pretrained \
  --model_name $MODEL_NAME \
  --mask_type $MASK_TYPE \
  --band $BAND \
  --ratio 15 \
  --batch_size $BATCH_SIZE \
  --save_dir ./features/ \
  --checkpoint_path /storage/datasets/gabriela.barreto/rn50_modft_spectralmask.pth

  
