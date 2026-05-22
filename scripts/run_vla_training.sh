#!/bin/bash
# VLA Training Pipeline Script
set -e

CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate isaaclab

DATA_DIR=${1:-"data/cosmos_sdg"}
MODEL_DIR=${2:-"models/vla"}
CONFIG=${3:-"config/vla_training_config.yaml"}

mkdir -p $MODEL_DIR/bc $MODEL_DIR/ppo $MODEL_DIR/export

# Phase 1: Behavior Cloning
echo "=== Phase 1: Behavior Cloning ==="
python src/isaac_lab/train_vla.py \
    --mode bc \
    --data_dir $DATA_DIR \
    --output_dir results/vla_bc \
    --batch_size 256 \
    --epochs 100 \
    --lr 3e-4 \
    --model_dir $MODEL_DIR/bc

# Phase 2: PPO Fine-tuning
echo "=== Phase 2: PPO Fine-tuning ==="
python src/isaac_lab/train_vla.py \
    --mode ppo \
    --checkpoint $MODEL_DIR/bc/best_model.pt \
    --output_dir results/vla_ppo \
    --num_envs 64 \
    --ppo_epochs 10 \
    --model_dir $MODEL_DIR/ppo

# Phase 3: ONNX Export
echo "=== Phase 3: ONNX Export ==="
python src/vla_policy/export_onnx.py \
    --checkpoint $MODEL_DIR/ppo/best_model.pt \
    --output $MODEL_DIR/export/vla_model.onnx

# Phase 4: TensorRT Build
echo "=== Phase 4: TensorRT Engine ==="
/usr/src/tensorrt/bin/trtexec \
    --onnx=$MODEL_DIR/export/vla_model.onnx \
    --saveEngine=$MODEL_DIR/export/vla_model_fp16.plan \
    --fp16 --workspace=8192

echo "=== Training Complete ==="
ls -lh $MODEL_DIR/export/
