#!/bin/bash

#SBATCH --job-name=qwen3vl-8b-sft
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --partition=002-partition-debug
#SBATCH --output=logs/qwen3vl_8b_sft_%j.out
#SBATCH --error=logs/qwen3vl_8b_sft_%j.err
#SBATCH --exclusive

# Workspace directory for checkpoints and results
# WORKSPACE="/home/user02037/work/working_branch/Megatron-Bridge-Beta"
# WORKSPACE=${WORKSPACE:-/workspace}

# Container and Environment setup
CONTAINER_IMAGE="/lustre/container_cache/nvidia/nemo:26.02.00.sqsh"
export WANDB_MODE=disabled

# Local Model Path
# LOCAL_MODEL_PATH=${LOCAL_MODEL_PATH:-"${WORKSPACE}/models/Qwen3-VL-8B-Instruct"}
# LOCAL_MODEL_PATH="/lustre/share/downloaded/models/Qwen/Qwen3-VL-8B-Instruct"

# Training configurations
# MODEL_NAME=qwen3_vl_8b
# SEQ_LENGTH=4096
# TRAIN_ITERS=50
# GLOBAL_BATCH_SIZE=16
# MICRO_BATCH_SIZE=2
# EVAL_ITERS=10
# LR=0.00005
# MIN_LR=0.000005
# LR_WARMUP_ITERS=10
# LOG_INTERVAL=1

# For Megatron-LM
export CUDA_DEVICE_MAX_CONNECTIONS=1
# For NCCL
export NCCL_DEBUG=WARN
export NCCL_IB_TIMEOUT=22 # approx. 16s = 4us * 2^22
export NCCL_NVLS_ENABLE=0 # reduce memory usage but slows down intra-node communication
# For OpenMP
export OMP_NUM_THREADS=14 # 112 cores per node / 8 proceses
# Distributed initialization
export MASTER_ADDR=$(hostname)
export MASTER_PORT=25091
export TORCH_MASTER_ADDR=$MASTER_ADDR
export TORCH_MASTER_PORT=$MASTER_PORT


mkdir -p logs


# EP/TP/PP/CP combinations: "EP,TP,PP,CP" configurations

echo "Running 8B finetuning"

# Build the command
# dataset_type=mock: uses mock data
# hf_path=$LOCAL_MODEL_PATH: uses local model
# CMD="uv run python scripts/training/run_recipe.py \
    # --recipe ${MODEL_NAME}_finetune_config \
    #         CMD="torchrun --nproc_per_node=8 \
    # --master_addr=$TORCH_MASTER_ADDR \
    # --master_port=$TORCH_MASTER_PORT \
    # scripts/training/run_recipe.py \
    # hf_path=$LOCAL_MODEL_PATH \
CMD="uv run torchrun --nproc_per_node=8 experiments/train_qwenvl/finetune_qwen3_vl.py \
    --recipe qwen3_vl_8b_finetune_config \
    --debug"
    # --config-file experiments/train_qwenvl/conf/qwen3_vl_8b_pretrain_wsd.yaml"
    

# Execute with srun using the specified container
srun --container-image=$CONTAINER_IMAGE \
        --container-mounts /lustre:/lustre,/store:/store \
        --container-workdir=/home/user02037/work/working_branch/Megatron-Bridge-Beta \
        bash -c "$CMD"
             
