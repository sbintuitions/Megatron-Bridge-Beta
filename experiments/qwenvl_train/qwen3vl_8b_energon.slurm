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
WORKSPACE="/home/user02037/work/working_branch/Megatron-Bridge-Beta"
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

PRETRAINED_CHECKPOINT=/lustre/users/spliew/megatron_bridge_vlm/models/qwen3vl8b
MODEL_NAME=qwen3_vl_8b
SEQ_LENGTH=4096
TRAIN_ITERS=50
GLOBAL_BATCH_SIZE=32
MICRO_BATCH_SIZE=2
EVAL_ITERS=10
LR=0.00005
MIN_LR=0.000005
LR_WARMUP_ITERS=10
LOG_INTERVAL=1

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
CMD="uv run torchrun --nproc_per_node=8 scripts/training/run_recipe.py \
            --recipe ${MODEL_NAME}_finetune_config \
            --step_func qwen3_vl_step \
            --dataset_type energon \
            checkpoint.pretrained_checkpoint=$PRETRAINED_CHECKPOINT \
            model.seq_length=$SEQ_LENGTH \
            train.train_iters=$TRAIN_ITERS \
            train.global_batch_size=$GLOBAL_BATCH_SIZE \
            train.micro_batch_size=$MICRO_BATCH_SIZE \
            train.eval_iters=$EVAL_ITERS \
            optimizer.lr=$LR \
            optimizer.min_lr=$MIN_LR \
            scheduler.lr_warmup_iters=$LR_WARMUP_ITERS \
            checkpoint.save=/lustre/users/spliew/megatron_bridge_vlm/checkpoints/qwen3vl8b \
            logger.log_interval=$LOG_INTERVAL \
            dataset.seq_length=$SEQ_LENGTH \
            dataset.path=/lustre/users/spliew/megatron_bridge_vlm/dataset/LLaVA-Pretrain \
            dataset.pack_sequences_in_batch=True \
            model.expert_model_parallel_size=1 \
            model.tensor_model_parallel_size=1 \
            model.pipeline_model_parallel_size=1 \
            model.context_parallel_size=1 \
            model.calculate_per_token_loss=True \
            ddp.average_in_collective=False \
            ddp.grad_reduce_in_fp32=True
"
    # --config-file experiments/train_qwenvl/conf/qwen3_vl_8b_pretrain_wsd.yaml"
    

# Execute with srun using the specified container
srun --container-image=$CONTAINER_IMAGE \
        --container-mounts /lustre:/lustre,/store:/store \
        --container-workdir=/home/user02037/work/working_branch/Megatron-Bridge-Beta \
        bash -c "$CMD"
             
