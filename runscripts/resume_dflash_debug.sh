#!/bin/bash

# Resume training from checkpoint: epoch_4_step_75000

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)
export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels

# support tp8 train eagle3 for Qwen3-4B/8B/32B up to tp_size = 8
NUM_GPUS=${1:-4}
TP_SIZE=${2:-1}
BUILD_DATASET_NUM_PROC=${BUILD_DATASET_NUM_PROC:-1}

mkdir -p $ROOT_DIR/logs

# CONFIG_NAME=qwen3-8b-jacobi-1layer
# CONFIG_NAME=qwen3-8b-jacobi-3layer-qwen3-5target
CONFIG_NAME=dflash
LR=1e-4
DATA=ultrachat_debug
DP=$(( NUM_GPUS / TP_SIZE ))
RUNNAME=$CONFIG_NAME-${DATA}-$LR-onestep-full-BS$DP
echo "====================RUN====================="
echo "$RUNNAME (RESUMED)"
echo "============================================"

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_eagle3.py \
    --target-model-path Qwen/Qwen3-8B \
    --draft-model-config $ROOT_DIR/dflash/z-lab/Qwen3-8B-DFlash-b16/config.json \
    --attention-backend one_step_flex_attention \
    --is-jacobi \
    --use-causal-attention false \
    --train-data-path $ROOT_DIR/cache/dataset/${DATA}_train.jsonl \
    --build-dataset-num-proc $BUILD_DATASET_NUM_PROC \
    --ckpt-dir $ROOT_DIR/dflash/z-lab/Qwen3-8B-DFlash-b16 \
    --output-dir $ROOT_DIR/dflash/z-lab/output \
    --num-epochs 1 \
    --batch-size 1 \
    --learning-rate $LR \
    --max-length 4096 \
    --chat-template qwen \
    --cache-dir $ROOT_DIR/cache \
    --embedding-key model.embed_tokens.weight \
    --tp-size $TP_SIZE \
    --target-model-backend sglang \
    --save-interval 99999999 \
    --eval-interval 500 \
    --report-to none \
    --wandb-project specforge \
    --wandb-name ${RUNNAME}-resumed \
    2>&1 | tee $ROOT_DIR/logs/${RUNNAME}-resumed.log
