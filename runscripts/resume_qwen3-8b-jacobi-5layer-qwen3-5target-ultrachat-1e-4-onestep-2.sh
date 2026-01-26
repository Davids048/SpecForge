#!/bin/bash

# Resume training from checkpoint: epoch_1_step_70000

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)
export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels

# support tp8 train eagle3 for Qwen3-4B/8B/32B up to tp_size = 8
NUM_GPUS=${1:-4}
TP_SIZE=${2:-1}
BUILD_DATASET_NUM_PROC=${BUILD_DATASET_NUM_PROC:-64}

mkdir -p $ROOT_DIR/logs

CONFIG_NAME=qwen3-8b-jacobi-5layer-qwen3-5target
LR=1e-4
DATA=ultrachat
DP=$(( NUM_GPUS / TP_SIZE ))
RUNNAME=$CONFIG_NAME-${DATA}-$LR-onestep-$DP

echo "====================RUN====================="
echo "$RUNNAME (RESUMED)"
echo "============================================"

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_eagle3.py \
    --target-model-path Qwen/Qwen3-8B \
    --draft-model-config $ROOT_DIR/outputs/qwen3-8b-jacobi-5layer-qwen3-5target-ultrachat-1e-4-onestep-2/epoch_1_step_70000/config.json \
    --attention-backend one_step_flex_attention \
    --is-jacobi \
    --train-data-path $ROOT_DIR/cache/dataset/${DATA}_train.jsonl \
    --eval-data-path $ROOT_DIR/cache/dataset/${DATA}_test.jsonl \
    --build-dataset-num-proc $BUILD_DATASET_NUM_PROC \
    --ckpt-dir $ROOT_DIR/outputs/qwen3-8b-jacobi-5layer-qwen3-5target-ultrachat-1e-4-onestep-2/epoch_1_step_70000 \
    --output-dir $ROOT_DIR/outputs/${RUNNAME}-resumed \
    --num-epochs 10 \
    --batch-size 1 \
    --learning-rate $LR \
    --max-length 4096 \
    --chat-template qwen \
    --cache-dir $ROOT_DIR/cache \
    --embedding-key model.embed_tokens.weight \
    --tp-size $TP_SIZE \
    --target-model-backend sglang \
    --save-interval 9999999 \
    --eval-interval 500 \
    --report-to wandb \
    --wandb-project specforge \
    --wandb-name ${RUNNAME}-resumed \
    2>&1 | tee $ROOT_DIR/logs/${RUNNAME}-resumed.log
