#!/bin/bash
# Stage 1: decoder pre-training on image-text shards.
export MASTER_PORT=23456
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Directory containing the WebDataset .tar shards (image-text pairs).
LAION_DIR="${LAION_DIR:-datasets/laion400m-shards}"
NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l | tr -d ' ')

torchrun --nproc_per_node="${NUM_GPUS}" --master_port="${MASTER_PORT}" main_ddp.py \
    --tar_dir "${LAION_DIR}" \
    --lr 1e-4 \
    --epoch 10 \
    --batch_size 600 \
    --dist_url "tcp://127.0.0.1:${MASTER_PORT}"
