#!/bin/bash
# Stage 2: spatially guided decoder fine-tuning (paper defaults).
export MASTER_PORT=23456
export OMP_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=0,1

DATASET_BASE_PATH="${DATASET_BASE_PATH:-datasets}"
NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l | tr -d ' ')

# dataset: coco_retrieval | flickr30k | coco_caption | snli_ve
# criterion: BiContrastiveLoss | Cosine
torchrun --nproc_per_node="${NUM_GPUS}" --master_port="${MASTER_PORT}" finetune_ddp.py \
    --lr 1e-4 \
    --epoch 20 \
    --batch_size 74 \
    --dataset coco_retrieval \
    --criterion BiContrastiveLoss \
    --checkpoint checkpoints/pre-trained.pt \
    --data_dir "${DATASET_BASE_PATH}/mscoco" \
    --imagenet "${DATASET_BASE_PATH}/ILSVRC2012/train" \
    --priority_mode triple \
    --topk_ratio 0.35 \
    --low_region_weight 0.385 \
    --lambda_sparse 0.20 \
    --w_noise 0.45 --w_host 0.30 --w_target 0.25
