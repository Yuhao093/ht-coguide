#!/bin/bash
# Image-text retrieval evaluation across CLIP backbones.
export CUDA_VISIBLE_DEVICES=0

DATA="${DATA:-datasets}"
PROJECT_PATH="${PROJECT_PATH:-.}"
TAG="ht_coguide"
dataset="coco_retrieval"
backbones=("vitb16" "vitl14" "vitl14x336")

for b in "${backbones[@]}"; do
    python retrieval.py \
        --cache_path "${DATA}" \
        --cfg_path "lavis_tool/clip/ret_coco_retrieval_eval_${b}.yaml" \
        --image_path "${PROJECT_PATH}/outputs/${dataset}/${TAG}" \
        --json_path "${PROJECT_PATH}/outputs/${dataset}/${TAG}/annotations.json" \
        --output_dir "output/retrieval/${dataset}/${b}"
done
