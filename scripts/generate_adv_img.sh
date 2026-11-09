#!/bin/bash
# Generate the fixed 1000 adversarial pairs for each benchmark.
export CUDA_VISIBLE_DEVICES=0

DATASET_BASE_PATH="${DATASET_BASE_PATH:-datasets}"
PROJECT_PATH="${PROJECT_PATH:-.}"
MANIFEST_DIR="${MANIFEST_DIR:-manifests}"
TAG="ht_coguide"

datasets=("coco_retrieval" "flickr30k" "snli_ve" "coco_caption")
decoders=("checkpoints/coco_retrieval_BiContrastiveLoss_priority_top35_triple_epoch08.pt"
          "checkpoints/flickr30k_BiContrastiveLoss_priority_top35_triple_epoch08.pt"
          "checkpoints/snli_ve_Cosine_priority_top35_triple_epoch08.pt"
          "checkpoints/coco_caption_Cosine_priority_top35_triple_epoch08.pt")
target_roots=("${DATASET_BASE_PATH}/mscoco"
              "${DATASET_BASE_PATH}/flickr30k/images/flickr30k-images"
              "${DATASET_BASE_PATH}/flickr30k/images/flickr30k-images"
              "${DATASET_BASE_PATH}/mscoco")

for i in 0 1 2 3; do
    python generate_adv_img.py \
        --model_name ViT-B/32 \
        --decoder_path "${PROJECT_PATH}/${decoders[$i]}" \
        --pairs "${MANIFEST_DIR}/${datasets[$i]}_pairs.json" \
        --host_root "${DATASET_BASE_PATH}/ILSVRC2012/val" \
        --target_root "${target_roots[$i]}" \
        --count 1000 \
        --batch_size 50 \
        --device cuda:0 \
        --output_path outputs \
        --dataset "${datasets[$i]}" \
        --adv_imgs "${TAG}" \
        --priority_mode triple \
        --topk_ratio 0.35 \
        --low_region_weight 0.385 \
        --w_noise 0.45 --w_host 0.30 --w_target 0.25
done
