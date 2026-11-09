"""Run captioning evaluation on the generated adversarial images."""
import os
import subprocess
import sys

DATASET = os.environ.get("DATASET_BASE_PATH", "datasets")
PROJECT_PATH = os.environ.get("PROJECT_PATH", ".")
TAG = "ht_coguide"
datasets = ["coco_caption"]

# Alternatives:
# lavis_tool/blip2/caption_coco_opt2.7b_eval.yaml
# lavis_tool/blip/caption_coco_eval.yaml
# lavis_tool/blip2/caption_coco_vicuna7b_instruct_eval.yaml
config = "lavis_tool/blip/caption_coco_eval.yaml"

for dataset in datasets:
    image_path = f"{PROJECT_PATH}/outputs/{dataset}/{TAG}"
    command = [
        sys.executable, "caption.py",
        "--cache_path", DATASET,
        "--image_path", image_path,
        "--json_path", f"{image_path}/annotations.json",
        "--gt_path", f"{PROJECT_PATH}/json/coco_caption_test_gt_adv.json",
        "--cfg_path", config,
        "--output_dir", f"output/caption/{dataset}",
    ]
    print(" ".join(command))
    subprocess.run(command, check=True)
