"""Run visual entailment evaluation on the generated adversarial images."""
import os
import subprocess
import sys

DATASET = os.environ.get("DATASET_BASE_PATH", "datasets")
PROJECT_PATH = os.environ.get("PROJECT_PATH", ".")
TAG = "ht_coguide"
dataset = "snli_ve"

image_path = f"{PROJECT_PATH}/outputs/{dataset}/{TAG}"
command = [
    sys.executable, "classification.py",
    "--cache_path", DATASET,
    "--cfg_path", "lavis_tool/blip2/ve_snlive_vicuna7b_instruct_eval.yaml",
    "--image_path", image_path,
    "--json_path", f"{image_path}/annotations.json",
    "--output_dir", f"output/classification/{dataset}",
]
print(" ".join(command))
subprocess.run(command, check=True)
