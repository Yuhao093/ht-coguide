"""Image captioning evaluation through the LAVIS RunnerBase."""
import argparse
import os
import random
import warnings

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from omegaconf import OmegaConf
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

import lavis.common.utils as utils
import lavis.tasks as tasks
from lavis.common.config import Config
from lavis.common.dist_utils import get_rank, init_distributed_mode
from lavis.common.logger import setup_logger
from lavis.common.registry import registry
from lavis.common.utils import now
from lavis.datasets.builders import *  # noqa: F401,F403
from lavis.datasets.datasets.caption_datasets import CaptionDataset
from lavis.datasets.datasets.coco_caption_datasets import COCOCapEvalDataset, NoCapsEvalDataset
from lavis.models import *  # noqa: F401,F403
from lavis.processors import *  # noqa: F401,F403
from lavis.processors.blip_processors import BlipImageBaseProcessor
from lavis.processors.clip_processors import _convert_to_rgb
from lavis.processors.randaugment import RandomAugment
from lavis.runners.runner_base import RunnerBase
from lavis.tasks import *  # noqa: F401,F403
from lavis_tool.caption import CaptionTask


def parse_args():
    parser = argparse.ArgumentParser(description="Image captioning evaluation")
    parser.add_argument("--cfg_path", required=True)
    parser.add_argument("--cache_path", default="datasets")
    parser.add_argument("--json_path", required=True)
    parser.add_argument("--gt_path", default="")
    parser.add_argument("--image_path", required=True)
    parser.add_argument("--output_dir", default="output/caption")
    parser.add_argument("--options", nargs="+")
    return parser.parse_args()


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


class BlipImageTrainProcessor(BlipImageBaseProcessor):
    def __init__(self, image_size=384, transform=None, mean=None, std=None,
                 min_scale=0.5, max_scale=1.0):
        super().__init__(mean=mean, std=std)
        if transform is None:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size, scale=(min_scale, max_scale),
                                             interpolation=InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(),
                RandomAugment(2, 5, isPIL=True, augs=[
                    "Identity", "AutoContrast", "Brightness", "Sharpness", "Equalize",
                    "ShearX", "ShearY", "TranslateX", "TranslateY", "Rotate"]),
                transforms.ToTensor(),
                self.normalize,
            ])
        else:
            self.transform = transform

    def __call__(self, item):
        return self.transform(item)

    @classmethod
    def from_config(cls, cfg=None):
        cfg = cfg or OmegaConf.create()
        return cls(image_size=cfg.get("image_size", 384), mean=cfg.get("mean", None),
                   std=cfg.get("std", None), min_scale=cfg.get("min_scale", 0.5),
                   max_scale=cfg.get("max_scale", 1.0))


class BlipImageEvalProcessor(BlipImageBaseProcessor):
    def __init__(self, image_size=384, transform=None, mean=None, std=None):
        super().__init__(mean=mean, std=std)
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size),
                                  interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                self.normalize,
            ])
        else:
            self.transform = transform

    def __call__(self, item):
        return self.transform(item)

    @classmethod
    def from_config(cls, cfg=None):
        cfg = cfg or OmegaConf.create()
        return cls(image_size=cfg.get("image_size", 384),
                   mean=cfg.get("mean", None), std=cfg.get("std", None))


def build(cfg, transform=None):
    image_size = cfg.config.get("preprocess", {}).get("vis_processor", {}) \
        .get("eval", {}).get("image_size", 384)
    config = cfg.config["datasets"]
    key = list(config.keys())[0]
    vis_processors = {
        "train": BlipImageTrainProcessor(image_size=image_size, transform=transform),
        "eval": BlipImageEvalProcessor(image_size=image_size, transform=transform)}
    text_processors = {
        split: registry.get_processor_class("blip_caption").from_config({"name": "blip_caption"})
        for split in ("train", "eval")}

    build_info = config[key]["build_info"]
    ann_info, data_type = build_info["annotations"], config[key]["data_type"]
    vis_info = build_info[data_type]
    datasets = {}
    for split in ann_info.keys():
        if split not in ("train", "val", "test"):
            continue
        is_train = split == "train"
        ann_paths = ann_info.get(split).storage
        if isinstance(ann_paths, str):
            ann_paths = [ann_paths]
        ann_paths = [path if os.path.isabs(path) else utils.get_cache_path(path)
                     for path in ann_paths]
        vis_path = vis_info.storage
        if not os.path.isabs(vis_path):
            vis_path = utils.get_cache_path(vis_path)
        if not os.path.exists(vis_path):
            warnings.warn(f"storage path {vis_path} does not exist.")
        if key == "nocaps":
            dataset_cls = NoCapsEvalDataset
        else:
            dataset_cls = CaptionDataset if is_train else COCOCapEvalDataset
        datasets[split] = dataset_cls(
            vis_processor=vis_processors["train" if is_train else "eval"],
            text_processor=text_processors["train" if is_train else "eval"],
            ann_paths=ann_paths, vis_root=vis_path)
    return {key: datasets}


def main():
    args = parse_args()
    registry.mapping["paths"]["cache_root"] = args.cache_path
    job_id = now()
    cfg = Config(args)
    registry.mapping["paths"]["coco_gt"] = args.gt_path

    dataset_name = list(cfg.config["datasets"].keys())[0]
    if args.image_path:
        cfg.config["datasets"][dataset_name]["build_info"]["images"]["storage"] = args.image_path
    if args.json_path:
        cfg.config["datasets"][dataset_name]["build_info"]["annotations"]["test"]["storage"] = args.json_path
    if args.output_dir:
        cfg.config["run"]["output_dir"] = args.output_dir

    init_distributed_mode(cfg.run_cfg)
    setup_seeds(cfg)
    setup_logger()

    task = tasks.setup_task(cfg) if dataset_name == "nocaps" else CaptionTask.setup_task(cfg=cfg)
    image_size = cfg.config.get("preprocess", {}).get("vis_processor", {}) \
        .get("eval", {}).get("image_size", 384)
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                     (0.26862954, 0.26130258, 0.27577711))
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        _convert_to_rgb,
        transforms.ToTensor(),
        normalize,
    ])
    datasets = build(cfg, transform=transform)
    model = task.build_model(cfg)
    runner = RunnerBase(cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets)

    output_dir = os.path.join(cfg.run_cfg["output_dir"], job_id)
    os.makedirs(output_dir, exist_ok=True)
    registry.mapping["paths"]["output_dir"] = output_dir
    registry.mapping["paths"]["result_dir"] = output_dir
    runner.evaluate(skip_reload=True)


if __name__ == "__main__":
    main()
