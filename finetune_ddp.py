"""Stage 2: spatially guided decoder fine-tuning (DDP).

The decoder learns to place perturbation using three fused priority sources
and is supervised through CLIP, EVA-02 and ImageNet ViT-B/16 encoders.
"""
import argparse
import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from torchvision import transforms

import timm
from lavis.common.registry import registry
from lavis.datasets.builders import load_dataset

from models.model import CLIPEncoder, Decoder
from priority import MODES, PriorityConfig, build_priority_mask
from util import cycle, train_collate_fn


class DirectMatchingLoss(nn.Module):
    def forward(self, image_features, text_features):
        return -torch.cosine_similarity(image_features, text_features, dim=-1).mean()


class BiContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, image_features, text_features):
        image_features = F.normalize(image_features, p=2, dim=1)
        text_features = F.normalize(text_features, p=2, dim=1)
        logits = image_features @ text_features.t() / self.temperature
        labels = torch.eye(logits.size(0), device=logits.device)
        loss_i2t = -torch.sum(labels * F.log_softmax(logits, dim=1), dim=1).mean()
        loss_t2i = -torch.sum(labels * F.log_softmax(logits.t(), dim=1), dim=1).mean()
        loss = (loss_i2t + loss_t2i) / 2
        observer = torch.diag(image_features @ text_features.t()).mean()
        return loss, observer


def select_criterion(name, criterion, embed_adv, embed_target):
    if name == "BiContrastiveLoss":
        return criterion(embed_adv, embed_target)
    if name == "Cosine":
        loss = criterion(embed_adv, embed_target)
        return loss, -loss
    raise ValueError(f"unknown criterion {name}")


def make_loaders(args, rank, world_size):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    target_dataset = load_dataset(args.dataset, vis_path=args.data_dir)["train"]
    target_sampler = torch.utils.data.distributed.DistributedSampler(
        target_dataset, num_replicas=world_size, rank=rank)
    target_loader = DataLoader(target_dataset, batch_size=args.batch_size, num_workers=4,
                               pin_memory=True, shuffle=False, sampler=target_sampler,
                               collate_fn=train_collate_fn, drop_last=True)

    host_dataset = torchvision.datasets.ImageFolder(args.imagenet, transform=transform)
    host_sampler = torch.utils.data.distributed.DistributedSampler(
        host_dataset, num_replicas=world_size, rank=rank)
    host_loader = DataLoader(host_dataset, batch_size=args.batch_size, num_workers=4,
                             pin_memory=True, shuffle=False, sampler=host_sampler, drop_last=True)
    return target_loader, host_loader


def train(args):
    dist.init_process_group(backend="nccl")
    rank, world_size = dist.get_rank(), dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    clip_encoder = CLIPEncoder("ViT-B/32").to(device).eval()
    eva_encoder = timm.create_model(
        "hf_hub:timm/eva02_large_patch14_448.mim_m38m_ft_in1k",
        num_classes=0, pretrained=True).to(device).eval()
    imagenet_encoder = torchvision.models.vit_b_16(pretrained=True).to(device).eval()
    imagenet_encoder.head = torch.nn.Identity()

    decoder = Decoder(embed_dim=512).to(device)
    decoder = DDP(decoder, device_ids=[rank])
    optimizer = torch.optim.AdamW(decoder.parameters(), args.lr)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5000, T_mult=1)
    scaler = GradScaler()
    start_epoch = 0

    if args.checkpoint != "scratch":
        checkpoint = torch.load(args.checkpoint, map_location=device)
        if "decoder_state_dict" in checkpoint:
            decoder.module.load_state_dict(checkpoint["decoder_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
            start_epoch = int(checkpoint["epoch"]) + 1
        else:
            decoder.module.load_state_dict(checkpoint)

    criterion = BiContrastiveLoss() if args.criterion == "BiContrastiveLoss" else DirectMatchingLoss()
    config = PriorityConfig(epsilon=args.eps, ratio=args.topk_ratio,
                            low_region_weight=args.low_region_weight, w_noise=args.w_noise,
                            w_host=args.w_host, w_target=args.w_target, mode=args.priority_mode)
    config.validate()

    target_loader, host_loader = make_loaders(args, rank, world_size)
    host_cycle = cycle(host_loader)

    for epoch in range(start_epoch, args.epoch):
        target_loader.sampler.set_epoch(epoch)
        stats = {"loss": 0.0, "clip": 0.0, "eva": 0.0, "imnet": 0.0, "sparse": 0.0}
        global_step = 0
        for batch_idx, batch in enumerate(target_loader):
            target = batch["image"].to(device)
            host, _ = next(host_cycle)
            host = host.to(device)

            optimizer.zero_grad(set_to_none=True)
            with autocast():
                with torch.no_grad():
                    target_clip = clip_encoder.encode_img(target)
                    target_eva = eva_encoder(F.interpolate(target, size=(448, 448), mode="bilinear"))
                    target_imnet = imagenet_encoder(target)

                raw = decoder(target_clip).clamp(-args.eps, args.eps)
                random_priority = torch.rand(len(host), 1, host.shape[-2], host.shape[-1],
                                             device=device) if args.priority_mode == "random_topk" else None
                bounded, _, hard_mask, soft_mask = build_priority_mask(
                    raw, host, target, clip_encoder, config, random_priority)
                adversarial = torch.clamp(host + bounded * soft_mask, 0, 1)

                adv_clip = clip_encoder.encode_img(adversarial)
                adv_eva = eva_encoder(F.interpolate(adversarial, size=(448, 448), mode="bilinear"))
                adv_imnet = imagenet_encoder(adversarial)

                loss_clip, obs_clip = select_criterion(args.criterion, criterion, adv_clip, target_clip)
                loss_eva, obs_eva = select_criterion(args.criterion, criterion, adv_eva, target_eva)
                loss_imnet, obs_imnet = select_criterion(args.criterion, criterion, adv_imnet, target_imnet)
                sparse_penalty = (raw.abs() * (1 - hard_mask)).mean()
                total = loss_clip + loss_eva + loss_imnet + args.lambda_sparse * sparse_penalty

            scaler.scale(total).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            stats["loss"] += total.item()
            stats["clip"] += obs_clip.item()
            stats["eva"] += obs_eva.item()
            stats["imnet"] += obs_imnet.item()
            stats["sparse"] += sparse_penalty.item()
            global_step += 1
            if batch_idx % 100 == 0 and rank == 0:
                print(f"Epoch {epoch} batch {batch_idx} loss {stats['loss'] / global_step:.6f} "
                      f"CLIP {stats['clip'] / global_step:.4f} EVA {stats['eva'] / global_step:.4f} "
                      f"IMNET {stats['imnet'] / global_step:.4f} sparse {stats['sparse'] / global_step:.6f}",
                      flush=True)

        if rank == 0:
            os.makedirs("checkpoints", exist_ok=True)
            save_name = (f"checkpoints/{args.dataset}_{args.criterion}_priority_"
                         f"top{int(args.topk_ratio * 100)}_{args.priority_mode}_epoch{epoch + 1:02d}.pt")
            torch.save({
                "epoch": epoch,
                "global_step": global_step,
                "decoder_state_dict": decoder.module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "args": vars(args),
            }, save_name)
            print(f"[checkpoint] {save_name}", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="datasets/mscoco")
    parser.add_argument("--dataset", type=str, default="coco_retrieval")
    parser.add_argument("--epoch", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=74)
    parser.add_argument("--eps", type=float, default=16 / 255)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/pre-trained.pt")
    parser.add_argument("--criterion", type=str, default="BiContrastiveLoss",
                        choices=["BiContrastiveLoss", "Cosine"])
    parser.add_argument("--imagenet", type=str, required=True)
    parser.add_argument("--cache_path", type=str, default="datasets")
    parser.add_argument("--topk_ratio", type=float, default=0.35)
    parser.add_argument("--low_region_weight", type=float, default=0.385)
    parser.add_argument("--lambda_sparse", type=float, default=0.20)
    parser.add_argument("--priority_mode", type=str, default="triple", choices=list(MODES))
    parser.add_argument("--w_noise", type=float, default=0.45)
    parser.add_argument("--w_host", type=float, default=0.30)
    parser.add_argument("--w_target", type=float, default=0.25)
    args = parser.parse_args()
    registry.mapping["paths"]["cache_root"] = args.cache_path
    train(args)
