"""Stage 1: decoder pre-training on web-scale image-text shards (DDP)."""
import argparse
import glob
import os

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import webdataset as wds
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from torchvision import transforms

from models.model import CLIPEncoder, Decoder


class DynamicInfoNCELoss(nn.Module):
    """InfoNCE whose temperature decays exponentially from 1.0 to 0.07."""

    def __init__(self, initial_temp=1.0, final_temp=0.07, total_batches=10000):
        super().__init__()
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.total_batches = total_batches
        self.current_temp = initial_temp
        self.decay_rate = -np.log(final_temp / initial_temp) / total_batches

    def forward(self, embeddings1, embeddings2):
        embeddings1 = F.normalize(embeddings1, p=2, dim=1)
        embeddings2 = F.normalize(embeddings2, p=2, dim=1)
        logits = embeddings1 @ embeddings2.T / self.current_temp
        labels = torch.arange(logits.shape[0], device=logits.device)
        return F.cross_entropy(logits, labels)

    def update_temperature(self, batch_count):
        self.current_temp = self.initial_temp * np.exp(-self.decay_rate * batch_count)


def make_dataloader(tar_dir, batch_size):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
    ])

    def handle_sample(sample):
        image, text = sample
        return transform(image), text

    tar_files = sorted(glob.glob(os.path.join(tar_dir, "*.tar")))
    if not tar_files:
        raise FileNotFoundError(f"no .tar shards found under {tar_dir}")
    dataset = (wds.WebDataset(tar_files, resampled=True, shardshuffle=True)
               .shuffle(5000)
               .decode("pil")
               .to_tuple("jpg", "txt")
               .map(handle_sample)
               .batched(batch_size)
               .with_epoch(len(tar_files)))
    return wds.WebLoader(dataset, batch_size=None, num_workers=8, pin_memory=True)


def main_worker(args):
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend="gloo", init_method=args.dist_url,
                            world_size=args.world_size, rank=local_rank)
    torch.cuda.set_device(local_rank)

    clip_encoder = CLIPEncoder("ViT-B/32").to(local_rank)
    decoder = Decoder(embed_dim=512).to(local_rank)
    clip_encoder = DistributedDataParallel(clip_encoder, device_ids=[local_rank])
    decoder = DistributedDataParallel(decoder, device_ids=[local_rank])
    clip_encoder.requires_grad_(False)

    optimizer = torch.optim.AdamW(decoder.parameters(), args.lr)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5000, T_mult=1)
    scaler = GradScaler()
    criterion = DynamicInfoNCELoss()
    train_loader = make_dataloader(args.tar_dir, args.batch_size)

    global_step = 0
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=f"cuda:{local_rank}")
        decoder.module.load_state_dict(checkpoint["decoder_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        global_step = checkpoint.get("global_step", 0)

    os.makedirs("checkpoints", exist_ok=True)
    for epoch in range(args.epoch):
        running_loss, count = 0.0, 0
        for batch_idx, (images, _) in enumerate(train_loader):
            images = images.to(local_rank)
            optimizer.zero_grad()
            with torch.no_grad():
                e1 = clip_encoder.module.encode_img(images)
            with autocast():
                noise = decoder(e1).clamp(-args.eps, args.eps)
                attacked = []
                for _ in range(args.chunk):
                    permuted = images[torch.randperm(images.size(0))]
                    attacked.append(torch.clamp(noise + permuted, 0, 1))
                attacked = torch.cat(attacked, dim=0)
                e2 = clip_encoder.module.encode_img(attacked)
                e2 = torch.stack(torch.chunk(e2, args.chunk, dim=0)).mean(dim=0)
                loss = criterion(e1, e2)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.item()
            count += 1
            global_step += 1
            if batch_idx <= 10000:
                criterion.update_temperature(batch_idx)

            if batch_idx % 500 == 0 and local_rank == 0:
                print(f"Epoch {epoch} batch {batch_idx} loss {running_loss / count:.6f} "
                      f"lr {optimizer.param_groups[0]['lr']:.2e}", flush=True)
                torch.save({
                    "global_step": global_step,
                    "decoder_state_dict": decoder.module.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                }, "checkpoints/pre-trained.pt")

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tar_dir", type=str, required=True)
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=600)
    parser.add_argument("--dist_url", type=str, default="tcp://127.0.0.1:23456")
    parser.add_argument("--chunk", type=int, default=5)
    parser.add_argument("--eps", type=float, default=16 / 255)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()
    args.world_size = torch.cuda.device_count()
    main_worker(args)
