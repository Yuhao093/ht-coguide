"""Spatially guided perturbation allocation.

Three priority sources are fused per pixel: the decoder perturbation itself,
host-image edges and target-image feature gradients. A hard top-k mask keeps
full strength inside the selected region and attenuates the rest.
"""
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F


MODES = ("triple", "host_target", "host_only", "target_only",
         "noise_only", "random_topk", "full_image")


@dataclass(frozen=True)
class PriorityConfig:
    epsilon: float = 16 / 255
    ratio: float = 0.35
    low_region_weight: float = 0.385
    w_noise: float = 0.45
    w_host: float = 0.30
    w_target: float = 0.25
    mode: str = "triple"

    def validate(self):
        numbers = (self.epsilon, self.ratio, self.low_region_weight,
                   self.w_noise, self.w_host, self.w_target)
        if not all(math.isfinite(v) for v in numbers):
            raise ValueError("priority parameters must be finite")
        if not 0 < self.epsilon <= 1:
            raise ValueError("epsilon must be in (0, 1]")
        if not 0 < self.ratio <= 1:
            raise ValueError("ratio must be in (0, 1]")
        if not 0 <= self.low_region_weight <= 1:
            raise ValueError("low_region_weight must be in [0, 1]")
        if min(self.w_noise, self.w_host, self.w_target) < 0:
            raise ValueError("fusion weights must be nonnegative")
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if self.mode == "triple" and self.w_noise + self.w_host + self.w_target <= 0:
            raise ValueError("triple mode requires at least one positive weight")
        if self.mode == "host_target" and self.w_host + self.w_target <= 0:
            raise ValueError("host_target mode requires a positive image weight")


def normalize_map(value):
    lo = value.amin(dim=(2, 3), keepdim=True)
    hi = value.amax(dim=(2, 3), keepdim=True)
    return (value - lo) / (hi - lo + 1e-6)


def compute_host_priority(host):
    """Grayscale spatial edge magnitude (dx + dy)."""
    gray = host.mean(dim=1, keepdim=True)
    dx = F.pad((gray[:, :, :, 1:] - gray[:, :, :, :-1]).abs(), (0, 1, 0, 0))
    dy = F.pad((gray[:, :, 1:, :] - gray[:, :, :-1, :]).abs(), (0, 0, 0, 1))
    return normalize_map(dx + dy)


def compute_target_priority(encoder, target):
    """Mean absolute gradient of the mean target feature norm."""
    with torch.enable_grad():
        image = target.detach().clone().requires_grad_(True)
        objective = encoder.encode_img(image).norm(dim=1).mean()
        gradient = torch.autograd.grad(objective, image, create_graph=False)[0]
    return normalize_map(gradient.abs().mean(dim=1, keepdim=True).detach())


def topk_mask(priority, ratio):
    if priority.shape[1] != 1:
        raise ValueError("priority map must have one spatial channel")
    batch, _, height, width = priority.shape
    if not 0 < ratio <= 1:
        raise ValueError("ratio must be in (0, 1]")
    k = max(1, int(height * width * ratio))
    flat = priority.reshape(batch, -1)
    indices = flat.topk(k, dim=1, sorted=False).indices
    mask = torch.zeros_like(flat).scatter_(1, indices, 1.0)
    return mask.reshape(batch, 1, height, width)


def fuse_priorities(raw, host, target, encoder, cfg, random_priority):
    noise = normalize_map(raw.detach().abs().mean(dim=1, keepdim=True))
    mode = cfg.mode
    if mode == "full_image":
        return noise, torch.ones_like(noise)
    host_map = compute_host_priority(host) if mode in ("triple", "host_target", "host_only") else None
    target_map = compute_target_priority(encoder, target) if mode in ("triple", "host_target", "target_only") else None
    if mode == "triple":
        fused = cfg.w_noise * noise + cfg.w_host * host_map + cfg.w_target * target_map
        fused = normalize_map(fused)
    elif mode == "host_target":
        fused = normalize_map(cfg.w_host * host_map + cfg.w_target * target_map)
    elif mode == "host_only":
        fused = host_map
    elif mode == "target_only":
        fused = target_map
    elif mode == "noise_only":
        fused = noise
    else:  # random_topk
        if random_priority is None or random_priority.shape != noise.shape:
            raise ValueError("random_topk requires a deterministic per-pair random map")
        fused = random_priority.to(device=raw.device, dtype=raw.dtype)
    return fused, topk_mask(fused, cfg.ratio)


def build_priority_mask(raw, host, target, encoder, cfg, random_priority=None):
    """Return bounded noise, fused priority, hard mask and attenuated soft mask."""
    cfg.validate()
    if not (raw.shape == host.shape == target.shape):
        raise ValueError("raw, host and target tensors must share the same BCHW shape")
    bounded = raw.clamp(-cfg.epsilon, cfg.epsilon)
    priority, hard_mask = fuse_priorities(raw, host, target, encoder, cfg, random_priority)
    soft_mask = hard_mask + (1 - hard_mask) * cfg.low_region_weight
    return bounded, priority, hard_mask, soft_mask


def allocate(raw, host, target, encoder, cfg, random_priority=None):
    """Apply the guided noise to the host and clip the result to [0, 1]."""
    bounded, priority, hard_mask, soft_mask = build_priority_mask(
        raw, host, target, encoder, cfg, random_priority)
    scaled = bounded * soft_mask
    adversarial = (host + scaled).clamp(0, 1)
    return {"adversarial": adversarial, "bounded": bounded, "scaled": scaled,
            "priority": priority, "hard_mask": hard_mask, "soft_mask": soft_mask}
