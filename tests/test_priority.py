"""Numerical checks for the priority allocation logic (toy encoder, no heavy deps)."""
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from priority import (  # noqa: E402
    MODES, PriorityConfig, allocate, build_priority_mask, compute_host_priority,
    compute_target_priority, normalize_map, topk_mask)
from util import PAIR_COUNT, validate_count  # noqa: E402


class ToyEncoder:
    """Deterministic differentiable image encoder used in place of CLIP."""

    def __init__(self, channels=3, dim=16):
        self.weight = torch.arange(1, channels + 1, dtype=torch.float32).view(1, channels, 1, 1)
        self.dim = dim

    def encode_img(self, image):
        pooled = (image * self.weight.to(image.device)).mean(dim=(2, 3))
        return pooled.repeat(1, self.dim // pooled.shape[1])


@pytest.fixture
def batch():
    torch.manual_seed(0)
    host = torch.rand(4, 3, 224, 224)
    target = torch.rand(4, 3, 224, 224)
    raw = (torch.rand_like(host) - 0.5) * 0.2
    return host, target, raw


@pytest.fixture
def encoder():
    return ToyEncoder()


def test_normalize_map_spans_unit_range():
    value = torch.randn(3, 1, 8, 8) * 5
    normalized = normalize_map(value)
    assert torch.allclose(normalized.amin(dim=(2, 3)), torch.zeros(3, 1, 1), atol=1e-5)
    assert torch.allclose(normalized.amax(dim=(2, 3)), torch.ones(3, 1, 1), atol=1e-5)


def test_host_and_target_priority_shapes(batch, encoder):
    host, target, _ = batch
    host_map = compute_host_priority(host)
    target_map = compute_target_priority(encoder, target)
    assert host_map.shape == (4, 1, 224, 224)
    assert target_map.shape == (4, 1, 224, 224)
    assert torch.isfinite(target_map).all()


def test_topk_density_matches_ratio():
    priority = torch.rand(2, 1, 10, 10)
    mask = topk_mask(priority, 0.35)
    assert mask.sum().item() == 2 * max(1, int(100 * 0.35))
    assert set(torch.unique(mask).tolist()) == {0.0, 1.0}


@pytest.mark.parametrize("mode", MODES)
def test_every_mode_is_valid_and_bounded(batch, encoder, mode):
    host, target, raw = batch
    config = PriorityConfig(mode=mode)
    random_priority = torch.rand(4, 1, 224, 224) if mode == "random_topk" else None
    bounded, _, hard_mask, soft_mask = build_priority_mask(
        raw, host, target, encoder, config, random_priority)
    assert bounded.abs().max() <= config.epsilon + 1e-6
    assert hard_mask.shape == (4, 1, 224, 224)
    assert set(torch.unique(hard_mask).tolist()).issubset({0.0, 1.0})
    assert torch.count_nonzero(hard_mask).item() > 0
    assert torch.logical_and(soft_mask >= config.low_region_weight - 1e-6,
                             soft_mask <= 1 + 1e-6).all()


def test_full_image_keeps_every_pixel(batch, encoder):
    host, target, raw = batch
    config = PriorityConfig(mode="full_image")
    _, _, hard_mask, soft_mask = build_priority_mask(raw, host, target, encoder, config)
    assert torch.count_nonzero(hard_mask).item() == hard_mask.numel()
    assert torch.allclose(soft_mask, torch.ones_like(soft_mask))


def test_allocate_clips_and_respects_budget(batch, encoder):
    host, target, raw = batch
    config = PriorityConfig()
    result = allocate(raw, host, target, encoder, config)
    assert result["adversarial"].min() >= 0 and result["adversarial"].max() <= 1
    assert (result["adversarial"] - host).abs().max() <= config.epsilon + 1e-6
    # Attenuated pixels never exceed the raw perturbation magnitude.
    assert result["scaled"].abs().max() <= config.epsilon + 1e-6


def test_random_topk_is_deterministic_with_seed(batch, encoder):
    host, target, raw = batch
    config = PriorityConfig(mode="random_topk")
    generator = torch.Generator().manual_seed(123)
    first = torch.rand(4, 1, 224, 224, generator=generator)
    generator = torch.Generator().manual_seed(123)
    second = torch.rand(4, 1, 224, 224, generator=generator)
    _, _, mask_a, _ = build_priority_mask(raw, host, target, encoder, config, first)
    _, _, mask_b, _ = build_priority_mask(raw, host, target, encoder, config, second)
    assert torch.equal(mask_a, mask_b)


def test_invalid_mode_rejected():
    with pytest.raises(ValueError):
        PriorityConfig(mode="nope").validate()


def test_pair_count_constant_and_validation():
    assert PAIR_COUNT == 1000
    validate_count([object()] * 1000)
    with pytest.raises(ValueError):
        validate_count([object()] * 999)
