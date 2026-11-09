"""Generate adversarial images for a fixed, ordered set of host-target pairs."""
import argparse
import os
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from dataset import PairDataset
from models.model import CLIPEncoder, Decoder
from priority import MODES, PriorityConfig, allocate
from util import lavis_annotation, load_decoder, prepare_output, sha256, write_json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eps", type=float, default=16 / 255)
    parser.add_argument("--model_name", type=str, default="ViT-B/32")
    parser.add_argument("--decoder_path", type=str, required=True)
    parser.add_argument("--pairs", type=Path, required=True, help="fixed host-target manifest")
    parser.add_argument("--host_root", type=Path, required=True)
    parser.add_argument("--target_root", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output_path", type=str, default="outputs")
    parser.add_argument("--adv_imgs", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--priority_mode", type=str, default="triple", choices=list(MODES))
    parser.add_argument("--topk_ratio", type=float, default=0.35)
    parser.add_argument("--low_region_weight", type=float, default=0.385)
    parser.add_argument("--w_noise", type=float, default=0.45)
    parser.add_argument("--w_host", type=float, default=0.30)
    parser.add_argument("--w_target", type=float, default=0.25)
    args = parser.parse_args()
    if args.batch_size < 1 or args.count < 1:
        parser.error("batch_size and count must be positive")
    return args


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    config = PriorityConfig(epsilon=args.eps, ratio=args.topk_ratio,
                            low_region_weight=args.low_region_weight, w_noise=args.w_noise,
                            w_host=args.w_host, w_target=args.w_target, mode=args.priority_mode)
    config.validate()

    encoder = CLIPEncoder(args.model_name, device=device).to(device).eval().requires_grad_(False)
    decoder = Decoder(embed_dim=512)
    load_decoder(args.decoder_path, decoder, device=device)
    decoder = decoder.to(device).eval().requires_grad_(False)

    dataset = PairDataset(args.pairs, args.host_root, args.target_root, args.count)
    out_dir = prepare_output(Path(args.output_path) / args.dataset / args.adv_imgs)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    mapping = []
    for host, target, indices in loader:
        host, target = host.to(device), target.to(device)
        with torch.no_grad():
            raw = decoder(encoder.encode_img(target))
        random_priority = None
        if args.priority_mode == "random_topk":
            random_priority = torch.stack([
                torch.rand(1, host.shape[-2], host.shape[-1],
                           generator=torch.Generator().manual_seed(args.seed + int(index)))
                for index in indices
            ]).to(device)
        result = allocate(raw, host, target, encoder, config, random_priority)
        for j, index in enumerate(indices.tolist()):
            filename = f"{index:05d}.png"
            save_image(result["adversarial"][j].detach().cpu(), out_dir / filename)
            row = dict(dataset.rows[index])
            row.update({
                "generated": filename,
                "hard_ratio": float(result["hard_mask"][j].mean()),
                "float_linf": float((result["adversarial"][j] - host[j]).abs().max()),
            })
            mapping.append(row)

    if len(mapping) != args.count:
        raise RuntimeError(f"generated {len(mapping)} images but expected {args.count}")
    write_json(out_dir / "mapping.json", mapping)
    write_json(out_dir / "annotations.json", lavis_annotation(mapping))
    write_json(out_dir / "run.json", {
        "count": len(mapping),
        "seed": args.seed,
        "batch_size": args.batch_size,
        "pairs_sha256": sha256(args.pairs),
        "decoder_sha256": sha256(args.decoder_path),
        "config": asdict(config),
    })
    print(f"Saved {len(mapping)} adversarial images to {out_dir}")


if __name__ == "__main__":
    main()
