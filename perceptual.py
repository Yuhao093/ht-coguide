"""Perceptual evaluation of generated images against their mapped clean hosts."""
import argparse
import json
import math
from pathlib import Path

from dataset import EVAL_TRANSFORM, rooted_path
from util import validate_count, write_json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--host_root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lpips_net", choices=("alex", "vgg", "squeeze"), default="alex")
    args = parser.parse_args()
    if args.batch_size < 1 or args.count < 1:
        parser.error("batch_size and count must be positive")
    return args


def main():
    args = parse_args()
    mapping = validate_count(json.loads((args.generated / "mapping.json").read_text(encoding="utf-8")),
                             args.count)
    run = json.loads((args.generated / "run.json").read_text(encoding="utf-8"))
    if run["count"] != args.count:
        raise ValueError("run count differs from mapping count")
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")

    import numpy as np
    import lpips
    import torch
    from PIL import Image
    from skimage.metrics import structural_similarity
    from torchvision.transforms.functional import to_tensor

    epsilon = run["config"]["epsilon"]
    model = lpips.LPIPS(net=args.lpips_net).to(args.device).eval()
    rows, hosts, adversarial, pending = [], [], [], []

    def flush():
        if not pending:
            return
        with torch.no_grad():
            values = model(torch.stack(hosts).to(args.device) * 2 - 1,
                           torch.stack(adversarial).to(args.device) * 2 - 1).flatten().tolist()
        for row, value in zip(pending, values):
            row["lpips"] = value
        hosts.clear()
        adversarial.clear()
        pending.clear()

    for item in mapping:
        with Image.open(rooted_path(args.host_root, item["host"])) as image:
            clean = EVAL_TRANSFORM(image.convert("RGB"))
        with Image.open(rooted_path(args.generated, item["generated"])) as image:
            adv = to_tensor(image.convert("RGB"))
        if clean.shape != adv.shape:
            raise ValueError("generated and preprocessed clean image shapes differ")
        delta = adv - clean
        mse = float(delta.square().mean())
        row = {
            "id": item["id"],
            "psnr": None if mse == 0 else 10 * math.log10(1 / mse),
            "psnr_infinite": mse == 0,
            "linf": float(delta.abs().max()),
            "ssim": float(structural_similarity(clean.permute(1, 2, 0).numpy(),
                                                 adv.permute(1, 2, 0).numpy(),
                                                 data_range=1, channel_axis=2)),
        }
        rows.append(row)
        hosts.append(clean)
        adversarial.append(adv)
        pending.append(row)
        if len(hosts) >= args.batch_size:
            flush()
    flush()

    finite_psnr = [row["psnr"] for row in rows if not row["psnr_infinite"]]
    result = {
        "count": len(rows),
        "lpips": float(np.mean([row["lpips"] for row in rows])),
        "psnr": float(np.mean(finite_psnr)) if finite_psnr else None,
        "psnr_infinite": not finite_psnr,
        "ssim": float(np.mean([row["ssim"] for row in rows])),
        "linf_max": max(row["linf"] for row in rows),
        "budget_violations": sum(row["linf"] > epsilon + 1e-6 for row in rows),
        "budget_violations_with_8bit_tolerance":
            sum(row["linf"] > epsilon + 1 / 255 + 1e-6 for row in rows),
        "lpips_net": args.lpips_net,
        "rows": rows,
    }
    write_json(args.output, result)
    print({key: value for key, value in result.items() if key != "rows"})


if __name__ == "__main__":
    main()
