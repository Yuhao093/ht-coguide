"""Fixed-pair allocation sweeps over budget, components, ratio and weights."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from util import prepare_output, sha256, validate_count, write_json

ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=("budget", "components", "ratio", "weights"),
                        required=True)
    parser.add_argument("--decoder_path", type=Path, required=True)
    parser.add_argument("--component_checkpoint", type=Path,
                        help="decoder used by non-triple component variants")
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--host_root", type=Path, required=True)
    parser.add_argument("--target_root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=str, default="coco_retrieval")
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.count < 1 or args.batch_size < 1:
        parser.error("count and batch_size must be positive")
    return args


def build_runs(args):
    if args.experiment == "budget":
        return [(f"budget_{value:02d}", args.decoder_path, ["--eps", str(value / 255)])
                for value in (4, 8, 12, 16)]
    if args.experiment == "components":
        if args.component_checkpoint is None:
            raise ValueError("components requires --component_checkpoint")
        runs = []
        for mode in ("triple", "host_target", "host_only", "target_only", "noise_only", "random_topk"):
            checkpoint = args.decoder_path if mode == "triple" else args.component_checkpoint
            runs.append((mode, checkpoint, ["--priority_mode", mode]))
        return runs
    if args.experiment == "ratio":
        return [(f"ratio_{value:.2f}", args.decoder_path, ["--topk_ratio", str(value)])
                for value in (.10, .20, .25, .30, .35, .40, .50, .60, .80)]
    runs = []
    for host_weight in (.45, .40, .35, .30, .25, .20, .15, .10):
        runs.append((f"host_target_{host_weight:.2f}", args.decoder_path,
                     ["--w_host", str(host_weight), "--w_target", str(0.55 - host_weight)]))
    for decoder_weight in (0, .15, .25, .35, .45, .55, .65, .80):
        runs.append((f"decoder_{decoder_weight:.2f}", args.decoder_path,
                     ["--w_noise", str(decoder_weight),
                      "--w_host", str((1 - decoder_weight) * 6 / 11),
                      "--w_target", str((1 - decoder_weight) * 5 / 11)]))
    return runs


def main():
    args = parse_args()
    validate_count(json.loads(args.pairs.read_text(encoding="utf-8")), args.count)
    base = [sys.executable, str(ROOT / "generate_adv_img.py"),
            "--pairs", str(args.pairs.resolve()),
            "--host_root", str(args.host_root.resolve()),
            "--target_root", str(args.target_root.resolve()),
            "--dataset", args.dataset, "--batch_size", str(args.batch_size),
            "--count", str(args.count), "--device", args.device]
    prepare_output(args.output)
    recorded = []
    for tag, checkpoint, options in build_runs(args):
        output = args.output / tag
        command = base + ["--decoder_path", str(checkpoint.resolve()),
                          "--adv_imgs", tag,
                          "--output_path", str(args.output.resolve())] + options
        subprocess.run(command, cwd=ROOT, check=True)
        recorded.append({"name": tag, "command": command})
    write_json(args.output / "sweep.json",
               {"experiment": args.experiment, "count": args.count,
                "pairs_sha256": sha256(args.pairs), "runs": recorded})


if __name__ == "__main__":
    main()
