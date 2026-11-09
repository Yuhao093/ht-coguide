"""Build a portable manifest of fixed host-target pairs.

Hosts are the first ``--count`` images of the lexicographically sorted host
root (ImageFolder order), so the same root always yields the same pairs
without any random state. Targets follow the annotation file order.
"""
import argparse
import json
from pathlib import Path

from util import write_json

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=Path, required=True,
                        help="LAVIS-style annotation JSON with image and caption fields")
    parser.add_argument("--host_root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--existing_mapping", type=Path,
                        help="reuse clean_path entries from an earlier mapping")
    parser.add_argument("--original_host_root", type=Path,
                        help="root prefix of clean_path values in the earlier mapping")
    args = parser.parse_args()
    if args.count < 1:
        parser.error("count must be positive")
    return args


def select_hosts(args):
    root = args.host_root.resolve()
    if args.existing_mapping:
        if args.original_host_root is None:
            raise ValueError("--original_host_root is required with --existing_mapping")
        old = json.loads(args.existing_mapping.read_text(encoding="utf-8"))
        if not isinstance(old, list) or len(old) != args.count:
            raise ValueError(f"existing mapping must contain exactly {args.count} rows")
        prefix = str(args.original_host_root).replace("\\", "/").rstrip("/") + "/"
        hosts = []
        for row in old:
            value = row["clean_path"].replace("\\", "/")
            if not value.startswith(prefix):
                raise ValueError("mapping clean_path does not match original_host_root")
            hosts.append(value[len(prefix):])
        return hosts, root

    paths = sorted(str(p.relative_to(root)) for p in root.rglob("*")
                   if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if len(paths) < args.count:
        raise ValueError(f"host root contains only {len(paths)} images, need {args.count}")
    return paths[:args.count], root


def main():
    args = parse_args()
    targets = json.loads(args.targets.read_text(encoding="utf-8"))
    if not isinstance(targets, list) or len(targets) < args.count:
        raise ValueError(f"target annotations must contain at least {args.count} entries")
    hosts, root = select_hosts(args)

    rows = []
    for index, (host, target) in enumerate(zip(hosts, targets[:args.count])):
        captions = target.get("caption")
        if isinstance(captions, str):
            captions = [captions]
        if not captions or not all(isinstance(text, str) and text for text in captions):
            raise ValueError("each target must carry caption text")
        for value in (host, target["image"]):
            path = Path(value)
            if not value or path.is_absolute() or ".." in path.parts:
                raise ValueError("image paths must be non-empty relative paths")
        host_path = root / host
        if not host_path.is_file() or root not in host_path.resolve().parents:
            raise ValueError("host image is missing or outside host_root")
        rows.append({"id": f"{index:05d}", "host": host,
                     "target": target["image"], "captions": captions})

    write_json(args.output, rows)
    print(f"Wrote {len(rows)} fixed pairs to {args.output}")


if __name__ == "__main__":
    main()
