"""Datasets and transforms for fixed host-target pair generation."""
import json
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from util import validate_count

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])


def rooted_path(root, relative):
    """Resolve a manifest path while keeping it inside its dataset root."""
    relative = Path(relative)
    root = Path(root).resolve()
    if relative.is_absolute():
        raise ValueError("manifest paths must be relative to their dataset root")
    path = (root / relative).resolve()
    if path != root and root not in path.parents:
        raise ValueError("manifest path escapes its dataset root")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def read_manifest(path, count=1000):
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_count(rows, count)
    identifiers = set()
    for row in rows:
        if not isinstance(row, dict) or not all(k in row for k in ("id", "host", "target", "captions")):
            raise ValueError("each row needs id, host, target and captions")
        if not all(isinstance(row[k], str) and row[k] for k in ("id", "host", "target")):
            raise ValueError("id, host and target must be non-empty strings")
        if row["id"] in identifiers:
            raise ValueError("duplicate pair identifier")
        identifiers.add(row["id"])
        captions = row["captions"]
        if not isinstance(captions, list) or not captions or \
                not all(isinstance(text, str) and text for text in captions):
            raise ValueError("captions must be a non-empty list of non-empty strings")
    return rows


class PairDataset(Dataset):
    """Deterministically ordered fixed host-target pairs (no shuffling)."""

    def __init__(self, manifest, host_root, target_root, count=1000):
        self.rows = read_manifest(manifest, count)
        self.host_root = Path(host_root)
        self.target_root = Path(target_root)
        for row in self.rows:
            rooted_path(self.host_root, row["host"])
            rooted_path(self.target_root, row["target"])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(rooted_path(self.host_root, row["host"])) as image:
            host = EVAL_TRANSFORM(image.convert("RGB"))
        with Image.open(rooted_path(self.target_root, row["target"])) as image:
            target = EVAL_TRANSFORM(image.convert("RGB"))
        return host, target, index
