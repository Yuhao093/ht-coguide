"""Shared helpers: fixed-size validation, JSON IO, checkpoint loading."""
import hashlib
import json
from itertools import chain, repeat
from pathlib import Path

PAIR_COUNT = 1000


def validate_count(rows, count=PAIR_COUNT):
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("count must be a positive integer")
    if not isinstance(rows, list) or len(rows) != count:
        received = len(rows) if isinstance(rows, list) else "invalid input"
        raise ValueError(f"expected exactly {count} pairs, received {received}")
    return rows


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_output(path, empty=True):
    path = Path(path)
    if empty and path.exists() and any(path.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_decoder(path, decoder, device="cpu"):
    import torch
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    state = checkpoint.get("decoder_state_dict", checkpoint)
    state = {key.removeprefix("module."): value for key, value in state.items()}
    decoder.load_state_dict(state, strict=True)
    return decoder


def cycle(loader):
    """Endlessly repeat a DataLoader."""
    while True:
        for batch in loader:
            yield batch


def train_collate_fn(batch):
    """Stack LAVIS training samples (PIL images) into an image tensor batch."""
    import torch
    from PIL import Image
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    images = []
    for item in batch:
        image = item["image"]
        if isinstance(image, Image.Image):
            image = transform(image)
        images.append(image)
    return {"image": torch.stack(images), "caption": None}


def lavis_annotation(mapping):
    """Convert a fixed-pair mapping to the LAVIS [{image, caption}] format."""
    return [{"image": row["generated"], "caption": row["captions"]} for row in mapping]
