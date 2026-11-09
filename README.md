# HT-CoGuide

Spatially guided adversarial example generation for vision-language models.
The decoder is pre-trained and then fine-tuned to allocate perturbation budget
from host, target and noise priorities.

All benchmark runs use a fixed set of **1000 host-target pairs**.

## Requirements

- Python 3.9+
- PyTorch 2.2+ with CUDA

```bash
pip install -r requirements.txt            # generation + perceptual metrics
pip install -r requirements-train.txt      # pre-training / fine-tuning
pip install -r requirements-retrieval.txt  # LAVIS-based evaluation
```

## Layout

```
main_ddp.py            decoder pre-training (WebDataset shards)
finetune_ddp.py        guided decoder fine-tuning (DDP)
generate_adv_img.py    adversarial image generation over fixed pairs
prepare_pairs.py       build the fixed host-target manifest
perceptual.py          PSNR / SSIM / LPIPS / L-inf checks
priority.py            priority maps, masks and budget allocation
dataset.py             manifest-driven paired dataset
models/                CLIP encoder, decoder and vendored CLIP package
lavis_tool/            LAVIS task glue and evaluation configs
scripts/               training, generation and evaluation entrypoints
retrieval.py           image-text retrieval evaluation
caption.py             image captioning evaluation
classification.py      visual entailment evaluation
demo.py                single-pair demonstration
sweep.py               parameter sweeps / ablations
configs/paper.json     default hyper-parameters
```

## Data

Prepare the datasets separately; no data, weights or logs are included.

- MSCOCO and Flickr30K can be obtained through the LAVIS benchmark downloader:
  <https://opensource.salesforce.com/LAVIS//latest/benchmark>
- ImageNet is required for host images.
- Optional pre-training shards (WebDataset `.tar`) can be built from LAION-400M:
  <https://github.com/rom1504/img2dataset/blob/main/dataset_examples/laion400m.md>
- LAVIS: <https://github.com/salesforce/LAVIS>

## Usage

### 1. Build the fixed pair manifest

```bash
python prepare_pairs.py \
    --targets datasets/mscoco/annotations/test.json \
    --host_root datasets/ILSVRC2012/val \
    --output manifests/coco_retrieval_pairs.json
```

The manifest stores relative paths and always contains 1000 entries.

### 2. Pre-train the decoder

```bash
LAION_DIR=datasets/laion400m-shards bash scripts/main.sh
```

### 3. Fine-tune the guided decoder

```bash
DATASET_BASE_PATH=datasets bash scripts/finetune_ddp.sh
```

### 4. Generate adversarial images

```bash
DATASET_BASE_PATH=datasets bash scripts/generate_adv_img.sh
```

Each benchmark writes 1000 numbered PNGs plus `mapping.json`,
`annotations.json` and `run.json` under `outputs/<dataset>/`.

### 5. Perceptual checks

```bash
python perceptual.py \
    --generated outputs/coco_retrieval/ht_coguide \
    --host_root datasets/ILSVRC2012/val \
    --output outputs/coco_retrieval/perceptual.json
```

### 6. Downstream evaluation

```bash
bash scripts/retrieval.sh
python scripts/caption_shell.py
python scripts/classification_shell.py
```

For a single image pair use `demo.py`; for hyper-parameter ablations use `sweep.py`.

## Defaults

The released defaults match `configs/paper.json`: CLIP ViT-B/32,
epsilon = 16/255, top-k ratio 0.35, low-region weight 0.385,
noise/host/target weights 0.45/0.30/0.25, sparse regularizer 0.20,
contrastive temperature 0.07, and the seven allocation modes
`triple`, `host_target`, `host_only`, `target_only`, `noise_only`,
`random_topk`, and `full_image`.

## Tests

```bash
pytest tests
```

## License

MIT, see [LICENSE](LICENSE).
