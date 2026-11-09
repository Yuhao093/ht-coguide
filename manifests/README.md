# Fixed host-target pair manifests

This directory archives the frozen evaluation records of the study: one
manifest per benchmark, each containing exactly 1,000 fixed host-target pairs.

## Record schema

Every row is a JSON object with four fields:

- `id`: zero-padded pair identifier from `00000` to `00999`;
- `host`: image path relative to the shared host image root;
- `target`: image path relative to the task-specific target root;
- `captions`: caption text carried verbatim by the task annotation.

All four manifests share the identical 1,000-image host sequence in the same
order; only the targets differ. Pair `id` therefore identifies the same host
image across every benchmark.

## Deterministic construction

The pair list contains no random state and is reproducible byte for byte:

1. Hosts are collected by recursively listing the host root for image files
   (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`), sorting the relative paths
   lexicographically (ImageFolder order), and taking the first 1,000 entries.
2. Targets are the first 1,000 entries of the task annotation file, preserving
   its stored order.
3. Hosts and targets are zipped by position and numbered `00000`-`00999`.

The host root is the 10-class ImageFolder-ordered ImageNet subset distributed
as imagenette (3,925 images in WNID subdirectories). Only relative paths are
stored; no absolute machine path appears in any manifest.

```bash
python prepare_pairs.py \
    --targets <task annotation json> \
    --host_root <imagenette image root> \
    --output manifests/<manifest name>.json
```

Determinism and the 1,000-pair structure are enforced by `tests/`.

## Annotation provenance

| Manifest | Task annotation (first 1,000 rows) | Public source |
|---|---|---|
| `coco_retrieval_pairs.json` | `coco_karpathy_test.json` (5,000 rows) | https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test.json |
| `coco_caption_pairs.json` | `coco_karpathy_test.json` (5,000 rows) | https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test.json |
| `flickr30k_pairs.json` | `flickr30k_test.json` (1,000 rows, full test split) | https://storage.googleapis.com/sfr-vision-language-research/datasets/flickr30k_test.json |
| `snli_ve_pairs.json` | SNLI-VE test annotation in LAVIS JSON format (1,000 rows) | https://github.com/necla-ml/SNLI-VE |

The COCO Karpathy test file matches the official LAVIS checksum
`md5:3ff34b0ef2db02d01c37399f6a2a6cd1`. Both COCO tasks read the same
annotation, so their two manifests are identical. The Flickr30k and SNLI-VE
target images reside in the Flickr30k image directory; SNLI-VE rows reference
them by bare file names.

## Integrity (SHA-256)

- `coco_caption_pairs.json`
  `32f78c2b32fc533db01e0dd399d72a344f7aeb90b36213e7e907f6b0d81fa35c`
- `coco_retrieval_pairs.json`
  `32f78c2b32fc533db01e0dd399d72a344f7aeb90b36213e7e907f6b0d81fa35c`
- `flickr30k_pairs.json`
  `c99cf82e4a51ca2c60c6bc7b4676ce5dcd5b09b3db469548beff25e053d81a50`
- `snli_ve_pairs.json`
  `67ed14b205e84a7480184c7d4e9a2967c012fba5b332a0e056ad27d595cb0794`
