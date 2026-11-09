"""Release hygiene checks: layout, forbidden assets, private paths and syntax."""
import py_compile
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ROOT_FILES = [
    "caption.py", "classification.py", "dataset.py", "demo.py", "finetune_ddp.py",
    "generate_adv_img.py", "main_ddp.py", "perceptual.py", "prepare_pairs.py",
    "priority.py", "retrieval.py", "sweep.py", "util.py",
    "README.md", "LICENSE", "requirements.txt", "requirements-train.txt",
    "requirements-retrieval.txt", ".gitignore",
]
ROOT_DIRS = ["models", "lavis_tool", "scripts", "tests", "configs"]
SCRIPT_FILES = [
    "main.sh", "finetune_ddp.sh", "generate_adv_img.sh", "retrieval.sh",
    "caption_shell.py", "classification_shell.py",
]
TEXT_SUFFIXES = {".py", ".sh", ".md", ".txt", ".json", ".yaml", ".yml"}
FORBIDDEN_BINARY = {".pt", ".pth", ".ckpt", ".safetensors", ".tar", ".log",
                    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".xlsx", ".csv", ".pdf"}
FORBIDDEN_NAME_PARTS = ("draw", "plot", "fig", "compose", "workbook", "writeback", "rerun")
FORBIDDEN_TOKENS = (
    "anyattack", "notice", "upstream-root", "upstream_root", "/new_data/", "/export/share",
    "dycpu", "yifei", "junhong", "/home/", "未复现", "未验证", "not reproduced",
    "minigpt", "attacked_model",
)


def all_files():
    return [path for path in ROOT.rglob("*") if path.is_file()]


def test_flat_layout_matches_reference():
    for name in ROOT_FILES:
        assert (ROOT / name).is_file(), f"missing root file {name}"
    for name in ROOT_DIRS:
        assert (ROOT / name).is_dir(), f"missing directory {name}"
    for name in SCRIPT_FILES:
        assert (ROOT / "scripts" / name).is_file(), f"missing script {name}"


def test_vendored_clip_and_lavis_present():
    for name in ("__init__.py", "clip.py", "model.py", "simple_tokenizer.py",
                 "bpe_simple_vocab_16e6.txt.gz"):
        assert (ROOT / "models" / "clip" / name).is_file()
    yamls = list((ROOT / "lavis_tool").rglob("*.yaml"))
    assert len(yamls) == 25


def test_no_weights_data_logs_or_figures():
    for path in all_files():
        suffix = path.suffix.lower()
        if suffix in FORBIDDEN_BINARY:
            pytest.fail(f"forbidden binary/data asset shipped: {path.relative_to(ROOT)}")
        lowered = path.name.lower()
        if any(part in lowered for part in FORBIDDEN_NAME_PARTS):
            pytest.fail(f"plotting/experiment artifact not allowed: {path.relative_to(ROOT)}")


def test_no_private_paths_or_attribution_tokens():
    offenders = []
    for path in all_files():
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if path.name == "test_release.py":
            continue
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
        for token in FORBIDDEN_TOKENS:
            if token in text:
                offenders.append((path.relative_to(ROOT), token))
    assert not offenders, offenders


def test_all_python_compiles():
    for path in ROOT.rglob("*.py"):
        py_compile.compile(str(path), doraise=True)


def test_fixed_pair_count_is_1000():
    util_text = (ROOT / "util.py").read_text(encoding="utf-8")
    assert "PAIR_COUNT = 1000" in util_text
    for entry in ("generate_adv_img.py", "perceptual.py", "prepare_pairs.py"):
        text = (ROOT / entry).read_text(encoding="utf-8")
        assert "default=1000" in text, f"{entry} must default to 1000 pairs"


def test_seven_allocation_modes_and_paper_defaults():
    ns = {}
    exec((ROOT / "priority.py").read_text(encoding="utf-8"), ns)
    assert ns["MODES"] == ("triple", "host_target", "host_only", "target_only",
                           "noise_only", "random_topk", "full_image")
    config = ns["PriorityConfig"]()
    assert config.ratio == 0.35
    assert config.low_region_weight == 0.385
    assert (config.w_noise, config.w_host, config.w_target) == (0.45, 0.30, 0.25)
    assert config.mode == "triple"


def test_yamls_use_relative_storage_only():
    for path in (ROOT / "lavis_tool").rglob("*.yaml"):
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.split("#", 1)[0]
            if "storage:" in stripped:
                value = stripped.split("storage:", 1)[1].strip()
                assert not value.startswith("/"), f"absolute storage in {path}"
