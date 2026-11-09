"""Tests for deterministic fixed-pair construction."""
import json
import sys
import tempfile
from argparse import Namespace
from pathlib import Path

import prepare_pairs


IMAGE_SUFFIX = ".JPEG"


def _make_host_root(base):
    # Create files in an order that is deliberately not lexicographic.
    paths = [
        base / "n02102040" / "n02102040_9.JPEG",
        base / "n01440764" / "ILSVRC2012_val_00000002.JPEG",
        base / "n01440764" / "ILSVRC2012_val_00000001.JPEG",
        base / "n02102040" / "n02102040_1.JPEG",
        base / "n01440764" / "n01440764_3.JPEG",
    ]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
    return paths


def _make_targets(path, count):
    rows = [{"image": f"images/target_{index:05d}.jpg",
             "caption": [f"caption {index}"]} for index in range(count)]
    path.write_text(json.dumps(rows), encoding="utf-8")
    return rows


def _run_main(repo_root, targets_path, host_root, output_path, count):
    old_argv = sys.argv
    try:
        sys.argv = [
            "prepare_pairs.py",
            "--targets", str(targets_path),
            "--host_root", str(host_root),
            "--output", str(output_path),
            "--count", str(count),
        ]
        prepare_pairs.main()
    finally:
        sys.argv = old_argv


def test_hosts_are_sorted_prefix_not_a_sample():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        host_root = tmp / "val"
        _make_host_root(host_root)
        targets_path = tmp / "targets.json"
        _make_targets(targets_path, 3)
        output_path = tmp / "pairs.json"
        _run_main(tmp, targets_path, host_root, output_path, 3)

        rows = json.loads(output_path.read_text(encoding="utf-8"))
        expected_hosts = sorted(
            str(p.relative_to(host_root))
            for p in host_root.rglob("*") if p.is_file()
        )[:3]
        assert [row["host"] for row in rows] == expected_hosts
        assert [row["id"] for row in rows] == ["00000", "00001", "00002"]
        assert [row["target"] for row in rows] == [
            "images/target_00000.jpg",
            "images/target_00001.jpg",
            "images/target_00002.jpg",
        ]


def test_pair_construction_is_byte_stable():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        host_root = tmp / "val"
        _make_host_root(host_root)
        targets_path = tmp / "targets.json"
        _make_targets(targets_path, 4)
        first = tmp / "first.json"
        second = tmp / "second.json"
        _run_main(tmp, targets_path, host_root, first, 4)
        _run_main(tmp, targets_path, host_root, second, 4)
        assert first.read_bytes() == second.read_bytes()


def test_small_host_root_is_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        host_root = tmp / "val"
        (host_root / "n01440764").mkdir(parents=True)
        (host_root / "n01440764" / "only.JPEG").write_bytes(b"x")
        args = Namespace(host_root=host_root, count=2, existing_mapping=None,
                         original_host_root=None)
        try:
            prepare_pairs.select_hosts(args)
        except ValueError as exc:
            assert "need 2" in str(exc)
        else:
            raise AssertionError("expected ValueError for a small host root")


def test_existing_mapping_reuses_relative_hosts():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        host_root = tmp / "val"
        original_root = tmp / "original"
        mapping_path = tmp / "mapping.json"
        rel = "n01440764/ILSVRC2012_val_00000001.JPEG"
        (host_root / "n01440764").mkdir(parents=True)
        (host_root / rel).write_bytes(b"x")
        mapping_path.write_text(json.dumps([
            {"clean_path": str(original_root / rel)}
        ]), encoding="utf-8")
        args = Namespace(host_root=host_root, count=1,
                         existing_mapping=mapping_path,
                         original_host_root=original_root)
        hosts, resolved_root = prepare_pairs.select_hosts(args)
        assert hosts == [rel]
        assert resolved_root == host_root.resolve()
