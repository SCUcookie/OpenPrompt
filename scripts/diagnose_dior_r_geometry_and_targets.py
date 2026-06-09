#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
OPENRSD_ROOT = WORKSPACE_ROOT / "OpenRSD"

DIOR_R_CLASSES = [
    "airplane",
    "airport",
    "baseballfield",
    "basketballcourt",
    "bridge",
    "chimney",
    "dam",
    "Expressway-Service-area",
    "Expressway-toll-station",
    "golffield",
    "groundtrackfield",
    "harbor",
    "overpass",
    "ship",
    "stadium",
    "storagetank",
    "tenniscourt",
    "trainstation",
    "vehicle",
    "windmill",
]

DEFAULT_CONFIGS = [
    OPENRSD_ROOT / "M_configs/G02_Baselines/Data2_DIOR_R/G02_Baselines_Data2_DIOR_R_M2_RoITrans.py",
    OPENRSD_ROOT / "M_configs/G02_Baselines/Data2_DIOR_R/G02_Baselines_Data2_DIOR_R_M5_ORCNN_R50.py",
]


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return WORKSPACE_ROOT / path


def _decode_image(path: Path) -> tuple[bool, dict[str, Any]]:
    try:
        from PIL import Image

        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            return True, {"width": image.width, "height": image.height, "mode": image.mode}
    except Exception as pil_error:
        try:
            import cv2

            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if image is None:
                return False, {"error": f"PIL={pil_error}; cv2.imread returned None"}
            height, width = image.shape[:2]
            return True, {"width": int(width), "height": int(height), "mode": f"cv2:{image.shape}"}
        except Exception as cv_error:
            return False, {"error": f"PIL={pil_error}; cv2={cv_error}"}


def _polygon_area(points: list[tuple[float, float]]) -> float:
    area = 0.0
    for idx, (x1, y1) in enumerate(points):
        x2, y2 = points[(idx + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return abs(area) * 0.5


def _edge_lengths(points: list[tuple[float, float]]) -> list[float]:
    lengths = []
    for idx, (x1, y1) in enumerate(points):
        x2, y2 = points[(idx + 1) % len(points)]
        lengths.append(math.hypot(x2 - x1, y2 - y1))
    return lengths


def _fallback_qbox_to_rbox(coords: list[float]) -> list[float]:
    points = [(coords[idx], coords[idx + 1]) for idx in range(0, 8, 2)]
    cx = sum(point[0] for point in points) / 4.0
    cy = sum(point[1] for point in points) / 4.0
    lengths = _edge_lengths(points)
    width = max(lengths[0], lengths[2])
    height = max(lengths[1], lengths[3])
    angle = math.atan2(points[1][1] - points[0][1], points[1][0] - points[0][0])
    return [cx, cy, width, height, angle]


def _mmrotate_qbox_to_rbox(coords: list[float]) -> tuple[bool, list[float] | str]:
    try:
        sys.path.insert(0, str(OPENRSD_ROOT))
        import torch
        from mmrotate.structures.bbox import qbox2rbox

        qbox = torch.tensor([coords], dtype=torch.float32)
        rbox = qbox2rbox(qbox)
        values = [float(value) for value in rbox.reshape(-1).tolist()]
        return True, values
    except Exception as error:
        try:
            return False, _fallback_qbox_to_rbox(coords)
        except Exception:
            return False, str(error)


def _parse_label_file(path: Path, class_set: set[str]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": path,
        "num_objects": 0,
        "class_counts": Counter(),
        "bad_records": [],
        "first_bad_conversion": None,
    }
    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 9:
            result["bad_records"].append({"line": line_no, "reason": "expected at least 9 fields", "text": line})
            continue
        try:
            coords = [float(value) for value in parts[:8]]
        except ValueError as error:
            result["bad_records"].append({"line": line_no, "reason": f"coordinate parse failed: {error}", "text": line})
            continue
        class_name = parts[8]
        points = [(coords[idx], coords[idx + 1]) for idx in range(0, 8, 2)]
        area = _polygon_area(points)
        lengths = _edge_lengths(points)
        finite = all(math.isfinite(value) for value in coords)
        positive_edges = all(length > 0 for length in lengths)
        known_class = class_name in class_set
        conversion_ok, conversion = _mmrotate_qbox_to_rbox(coords)
        conversion_finite = (
            isinstance(conversion, list)
            and len(conversion) == 5
            and all(math.isfinite(value) for value in conversion)
        )
        if not finite or area <= 0 or not positive_edges or not known_class or not conversion_finite:
            bad = {
                "line": line_no,
                "class_name": class_name,
                "known_class": known_class,
                "finite_coords": finite,
                "area": area,
                "edge_lengths": lengths,
                "conversion_source": "mmrotate" if conversion_ok else "fallback_or_error",
                "conversion": conversion,
            }
            result["bad_records"].append(bad)
            if not conversion_finite and result["first_bad_conversion"] is None:
                result["first_bad_conversion"] = bad
        result["num_objects"] += 1
        result["class_counts"][class_name] += 1
    return result


def _scan_split(data_root: Path, split: str, max_images: int | None, max_labels: int | None) -> dict[str, Any]:
    image_dir = data_root / split / "images"
    label_dir = data_root / split / "labelTxt"
    image_paths = sorted(image_dir.glob("*"))
    label_paths = sorted(label_dir.glob("*.txt"))
    if max_images is not None:
        image_paths = image_paths[:max_images]
    if max_labels is not None:
        label_paths = label_paths[:max_labels]

    decode_bad = []
    image_shapes = Counter()
    for image_path in image_paths:
        ok, info = _decode_image(image_path)
        if ok:
            image_shapes[(info["width"], info["height"])] += 1
        else:
            decode_bad.append({"path": str(image_path), **info})

    class_counts: Counter[str] = Counter()
    bad_label_files = []
    first_bad_conversion = None
    total_objects = 0
    for label_path in label_paths:
        parsed = _parse_label_file(label_path, set(DIOR_R_CLASSES))
        total_objects += int(parsed["num_objects"])
        class_counts.update(parsed["class_counts"])
        if parsed["bad_records"]:
            bad_label_files.append(
                {
                    "path": str(label_path),
                    "sample_id": label_path.stem,
                    "bad_records": parsed["bad_records"][:5],
                }
            )
        if first_bad_conversion is None and parsed["first_bad_conversion"] is not None:
            first_bad_conversion = {"path": str(label_path), **parsed["first_bad_conversion"]}

    return {
        "split": split,
        "image_dir": image_dir,
        "label_dir": label_dir,
        "num_images_checked": len(image_paths),
        "num_label_files_checked": len(label_paths),
        "num_objects": total_objects,
        "image_decode_bad": decode_bad[:20],
        "num_image_decode_bad": len(decode_bad),
        "top_image_shapes": [
            {"width": width, "height": height, "count": count}
            for (width, height), count in image_shapes.most_common(10)
        ],
        "class_counts": {name: int(class_counts[name]) for name in DIOR_R_CLASSES},
        "unknown_class_counts": {
            name: int(count)
            for name, count in sorted(class_counts.items())
            if name not in DIOR_R_CLASSES
        },
        "num_bad_label_files": len(bad_label_files),
        "bad_label_files": bad_label_files[:20],
        "first_bad_conversion": first_bad_conversion,
    }


def _load_config_class_names(config_path: Path) -> list[str] | None:
    try:
        namespace = runpy_namespace(config_path)
        class_name = namespace.get("class_name")
        metainfo = namespace.get("metainfo")
        if isinstance(class_name, (list, tuple)):
            return [str(value) for value in class_name]
        if isinstance(metainfo, dict) and isinstance(metainfo.get("classes"), (list, tuple)):
            return [str(value) for value in metainfo["classes"]]
    except Exception:
        return None
    return None


def runpy_namespace(config_path: Path) -> dict[str, Any]:
    import runpy

    old_cwd = os.getcwd()
    try:
        os.chdir(str(OPENRSD_ROOT))
        return runpy.run_path(str(config_path))
    finally:
        os.chdir(old_cwd)


def _check_configs(config_paths: list[Path]) -> list[dict[str, Any]]:
    checks = []
    for config_path in config_paths:
        names = _load_config_class_names(config_path)
        checks.append(
            {
                "config": config_path,
                "exists": config_path.exists(),
                "class_names": names,
                "matches_dior_r_order": names == DIOR_R_CLASSES,
            }
        )
    return checks


def _optional_dataloader_check(config_path: Path, max_batches: int) -> dict[str, Any]:
    try:
        sys.path.insert(0, str(OPENRSD_ROOT))
        from mmengine.config import Config
        from mmrotate.registry import DATASETS
        from mmrotate.utils import register_all_modules

        register_all_modules(init_default_scope=True)
        cfg = Config.fromfile(str(config_path))
        dataset = DATASETS.build(cfg.train_dataloader.dataset)
        samples = []
        for index in range(min(max_batches, len(dataset))):
            sample = dataset[index]
            data_sample = sample.get("data_samples")
            instances = getattr(data_sample, "gt_instances", None)
            bboxes = getattr(instances, "bboxes", None)
            labels = getattr(instances, "labels", None)
            tensor = getattr(bboxes, "tensor", bboxes)
            finite = None
            shape = None
            if tensor is not None:
                finite = bool(tensor.isfinite().all().item())
                shape = list(tensor.shape)
            samples.append(
                {
                    "index": index,
                    "img_path": getattr(data_sample, "img_path", None),
                    "bbox_shape": shape,
                    "bbox_finite": finite,
                    "num_labels": int(labels.numel()) if labels is not None else None,
                }
            )
        return {"status": "ok", "config": config_path, "samples": samples}
    except Exception as error:
        return {"status": "error", "config": config_path, "error": repr(error)}


def _optional_first_loss_check(config_path: Path) -> dict[str, Any]:
    try:
        sys.path.insert(0, str(OPENRSD_ROOT))
        import torch
        from mmengine.config import Config
        from mmdet.registry import MODELS
        from mmengine.runner import Runner

        cfg = Config.fromfile(str(config_path))
        cfg.train_dataloader.num_workers = 0
        cfg.train_dataloader.persistent_workers = False
        cfg.train_dataloader.batch_size = min(1, int(cfg.train_dataloader.get("batch_size", 1)))
        dataloader = Runner.build_dataloader(cfg.train_dataloader)
        model = MODELS.build(cfg.model)
        model.train()
        batch = next(iter(dataloader))
        with torch.no_grad():
            losses = model(**batch, mode="loss")
        flat = {}
        finite = True
        for key, value in losses.items():
            if isinstance(value, list):
                value = sum(value)
            if hasattr(value, "detach"):
                scalar = float(value.detach().mean().cpu())
            else:
                scalar = float(value)
            flat[key] = scalar
            finite = finite and math.isfinite(scalar)
        return {"status": "ok", "config": config_path, "finite": finite, "losses": flat}
    except Exception as error:
        return {"status": "error", "config": config_path, "error": repr(error)}


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# DIOR-R Geometry And Target Diagnostics",
        "",
        f"- Data root: `{payload['data_root']}`",
        f"- Config class-order checks: `{payload['config_checks']}`",
        "",
        "## Split Summary",
    ]
    for split in payload["splits"]:
        lines.extend(
            [
                "",
                f"### {split['split']}",
                "",
                f"- Images checked: `{split['num_images_checked']}`",
                f"- Label files checked: `{split['num_label_files_checked']}`",
                f"- Objects: `{split['num_objects']}`",
                f"- Bad image decodes: `{split['num_image_decode_bad']}`",
                f"- Bad label files: `{split['num_bad_label_files']}`",
                f"- Unknown classes: `{split['unknown_class_counts']}`",
                f"- First bad conversion: `{split['first_bad_conversion']}`",
            ]
        )
    if payload.get("dataloader_checks"):
        lines.extend(["", "## Dataloader Checks", "", json.dumps(payload["dataloader_checks"], indent=2, default=_json_default)])
    if payload.get("first_loss_checks"):
        lines.extend(["", "## First-Loss Checks", "", json.dumps(payload["first_loss_checks"], indent=2, default=_json_default)])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose DIOR-R geometry, class mapping, and optional target/loss sanity.")
    parser.add_argument("--data-root", default=str(OPENRSD_ROOT / "data/DIOR_R_dota"))
    parser.add_argument("--config", action="append", default=None, help="DIOR-R config to class-check; repeatable.")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--max-label-files", type=int, default=None)
    parser.add_argument("--check-dataloader", action="store_true")
    parser.add_argument("--check-first-loss", action="store_true")
    parser.add_argument("--max-dataloader-samples", type=int, default=2)
    parser.add_argument("--output-json", default=str(REPO_ROOT / "artifacts/dior_r_diagnostics_20260609.json"))
    parser.add_argument("--output-md", default=str(REPO_ROOT / "artifacts/dior_r_diagnostics_20260609.md"))
    args = parser.parse_args()

    data_root = _resolve(args.data_root)
    config_paths = [_resolve(path) for path in args.config] if args.config else DEFAULT_CONFIGS

    payload: dict[str, Any] = {
        "data_root": data_root,
        "class_order_reference": DIOR_R_CLASSES,
        "config_checks": _check_configs(config_paths),
        "splits": [
            _scan_split(data_root, "train_val", args.max_images, args.max_label_files),
            _scan_split(data_root, "test", args.max_images, args.max_label_files),
        ],
    }
    if args.check_dataloader:
        payload["dataloader_checks"] = [
            _optional_dataloader_check(config_path, args.max_dataloader_samples)
            for config_path in config_paths
        ]
    if args.check_first_loss:
        payload["first_loss_checks"] = [
            _optional_first_loss_check(config_path)
            for config_path in config_paths
        ]

    output_json = _resolve(args.output_json)
    output_md = _resolve(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n", encoding="utf-8")
    _write_markdown(output_md, payload)
    print(json.dumps(payload, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
