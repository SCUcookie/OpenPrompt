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

_QBOX_CONVERTER_READY = False
_TORCH_MODULE: Any | None = None
_QBOX2RBOX_FN: Any | None = None
_QBOX_CONVERTER_ERROR: str | None = None


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return WORKSPACE_ROOT / path


def _decode_image(path: Path, check_mode: str = "verify") -> tuple[bool, dict[str, Any]]:
    try:
        from PIL import Image

        with Image.open(path) as image:
            if check_mode == "verify":
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


def _percentile(sorted_values: list[float], percent: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * (percent / 100.0)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _summarize_values(values: list[float]) -> dict[str, Any]:
    finite_values = sorted(value for value in values if math.isfinite(value))
    if not finite_values:
        return {"count": 0}
    return {
        "count": len(finite_values),
        "min": finite_values[0],
        "p01": _percentile(finite_values, 1),
        "p05": _percentile(finite_values, 5),
        "p25": _percentile(finite_values, 25),
        "p50": _percentile(finite_values, 50),
        "p75": _percentile(finite_values, 75),
        "p95": _percentile(finite_values, 95),
        "p99": _percentile(finite_values, 99),
        "max": finite_values[-1],
        "mean": sum(finite_values) / len(finite_values),
    }


def _fallback_qbox_to_rbox(coords: list[float]) -> list[float]:
    points = [(coords[idx], coords[idx + 1]) for idx in range(0, 8, 2)]
    cx = sum(point[0] for point in points) / 4.0
    cy = sum(point[1] for point in points) / 4.0
    lengths = _edge_lengths(points)
    width = max(lengths[0], lengths[2])
    height = max(lengths[1], lengths[3])
    angle = math.atan2(points[1][1] - points[0][1], points[1][0] - points[0][0])
    return [cx, cy, width, height, angle]


def _load_qbox_converter() -> tuple[Any | None, Any | None, str | None]:
    global _QBOX_CONVERTER_READY, _TORCH_MODULE, _QBOX2RBOX_FN, _QBOX_CONVERTER_ERROR
    if not _QBOX_CONVERTER_READY:
        _QBOX_CONVERTER_READY = True
        try:
            openrsd_root = str(OPENRSD_ROOT)
            if openrsd_root not in sys.path:
                sys.path.insert(0, openrsd_root)
            import torch
            from mmrotate.structures.bbox import qbox2rbox

            _TORCH_MODULE = torch
            _QBOX2RBOX_FN = qbox2rbox
        except Exception as error:
            _QBOX_CONVERTER_ERROR = repr(error)
    return _TORCH_MODULE, _QBOX2RBOX_FN, _QBOX_CONVERTER_ERROR


def _mmrotate_qbox_to_rbox(coords: list[float]) -> tuple[bool, list[float] | str]:
    torch, qbox2rbox, error = _load_qbox_converter()
    try:
        if torch is None or qbox2rbox is None:
            raise RuntimeError(error or "qbox2rbox unavailable")
        qbox = torch.tensor([coords], dtype=torch.float32)
        rbox = qbox2rbox(qbox)
        values = [float(value) for value in rbox.reshape(-1).tolist()]
        return True, values
    except Exception as error:
        try:
            return False, _fallback_qbox_to_rbox(coords)
        except Exception:
            return False, str(error)


def _qbox_to_rbox(coords: list[float], conversion_backend: str) -> tuple[bool, str, list[float] | str]:
    if conversion_backend == "fallback":
        try:
            return True, "fallback", _fallback_qbox_to_rbox(coords)
        except Exception as error:
            return False, "fallback", str(error)
    conversion_ok, conversion = _mmrotate_qbox_to_rbox(coords)
    return conversion_ok, "mmrotate" if conversion_ok else "fallback_or_error", conversion


def _label_image_path(label_path: Path, image_dir: Path) -> Path | None:
    for suffix in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"):
        image_path = image_dir / f"{label_path.stem}{suffix}"
        if image_path.exists():
            return image_path
    matches = sorted(image_dir.glob(f"{label_path.stem}.*"))
    return matches[0] if matches else None


def _parse_label_file(
    path: Path,
    class_set: set[str],
    image_info: dict[str, Any] | None = None,
    conversion_backend: str = "mmrotate",
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": path,
        "num_objects": 0,
        "class_counts": Counter(),
        "bad_records": [],
        "first_bad_conversion": None,
        "object_records": [],
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
        conversion_ok, conversion_source, conversion = _qbox_to_rbox(coords, conversion_backend)
        conversion_finite = (
            isinstance(conversion, list)
            and len(conversion) == 5
            and all(math.isfinite(value) for value in conversion)
        )
        width = float(conversion[2]) if conversion_finite else float("nan")
        height = float(conversion[3]) if conversion_finite else float("nan")
        rbox_area = width * height if width > 0 and height > 0 else float("nan")
        aspect_ratio = max(width, height) / min(width, height) if width > 0 and height > 0 else float("nan")
        qbox_min_x = min(point[0] for point in points)
        qbox_max_x = max(point[0] for point in points)
        qbox_min_y = min(point[1] for point in points)
        qbox_max_y = max(point[1] for point in points)
        out_of_bounds = None
        center_out_of_bounds = None
        if image_info is not None and "width" in image_info and "height" in image_info:
            image_width = float(image_info["width"])
            image_height = float(image_info["height"])
            out_of_bounds = qbox_min_x < 0 or qbox_min_y < 0 or qbox_max_x > image_width or qbox_max_y > image_height
            if conversion_finite:
                center_out_of_bounds = (
                    float(conversion[0]) < 0
                    or float(conversion[1]) < 0
                    or float(conversion[0]) > image_width
                    or float(conversion[1]) > image_height
                )
        result["object_records"].append(
            {
                "class_name": class_name,
                "qbox_area": area,
                "rbox_width": width,
                "rbox_height": height,
                "rbox_area": rbox_area,
                "aspect_ratio": aspect_ratio,
                "qbox_out_of_bounds": out_of_bounds,
                "rbox_center_out_of_bounds": center_out_of_bounds,
                "invalid_rbox_size": not (width > 0 and height > 0),
            }
        )
        if not finite or area <= 0 or not positive_edges or not known_class or not conversion_finite:
            bad = {
                "line": line_no,
                "class_name": class_name,
                "known_class": known_class,
                "finite_coords": finite,
                "area": area,
                "edge_lengths": lengths,
                "conversion_source": conversion_source,
                "conversion": conversion,
            }
            result["bad_records"].append(bad)
            if not conversion_finite and result["first_bad_conversion"] is None:
                result["first_bad_conversion"] = bad
        result["num_objects"] += 1
        result["class_counts"][class_name] += 1
    return result


def _scan_split(
    data_root: Path,
    split: str,
    max_images: int | None,
    max_labels: int | None,
    conversion_backend: str,
    image_check_mode: str,
    assume_image_size: tuple[int, int] | None,
) -> dict[str, Any]:
    image_dir = data_root / split / "images"
    label_dir = data_root / split / "labelTxt"
    label_paths = sorted(label_dir.glob("*.txt"))
    if max_labels is not None:
        label_paths = label_paths[:max_labels]
    if assume_image_size is None:
        image_paths = sorted(image_dir.glob("*"))
        if max_images is not None:
            image_paths = image_paths[:max_images]
    else:
        image_paths = []

    decode_bad = []
    image_shapes = Counter()
    image_info_by_stem: dict[str, dict[str, Any]] = {}
    if assume_image_size is not None:
        assumed_width, assumed_height = assume_image_size
        assumed_info = {"width": assumed_width, "height": assumed_height, "mode": "assumed"}
        image_shapes[(assumed_width, assumed_height)] = len(label_paths)
        image_info_by_stem = {label_path.stem: assumed_info for label_path in label_paths}
    else:
        for image_path in image_paths:
            ok, info = _decode_image(image_path, image_check_mode)
            if ok:
                image_shapes[(info["width"], info["height"])] += 1
                image_info_by_stem[image_path.stem] = info
            else:
                decode_bad.append({"path": str(image_path), **info})

    class_counts: Counter[str] = Counter()
    bad_label_files = []
    first_bad_conversion = None
    total_objects = 0
    qbox_areas: list[float] = []
    rbox_widths: list[float] = []
    rbox_heights: list[float] = []
    rbox_areas: list[float] = []
    aspect_ratios: list[float] = []
    qbox_out_of_bounds = 0
    qbox_oob_examples = []
    rbox_center_out_of_bounds = 0
    rbox_center_oob_examples = []
    invalid_rbox_size = 0
    missing_image_info = 0
    for label_path in label_paths:
        image_info = image_info_by_stem.get(label_path.stem)
        if image_info is None:
            image_path = _label_image_path(label_path, image_dir)
            if image_path is not None and assume_image_size is not None:
                assumed_width, assumed_height = assume_image_size
                image_info = {"width": assumed_width, "height": assumed_height, "mode": "assumed"}
                image_info_by_stem[label_path.stem] = image_info
            elif image_path is not None:
                ok, info = _decode_image(image_path, image_check_mode)
                if ok:
                    image_info = info
                    image_info_by_stem[label_path.stem] = info
            if image_info is None:
                missing_image_info += 1
        parsed = _parse_label_file(label_path, set(DIOR_R_CLASSES), image_info, conversion_backend)
        total_objects += int(parsed["num_objects"])
        class_counts.update(parsed["class_counts"])
        for record in parsed["object_records"]:
            qbox_areas.append(float(record["qbox_area"]))
            rbox_widths.append(float(record["rbox_width"]))
            rbox_heights.append(float(record["rbox_height"]))
            rbox_areas.append(float(record["rbox_area"]))
            aspect_ratios.append(float(record["aspect_ratio"]))
            if record["qbox_out_of_bounds"] is True:
                qbox_out_of_bounds += 1
                if len(qbox_oob_examples) < 10:
                    qbox_oob_examples.append({"path": str(label_path), "class_name": record["class_name"]})
            if record["rbox_center_out_of_bounds"] is True:
                rbox_center_out_of_bounds += 1
                if len(rbox_center_oob_examples) < 10:
                    rbox_center_oob_examples.append({"path": str(label_path), "class_name": record["class_name"]})
            if record["invalid_rbox_size"]:
                invalid_rbox_size += 1
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
        "conversion_backend": conversion_backend,
        "image_check_mode": image_check_mode,
        "assume_image_size": assume_image_size,
        "num_images_checked": len(image_paths),
        "num_images_assumed": len(label_paths) if assume_image_size is not None else 0,
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
        "missing_image_info_for_labels": missing_image_info,
        "rbox_stats": {
            "qbox_area": _summarize_values(qbox_areas),
            "rbox_width": _summarize_values(rbox_widths),
            "rbox_height": _summarize_values(rbox_heights),
            "rbox_area": _summarize_values(rbox_areas),
            "aspect_ratio": _summarize_values(aspect_ratios),
        },
        "bounds_checks": {
            "qbox_out_of_bounds": qbox_out_of_bounds,
            "qbox_out_of_bounds_examples": qbox_oob_examples,
            "rbox_center_out_of_bounds": rbox_center_out_of_bounds,
            "rbox_center_out_of_bounds_examples": rbox_center_oob_examples,
            "invalid_rbox_size": invalid_rbox_size,
        },
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
                f"- Conversion backend: `{split.get('conversion_backend')}`",
                f"- Image check mode: `{split.get('image_check_mode')}`",
                f"- Assumed image size: `{split.get('assume_image_size')}`",
                f"- Images checked: `{split['num_images_checked']}`",
                f"- Images assumed: `{split.get('num_images_assumed')}`",
                f"- Label files checked: `{split['num_label_files_checked']}`",
                f"- Objects: `{split['num_objects']}`",
                f"- Bad image decodes: `{split['num_image_decode_bad']}`",
                f"- Bad label files: `{split['num_bad_label_files']}`",
                f"- Unknown classes: `{split['unknown_class_counts']}`",
                f"- First bad conversion: `{split['first_bad_conversion']}`",
                f"- Missing image info for labels: `{split.get('missing_image_info_for_labels')}`",
                f"- Bounds checks: `{split.get('bounds_checks')}`",
                "",
                "RBox statistics:",
                "",
                json.dumps(split.get("rbox_stats", {}), indent=2, default=_json_default),
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
    parser.add_argument(
        "--conversion-backend",
        choices=("mmrotate", "fallback"),
        default="mmrotate",
        help="Use mmrotate qbox2rbox or a fast polygon-edge fallback for geometry scans.",
    )
    parser.add_argument(
        "--image-check-mode",
        choices=("verify", "dimensions"),
        default="verify",
        help="Use full PIL verify checks or only image header dimensions for bounds statistics.",
    )
    parser.add_argument(
        "--assume-image-size",
        default=None,
        metavar="WIDTHxHEIGHT",
        help="Skip image reads and use declared dimensions for label bounds checks, e.g. 800x800.",
    )
    parser.add_argument("--output-json", default=str(REPO_ROOT / "artifacts/dior_r_diagnostics_20260609.json"))
    parser.add_argument("--output-md", default=str(REPO_ROOT / "artifacts/dior_r_diagnostics_20260609.md"))
    parser.add_argument("--quiet", action="store_true", help="Write files without printing the full JSON payload.")
    args = parser.parse_args()

    data_root = _resolve(args.data_root)
    config_paths = [_resolve(path) for path in args.config] if args.config else DEFAULT_CONFIGS
    assume_image_size = None
    if args.assume_image_size is not None:
        try:
            width_text, height_text = args.assume_image_size.lower().split("x", 1)
            assume_image_size = (int(width_text), int(height_text))
        except ValueError as error:
            raise SystemExit(f"--assume-image-size must be WIDTHxHEIGHT, got {args.assume_image_size!r}") from error

    payload: dict[str, Any] = {
        "data_root": data_root,
        "class_order_reference": DIOR_R_CLASSES,
        "config_checks": _check_configs(config_paths),
        "splits": [
            _scan_split(
                data_root,
                "train_val",
                args.max_images,
                args.max_label_files,
                args.conversion_backend,
                args.image_check_mode,
                assume_image_size,
            ),
            _scan_split(
                data_root,
                "test",
                args.max_images,
                args.max_label_files,
                args.conversion_backend,
                args.image_check_mode,
                assume_image_size,
            ),
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
    if args.quiet:
        print(f"wrote {output_json}")
        print(f"wrote {output_md}")
    else:
        print(json.dumps(payload, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
