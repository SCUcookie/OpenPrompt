#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import runpy
import sys
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
OPENRSD_ROOT = WORKSPACE_ROOT / "OpenRSD"
DEFAULT_TAXONOMY = REPO_ROOT / "assets/hierarchies/fair1m_remote_sensing_taxonomy.json"
DEFAULT_CONFIG = OPENRSD_ROOT / "M_configs/G02_Baselines/Data3_FAIR1M/G02_Baselines_Data3_FAIR1M_M2_RoITrans.py"
_QBOX_CONVERTER: tuple[Any | None, Any | None, str | None] | None = None


def normalize_class_name(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def load_taxonomy(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [str(item["name"]) for item in payload["classes"]]


def decode_image(path: Path) -> tuple[bool, dict[str, Any]]:
    try:
        from PIL import Image

        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            return True, {"width": image.width, "height": image.height, "mode": image.mode}
    except Exception as error:
        return False, {"error": repr(error)}


def polygon_area(points: list[tuple[float, float]]) -> float:
    return abs(
        sum(
            x1 * points[(index + 1) % 4][1] - points[(index + 1) % 4][0] * y1
            for index, (x1, y1) in enumerate(points)
        )
    ) * 0.5


def edge_lengths(points: list[tuple[float, float]]) -> list[float]:
    return [
        math.hypot(points[(index + 1) % 4][0] - x1, points[(index + 1) % 4][1] - y1)
        for index, (x1, y1) in enumerate(points)
    ]


def load_qbox_converter() -> tuple[Any | None, Any | None, str | None]:
    global _QBOX_CONVERTER
    if _QBOX_CONVERTER is None:
        try:
            import torch
            from mmrotate.structures.bbox import qbox2rbox

            _QBOX_CONVERTER = (torch, qbox2rbox, None)
        except Exception as error:
            _QBOX_CONVERTER = (None, None, repr(error))
    return _QBOX_CONVERTER


def qbox_to_rbox(coords: list[float]) -> tuple[bool, list[float] | str]:
    torch, qbox2rbox, import_error = load_qbox_converter()
    try:
        if torch is None or qbox2rbox is None:
            raise RuntimeError(import_error or "MMRotate qbox2rbox is unavailable")
        values = qbox2rbox(torch.tensor([coords], dtype=torch.float32)).reshape(-1).tolist()
        return True, [float(value) for value in values]
    except Exception as error:
        return False, repr(error)


def find_image(image_dir: Path, stem: str) -> Path | None:
    for suffix in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"):
        candidate = image_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def summarize(values: list[float]) -> dict[str, float | int | None]:
    values = sorted(value for value in values if math.isfinite(value))
    if not values:
        return {"count": 0, "min": None, "mean": None, "max": None}
    return {
        "count": len(values),
        "min": values[0],
        "mean": sum(values) / len(values),
        "max": values[-1],
    }


def scan_split(
    name: str,
    image_dir: Path,
    label_dir: Path,
    taxonomy: list[str],
    max_files: int | None,
    decode_limit: int,
) -> dict[str, Any]:
    image_paths = sorted(path for path in image_dir.iterdir() if path.is_file()) if image_dir.is_dir() else []
    label_paths = sorted(label_dir.glob("*.txt")) if label_dir.is_dir() else []
    all_image_stems = {path.stem for path in image_paths}
    all_label_stems = {path.stem for path in label_paths}
    selected_labels = label_paths[:max_files] if max_files is not None else label_paths
    known = set(taxonomy)
    class_counts: Counter[str] = Counter()
    malformed: list[dict[str, Any]] = []
    malformed_count = 0
    decode_errors: list[dict[str, Any]] = []
    missing_images: list[str] = []
    areas: list[float] = []
    edges: list[float] = []
    out_of_bounds = 0
    invalid_rboxes = 0
    conversion_errors: list[dict[str, Any]] = []
    decoded = 0
    objects = 0

    for label_path in selected_labels:
        image_path = find_image(image_dir, label_path.stem)
        image_info = None
        if image_path is None:
            missing_images.append(label_path.stem)
        elif decoded < decode_limit:
            ok, image_info = decode_image(image_path)
            decoded += 1
            if not ok:
                decode_errors.append({"path": str(image_path), **image_info})
                image_info = None
        for line_no, raw_line in enumerate(label_path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            parts = raw_line.strip().split()
            if not parts:
                continue
            if len(parts) < 9:
                malformed_count += 1
                if len(malformed) < 100:
                    malformed.append({"path": str(label_path), "line": line_no, "reason": "fewer than 9 fields"})
                continue
            try:
                coords = [float(value) for value in parts[:8]]
            except ValueError:
                malformed_count += 1
                if len(malformed) < 100:
                    malformed.append({"path": str(label_path), "line": line_no, "reason": "non-numeric coordinate"})
                continue
            class_name = normalize_class_name(parts[8])
            points = [(coords[index], coords[index + 1]) for index in range(0, 8, 2)]
            area = polygon_area(points)
            lengths = edge_lengths(points)
            conversion_ok, rbox = qbox_to_rbox(coords)
            rbox_finite = conversion_ok and isinstance(rbox, list) and len(rbox) == 5 and all(math.isfinite(v) for v in rbox)
            if not rbox_finite:
                invalid_rboxes += 1
                if len(conversion_errors) < 20:
                    conversion_errors.append({"path": str(label_path), "line": line_no, "result": rbox})
            if (
                not all(math.isfinite(value) for value in coords)
                or area <= 0
                or any(length <= 0 for length in lengths)
                or class_name not in known
            ):
                malformed_count += 1
                if len(malformed) < 100:
                    malformed.append(
                        {
                            "path": str(label_path),
                            "line": line_no,
                            "class_name": class_name,
                            "area": area,
                            "edge_lengths": lengths,
                            "known_class": class_name in known,
                        }
                    )
            if image_info and "width" in image_info:
                xs = coords[0::2]
                ys = coords[1::2]
                if min(xs) < 0 or min(ys) < 0 or max(xs) > image_info["width"] or max(ys) > image_info["height"]:
                    out_of_bounds += 1
            class_counts[class_name] += 1
            areas.append(area)
            edges.extend(lengths)
            objects += 1

    return {
        "name": name,
        "image_dir": str(image_dir),
        "label_dir": str(label_dir),
        "num_images": len(image_paths),
        "num_label_files": len(label_paths),
        "num_label_files_checked": len(selected_labels),
        "num_objects": objects,
        "missing_image_stems": sorted(all_label_stems - all_image_stems)[:100],
        "missing_label_stems": sorted(all_image_stems - all_label_stems)[:100],
        "num_missing_image_stems": len(all_label_stems - all_image_stems),
        "num_missing_label_stems": len(all_image_stems - all_label_stems),
        "images_decoded": decoded,
        "decode_errors": decode_errors,
        "class_counts": dict(sorted(class_counts.items())),
        "missing_taxonomy_classes": [name for name in taxonomy if class_counts[name] == 0],
        "unknown_class_counts": {name: count for name, count in class_counts.items() if name not in known},
        "malformed_records": malformed,
        "num_malformed_records": malformed_count,
        "qbox_area": summarize(areas),
        "edge_length": summarize(edges),
        "qboxes_out_of_bounds_for_decoded_images": out_of_bounds,
        "invalid_mmrotate_rboxes": invalid_rboxes,
        "mmrotate_conversion_errors": conversion_errors,
    }


def config_class_check(config_path: Path, taxonomy: list[str]) -> dict[str, Any]:
    namespace = runpy.run_path(str(config_path))
    raw_names = namespace.get("class_name") or namespace.get("metainfo", {}).get("classes")
    names = [normalize_class_name(str(name)) for name in raw_names] if raw_names else None
    return {
        "path": str(config_path),
        "raw_class_names": list(raw_names) if raw_names else None,
        "normalized_class_names": names,
        "same_class_set": set(names or []) == set(taxonomy),
        "exact_taxonomy_order": names == taxonomy,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan FAIR1M DOTA labels, images, geometry, and class order.")
    parser.add_argument("--taxonomy", default=str(DEFAULT_TAXONOMY))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--split", action="append", nargs=3, metavar=("NAME", "IMAGE_DIR", "LABEL_DIR"), required=True)
    parser.add_argument("--max-label-files", type=int, default=None)
    parser.add_argument("--decode-limit", type=int, default=100)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    taxonomy_path = Path(args.taxonomy).resolve()
    config_path = Path(args.config).resolve()
    taxonomy = load_taxonomy(taxonomy_path)
    payload = {
        "taxonomy_path": str(taxonomy_path),
        "taxonomy_class_names": taxonomy,
        "config_class_check": config_class_check(config_path, taxonomy),
        "mmrotate_converter_import_error": load_qbox_converter()[2],
        "splits": [
            scan_split(name, Path(image_dir), Path(label_dir), taxonomy, args.max_label_files, args.decode_limit)
            for name, image_dir, label_dir in args.split
        ],
    }
    output_json = Path(args.output_json).resolve()
    output_md = Path(args.output_md).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = ["# FAIR1M Geometry And Dataloader Gate", "", f"- Taxonomy: `{taxonomy_path}`", f"- Config: `{config_path}`", f"- Config class check: `{payload['config_class_check']}`"]
    for split in payload["splits"]:
        lines.extend(["", f"## {split['name']}", "", "```json", json.dumps(split, indent=2), "```"])
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_json": str(output_json), "output_md": str(output_md), "splits": payload["splits"]}, indent=2))


if __name__ == "__main__":
    main()
