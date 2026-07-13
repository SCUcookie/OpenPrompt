#!/usr/bin/env python3
"""Normalize and sanitize a FAIR1M DOTA-format split non-destructively."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

from PIL import Image


RECONSTRUCT = Path(__file__).with_name("reconstruct_fair1m_tiled_annotations.py")
SPEC = importlib.util.spec_from_file_location("fair_reconstruct", RECONSTRUCT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def find_image(image_dir: Path, stem: str) -> Path | None:
    for suffix in (".png", ".jpg", ".JPG", ".tif", ".tiff"):
        path = image_dir / f"{stem}{suffix}"
        if path.is_file():
            return path
    return None


def sanitize(image_dir: Path, input_dir: Path, output_dir: Path, report_path: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = [path for path in image_dir.iterdir() if path.is_file()]
    image_stems = {path.stem for path in image_paths}
    label_paths = sorted(input_dir.glob("*.txt"))
    label_stems = {path.stem for path in label_paths}
    rejections = []
    class_counts: Counter[str] = Counter()
    objects_written = 0
    for stem in sorted(image_stems | label_stems):
        image_path = find_image(image_dir, stem)
        label_path = input_dir / f"{stem}.txt"
        lines = []
        if image_path is None or not label_path.is_file():
            rejections.append({"source": stem, "line": None, "reason": "missing_pair",
                               "detail": f"image={image_path}, label_exists={label_path.is_file()}"})
        else:
            with Image.open(image_path) as image:
                width, height = image.size
            for line_no, line in enumerate(label_path.read_text(encoding="utf-8-sig").splitlines(), 1):
                if not line.strip() or line.lower().startswith(("imagesource", "gsd")):
                    continue
                try:
                    obj = MODULE.parse_raw_line(line, stem, line_no, width, height)
                except ValueError as exc:
                    rejections.append({"source": stem, "line": line_no,
                                       "reason": "invalid_record", "detail": str(exc)})
                    continue
                coords = [coordinate for point in obj.points for coordinate in point]
                difficulty = int(line.split()[9]) if len(line.split()) > 9 else 0
                lines.append(" ".join(f"{value:g}" for value in coords) +
                             f" {obj.class_name} {difficulty}")
                class_counts[obj.class_name] += 1
        (output_dir / f"{stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        objects_written += len(lines)
    report = {
        "image_dir": str(image_dir), "input_dir": str(input_dir), "output_dir": str(output_dir),
        "num_images": len(image_stems), "num_input_annotations": len(label_stems),
        "num_output_annotations": len(list(output_dir.glob("*.txt"))),
        "num_objects_written": objects_written, "class_counts": dict(sorted(class_counts.items())),
        "num_rejections": len(rejections),
        "rejection_reason_counts": dict(Counter(item["reason"] for item in rejections)),
        "rejections": rejections,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--report-path", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(sanitize(args.image_dir, args.input_dir, args.output_dir, args.report_path), indent=2))


if __name__ == "__main__":
    main()
