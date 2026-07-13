#!/usr/bin/env python3
"""Reconstruct sanitized DOTA annotations for an archived FAIR1M tile set."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image


CANONICAL_CLASSES = (
    "a220", "a321", "a330", "a350", "arj21", "baseball-field",
    "basketball-court", "boeing737", "boeing747", "boeing777", "boeing787",
    "bridge", "bus", "c919", "cargo-truck", "dry-cargo-ship", "dump-truck",
    "engineering-ship", "excavator", "fishing-boat", "football-field",
    "intersection", "liquid-cargo-ship", "motorboat", "other-airplane",
    "other-ship", "other-vehicle", "passenger-ship", "roundabout", "small-car",
    "tennis-court", "tractor", "trailer", "truck-tractor", "tugboat", "van",
    "warship",
)
KNOWN_CLASSES = frozenset(CANONICAL_CLASSES)
TILE_RE = re.compile(r"^(?P<source>.+)__(?P<size>\d+)__(?P<x>-?\d+)___(?P<y>-?\d+)$")


@dataclass(frozen=True)
class RawObject:
    points: tuple[tuple[float, float], ...]
    class_name: str
    source_line: int


@dataclass(frozen=True)
class Rejection:
    source: str
    line: int | None
    reason: str
    detail: str


def normalize_class_name(name: str) -> str:
    return re.sub(r"[-_\s]+", "-", name.strip().lower())


def polygon_area(points: Iterable[tuple[float, float]]) -> float:
    pts = tuple(points)
    return abs(sum(x * pts[(i + 1) % len(pts)][1] - pts[(i + 1) % len(pts)][0] * y
                   for i, (x, y) in enumerate(pts))) * 0.5


def parse_raw_line(line: str, source: str, line_no: int, width: int, height: int) -> RawObject:
    parts = line.split()
    if len(parts) < 9:
        raise ValueError(f"expected at least 9 fields, got {len(parts)}")
    try:
        coords = tuple(float(value) for value in parts[:8])
    except ValueError as exc:
        raise ValueError("non-numeric coordinate") from exc
    if not all(math.isfinite(value) for value in coords):
        raise ValueError("non-finite coordinate")
    points = tuple(zip(coords[0::2], coords[1::2]))
    class_name = normalize_class_name(parts[8])
    if class_name not in KNOWN_CLASSES:
        raise ValueError(f"unknown class {parts[8]!r}")
    if polygon_area(points) <= 0:
        raise ValueError("zero-area polygon")
    if any(math.hypot(points[(i + 1) % 4][0] - x, points[(i + 1) % 4][1] - y) <= 0
           for i, (x, y) in enumerate(points)):
        raise ValueError("zero-length polygon edge")
    if any(x < 0 or y < 0 or x > width or y > height for x, y in points):
        raise ValueError(f"vertex outside source image {width}x{height}")
    return RawObject(points=points, class_name=class_name, source_line=line_no)


def _clip_edge(points: list[tuple[float, float]], axis: int, bound: float,
               keep_greater: bool) -> list[tuple[float, float]]:
    if not points:
        return []
    output: list[tuple[float, float]] = []
    previous = points[-1]
    previous_inside = previous[axis] >= bound if keep_greater else previous[axis] <= bound
    for current in points:
        current_inside = current[axis] >= bound if keep_greater else current[axis] <= bound
        if current_inside != previous_inside:
            delta = current[axis] - previous[axis]
            ratio = 0.0 if delta == 0 else (bound - previous[axis]) / delta
            intersection = (previous[0] + ratio * (current[0] - previous[0]),
                            previous[1] + ratio * (current[1] - previous[1]))
            output.append(intersection)
        if current_inside:
            output.append(current)
        previous, previous_inside = current, current_inside
    return output


def clip_polygon(points: Iterable[tuple[float, float]], x0: float, y0: float,
                 x1: float, y1: float) -> list[tuple[float, float]]:
    clipped = list(points)
    for axis, bound, keep_greater in ((0, x0, True), (0, x1, False),
                                       (1, y0, True), (1, y1, False)):
        clipped = _clip_edge(clipped, axis, bound, keep_greater)
    return clipped


def reconstruct_tile(objects: Iterable[RawObject], x: int, y: int, size: int,
                     iof_threshold: float = 0.7) -> list[str]:
    lines: list[str] = []
    for obj in objects:
        original_area = polygon_area(obj.points)
        clipped = clip_polygon(obj.points, x, y, x + size, y + size)
        iof = polygon_area(clipped) / original_area if len(clipped) >= 3 else 0.0
        if iof + 1e-12 < iof_threshold:
            continue
        # Match the archived splitter: retain the original quadrilateral and translate it.
        coords = [coordinate for px, py in obj.points for coordinate in (px - x, py - y)]
        difficulty = 0 if math.isclose(iof, 1.0, rel_tol=0.0, abs_tol=1e-9) else 2
        lines.append(" ".join(f"{value:g}" for value in coords) +
                     f" {obj.class_name} {difficulty}")
    return lines


def load_tile_stems(path: Path) -> list[str]:
    stems: set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        value = raw.strip()
        if value:
            stems.add(Path(value).stem)
    return sorted(stems)


def load_tile_image_stems(path: Path) -> list[str]:
    return sorted(image.stem for image in path.iterdir()
                  if image.is_file() and image.suffix.lower() in {".png", ".jpg", ".tif", ".tiff"})


def source_image_path(image_dir: Path, source: str) -> Path | None:
    candidates = (source, source.lstrip("P"), source[2:] if source.startswith("P0") else source)
    for stem in dict.fromkeys(candidates):
        for suffix in (".tif", ".tiff", ".png", ".jpg", ".JPG"):
            candidate = image_dir / f"{stem}{suffix}"
            if candidate.is_file():
                return candidate
    return None


def reconstruct(raw_image_dir: Path, raw_label_dir: Path, tile_stems_path: Path | None,
                tile_image_dir: Path | None,
                output_dir: Path, report_path: Path, iof_threshold: float = 0.7) -> dict:
    tile_stems = (load_tile_stems(tile_stems_path) if tile_stems_path is not None
                  else load_tile_image_stems(tile_image_dir))
    grouped: dict[str, list[tuple[str, int, int, int]]] = defaultdict(list)
    rejections: list[Rejection] = []
    for stem in tile_stems:
        match = TILE_RE.match(stem)
        if not match:
            rejections.append(Rejection(stem, None, "malformed_tile_stem", stem))
            continue
        grouped[match["source"]].append(
            (stem, int(match["size"]), int(match["x"]), int(match["y"])))

    output_dir.mkdir(parents=True, exist_ok=True)
    class_counts: Counter[str] = Counter()
    difficulty_counts: Counter[int] = Counter()
    written_objects = 0
    for source, tiles in sorted(grouped.items()):
        image_path = source_image_path(raw_image_dir, source)
        label_candidates = (raw_label_dir / f"{source}.txt",
                            raw_label_dir / f"{source.lstrip('P')}.txt")
        label_path = next((path for path in label_candidates if path.is_file()), None)
        if image_path is None or label_path is None:
            rejections.append(Rejection(source, None, "missing_source_file",
                                        f"image={image_path}, label={label_path}"))
            objects: list[RawObject] = []
        else:
            with Image.open(image_path) as image:
                width, height = image.size
            objects = []
            for line_no, line in enumerate(label_path.read_text(encoding="utf-8-sig").splitlines(), 1):
                if not line.strip() or line.lower().startswith(("imagesource", "gsd")):
                    continue
                try:
                    obj = parse_raw_line(line, source, line_no, width, height)
                except ValueError as exc:
                    rejections.append(Rejection(source, line_no, "invalid_raw_record", str(exc)))
                else:
                    objects.append(obj)
        for stem, size, x, y in tiles:
            lines = reconstruct_tile(objects, x, y, size, iof_threshold)
            (output_dir / f"{stem}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            written_objects += len(lines)
            for line in lines:
                parts = line.split()
                class_counts[parts[8]] += 1
                difficulty_counts[int(parts[9])] += 1

    report = {
        "raw_image_dir": str(raw_image_dir), "raw_label_dir": str(raw_label_dir),
        "tile_stems": str(tile_stems_path) if tile_stems_path else None,
        "tile_image_dir": str(tile_image_dir) if tile_image_dir else None,
        "output_dir": str(output_dir),
        "iof_threshold": iof_threshold, "num_tile_stems": len(tile_stems),
        "num_annotations_written": len(list(output_dir.glob("*.txt"))),
        "num_objects_written": written_objects,
        "class_counts": dict(sorted(class_counts.items())),
        "difficulty_counts": dict(sorted(difficulty_counts.items())),
        "num_rejections": len(rejections),
        "rejection_reason_counts": dict(Counter(item.reason for item in rejections)),
        "rejections": [asdict(item) for item in rejections],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-image-dir", required=True, type=Path)
    parser.add_argument("--raw-label-dir", required=True, type=Path)
    tile_source = parser.add_mutually_exclusive_group(required=True)
    tile_source.add_argument("--tile-stems", type=Path)
    tile_source.add_argument("--tile-image-dir", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--report-path", required=True, type=Path)
    parser.add_argument("--iof-threshold", type=float, default=0.7)
    args = parser.parse_args()
    if not 0 <= args.iof_threshold <= 1:
        parser.error("--iof-threshold must be between 0 and 1")
    print(json.dumps(reconstruct(args.raw_image_dir, args.raw_label_dir, args.tile_stems,
                                 args.tile_image_dir,
                                 args.output_dir, args.report_path, args.iof_threshold), indent=2))


if __name__ == "__main__":
    main()
