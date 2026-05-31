#!/usr/bin/env python3
"""Build DOTA2 tiled annfiles from original DOTA labelTxt and tile filenames."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import zipfile


def parse_tile_name(path: Path) -> tuple[str, int, int, int]:
    stem = path.stem
    original, size, rest = stem.split("__", 2)
    x_str, y_str = rest.split("___", 1)
    return original, int(size), int(x_str), int(y_str)


def parse_label_line(line: str):
    parts = line.strip().split()
    if len(parts) < 9:
        return None
    try:
        coords = [float(v) for v in parts[:8]]
    except ValueError:
        return None
    label = parts[8]
    difficult = parts[9] if len(parts) > 9 else "0"
    return coords, label, difficult


def polygon_area(poly: list[tuple[float, float]]) -> float:
    if len(poly) < 3:
        return 0.0
    acc = 0.0
    for i, (x1, y1) in enumerate(poly):
        x2, y2 = poly[(i + 1) % len(poly)]
        acc += x1 * y2 - x2 * y1
    return abs(acc) * 0.5


def clip_polygon(poly: list[tuple[float, float]], xmin: float, ymin: float,
                 xmax: float, ymax: float) -> list[tuple[float, float]]:
    def clip_edge(points, inside, intersect):
        if not points:
            return []
        out = []
        prev = points[-1]
        prev_inside = inside(prev)
        for cur in points:
            cur_inside = inside(cur)
            if cur_inside:
                if not prev_inside:
                    out.append(intersect(prev, cur))
                out.append(cur)
            elif prev_inside:
                out.append(intersect(prev, cur))
            prev, prev_inside = cur, cur_inside
        return out

    def intersect_x(x):
        def fn(p1, p2):
            x1, y1 = p1
            x2, y2 = p2
            if x2 == x1:
                return (x, y1)
            t = (x - x1) / (x2 - x1)
            return (x, y1 + t * (y2 - y1))
        return fn

    def intersect_y(y):
        def fn(p1, p2):
            x1, y1 = p1
            x2, y2 = p2
            if y2 == y1:
                return (x1, y)
            t = (y - y1) / (y2 - y1)
            return (x1 + t * (x2 - x1), y)
        return fn

    poly = clip_edge(poly, lambda p: p[0] >= xmin, intersect_x(xmin))
    poly = clip_edge(poly, lambda p: p[0] <= xmax, intersect_x(xmax))
    poly = clip_edge(poly, lambda p: p[1] >= ymin, intersect_y(ymin))
    poly = clip_edge(poly, lambda p: p[1] <= ymax, intersect_y(ymax))
    return poly


def format_num(v: float) -> str:
    if abs(v) < 1e-6:
        v = 0.0
    return f"{v:.1f}"


def load_labels(label_zip: Path) -> dict[str, list[tuple[list[float], str, str]]]:
    labels = {}
    with zipfile.ZipFile(label_zip) as zf:
        for name in zf.namelist():
            if not name.endswith(".txt") or name.endswith("/"):
                continue
            image_id = Path(name).stem
            entries = []
            for raw in zf.read(name).decode("utf-8-sig").splitlines():
                parsed = parse_label_line(raw)
                if parsed is not None:
                    entries.append(parsed)
            labels[image_id] = entries
    return labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-zip", required=True, type=Path)
    parser.add_argument("--tile-image-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--iof-thr", type=float, default=0.7)
    args = parser.parse_args()

    labels = load_labels(args.label_zip)
    tiles_by_original: dict[str, list[tuple[Path, int, int, int]]] = defaultdict(list)
    for img_path in args.tile_image_dir.glob("*.png"):
        original, size, x_start, y_start = parse_tile_name(img_path)
        tiles_by_original[original].append((img_path, size, x_start, y_start))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    total_tiles = 0
    total_objects = 0
    missing_label_images = 0

    for original, tiles in sorted(tiles_by_original.items()):
        objects = labels.get(original, [])
        if not objects:
            missing_label_images += 1
        for img_path, size, x_start, y_start in tiles:
            xmin, ymin = float(x_start), float(y_start)
            xmax, ymax = xmin + size, ymin + size
            lines = []
            for coords, label, difficult in objects:
                poly = list(zip(coords[0::2], coords[1::2]))
                area = polygon_area(poly)
                if area <= 0:
                    continue
                clipped = clip_polygon(poly, xmin, ymin, xmax, ymax)
                if polygon_area(clipped) / area < args.iof_thr:
                    continue
                translated = []
                for x, y in poly:
                    translated.extend([x - xmin, y - ymin])
                diff = "2" if polygon_area(clipped) / area < 1.0 - 1e-6 else difficult
                lines.append(" ".join(format_num(v) for v in translated) + f" {label} {diff}")
            (args.out_dir / f"{img_path.stem}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            total_tiles += 1
            total_objects += len(lines)

    print(f"label_images={len(labels)}")
    print(f"tile_original_images={len(tiles_by_original)}")
    print(f"missing_label_images={missing_label_images}")
    print(f"total_tiles={total_tiles}")
    print(f"total_objects={total_objects}")
    print(f"out_dir={args.out_dir}")


if __name__ == "__main__":
    main()
