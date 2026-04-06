from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from openprompt_rs.data.base import BaseDetectionDataset


def point_distance(left: tuple[float, float], right: tuple[float, float]) -> float:
    dx = left[0] - right[0]
    dy = left[1] - right[1]
    return math.sqrt(dx * dx + dy * dy)


def polygon_to_obb(points: list[tuple[float, float]], width: int, height: int) -> list[float]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    cx = sum(xs) / 4.0 / width
    cy = sum(ys) / 4.0 / height

    edge_a = point_distance(points[0], points[1])
    edge_b = point_distance(points[1], points[2])
    long_edge = max(edge_a, edge_b)
    short_edge = min(edge_a, edge_b)
    if edge_a >= edge_b:
        theta = math.atan2(points[1][1] - points[0][1], points[1][0] - points[0][0])
    else:
        theta = math.atan2(points[2][1] - points[1][1], points[2][0] - points[1][0])
    return [cx, cy, long_edge / width, short_edge / height, theta]


class DotaOBBDataset(BaseDetectionDataset):
    def __init__(
        self,
        image_dir: str,
        label_dir: str,
        class_names: list[str],
        image_size: int,
        tile_size: int | None = None,
        tile_stride: int | None = None,
        include_empty_tiles: bool = True,
        max_tiles_per_image: int | None = None,
    ) -> None:
        super().__init__(class_names=class_names, image_size=image_size)
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.tile_size = int(tile_size) if tile_size is not None else None
        self.tile_stride = int(tile_stride or tile_size or 0) if tile_size is not None else None
        self.include_empty_tiles = include_empty_tiles
        self.max_tiles_per_image = int(max_tiles_per_image) if max_tiles_per_image is not None else None
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        self.image_paths = sorted(
            [
                path
                for path in self.image_dir.iterdir()
                if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".bmp"}
            ]
        )
        self.samples = self._build_samples()

    def __len__(self) -> int:
        return len(self.samples)

    def _parse_original_dota_line(
        self,
        parts: list[str],
        width: int,
        height: int,
    ) -> tuple[list[float], int] | None:
        if len(parts) < 9:
            return None
        coords = [float(value) for value in parts[:8]]
        class_name = parts[8]
        if class_name not in self.class_to_idx:
            return None
        points = [(coords[i], coords[i + 1]) for i in range(0, 8, 2)]
        return polygon_to_obb(points, width=width, height=height), self.class_to_idx[class_name]

    def _parse_numeric_polygon_line(
        self,
        parts: list[str],
        width: int,
        height: int,
    ) -> tuple[list[float], int] | None:
        if len(parts) < 9:
            return None
        try:
            class_idx = int(parts[0])
            coords = [float(value) for value in parts[1:9]]
        except ValueError:
            return None
        if class_idx < 0 or class_idx >= len(self.class_names):
            return None

        # Converted DOTA labels in this workspace store normalized polygon
        # coordinates, so recover pixel coordinates before OBB conversion.
        points = [(coords[i] * width, coords[i + 1] * height) for i in range(0, 8, 2)]
        return polygon_to_obb(points, width=width, height=height), class_idx

    def _parse_target(self, label_path: Path, width: int, height: int) -> dict[str, torch.Tensor]:
        boxes = []
        labels = []
        if not label_path.exists():
            return {"boxes": torch.zeros((0, 5), dtype=torch.float32), "labels": torch.zeros((0,), dtype=torch.long)}

        with label_path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("imagesource") or line.startswith("gsd"):
                    continue
                parts = line.split()
                parsed = self._parse_original_dota_line(parts, width=width, height=height)
                if parsed is None:
                    parsed = self._parse_numeric_polygon_line(parts, width=width, height=height)
                if parsed is None:
                    continue
                box, label = parsed
                boxes.append(box)
                labels.append(label)

        if not boxes:
            return {"boxes": torch.zeros((0, 5), dtype=torch.float32), "labels": torch.zeros((0,), dtype=torch.long)}
        return {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def _tile_positions(self, length: int) -> list[tuple[int, int]]:
        if self.tile_size is None or self.tile_stride is None or self.tile_size <= 0 or self.tile_stride <= 0:
            return [(0, length)]
        if length <= self.tile_size:
            return [(0, length)]
        if self.tile_stride == self.tile_size:
            starts = list(range(0, length, self.tile_stride))
            return [(start, min(self.tile_size, length - start)) for start in starts]

        starts = list(range(0, length - self.tile_size + 1, self.tile_stride))
        last_start = length - self.tile_size
        if starts[-1] != last_start:
            starts.append(last_start)
        return [(start, min(self.tile_size, length - start)) for start in starts]

    def _select_best_tile(
        self,
        center_x: float,
        center_y: float,
        tile_records: list[dict[str, Any]],
    ) -> int | None:
        best_index = None
        best_key: tuple[float, int, int, int] | None = None

        for tile_index, tile_record in enumerate(tile_records):
            tile_x = int(tile_record["tile_x"])
            tile_y = int(tile_record["tile_y"])
            tile_w = int(tile_record["tile_w"])
            tile_h = int(tile_record["tile_h"])
            if not (tile_x <= center_x < tile_x + tile_w and tile_y <= center_y < tile_y + tile_h):
                continue

            margin = min(
                center_x - tile_x,
                tile_x + tile_w - center_x,
                center_y - tile_y,
                tile_y + tile_h - center_y,
            )
            key = (margin, tile_w * tile_h, -tile_y, -tile_x)
            if best_key is None or key > best_key:
                best_key = key
                best_index = tile_index
        return best_index

    def _build_samples(self) -> list[dict[str, Any]]:
        samples: list[dict[str, Any]] = []
        for image_path in self.image_paths:
            label_path = self.label_dir / f"{image_path.stem}.txt"
            with Image.open(image_path) as image:
                width, height = image.size
            target = self._parse_target(label_path, width=width, height=height)
            base_record = {
                "image_path": image_path,
                "label_path": label_path,
                "width": width,
                "height": height,
                "target": target,
            }
            if self.tile_size is None:
                samples.append(base_record)
                continue
            samples.extend(self._build_tiled_samples(base_record))
        return samples

    def _build_tiled_samples(self, base_record: dict[str, Any]) -> list[dict[str, Any]]:
        width = int(base_record["width"])
        height = int(base_record["height"])
        target = base_record["target"]
        x_positions = self._tile_positions(width)
        y_positions = self._tile_positions(height)

        tile_records: list[dict[str, Any]] = []
        for tile_y, tile_h in y_positions:
            for tile_x, tile_w in x_positions:
                tile_records.append(
                    {
                        **base_record,
                        "tile_x": tile_x,
                        "tile_y": tile_y,
                        "tile_w": tile_w,
                        "tile_h": tile_h,
                        "tile_indices": [],
                        "object_count": 0,
                    }
                )

        if target["boxes"].numel() > 0:
            epsilon = 1e-6
            center_x = (target["boxes"][:, 0].clamp(0.0, 1.0 - epsilon) * width).tolist()
            center_y = (target["boxes"][:, 1].clamp(0.0, 1.0 - epsilon) * height).tolist()
            for object_index, (object_x, object_y) in enumerate(zip(center_x, center_y)):
                tile_index = self._select_best_tile(object_x, object_y, tile_records)
                if tile_index is None:
                    continue
                tile_records[tile_index]["tile_indices"].append(object_index)

        for tile_record in tile_records:
            tile_record["object_count"] = len(tile_record["tile_indices"])
        if not self.include_empty_tiles:
            tile_records = [tile_record for tile_record in tile_records if tile_record["object_count"] > 0]

        if not tile_records:
            tile_records.append(
                {
                    **base_record,
                    "tile_x": 0,
                    "tile_y": 0,
                    "tile_w": width,
                    "tile_h": height,
                    "tile_indices": [],
                    "object_count": 0,
                }
            )

        if self.max_tiles_per_image is not None and len(tile_records) > self.max_tiles_per_image:
            ranked = sorted(
                tile_records,
                key=lambda record: (
                    record["object_count"],
                    -record["tile_h"] * record["tile_w"],
                    -record["tile_y"],
                    -record["tile_x"],
                ),
                reverse=True,
            )
            tile_records = ranked[: self.max_tiles_per_image]
            tile_records = sorted(tile_records, key=lambda record: (record["tile_y"], record["tile_x"]))
        return tile_records

    def _slice_tile_target(self, sample: dict[str, Any]) -> dict[str, torch.Tensor]:
        indices = sample.get("tile_indices")
        target = sample["target"]
        if not indices:
            return {"boxes": torch.zeros((0, 5), dtype=torch.float32), "labels": torch.zeros((0,), dtype=torch.long)}

        boxes = target["boxes"][indices].clone()
        labels = target["labels"][indices].clone()
        width = float(sample["width"])
        height = float(sample["height"])
        tile_x = float(sample["tile_x"])
        tile_y = float(sample["tile_y"])
        tile_w = float(sample["tile_w"])
        tile_h = float(sample["tile_h"])

        boxes[:, 0] = (boxes[:, 0] * width - tile_x) / tile_w
        boxes[:, 1] = (boxes[:, 1] * height - tile_y) / tile_h
        boxes[:, 2] = boxes[:, 2] * width / tile_w
        boxes[:, 3] = boxes[:, 3] * height / tile_h
        boxes[:, 0] = boxes[:, 0].clamp(0.0, 1.0)
        boxes[:, 1] = boxes[:, 1].clamp(0.0, 1.0)
        return {"boxes": boxes, "labels": labels}

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        image_path = sample["image_path"]
        label_path = sample["label_path"]
        if self.tile_size is None:
            tensor = self._load_image(image_path)
            target = sample["target"]
        else:
            with Image.open(image_path) as handle:
                image = handle.convert("RGB")
                image = image.crop(
                    (
                        int(sample["tile_x"]),
                        int(sample["tile_y"]),
                        int(sample["tile_x"] + sample["tile_w"]),
                        int(sample["tile_y"] + sample["tile_h"]),
                    )
                )
                tensor = self._image_to_tensor(image)
            target = self._slice_tile_target(sample)
        meta = {
            "image_path": str(image_path),
            "label_path": str(label_path),
            "source_width": int(sample["width"]),
            "source_height": int(sample["height"]),
        }
        if self.tile_size is not None:
            meta.update(
                {
                    "tile_x": int(sample["tile_x"]),
                    "tile_y": int(sample["tile_y"]),
                    "tile_w": int(sample["tile_w"]),
                    "tile_h": int(sample["tile_h"]),
                    "tile_object_count": int(sample["object_count"]),
                }
            )
        return {"image": tensor, "target": target, "meta": meta}
