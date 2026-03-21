from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from openprompt_rs.data.base import BaseDetectionDataset


def polygon_to_obb(points: list[tuple[float, float]], width: int, height: int) -> list[float]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    cx = sum(xs) / 4.0 / width
    cy = sum(ys) / 4.0 / height

    edge_a = math.dist(points[0], points[1])
    edge_b = math.dist(points[1], points[2])
    long_edge = max(edge_a, edge_b)
    short_edge = min(edge_a, edge_b)
    if edge_a >= edge_b:
        theta = math.atan2(points[1][1] - points[0][1], points[1][0] - points[0][0])
    else:
        theta = math.atan2(points[2][1] - points[1][1], points[2][0] - points[1][0])
    return [cx, cy, long_edge / width, short_edge / height, theta]


class DotaOBBDataset(BaseDetectionDataset):
    def __init__(self, image_dir: str, label_dir: str, class_names: list[str], image_size: int) -> None:
        super().__init__(class_names=class_names, image_size=image_size)
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        self.image_paths = sorted(
            [
                path
                for path in self.image_dir.iterdir()
                if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".bmp"}
            ]
        )

    def __len__(self) -> int:
        return len(self.image_paths)

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
                if len(parts) < 9:
                    continue
                coords = [float(value) for value in parts[:8]]
                class_name = parts[8]
                if class_name not in self.class_to_idx:
                    continue
                points = [(coords[i], coords[i + 1]) for i in range(0, 8, 2)]
                boxes.append(polygon_to_obb(points, width=width, height=height))
                labels.append(self.class_to_idx[class_name])

        if not boxes:
            return {"boxes": torch.zeros((0, 5), dtype=torch.float32), "labels": torch.zeros((0,), dtype=torch.long)}
        return {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        image_path = self.image_paths[index]
        label_path = self.label_dir / f"{image_path.stem}.txt"
        with Image.open(image_path) as image:
            width, height = image.size
        tensor = self._load_image(image_path)
        target = self._parse_target(label_path, width=width, height=height)
        meta = {"image_path": str(image_path), "label_path": str(label_path)}
        return {"image": tensor, "target": target, "meta": meta}
