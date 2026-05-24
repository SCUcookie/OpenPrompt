from __future__ import annotations

import math
import random
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

from openprompt_rs.data.base import BaseDetectionDataset
from openprompt_rs.data.structures import obb_to_polygon


class SyntheticRemoteSensingDataset(BaseDetectionDataset):
    def __init__(
        self,
        class_names: list[str],
        image_size: int,
        num_samples: int,
        max_objects: int,
        seed: int,
    ) -> None:
        super().__init__(class_names=class_names, image_size=image_size)
        self.num_samples = num_samples
        self.max_objects = max_objects
        self.seed = seed
        self.palette = [
            (230, 57, 70),
            (29, 53, 87),
            (69, 123, 157),
            (42, 157, 143),
            (241, 196, 15),
            (188, 108, 37),
            (106, 76, 147),
            (38, 70, 83),
        ]

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> dict[str, Any]:
        rng = random.Random(self.seed + index)
        image = Image.new("RGB", (self.image_size, self.image_size), (240, 240, 240))
        draw = ImageDraw.Draw(image)
        num_objects = rng.randint(1, self.max_objects)

        boxes = []
        labels = []
        scene_tag = []

        for obj_idx in range(num_objects):
            label = rng.randrange(len(self.class_names))
            class_name = self.class_names[label]
            cx = rng.uniform(0.15, 0.85)
            cy = rng.uniform(0.15, 0.85)
            w = rng.uniform(0.10, 0.26)
            h = rng.uniform(0.06, 0.18)
            theta = rng.uniform(-math.pi / 2.0, math.pi / 2.0)

            if class_name in {"ship", "bridge", "large-vehicle", "plane"}:
                w = max(w, h * 1.5)
            if class_name == "storage-tank":
                w = h = rng.uniform(0.09, 0.16)

            polygon = obb_to_polygon(
                cx * self.image_size,
                cy * self.image_size,
                w * self.image_size,
                h * self.image_size,
                theta,
            )
            draw.polygon(polygon, fill=self.palette[obj_idx % len(self.palette)], outline=(30, 30, 30))

            boxes.append([cx, cy, w, h, theta])
            labels.append(label)
            scene_tag.append(class_name)

        tensor = torch.from_numpy(np.asarray(image).astype("float32") / 255.0).permute(2, 0, 1)
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
        meta = {"index": index, "scene_hint": ",".join(scene_tag)}
        return {"image": tensor, "target": target, "meta": meta}
