from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class BaseDetectionDataset(Dataset):
    def __init__(self, class_names: list[str], image_size: int) -> None:
        self.class_names = class_names
        self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        self.image_size = image_size

    def _load_image(self, path: str | Path) -> torch.Tensor:
        with Image.open(path) as handle:
            image = handle.convert("RGB")
            return self._image_to_tensor(image)

    def _image_to_tensor(self, image: Image.Image) -> torch.Tensor:
        image = image.resize((self.image_size, self.image_size))
        array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1)


def collate_detection_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    images = torch.stack([sample["image"] for sample in batch], dim=0)
    targets = [sample["target"] for sample in batch]
    metadata = [sample["meta"] for sample in batch]
    return {"images": images, "targets": targets, "meta": metadata}
