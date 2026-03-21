from __future__ import annotations

from openprompt_rs.data.dota import DotaOBBDataset
from openprompt_rs.data.synthetic import SyntheticRemoteSensingDataset


def build_dataset(dataset_cfg: dict) -> object:
    dataset_type = dataset_cfg["type"]
    if dataset_type == "synthetic":
        return SyntheticRemoteSensingDataset(
            class_names=dataset_cfg["class_names"],
            image_size=dataset_cfg["image_size"],
            num_samples=dataset_cfg["num_samples"],
            max_objects=dataset_cfg["max_objects"],
            seed=dataset_cfg.get("seed", 7),
        )
    if dataset_type == "dota":
        return DotaOBBDataset(
            image_dir=dataset_cfg["image_dir"],
            label_dir=dataset_cfg["label_dir"],
            class_names=dataset_cfg["class_names"],
            image_size=dataset_cfg["image_size"],
        )
    raise ValueError(f"Unsupported dataset type: {dataset_type}")

