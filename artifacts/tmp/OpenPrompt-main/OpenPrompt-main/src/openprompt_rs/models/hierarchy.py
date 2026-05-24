from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from openprompt_rs.utils.io import load_json


@dataclass
class ClassMetadata:
    name: str
    parent: str
    synonyms: list[str]
    confusing_classes: list[str]
    scene_priors: list[str]
    geometry: list[str]
    negative_cues: list[str]


class HierarchyGraph:
    def __init__(self, metadata: dict[str, ClassMetadata], class_names: list[str]) -> None:
        self.metadata = metadata
        self.class_names = class_names
        self.class_to_index = {name: idx for idx, name in enumerate(class_names)}
        self.relation_matrix = self._build_relation_matrix()
        self.confusing_matrix = self._build_confusing_matrix()

    @classmethod
    def from_json(cls, path: str | Path, class_names: list[str] | None = None) -> "HierarchyGraph":
        raw = load_json(path)
        metadata = {
            entry["name"]: ClassMetadata(
                name=entry["name"],
                parent=entry.get("parent", ""),
                synonyms=entry.get("synonyms", []),
                confusing_classes=entry.get("confusing_classes", []),
                scene_priors=entry.get("scene_priors", []),
                geometry=entry.get("geometry", []),
                negative_cues=entry.get("negative_cues", []),
            )
            for entry in raw["classes"]
        }
        ordered_names = class_names or list(metadata.keys())
        metadata = {name: metadata[name] for name in ordered_names if name in metadata}
        return cls(metadata=metadata, class_names=list(metadata.keys()))

    def _build_relation_matrix(self) -> torch.Tensor:
        num_classes = len(self.class_names)
        matrix = torch.eye(num_classes, dtype=torch.float32)
        for left_name, left_idx in self.class_to_index.items():
            left_meta = self.metadata[left_name]
            left_scene = set(left_meta.scene_priors)
            for right_name, right_idx in self.class_to_index.items():
                if left_idx == right_idx:
                    continue
                right_meta = self.metadata[right_name]
                weight = 0.0
                if left_meta.parent and left_meta.parent == right_meta.parent:
                    weight += 0.35
                if right_name in left_meta.confusing_classes or left_name in right_meta.confusing_classes:
                    weight += 0.25
                if left_scene and set(right_meta.scene_priors) & left_scene:
                    weight += 0.15
                if weight > 0.0:
                    matrix[left_idx, right_idx] = weight
        row_sums = matrix.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return matrix / row_sums

    def _build_confusing_matrix(self) -> torch.Tensor:
        num_classes = len(self.class_names)
        matrix = torch.zeros((num_classes, num_classes), dtype=torch.float32)
        for name, idx in self.class_to_index.items():
            for confusing_name in self.metadata[name].confusing_classes:
                if confusing_name in self.class_to_index:
                    matrix[idx, self.class_to_index[confusing_name]] = 1.0
        return matrix

    def relation_bonus(self, label_indices: torch.Tensor, probabilities: torch.Tensor) -> torch.Tensor:
        relations = self.relation_matrix.to(probabilities.device)[label_indices]
        return (relations * probabilities).sum(dim=-1)

    def confusing_penalty(self, label_indices: torch.Tensor, probabilities: torch.Tensor) -> torch.Tensor:
        confusing = self.confusing_matrix.to(probabilities.device)[label_indices]
        return (confusing * probabilities).sum(dim=-1)

