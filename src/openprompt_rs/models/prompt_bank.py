from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from openprompt_rs.models.hierarchy import HierarchyGraph
from openprompt_rs.utils.embeddings import HashTextEmbedder
from openprompt_rs.utils.io import load_json


class PromptBank(nn.Module):
    def __init__(
        self,
        class_names: list[str],
        base_embeddings: torch.Tensor,
        relation_matrix: torch.Tensor,
        prompt_strings: dict[str, list[str]],
        hierarchy: HierarchyGraph,
        hierarchy_lambda: float = 0.1,
        use_class_offsets: bool = True,
    ) -> None:
        super().__init__()
        self.class_names = class_names
        self.prompt_strings = prompt_strings
        self.hierarchy = hierarchy
        self.hierarchy_lambda = hierarchy_lambda
        self.register_buffer("base_embeddings", F.normalize(base_embeddings, dim=-1))
        self.register_buffer("relation_matrix", relation_matrix)
        if use_class_offsets:
            self.class_offsets = nn.Parameter(torch.zeros_like(base_embeddings))
        else:
            self.register_parameter("class_offsets", None)

    @classmethod
    def build_from_files(
        cls,
        taxonomy_path: str | Path,
        template_path: str | Path,
        embedding_dim: int,
        class_names: list[str] | None = None,
        hierarchy_lambda: float = 0.1,
        use_class_offsets: bool = True,
    ) -> "PromptBank":
        hierarchy = HierarchyGraph.from_json(taxonomy_path, class_names=class_names)
        templates = load_json(template_path)["templates"]
        prompt_strings = cls._build_prompt_strings(hierarchy, templates)
        embedder = HashTextEmbedder(embedding_dim=embedding_dim)
        embeddings = []
        for class_name in hierarchy.class_names:
            encoded = embedder.embed_texts(prompt_strings[class_name])
            embeddings.append(encoded.mean(dim=0))
        embedding_tensor = torch.stack(embeddings, dim=0)
        return cls(
            class_names=hierarchy.class_names,
            base_embeddings=embedding_tensor,
            relation_matrix=hierarchy.relation_matrix,
            prompt_strings=prompt_strings,
            hierarchy=hierarchy,
            hierarchy_lambda=hierarchy_lambda,
            use_class_offsets=use_class_offsets,
        )

    @staticmethod
    def _build_prompt_strings(hierarchy: HierarchyGraph, templates: list[str]) -> dict[str, list[str]]:
        prompt_strings: dict[str, list[str]] = {}
        for class_name in hierarchy.class_names:
            meta = hierarchy.metadata[class_name]
            geometry = ", ".join(meta.geometry) if meta.geometry else "none"
            scene_priors = ", ".join(meta.scene_priors) if meta.scene_priors else "none"
            text_bank = []
            names = [class_name, *meta.synonyms]
            for name_variant in names:
                for template in templates:
                    text_bank.append(
                        template.format(name=name_variant, geometry=geometry, scene_priors=scene_priors)
                    )
            prompt_strings[class_name] = text_bank
        return prompt_strings

    def forward(self) -> torch.Tensor:
        embeddings = self.base_embeddings
        if self.class_offsets is not None:
            embeddings = embeddings + self.class_offsets
        embeddings = embeddings + self.hierarchy_lambda * self.relation_matrix @ embeddings
        return F.normalize(embeddings, dim=-1)

    def export_artifact(self) -> dict[str, Any]:
        return {
            "class_names": self.class_names,
            "prompt_strings": self.prompt_strings,
            "embeddings": self().detach().cpu(),
            "relation_matrix": self.relation_matrix.detach().cpu(),
        }

