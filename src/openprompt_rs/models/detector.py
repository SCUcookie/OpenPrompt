from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from openprompt_rs.models.backbone import QueryGenerator, TinyBackbone
from openprompt_rs.models.heads import AlignmentHead, FusionHead
from openprompt_rs.models.innovations import build_innovation_modules
from openprompt_rs.models.prompt_bank import PromptBank


class ModularPromptDetector(nn.Module):
    def __init__(self, prompt_bank: PromptBank, model_cfg: dict[str, Any]) -> None:
        super().__init__()
        embedding_dim = model_cfg["embedding_dim"]
        self.prompt_bank = prompt_bank
        self.backbone = TinyBackbone(output_dim=model_cfg["backbone_dim"])
        self.query_generator = QueryGenerator(feature_dim=model_cfg["backbone_dim"], grid_size=model_cfg["grid_size"])
        self.alignment_head = AlignmentHead(embedding_dim=embedding_dim)
        self.fusion_head = FusionHead(embedding_dim=embedding_dim)
        self.alignment_weight = float(model_cfg["alignment_weight"])
        self.fusion_weight = float(model_cfg["fusion_weight"])
        self.innovation_cfg, self.innovations = build_innovation_modules(
            model_cfg=model_cfg,
            num_classes=len(prompt_bank.class_names),
        )

    def _merge_outputs(self, alignment: dict[str, torch.Tensor], fusion: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        logits = self.alignment_weight * alignment["logits"] + self.fusion_weight * fusion["logits"]
        boxes = self.alignment_weight * alignment["boxes"] + self.fusion_weight * fusion["boxes"]
        query_embeddings = self.alignment_weight * alignment["query_embeddings"] + self.fusion_weight * fusion["query_embeddings"]
        return {"logits": logits, "boxes": boxes, "query_embeddings": query_embeddings}

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        feature_map = self.backbone(images)
        query_tokens, scene_feature, query_centers = self.query_generator(feature_map)

        base_prompt_embeddings = self.prompt_bank()
        prompt_embeddings = base_prompt_embeddings
        scene_scores = None

        if "scene_adapter" in self.innovations:
            prompt_embeddings, scene_scores = self.innovations["scene_adapter"](prompt_embeddings, scene_feature)

        alignment = self.alignment_head(query_tokens, prompt_embeddings)
        fusion = self.fusion_head(query_tokens, prompt_embeddings)

        if "router" in self.innovations:
            route = self.innovations["router"](query_tokens, alignment["logits"], alignment["boxes"])
            logits = (1.0 - route) * alignment["logits"] + route * fusion["logits"]
            boxes = (1.0 - route) * alignment["boxes"] + route * fusion["boxes"]
            query_embeddings = (1.0 - route) * alignment["query_embeddings"] + route * fusion["query_embeddings"]
        else:
            route = None
            merged = self._merge_outputs(alignment, fusion)
            logits = merged["logits"]
            boxes = merged["boxes"]
            query_embeddings = merged["query_embeddings"]

        scene_temperature = None
        if "scene_temperature" in self.innovations:
            logits, scene_temperature = self.innovations["scene_temperature"](logits, scene_feature)

        return {
            "logits": logits,
            "boxes": boxes,
            "query_embeddings": query_embeddings,
            "alignment_logits": alignment["logits"],
            "fusion_logits": fusion["logits"],
            "query_centers": query_centers,
            "scene_feature": scene_feature,
            "scene_scores": scene_scores,
            "prompt_embeddings": prompt_embeddings,
            "base_prompt_embeddings": base_prompt_embeddings,
            "route": route,
            "scene_temperature": scene_temperature,
        }


def build_model(model_cfg: dict[str, Any], prompt_bank: PromptBank) -> nn.Module:
    model_type = model_cfg["type"]
    if model_type in {"baseline", "geonexus", "modular"}:
        return ModularPromptDetector(prompt_bank=prompt_bank, model_cfg=model_cfg)
    raise ValueError(f"Unsupported model type: {model_type}")
