from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from openprompt_rs.models.backbone import QueryGenerator, TinyBackbone
from openprompt_rs.models.context_adapter import SceneContextAdapter
from openprompt_rs.models.heads import AlignmentHead, FusionHead
from openprompt_rs.models.prompt_bank import PromptBank
from openprompt_rs.models.routing import ScaleRotationRouter


class OpenRSDLikeDetector(nn.Module):
    def __init__(self, prompt_bank: PromptBank, model_cfg: dict[str, Any]) -> None:
        super().__init__()
        embedding_dim = model_cfg["embedding_dim"]
        self.prompt_bank = prompt_bank
        self.backbone = TinyBackbone(output_dim=model_cfg["backbone_dim"])
        self.query_generator = QueryGenerator(feature_dim=model_cfg["backbone_dim"], grid_size=model_cfg["grid_size"])
        self.alignment_head = AlignmentHead(embedding_dim=embedding_dim)
        self.fusion_head = FusionHead(embedding_dim=embedding_dim)
        self.alignment_weight = model_cfg["alignment_weight"]
        self.fusion_weight = model_cfg["fusion_weight"]

    def _merge_outputs(self, alignment: dict[str, torch.Tensor], fusion: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        logits = self.alignment_weight * alignment["logits"] + self.fusion_weight * fusion["logits"]
        boxes = self.alignment_weight * alignment["boxes"] + self.fusion_weight * fusion["boxes"]
        query_embeddings = self.alignment_weight * alignment["query_embeddings"] + self.fusion_weight * fusion["query_embeddings"]
        return {"logits": logits, "boxes": boxes, "query_embeddings": query_embeddings}

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        feature_map = self.backbone(images)
        query_tokens, scene_feature, query_centers = self.query_generator(feature_map)
        prompt_embeddings = self.prompt_bank()

        alignment = self.alignment_head(query_tokens, prompt_embeddings)
        fusion = self.fusion_head(query_tokens, prompt_embeddings)
        merged = self._merge_outputs(alignment, fusion)

        return {
            **merged,
            "alignment_logits": alignment["logits"],
            "fusion_logits": fusion["logits"],
            "query_centers": query_centers,
            "scene_feature": scene_feature,
            "scene_scores": None,
            "prompt_embeddings": prompt_embeddings,
        }


class GeoNexusRSDDetector(OpenRSDLikeDetector):
    def __init__(self, prompt_bank: PromptBank, model_cfg: dict[str, Any]) -> None:
        super().__init__(prompt_bank=prompt_bank, model_cfg=model_cfg)
        embedding_dim = model_cfg["embedding_dim"]
        num_classes = len(prompt_bank.class_names)
        self.scene_adapter = SceneContextAdapter(embedding_dim=embedding_dim, num_classes=num_classes)
        self.router = ScaleRotationRouter(
            embedding_dim=embedding_dim,
            hidden_dim=model_cfg.get("router_hidden_dim", embedding_dim),
        )
        self.use_router = model_cfg.get("use_router", True)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        feature_map = self.backbone(images)
        query_tokens, scene_feature, query_centers = self.query_generator(feature_map)
        prompt_embeddings = self.prompt_bank()
        adapted_prompts, scene_scores = self.scene_adapter(prompt_embeddings, scene_feature)

        alignment = self.alignment_head(query_tokens, adapted_prompts)
        fusion = self.fusion_head(query_tokens, adapted_prompts)

        if self.use_router:
            route = self.router(query_tokens, alignment["logits"], alignment["boxes"])
            logits = (1.0 - route) * alignment["logits"] + route * fusion["logits"]
            boxes = (1.0 - route) * alignment["boxes"] + route * fusion["boxes"]
            query_embeddings = (1.0 - route) * alignment["query_embeddings"] + route * fusion["query_embeddings"]
        else:
            route = None
            merged = self._merge_outputs(alignment, fusion)
            logits = merged["logits"]
            boxes = merged["boxes"]
            query_embeddings = merged["query_embeddings"]

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
            "route": route,
        }


def build_model(model_cfg: dict[str, Any], prompt_bank: PromptBank) -> nn.Module:
    model_type = model_cfg["type"]
    if model_type == "baseline":
        return OpenRSDLikeDetector(prompt_bank=prompt_bank, model_cfg=model_cfg)
    if model_type == "geonexus":
        return GeoNexusRSDDetector(prompt_bank=prompt_bank, model_cfg=model_cfg)
    raise ValueError(f"Unsupported model type: {model_type}")

