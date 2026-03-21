from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from openprompt_rs.models.hierarchy import HierarchyGraph


@dataclass
class PseudoLabelConfig:
    score_threshold: float = 0.55
    semantic_threshold: float = 0.50
    scene_threshold: float = 0.40
    hierarchy_bonus: float = 0.10
    negative_penalty: float = 0.15
    final_threshold: float = 0.60


class HierarchyConsistentPseudoLabeler:
    def __init__(self, hierarchy: HierarchyGraph, config: PseudoLabelConfig) -> None:
        self.hierarchy = hierarchy
        self.config = config

    @torch.no_grad()
    def filter(
        self,
        outputs: dict[str, torch.Tensor],
        prompt_embeddings: torch.Tensor,
        scene_scores: torch.Tensor | None = None,
    ) -> list[dict[str, torch.Tensor]]:
        probabilities = torch.sigmoid(outputs["logits"])
        confidence, labels = probabilities.max(dim=-1)
        query_embeddings = F.normalize(outputs["query_embeddings"], dim=-1)
        prompt_embeddings = F.normalize(prompt_embeddings, dim=-1)
        gathered_prompts = prompt_embeddings[labels]
        semantic = (query_embeddings * gathered_prompts).sum(dim=-1)

        if scene_scores is None:
            scene = torch.full_like(confidence, 0.5)
        else:
            scene = torch.gather(torch.sigmoid(scene_scores), dim=-1, index=labels)

        relation = self.hierarchy.relation_bonus(labels, probabilities)
        confusing = self.hierarchy.confusing_penalty(labels, probabilities)
        composite = (
            0.45 * confidence
            + 0.20 * semantic
            + 0.15 * scene
            + self.config.hierarchy_bonus * relation
            - self.config.negative_penalty * confusing
        )

        keep = (
            (confidence >= self.config.score_threshold)
            & (semantic >= self.config.semantic_threshold)
            & (scene >= self.config.scene_threshold)
            & (composite >= self.config.final_threshold)
        )

        pseudo_targets = []
        for batch_idx in range(outputs["boxes"].size(0)):
            batch_keep = keep[batch_idx]
            pseudo_targets.append(
                {
                    "boxes": outputs["boxes"][batch_idx][batch_keep].detach(),
                    "labels": labels[batch_idx][batch_keep].detach(),
                    "score": composite[batch_idx][batch_keep].detach(),
                }
            )
        return pseudo_targets

