from __future__ import annotations

from dataclasses import dataclass, field

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
    # Class-adaptive thresholds for tail classes
    use_class_adaptive: bool = True
    class_counts: dict[str, int] = field(default_factory=dict)
    tail_class_ratio: float = 0.05
    head_class_ratio: float = 1.5
    min_threshold: float = 0.25


class HierarchyConsistentPseudoLabeler:
    def __init__(self, hierarchy: HierarchyGraph, config: PseudoLabelConfig) -> None:
        self.hierarchy = hierarchy
        self.config = config
        self._class_thresholds = None
        if config.use_class_adaptive and config.class_counts:
            self._class_thresholds = self._compute_class_adaptive_thresholds()

    def _compute_class_adaptive_thresholds(self) -> dict[str, float]:
        counts = self.config.class_counts
        if not counts:
            return {}
        total = sum(counts.values())
        thresholds = {}
        for cls_name, cnt in counts.items():
            ratio = cnt / max(total, 1)
            if ratio < 0.02:
                t = self.config.score_threshold * self.config.tail_class_ratio
            elif ratio < 0.05:
                t = self.config.score_threshold * 0.7
            elif ratio > 0.15:
                t = self.config.score_threshold * self.config.head_class_ratio
            else:
                t = self.config.score_threshold
            thresholds[cls_name] = max(t, self.config.min_threshold)
        return thresholds

    def _get_class_threshold(self, class_idx: int) -> float:
        if self._class_thresholds is None:
            return self.config.score_threshold
        cls_name = self.hierarchy.class_names[class_idx]
        return self._class_thresholds.get(cls_name, self.config.score_threshold)

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
        if prompt_embeddings.dim() == 2:
            gathered_prompts = prompt_embeddings[labels]
        else:
            gathered_prompts = torch.gather(
                prompt_embeddings,
                dim=1,
                index=labels.unsqueeze(-1).expand(-1, -1, prompt_embeddings.size(-1)),
            )
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

        # Class-adaptive score thresholds
        if self._class_thresholds is not None:
            class_thresh = torch.tensor(
                [self._get_class_threshold(i)
                 for i in range(len(self.hierarchy.class_names))],
                device=confidence.device, dtype=confidence.dtype,
            )
            score_thresholds = class_thresh[labels]
        else:
            score_thresholds = self.config.score_threshold

        keep = (
            (confidence >= score_thresholds)
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

    def compute_loss_weights(self, pseudo_scores: torch.Tensor) -> torch.Tensor:
        return torch.clamp(pseudo_scores / self.config.final_threshold, 0.1, 1.0)
