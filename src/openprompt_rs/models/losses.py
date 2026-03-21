from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_supervision_targets(
    query_centers: torch.Tensor,
    targets: list[dict[str, torch.Tensor]],
    num_classes: int,
) -> dict[str, torch.Tensor]:
    batch_size, num_queries, _ = query_centers.shape
    cls_targets = torch.zeros((batch_size, num_queries, num_classes), device=query_centers.device)
    box_targets = torch.zeros((batch_size, num_queries, 5), device=query_centers.device)
    positive_mask = torch.zeros((batch_size, num_queries), dtype=torch.bool, device=query_centers.device)
    label_indices = torch.full((batch_size, num_queries), -1, dtype=torch.long, device=query_centers.device)

    for batch_idx, target in enumerate(targets):
        if target["boxes"].numel() == 0:
            continue
        boxes = target["boxes"].to(query_centers.device)
        labels = target["labels"].to(query_centers.device)
        distances = torch.cdist(boxes[:, :2], query_centers[batch_idx], p=2)
        used_queries: set[int] = set()
        ordering = distances.min(dim=1).values.argsort()
        for gt_idx in ordering.tolist():
            for query_idx in distances[gt_idx].argsort().tolist():
                if query_idx not in used_queries:
                    used_queries.add(query_idx)
                    label = int(labels[gt_idx].item())
                    cls_targets[batch_idx, query_idx, label] = 1.0
                    box_targets[batch_idx, query_idx] = boxes[gt_idx]
                    positive_mask[batch_idx, query_idx] = True
                    label_indices[batch_idx, query_idx] = label
                    break

    return {
        "cls_targets": cls_targets,
        "box_targets": box_targets,
        "positive_mask": positive_mask,
        "label_indices": label_indices,
    }


def sigmoid_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float,
    gamma: float,
) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = probabilities * targets + (1.0 - probabilities) * (1.0 - targets)
    alpha_factor = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    modulating = (1.0 - p_t).pow(gamma)
    return (alpha_factor * modulating * ce_loss).mean()


def hierarchy_laplacian_loss(logits: torch.Tensor, relation_matrix: torch.Tensor) -> torch.Tensor:
    probabilities = torch.sigmoid(logits).mean(dim=1)
    adjacency = relation_matrix.to(logits.device)
    degree = torch.diag(adjacency.sum(dim=-1))
    laplacian = degree - adjacency
    return torch.einsum("bc,cd,bd->b", probabilities, laplacian, probabilities).mean()


class OpenPromptCriterion(nn.Module):
    def __init__(
        self,
        cls_weight: float,
        box_weight: float,
        hierarchy_weight: float,
        focal_alpha: float,
        focal_gamma: float,
    ) -> None:
        super().__init__()
        self.cls_weight = cls_weight
        self.box_weight = box_weight
        self.hierarchy_weight = hierarchy_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
        relation_matrix: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        supervision = build_supervision_targets(
            query_centers=outputs["query_centers"],
            targets=targets,
            num_classes=outputs["logits"].size(-1),
        )
        cls_loss = sigmoid_focal_loss(
            outputs["logits"],
            supervision["cls_targets"],
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
        )

        positive_mask = supervision["positive_mask"]
        if positive_mask.any():
            box_loss = F.smooth_l1_loss(
                outputs["boxes"][positive_mask],
                supervision["box_targets"][positive_mask],
                reduction="mean",
            )
        else:
            box_loss = outputs["boxes"].sum() * 0.0

        hierarchy_loss = outputs["logits"].sum() * 0.0
        if relation_matrix is not None and self.hierarchy_weight > 0.0:
            hierarchy_loss = hierarchy_laplacian_loss(outputs["logits"], relation_matrix)

        total = self.cls_weight * cls_loss + self.box_weight * box_loss + self.hierarchy_weight * hierarchy_loss
        return {
            "loss": total,
            "loss_cls": cls_loss.detach(),
            "loss_box": box_loss.detach(),
            "loss_hier": hierarchy_loss.detach(),
        }

