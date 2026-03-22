from __future__ import annotations

from collections import defaultdict

import torch

from openprompt_rs.models.losses import build_supervision_targets


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    relation_matrix: torch.Tensor | None,
    confusing_matrix: torch.Tensor | None,
    device: str,
) -> dict[str, float]:
    model.eval()
    meters: dict[str, float] = defaultdict(float)
    steps = 0
    positive_correct = 0.0
    positive_total = 0.0
    box_l1_total = 0.0

    for batch in dataloader:
        images = batch["images"].to(device)
        targets = batch["targets"]
        outputs = model(images)
        losses = criterion(
            outputs,
            targets,
            relation_matrix=relation_matrix,
            confusing_matrix=confusing_matrix,
        )
        supervision = build_supervision_targets(outputs["query_centers"], targets, outputs["logits"].size(-1))
        predictions = outputs["logits"].argmax(dim=-1)
        mask = supervision["positive_mask"]

        if mask.any():
            positive_correct += (
                predictions[mask] == supervision["label_indices"][mask]
            ).float().sum().item()
            positive_total += float(mask.sum().item())
            box_l1_total += torch.abs(outputs["boxes"][mask] - supervision["box_targets"][mask]).mean().item()

        for key, value in losses.items():
            meters[key] += float(value.item())
        steps += 1

    if steps == 0:
        return {"loss": 0.0, "positive_cls_acc": 0.0, "positive_box_l1": 0.0}

    metrics = {key: value / steps for key, value in meters.items()}
    metrics["positive_cls_acc"] = positive_correct / max(positive_total, 1.0)
    metrics["positive_box_l1"] = box_l1_total / max(steps, 1)
    return metrics
