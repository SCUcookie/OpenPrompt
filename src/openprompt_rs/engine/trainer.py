from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from openprompt_rs.data.base import collate_detection_batch
from openprompt_rs.engine.evaluator import evaluate_model
from openprompt_rs.models.losses import OpenPromptCriterion
from openprompt_rs.utils.io import ensure_dir


def build_dataloader(dataset: object, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_detection_batch)


def build_criterion(criterion_cfg: dict[str, Any]) -> OpenPromptCriterion:
    return OpenPromptCriterion(
        cls_weight=criterion_cfg["cls_weight"],
        box_weight=criterion_cfg["box_weight"],
        hierarchy_weight=criterion_cfg["hierarchy_weight"],
        focal_alpha=criterion_cfg["focal_alpha"],
        focal_gamma=criterion_cfg["focal_gamma"],
    )


def train_experiment(
    model: torch.nn.Module,
    train_dataset: object,
    experiment_cfg: dict[str, Any],
    criterion_cfg: dict[str, Any],
    relation_matrix: torch.Tensor | None,
    output_dir: str | Path,
) -> dict[str, float]:
    device = experiment_cfg["device"]
    batch_size = experiment_cfg["batch_size"]
    epochs = experiment_cfg["epochs"]
    learning_rate = experiment_cfg["learning_rate"]
    weight_decay = experiment_cfg["weight_decay"]

    train_loader = build_dataloader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_loader = build_dataloader(train_dataset, batch_size=batch_size, shuffle=False)

    model.to(device)
    criterion = build_criterion(criterion_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    last_metrics: dict[str, float] = {}
    for epoch in range(epochs):
        model.train()
        progress = tqdm(train_loader, desc=f"epoch {epoch + 1}/{epochs}", leave=False)
        for batch in progress:
            images = batch["images"].to(device)
            targets = batch["targets"]
            outputs = model(images)
            losses = criterion(outputs, targets, relation_matrix=relation_matrix)
            optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            optimizer.step()
            progress.set_postfix(loss=f"{losses['loss'].item():.4f}")

        last_metrics = evaluate_model(
            model=model,
            dataloader=eval_loader,
            criterion=criterion,
            relation_matrix=relation_matrix,
            device=device,
        )

    output_dir = ensure_dir(output_dir)
    torch.save({"model": model.state_dict(), "metrics": last_metrics}, output_dir / "last.pt")
    return last_metrics

