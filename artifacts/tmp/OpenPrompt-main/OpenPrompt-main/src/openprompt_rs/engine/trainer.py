from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from openprompt_rs.data.base import collate_detection_batch
from openprompt_rs.engine.evaluator import evaluate_model
from openprompt_rs.models.losses import OpenPromptCriterion
from openprompt_rs.utils.io import dump_json, ensure_dir


def build_dataloader(dataset: object, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_detection_batch)


def build_criterion(criterion_cfg: dict[str, Any]) -> OpenPromptCriterion:
    return OpenPromptCriterion(
        cls_weight=criterion_cfg.get("cls_weight", 1.0),
        box_weight=criterion_cfg.get("box_weight", 1.0),
        hierarchy_weight=criterion_cfg.get("hierarchy_weight", 0.0),
        focal_alpha=criterion_cfg.get("focal_alpha", 0.25),
        focal_gamma=criterion_cfg.get("focal_gamma", 2.0),
        margin_weight=criterion_cfg.get("margin_weight", 0.0),
        margin_value=criterion_cfg.get("margin_value", 0.2),
    )


def train_experiment(
    model: torch.nn.Module,
    train_dataset: object,
    eval_dataset: object | None,
    experiment_cfg: dict[str, Any],
    criterion_cfg: dict[str, Any],
    relation_matrix: torch.Tensor | None,
    confusing_matrix: torch.Tensor | None,
    output_dir: str | Path,
    resume_state: dict[str, Any] | None = None,
) -> dict[str, float]:
    device = experiment_cfg["device"]
    batch_size = experiment_cfg["batch_size"]
    epochs = experiment_cfg["epochs"]
    learning_rate = experiment_cfg["learning_rate"]
    weight_decay = experiment_cfg["weight_decay"]

    train_loader = build_dataloader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_loader = build_dataloader(eval_dataset or train_dataset, batch_size=batch_size, shuffle=False)
    output_dir = ensure_dir(output_dir)

    model.to(device)
    criterion = build_criterion(criterion_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    last_metrics: dict[str, float] = {}
    start_epoch = 0
    if resume_state is not None:
        model.load_state_dict(resume_state["model"])
        optimizer_state = resume_state.get("optimizer")
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
        start_epoch = int(resume_state.get("epoch", 0))
        last_metrics = dict(resume_state.get("metrics", {}))
        if start_epoch > epochs:
            raise ValueError(f"Checkpoint epoch {start_epoch} exceeds configured epochs {epochs}.")

    for epoch in range(start_epoch, epochs):
        model.train()
        progress = tqdm(train_loader, desc=f"epoch {epoch + 1}/{epochs}", leave=False)
        for batch in progress:
            images = batch["images"].to(device)
            targets = batch["targets"]
            outputs = model(images)
            losses = criterion(
                outputs,
                targets,
                relation_matrix=relation_matrix,
                confusing_matrix=confusing_matrix,
            )
            optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            optimizer.step()
            progress.set_postfix(loss=f"{losses['loss'].item():.4f}")

        last_metrics = evaluate_model(
            model=model,
            dataloader=eval_loader,
            criterion=criterion,
            relation_matrix=relation_matrix,
            confusing_matrix=confusing_matrix,
            device=device,
        )
        epoch_number = epoch + 1
        print(f"epoch {epoch_number}/{epochs} metrics: {last_metrics}", flush=True)
        torch.save(
            {
                "epoch": epoch_number,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "metrics": last_metrics,
            },
            output_dir / f"epoch_{epoch_number:03d}.pt",
        )
        dump_json(last_metrics, output_dir / f"metrics_epoch_{epoch_number:03d}.json")

    torch.save(
        {
            "epoch": epochs,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "metrics": last_metrics,
        },
        output_dir / "last.pt",
    )
    dump_json(last_metrics, output_dir / "metrics.json")
    return last_metrics
