from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch
import torch.nn as nn

from openprompt_rs.models.context_adapter import SceneContextAdapter
from openprompt_rs.models.routing import ScaleRotationRouter


class SceneConditionedTemperature(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, min_tau: float, max_tau: float) -> None:
        super().__init__()
        self.min_tau = min_tau
        self.max_tau = max_tau
        self.temperature_head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, logits: torch.Tensor, scene_feature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        temperature = torch.sigmoid(self.temperature_head(scene_feature))
        temperature = self.min_tau + (self.max_tau - self.min_tau) * temperature
        scaled_logits = logits / temperature.unsqueeze(-1)
        return scaled_logits, temperature.squeeze(-1)


def resolve_innovation_config(model_cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    model_type = model_cfg.get("type", "baseline")
    raw_config = deepcopy(model_cfg.get("innovations", {}))
    embedding_dim = int(model_cfg["embedding_dim"])

    defaults = {
        "scene_adapter": {
            "enabled": bool(model_cfg.get("use_scene_adapter", model_type == "geonexus")),
        },
        "router": {
            "enabled": bool(model_cfg.get("use_router", model_type == "geonexus")),
            "hidden_dim": int(model_cfg.get("router_hidden_dim", embedding_dim)),
        },
        "scene_temperature": {
            "enabled": False,
            "hidden_dim": embedding_dim,
            "min_tau": 0.75,
            "max_tau": 1.50,
        },
    }

    resolved: dict[str, dict[str, Any]] = {}
    for name, default_cfg in defaults.items():
        merged_cfg = deepcopy(default_cfg)
        override_cfg = raw_config.get(name, {})
        merged_cfg.update(override_cfg)
        resolved[name] = merged_cfg
    return resolved


def build_innovation_modules(
    model_cfg: dict[str, Any],
    num_classes: int,
) -> tuple[dict[str, dict[str, Any]], nn.ModuleDict]:
    innovation_cfg = resolve_innovation_config(model_cfg)
    embedding_dim = int(model_cfg["embedding_dim"])
    modules = nn.ModuleDict()

    if innovation_cfg["scene_adapter"]["enabled"]:
        modules["scene_adapter"] = SceneContextAdapter(embedding_dim=embedding_dim, num_classes=num_classes)

    if innovation_cfg["router"]["enabled"]:
        modules["router"] = ScaleRotationRouter(
            embedding_dim=embedding_dim,
            hidden_dim=int(innovation_cfg["router"]["hidden_dim"]),
        )

    if innovation_cfg["scene_temperature"]["enabled"]:
        modules["scene_temperature"] = SceneConditionedTemperature(
            embedding_dim=embedding_dim,
            hidden_dim=int(innovation_cfg["scene_temperature"]["hidden_dim"]),
            min_tau=float(innovation_cfg["scene_temperature"]["min_tau"]),
            max_tau=float(innovation_cfg["scene_temperature"]["max_tau"]),
        )

    return innovation_cfg, modules
