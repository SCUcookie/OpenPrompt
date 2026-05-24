from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SceneContextAdapter(nn.Module):
    def __init__(self, embedding_dim: int, num_classes: int) -> None:
        super().__init__()
        self.feature_gate = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.class_gate = nn.Sequential(
            nn.Linear(embedding_dim, num_classes),
            nn.Tanh(),
        )

    def forward(self, prompt_embeddings: torch.Tensor, scene_feature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if prompt_embeddings.dim() == 2:
            prompt_embeddings = prompt_embeddings.unsqueeze(0).expand(scene_feature.size(0), -1, -1)
        feature_gate = torch.sigmoid(self.feature_gate(scene_feature)).unsqueeze(1)
        class_gate = torch.sigmoid(self.class_gate(scene_feature)).unsqueeze(-1)
        adapted = prompt_embeddings * (1.0 + feature_gate) * (1.0 + class_gate)
        return F.normalize(adapted, dim=-1), class_gate.squeeze(-1)

