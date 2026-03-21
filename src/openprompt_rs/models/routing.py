from __future__ import annotations

import math

import torch
import torch.nn as nn


class ScaleRotationRouter(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.router = nn.Sequential(
            nn.Linear(embedding_dim + 5, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, query_tokens: torch.Tensor, alignment_logits: torch.Tensor, alignment_boxes: torch.Tensor) -> torch.Tensor:
        probabilities = torch.sigmoid(alignment_logits).clamp_min(1e-6)
        entropy = -(probabilities * probabilities.log()).sum(dim=-1, keepdim=True) / math.log(alignment_logits.size(-1) + 1)
        geometry = torch.cat(
            [
                alignment_boxes[..., 2:4],
                torch.sin(alignment_boxes[..., 4:5]),
                torch.cos(alignment_boxes[..., 4:5]),
            ],
            dim=-1,
        )
        features = torch.cat([query_tokens, entropy, geometry], dim=-1)
        return torch.sigmoid(self.router(features))

