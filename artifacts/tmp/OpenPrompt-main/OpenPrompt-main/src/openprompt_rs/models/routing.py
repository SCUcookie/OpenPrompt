from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaleRotationRouter(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        mode: str = "soft",
        temperature: float = 1.0,
        hard: bool = True,
    ) -> None:
        super().__init__()
        if mode not in {"soft", "gumbel", "random"}:
            raise ValueError(f"Unsupported router mode: {mode}")
        if temperature <= 0:
            raise ValueError("Router temperature must be positive.")
        self.mode = mode
        self.temperature = temperature
        self.hard = hard
        self.router = nn.Sequential(
            nn.Linear(embedding_dim + 5, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
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
        logits = self.router(features)

        if self.mode == "random":
            if self.training:
                route = torch.randint(0, 2, logits.shape[:-1], device=logits.device, dtype=torch.long)
                return F.one_hot(route, num_classes=2).to(logits.dtype)
            probabilities = torch.full_like(logits, 0.5)
            return probabilities

        if self.mode == "gumbel":
            return F.gumbel_softmax(logits, tau=self.temperature, hard=self.hard, dim=-1)

        return torch.softmax(logits / self.temperature, dim=-1)
