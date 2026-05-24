from __future__ import annotations

import torch
import torch.nn as nn

from openprompt_rs.data.structures import generate_query_centers


class TinyBackbone(nn.Module):
    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, output_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(output_dim),
            nn.GELU(),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.layers(images)


class QueryGenerator(nn.Module):
    def __init__(self, feature_dim: int, grid_size: int) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.pool = nn.AdaptiveAvgPool2d((grid_size, grid_size))
        self.proj = nn.Conv2d(feature_dim, feature_dim, kernel_size=1)

    def forward(self, feature_map: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = feature_map.shape[0]
        pooled = self.proj(self.pool(feature_map))
        query_tokens = pooled.flatten(2).transpose(1, 2)
        scene_feature = feature_map.mean(dim=(-1, -2))
        query_centers = generate_query_centers(self.grid_size, batch_size=batch_size, device=feature_map.device)
        return query_tokens, scene_feature, query_centers

