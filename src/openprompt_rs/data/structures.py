from __future__ import annotations

import math

import torch


def obb_to_polygon(cx: float, cy: float, w: float, h: float, theta: float) -> list[tuple[float, float]]:
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    half_w = w / 2.0
    half_h = h / 2.0
    corners = [(-half_w, -half_h), (half_w, -half_h), (half_w, half_h), (-half_w, half_h)]
    polygon = []
    for x_off, y_off in corners:
        x = cx + x_off * cos_t - y_off * sin_t
        y = cy + x_off * sin_t + y_off * cos_t
        polygon.append((x, y))
    return polygon


def generate_query_centers(grid_size: int, batch_size: int, device: torch.device | str) -> torch.Tensor:
    coords = torch.linspace(0.5 / grid_size, 1.0 - 0.5 / grid_size, grid_size, device=device)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    centers = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)
    return centers.unsqueeze(0).repeat(batch_size, 1, 1)

