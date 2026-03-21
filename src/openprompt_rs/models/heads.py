from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _ensure_prompt_batch(prompt_embeddings: torch.Tensor, batch_size: int) -> torch.Tensor:
    if prompt_embeddings.dim() == 2:
        return prompt_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
    return prompt_embeddings


def decode_box(raw_boxes: torch.Tensor) -> torch.Tensor:
    center = torch.sigmoid(raw_boxes[..., :2])
    size = torch.sigmoid(raw_boxes[..., 2:4]).clamp_min(1e-4)
    angle = torch.tanh(raw_boxes[..., 4:5]) * (math.pi / 2.0)
    return torch.cat([center, size, angle], dim=-1)


class AlignmentHead(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.query_proj = nn.Linear(embedding_dim, embedding_dim)
        self.box_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, 5),
        )
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0), dtype=torch.float32))

    def forward(self, query_tokens: torch.Tensor, prompt_embeddings: torch.Tensor) -> dict[str, torch.Tensor]:
        prompt_embeddings = _ensure_prompt_batch(prompt_embeddings, query_tokens.size(0))
        projected_queries = F.normalize(self.query_proj(query_tokens), dim=-1)
        normalized_prompts = F.normalize(prompt_embeddings, dim=-1)
        logits = torch.einsum("bqd,bcd->bqc", projected_queries, normalized_prompts) * self.logit_scale.exp()
        boxes = decode_box(self.box_head(query_tokens))
        return {"logits": logits, "boxes": boxes, "query_embeddings": projected_queries}


class FusionHead(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.query_proj = nn.Linear(embedding_dim, embedding_dim)
        self.prompt_proj = nn.Linear(embedding_dim, embedding_dim)
        self.value_proj = nn.Linear(embedding_dim, embedding_dim)
        self.fuse = nn.Sequential(
            nn.Linear(embedding_dim * 3, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
        )
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)
        self.box_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, 5),
        )
        self.logit_scale = nn.Parameter(torch.tensor(math.log(12.0), dtype=torch.float32))

    def forward(self, query_tokens: torch.Tensor, prompt_embeddings: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size = query_tokens.size(0)
        prompt_embeddings = _ensure_prompt_batch(prompt_embeddings, batch_size)
        normalized_queries = F.normalize(self.query_proj(query_tokens), dim=-1)
        normalized_prompts = F.normalize(self.prompt_proj(prompt_embeddings), dim=-1)
        attention = torch.softmax(
            torch.einsum("bqd,bcd->bqc", normalized_queries, normalized_prompts) / math.sqrt(query_tokens.size(-1)),
            dim=-1,
        )
        prompt_context = torch.einsum("bqc,bcd->bqd", attention, self.value_proj(prompt_embeddings))
        fused = self.fuse(torch.cat([query_tokens, prompt_context, query_tokens * prompt_context], dim=-1))
        fused_queries = F.normalize(self.out_proj(fused), dim=-1)
        logits = torch.einsum("bqd,bcd->bqc", fused_queries, F.normalize(prompt_embeddings, dim=-1))
        logits = logits * self.logit_scale.exp()
        boxes = decode_box(self.box_head(fused))
        return {"logits": logits, "boxes": boxes, "query_embeddings": fused_queries, "attention": attention}

