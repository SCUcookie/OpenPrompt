from __future__ import annotations

import hashlib
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F


class HashTextEmbedder:
    """Deterministic fallback embedder for offline smoke tests."""

    def __init__(self, embedding_dim: int) -> None:
        self.embedding_dim = embedding_dim

    def embed_texts(self, texts: Iterable[str]) -> torch.Tensor:
        vectors = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            seed = int.from_bytes(digest[:8], byteorder="big", signed=False)
            rng = np.random.default_rng(seed)
            vector = rng.standard_normal(self.embedding_dim).astype(np.float32)
            vectors.append(vector)
        tensor = torch.tensor(np.stack(vectors), dtype=torch.float32)
        return F.normalize(tensor, dim=-1)

