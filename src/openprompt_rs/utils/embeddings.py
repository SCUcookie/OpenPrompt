from __future__ import annotations

import hashlib
from pathlib import Path
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


class OpenClipTextEmbedder:
    """Text embedder backed by OpenCLIP/RemoteCLIP checkpoints."""

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        checkpoint: str | Path | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        try:
            import open_clip
        except ImportError as exc:
            raise RuntimeError(
                "open_clip is required for real VLM embeddings. Install "
                "`open-clip-torch` in the active environment or use "
                "embedding_backend='hash' for smoke tests."
            ) from exc

        self.open_clip = open_clip
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model_name = model_name
        self.checkpoint = str(checkpoint) if checkpoint else None
        self.model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=None)
        if checkpoint:
            state = torch.load(checkpoint, map_location="cpu")
            if isinstance(state, dict):
                state = state.get("state_dict", state.get("model", state))
            if isinstance(state, dict):
                state = {key.removeprefix("module."): value for key, value in state.items()}
            self.model.load_state_dict(state, strict=False)
        self.model.to(self.device)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)

    @torch.no_grad()
    def embed_texts(self, texts: Iterable[str]) -> torch.Tensor:
        text_list = list(texts)
        tokens = self.tokenizer(text_list).to(self.device)
        embeddings = self.model.encode_text(tokens)
        return F.normalize(embeddings.float(), dim=-1).cpu()


class ClipTextEmbedder:
    """Text embedder backed by OpenAI CLIP."""

    def __init__(
        self,
        model_name: str = "ViT-B/32",
        checkpoint: str | Path | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        try:
            import clip
        except ImportError as exc:
            raise RuntimeError(
                "clip is required for OpenAI CLIP embeddings. Install `clip` "
                "or `openai-clip` in the active environment or use "
                "embedding_backend='hash' for smoke tests."
            ) from exc

        self.clip = clip
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model_name = model_name
        self.checkpoint = str(checkpoint) if checkpoint else None
        self.model, _ = clip.load(model_name, device=self.device, download_root=None)
        if checkpoint:
            state = torch.load(checkpoint, map_location="cpu")
            if isinstance(state, dict):
                state = state.get("state_dict", state.get("model", state))
            if isinstance(state, dict):
                state = {key.removeprefix("module."): value for key, value in state.items()}
            self.model.load_state_dict(state, strict=False)
        self.model.eval()

    @torch.no_grad()
    def embed_texts(self, texts: Iterable[str]) -> torch.Tensor:
        text_list = list(texts)
        tokens = self.clip.tokenize(text_list).to(self.device)
        embeddings = self.model.encode_text(tokens)
        return F.normalize(embeddings.float(), dim=-1).cpu()


def build_text_embedder(
    backend: str,
    embedding_dim: int | None = None,
    model_name: str | None = None,
    checkpoint: str | Path | None = None,
    device: str | torch.device | None = None,
):
    backend = backend.lower()
    if backend == "hash":
        if embedding_dim is None:
            raise ValueError("embedding_dim is required for the hash embedder")
        return HashTextEmbedder(embedding_dim=embedding_dim)
    if backend in {"open_clip", "openclip", "skyclip", "remoteclip"}:
        return OpenClipTextEmbedder(
            model_name=model_name or "ViT-B-32",
            checkpoint=checkpoint,
            device=device,
        )
    if backend == "clip":
        return ClipTextEmbedder(
            model_name=model_name or "ViT-B/32",
            checkpoint=checkpoint,
            device=device,
        )
    raise ValueError(f"Unsupported embedding backend: {backend}")
