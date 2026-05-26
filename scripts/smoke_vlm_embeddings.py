from __future__ import annotations

import argparse
from pathlib import Path

import torch

from openprompt_rs.models.hierarchy import HierarchyGraph
from openprompt_rs.utils.embeddings import build_text_embedder


REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_repo_path(raw_path: str | None) -> Path | None:
    if raw_path is None:
        return None
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test real text embeddings for DOTA class prompts.")
    parser.add_argument("--taxonomy", default="assets/hierarchies/remote_sensing_taxonomy.json")
    parser.add_argument(
        "--embedding-backend",
        default="remoteclip",
        choices=["clip", "open_clip", "openclip", "remoteclip", "skyclip"],
    )
    parser.add_argument("--embedding-model-name", default="ViT-B-32")
    parser.add_argument("--embedding-checkpoint", default=None)
    parser.add_argument("--embedding-device", default=None)
    args = parser.parse_args()

    hierarchy = HierarchyGraph.from_json(resolve_repo_path(args.taxonomy))
    prompts = [f"a remote sensing image of {class_name}" for class_name in hierarchy.class_names]
    embedder = build_text_embedder(
        backend=args.embedding_backend,
        model_name=args.embedding_model_name,
        checkpoint=resolve_repo_path(args.embedding_checkpoint),
        device=args.embedding_device,
    )
    embeddings = embedder.embed_texts(prompts)
    norms = torch.linalg.vector_norm(embeddings, dim=-1)
    if embeddings.shape[0] != 16:
        raise RuntimeError(f"Expected 16 DOTA v1.5 classes, got {embeddings.shape[0]}")
    if not torch.isfinite(embeddings).all():
        raise RuntimeError("Embedding tensor contains non-finite values")
    if not torch.allclose(norms, torch.ones_like(norms), atol=1e-3):
        raise RuntimeError("Embedding tensor is not L2-normalized")
    print(
        {
            "backend": args.embedding_backend,
            "model_name": args.embedding_model_name,
            "classes": len(prompts),
            "embedding_shape": list(embeddings.shape),
        }
    )


if __name__ == "__main__":
    main()
