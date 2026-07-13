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
    parser = argparse.ArgumentParser(description="Smoke-test and optionally save real text embeddings for class prompts.")
    parser.add_argument("--taxonomy", default="assets/hierarchies/remote_sensing_taxonomy.json")
    parser.add_argument(
        "--embedding-backend",
        default="remoteclip",
        choices=["clip", "open_clip", "openclip", "remoteclip", "skyclip"],
    )
    parser.add_argument("--embedding-model-name", default="ViT-B-32")
    parser.add_argument("--embedding-checkpoint", default=None)
    parser.add_argument("--embedding-device", default=None)
    parser.add_argument(
        "--expected-class-count",
        type=int,
        default=None,
        help="Fail unless the taxonomy contains exactly this many classes.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional torch artifact path. Relative paths are resolved from the repository root.",
    )
    args = parser.parse_args()

    taxonomy_path = resolve_repo_path(args.taxonomy)
    checkpoint_path = resolve_repo_path(args.embedding_checkpoint)
    output_path = resolve_repo_path(args.output)
    hierarchy = HierarchyGraph.from_json(taxonomy_path)
    prompts = [f"a remote sensing image of {class_name}" for class_name in hierarchy.class_names]
    embedder = build_text_embedder(
        backend=args.embedding_backend,
        model_name=args.embedding_model_name,
        checkpoint=checkpoint_path,
        device=args.embedding_device,
    )
    embeddings = embedder.embed_texts(prompts)
    norms = torch.linalg.vector_norm(embeddings, dim=-1)
    if args.expected_class_count is not None and embeddings.shape[0] != args.expected_class_count:
        raise RuntimeError(f"Expected {args.expected_class_count} classes, got {embeddings.shape[0]}")
    if not torch.isfinite(embeddings).all():
        raise RuntimeError("Embedding tensor contains non-finite values")
    if not torch.allclose(norms, torch.ones_like(norms), atol=1e-3):
        raise RuntimeError("Embedding tensor is not L2-normalized")
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "class_names": list(hierarchy.class_names),
                "prompts": prompts,
                "embeddings": embeddings.detach().cpu(),
                "backend": args.embedding_backend,
                "model_name": args.embedding_model_name,
                "checkpoint": str(checkpoint_path) if checkpoint_path is not None else None,
            },
            output_path,
        )
    print(
        {
            "backend": args.embedding_backend,
            "model_name": args.embedding_model_name,
            "classes": len(prompts),
            "embedding_shape": list(embeddings.shape),
            "output": str(output_path) if output_path is not None else None,
        }
    )


if __name__ == "__main__":
    main()
