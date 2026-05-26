from __future__ import annotations

import argparse
from pathlib import Path

import torch

from openprompt_rs.models import PromptBank


REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_repo_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and export a prompt-bank artifact.")
    parser.add_argument("--taxonomy", required=True, help="Taxonomy JSON path.")
    parser.add_argument("--templates", default="assets/prompts/prompt_templates.json", help="Template JSON path.")
    parser.add_argument("--output", required=True, help="Artifact output path.")
    parser.add_argument("--embedding-dim", type=int, default=256, help="Prompt embedding dimension.")
    parser.add_argument(
        "--embedding-backend",
        default="hash",
        choices=["hash", "clip", "open_clip", "openclip", "remoteclip", "skyclip"],
        help="Text embedding backend. Use hash only for smoke tests.",
    )
    parser.add_argument("--embedding-model-name", default=None, help="VLM model name, e.g. ViT-B-32.")
    parser.add_argument("--embedding-checkpoint", default=None, help="Optional local VLM checkpoint path.")
    parser.add_argument("--embedding-cache-path", default=None, help="Optional generated embedding cache path.")
    parser.add_argument("--embedding-device", default=None, help="Torch device for real VLM encoding.")
    args = parser.parse_args()

    prompt_bank = PromptBank.build_from_files(
        taxonomy_path=resolve_repo_path(args.taxonomy),
        template_path=resolve_repo_path(args.templates),
        embedding_dim=args.embedding_dim,
        embedding_backend=args.embedding_backend,
        embedding_model_name=args.embedding_model_name,
        embedding_checkpoint=resolve_repo_path(args.embedding_checkpoint)
        if args.embedding_checkpoint
        else None,
        embedding_cache_path=resolve_repo_path(args.embedding_cache_path)
        if args.embedding_cache_path
        else None,
        embedding_device=args.embedding_device,
    )
    artifact = prompt_bank.export_artifact()
    output_path = resolve_repo_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, output_path)
    print({"classes": len(artifact["class_names"]), "output": str(output_path)})


if __name__ == "__main__":
    main()
