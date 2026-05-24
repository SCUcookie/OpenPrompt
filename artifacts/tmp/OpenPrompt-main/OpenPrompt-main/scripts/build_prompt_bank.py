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
    args = parser.parse_args()

    prompt_bank = PromptBank.build_from_files(
        taxonomy_path=resolve_repo_path(args.taxonomy),
        template_path=resolve_repo_path(args.templates),
        embedding_dim=args.embedding_dim,
    )
    artifact = prompt_bank.export_artifact()
    output_path = resolve_repo_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, output_path)
    print({"classes": len(artifact["class_names"]), "output": str(output_path)})


if __name__ == "__main__":
    main()

