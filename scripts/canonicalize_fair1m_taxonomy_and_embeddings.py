#!/usr/bin/env python3
"""Apply the canonical FAIR1M detector order to taxonomy and prompt embeddings."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import torch

from reconstruct_fair1m_tiled_annotations import CANONICAL_CLASSES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--taxonomy", required=True, type=Path)
    parser.add_argument("--input-artifact", required=True, type=Path)
    parser.add_argument("--output-artifact", required=True, type=Path)
    args = parser.parse_args()

    taxonomy = json.loads(args.taxonomy.read_text(encoding="utf-8"))
    by_name = {item["name"]: item for item in taxonomy["classes"]}
    if set(by_name) != set(CANONICAL_CLASSES):
        raise SystemExit("taxonomy class set does not match canonical FAIR1M classes")
    taxonomy["classes"] = [by_name[name] for name in CANONICAL_CLASSES]
    taxonomy["_provenance"]["canonical_order_updated"] = str(date.today())
    taxonomy["_provenance"]["canonical_order"] = list(CANONICAL_CLASSES)
    args.taxonomy.write_text(json.dumps(taxonomy, indent=2) + "\n", encoding="utf-8")

    artifact = torch.load(args.input_artifact, map_location="cpu")
    old_names = list(artifact["class_names"])
    if set(old_names) != set(CANONICAL_CLASSES):
        raise SystemExit("embedding class set does not match canonical FAIR1M classes")
    old_index = {name: index for index, name in enumerate(old_names)}
    order = [old_index[name] for name in CANONICAL_CLASSES]
    prompts = artifact["prompts"]
    artifact["class_names"] = list(CANONICAL_CLASSES)
    artifact["prompts"] = [prompts[index] for index in order]
    artifact["embeddings"] = artifact["embeddings"][order].clone()
    artifact["supersedes"] = str(args.input_artifact.resolve())
    artifact["canonical_order"] = True
    args.output_artifact.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, args.output_artifact)

    loaded = torch.load(args.output_artifact, map_location="cpu")
    assert loaded["class_names"] == list(CANONICAL_CLASSES)
    assert len(loaded["prompts"]) == len(CANONICAL_CLASSES)
    assert tuple(loaded["embeddings"].shape) == (37, 512)
    assert torch.isfinite(loaded["embeddings"]).all()
    print(json.dumps({
        "taxonomy": str(args.taxonomy.resolve()),
        "input_artifact": str(args.input_artifact.resolve()),
        "output_artifact": str(args.output_artifact.resolve()),
        "class_names": loaded["class_names"],
        "embedding_shape": list(loaded["embeddings"].shape),
        "finite": bool(torch.isfinite(loaded["embeddings"]).all()),
    }, indent=2))


if __name__ == "__main__":
    main()
