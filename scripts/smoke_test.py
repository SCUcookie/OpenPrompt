from __future__ import annotations

import argparse
from pathlib import Path

import torch

from openprompt_rs.config import load_config
from openprompt_rs.data import build_dataset
from openprompt_rs.data.base import collate_detection_batch
from openprompt_rs.models import PromptBank, build_model
from openprompt_rs.models.losses import OpenPromptCriterion
from openprompt_rs.models.pseudo_label import HierarchyConsistentPseudoLabeler, PseudoLabelConfig
from openprompt_rs.utils.io import seed_everything


REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_repo_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a fast smoke test for the rebuilt repo.")
    parser.add_argument("--config", required=True, help="Path to an experiment YAML config.")
    args = parser.parse_args()

    config = load_config(args.config)
    experiment_cfg = config["experiment"]
    dataset_cfg = config["dataset"]
    model_cfg = config["model"]
    criterion_cfg = config["criterion"]

    seed_everything(experiment_cfg["seed"])
    dataset = build_dataset(dataset_cfg)
    batch = collate_detection_batch([dataset[0], dataset[1]])

    prompt_bank = PromptBank.build_from_files(
        taxonomy_path=resolve_repo_path(experiment_cfg["taxonomy_path"]),
        template_path=resolve_repo_path(experiment_cfg["prompt_template_path"]),
        embedding_dim=model_cfg["embedding_dim"],
        class_names=dataset_cfg["class_names"],
        hierarchy_lambda=model_cfg.get("hierarchy_smoothing_lambda", 0.1),
        use_class_offsets=model_cfg.get("use_class_offsets", True),
    )
    model = build_model(model_cfg=model_cfg, prompt_bank=prompt_bank)
    criterion = OpenPromptCriterion(**criterion_cfg)

    outputs = model(batch["images"])
    losses = criterion(outputs, batch["targets"], relation_matrix=prompt_bank.hierarchy.relation_matrix)
    losses["loss"].backward()

    result = {
        "logits_shape": tuple(outputs["logits"].shape),
        "boxes_shape": tuple(outputs["boxes"].shape),
        "loss": float(losses["loss"].item()),
    }

    if "pseudo_label" in config:
        pseudo_labeler = HierarchyConsistentPseudoLabeler(
            hierarchy=prompt_bank.hierarchy,
            config=PseudoLabelConfig(**config["pseudo_label"]),
        )
        pseudo = pseudo_labeler.filter(
            outputs=outputs,
            prompt_embeddings=outputs["prompt_embeddings"],
            scene_scores=outputs["scene_scores"],
        )
        result["pseudo_labels"] = [int(item["labels"].numel()) for item in pseudo]

    print(result)


if __name__ == "__main__":
    main()

