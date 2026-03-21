from __future__ import annotations

from pathlib import Path

from openprompt_rs.config import load_config
from openprompt_rs.data import build_dataset
from openprompt_rs.data.base import collate_detection_batch
from openprompt_rs.models import PromptBank, build_model
from openprompt_rs.models.pseudo_label import HierarchyConsistentPseudoLabeler, PseudoLabelConfig


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_pseudo_labeler_runs() -> None:
    config = load_config(REPO_ROOT / "configs/experiments/geonexus_synthetic.yaml")
    dataset = build_dataset(config["dataset"])
    batch = collate_detection_batch([dataset[0], dataset[1]])

    prompt_bank = PromptBank.build_from_files(
        taxonomy_path=REPO_ROOT / config["experiment"]["taxonomy_path"],
        template_path=REPO_ROOT / config["experiment"]["prompt_template_path"],
        embedding_dim=config["model"]["embedding_dim"],
        class_names=config["dataset"]["class_names"],
        hierarchy_lambda=config["model"]["hierarchy_smoothing_lambda"],
        use_class_offsets=config["model"]["use_class_offsets"],
    )
    model = build_model(config["model"], prompt_bank)
    outputs = model(batch["images"])

    labeler = HierarchyConsistentPseudoLabeler(
        hierarchy=prompt_bank.hierarchy,
        config=PseudoLabelConfig(**config["pseudo_label"]),
    )
    pseudo = labeler.filter(outputs, outputs["prompt_embeddings"], outputs["scene_scores"])

    assert len(pseudo) == 2
    assert all("boxes" in item and "labels" in item for item in pseudo)

