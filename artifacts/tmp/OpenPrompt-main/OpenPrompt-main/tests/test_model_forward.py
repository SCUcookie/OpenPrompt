from __future__ import annotations

from pathlib import Path

from openprompt_rs.config import load_config
from openprompt_rs.data import build_dataset
from openprompt_rs.data.base import collate_detection_batch
from openprompt_rs.models import PromptBank, build_model


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_geonexus_forward_shapes() -> None:
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

    assert outputs["logits"].shape[:2] == outputs["boxes"].shape[:2]
    assert outputs["logits"].shape[-1] == len(config["dataset"]["class_names"])
    assert outputs["boxes"].shape[-1] == 5

