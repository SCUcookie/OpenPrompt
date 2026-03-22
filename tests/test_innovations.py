from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from openprompt_rs.config import load_config
from openprompt_rs.data import build_dataset
from openprompt_rs.data.base import collate_detection_batch
from openprompt_rs.models import PromptBank, build_model
from openprompt_rs.models.losses import OpenPromptCriterion


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_math_innovation_hooks_run() -> None:
    config = load_config(REPO_ROOT / "configs/experiments/geonexus_synthetic.yaml")
    dataset = build_dataset(config["dataset"])
    batch = collate_detection_batch([dataset[0], dataset[1]])

    model_cfg = deepcopy(config["model"])
    model_cfg["innovations"]["scene_temperature"]["enabled"] = True

    criterion_cfg = deepcopy(config["criterion"])
    criterion_cfg["margin_weight"] = 0.1
    criterion_cfg["margin_value"] = 0.2

    prompt_bank = PromptBank.build_from_files(
        taxonomy_path=REPO_ROOT / config["experiment"]["taxonomy_path"],
        template_path=REPO_ROOT / config["experiment"]["prompt_template_path"],
        embedding_dim=model_cfg["embedding_dim"],
        class_names=config["dataset"]["class_names"],
        hierarchy_lambda=model_cfg["hierarchy_smoothing_lambda"],
        use_class_offsets=model_cfg["use_class_offsets"],
    )
    model = build_model(model_cfg, prompt_bank)
    outputs = model(batch["images"])

    criterion = OpenPromptCriterion(**criterion_cfg)
    losses = criterion(
        outputs,
        batch["targets"],
        relation_matrix=prompt_bank.hierarchy.relation_matrix,
        confusing_matrix=prompt_bank.hierarchy.confusing_matrix,
    )

    assert outputs["scene_temperature"] is not None
    assert "loss_margin" in losses
