from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import torch

from openprompt_rs.config import load_config
from openprompt_rs.data import build_dataset
from openprompt_rs.data.base import collate_detection_batch
from openprompt_rs.models import PromptBank, build_model
from openprompt_rs.models.losses import OpenPromptCriterion
from openprompt_rs.models.routing import ScaleRotationRouter


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


def test_router_modes_return_two_branch_weights() -> None:
    torch.manual_seed(3)
    query_tokens = torch.randn(2, 5, 16)
    alignment_logits = torch.randn(2, 5, 4)
    alignment_boxes = torch.rand(2, 5, 5)

    for mode in ("soft", "gumbel", "random"):
        router = ScaleRotationRouter(
            embedding_dim=16,
            hidden_dim=8,
            mode=mode,
            temperature=0.7,
            hard=mode != "soft",
        )
        router.train()
        route = router(query_tokens, alignment_logits, alignment_boxes)

        assert route.shape == (2, 5, 2)
        assert torch.allclose(route.sum(dim=-1), torch.ones(2, 5), atol=1e-6)
        assert torch.isfinite(route).all()

        if mode in {"gumbel", "random"}:
            assert torch.equal(route, route.round())
