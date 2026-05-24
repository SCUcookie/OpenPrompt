from __future__ import annotations

import argparse
from pathlib import Path

import torch

from openprompt_rs.config import load_config
from openprompt_rs.data import build_dataset
from openprompt_rs.engine.trainer import train_experiment
from openprompt_rs.models import PromptBank, build_model
from openprompt_rs.utils.io import dump_json, seed_everything


REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_repo_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def resolve_resume_checkpoint(resume_arg: str | None, output_dir: Path) -> Path | None:
    if resume_arg is None:
        return None
    if resume_arg == "latest":
        checkpoints = sorted(output_dir.glob("epoch_*.pt"))
        if not checkpoints:
            raise FileNotFoundError(f"No epoch checkpoints found in {output_dir} for --resume latest.")
        return checkpoints[-1]
    return resolve_repo_path(resume_arg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the rebuilt OpenPrompt research scaffold.")
    parser.add_argument("--config", required=True, help="Path to an experiment YAML config.")
    parser.add_argument("--output-dir", default=None, help="Optional override for the experiment output directory.")
    parser.add_argument(
        "--resume",
        default=None,
        help="Optional checkpoint path, or 'latest' to resume from the newest checkpoint in the output directory.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    experiment_cfg = config["experiment"]
    dataset_cfg = config["dataset"]
    eval_dataset_cfg = config.get("val_dataset")
    model_cfg = config["model"]
    criterion_cfg = config["criterion"]

    seed_everything(experiment_cfg["seed"])
    if experiment_cfg["device"].startswith("cuda") and not torch.cuda.is_available():
        experiment_cfg["device"] = "cpu"

    prompt_bank = PromptBank.build_from_files(
        taxonomy_path=resolve_repo_path(experiment_cfg["taxonomy_path"]),
        template_path=resolve_repo_path(experiment_cfg["prompt_template_path"]),
        embedding_dim=model_cfg["embedding_dim"],
        class_names=dataset_cfg["class_names"],
        hierarchy_lambda=model_cfg.get("hierarchy_smoothing_lambda", 0.1),
        use_class_offsets=model_cfg.get("use_class_offsets", True),
    )
    model = build_model(model_cfg=model_cfg, prompt_bank=prompt_bank)
    dataset = build_dataset(dataset_cfg)
    eval_dataset = build_dataset(eval_dataset_cfg) if eval_dataset_cfg is not None else None

    output_dir = resolve_repo_path(args.output_dir or experiment_cfg["output_dir"])
    resume_checkpoint = resolve_resume_checkpoint(args.resume, output_dir)
    resume_state = None
    if resume_checkpoint is not None:
        resume_state = torch.load(resume_checkpoint, map_location="cpu")
        print(f"Resuming from checkpoint: {resume_checkpoint}", flush=True)
    metrics = train_experiment(
        model=model,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        experiment_cfg=experiment_cfg,
        criterion_cfg=criterion_cfg,
        relation_matrix=prompt_bank.hierarchy.relation_matrix,
        confusing_matrix=prompt_bank.hierarchy.confusing_matrix,
        output_dir=output_dir,
        resume_state=resume_state,
    )
    dump_json(metrics, output_dir / "metrics.json")
    print(metrics, flush=True)


if __name__ == "__main__":
    main()
