from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from openprompt_rs.config import load_config
from openprompt_rs.data import build_dataset
from openprompt_rs.engine.evaluator import collect_detection_diagnostics
from openprompt_rs.engine.trainer import build_dataloader
from openprompt_rs.models import PromptBank, build_model
from openprompt_rs.utils.io import dump_json, seed_everything


REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_repo_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _summarize_dataset(dataset: object, class_names: list[str]) -> dict[str, Any]:
    samples = getattr(dataset, "samples", None)
    image_paths = getattr(dataset, "image_paths", None)
    class_counts = Counter()
    tile_object_counts: list[int] = []
    source_sizes = Counter()

    if isinstance(samples, list):
        for sample in samples:
            source_sizes[(int(sample.get("width", 0)), int(sample.get("height", 0)))] += 1
            if "object_count" in sample:
                tile_object_counts.append(int(sample["object_count"]))
            target = sample.get("target", {})
            labels = target.get("labels") if isinstance(target, dict) else None
            if labels is None:
                continue
            if "tile_indices" in sample and sample["tile_indices"]:
                labels = labels[sample["tile_indices"]]
            for label in labels.tolist():
                if 0 <= int(label) < len(class_names):
                    class_counts[class_names[int(label)]] += 1
    else:
        for index in range(len(dataset)):  # type: ignore[arg-type]
            sample = dataset[index]  # type: ignore[index]
            labels = sample["target"]["labels"]
            for label in labels.tolist():
                if 0 <= int(label) < len(class_names):
                    class_counts[class_names[int(label)]] += 1

    nonempty_tiles = sum(1 for count in tile_object_counts if count > 0)
    return {
        "num_samples": len(dataset),  # type: ignore[arg-type]
        "num_source_images": len(image_paths) if image_paths is not None else None,
        "num_objects": int(sum(class_counts.values())),
        "num_empty_tiles": int(len(tile_object_counts) - nonempty_tiles) if tile_object_counts else None,
        "num_nonempty_tiles": int(nonempty_tiles) if tile_object_counts else None,
        "max_objects_per_tile": max(tile_object_counts) if tile_object_counts else None,
        "class_counts": {class_name: int(class_counts[class_name]) for class_name in class_names},
        "top_source_sizes": [
            {"width": width, "height": height, "count": count}
            for (width, height), count in source_sizes.most_common(10)
        ],
    }


def _load_model(config: dict[str, Any], dataset_cfg: dict[str, Any], checkpoint: str) -> torch.nn.Module:
    experiment_cfg = config["experiment"]
    model_cfg = config["model"]
    prompt_bank = PromptBank.build_from_files(
        taxonomy_path=resolve_repo_path(experiment_cfg["taxonomy_path"]),
        template_path=resolve_repo_path(experiment_cfg["prompt_template_path"]),
        embedding_dim=model_cfg["embedding_dim"],
        class_names=dataset_cfg["class_names"],
        hierarchy_lambda=model_cfg.get("hierarchy_smoothing_lambda", 0.1),
        use_class_offsets=model_cfg.get("use_class_offsets", True),
    )
    model = build_model(model_cfg=model_cfg, prompt_bank=prompt_bank)
    state = torch.load(resolve_repo_path(checkpoint), map_location="cpu")
    model.load_state_dict(state["model"])
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose weak DOTA baseline outputs.")
    parser.add_argument("--config", required=True, help="Path to an experiment YAML config.")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint for prediction diagnostics.")
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--score-thresholds", default="0.05,0.01,0.001")
    parser.add_argument("--nms-iou-threshold", type=float, default=0.3)
    parser.add_argument("--max-detections", type=int, default=100)
    parser.add_argument("--max-batches", type=int, default=None, help="Limit batches for quick server checks.")
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    config = load_config(args.config)
    experiment_cfg = config["experiment"]
    split = args.split
    dataset_cfg = config["val_dataset"] if split == "val" and "val_dataset" in config else config["dataset"]
    seed_everything(experiment_cfg["seed"])
    if experiment_cfg["device"].startswith("cuda") and not torch.cuda.is_available():
        experiment_cfg["device"] = "cpu"

    dataset = build_dataset(dataset_cfg)
    payload: dict[str, Any] = {
        "split": split,
        "config": args.config,
        "checkpoint": args.checkpoint,
        "dataset": _summarize_dataset(dataset, dataset_cfg["class_names"]),
    }

    if args.checkpoint is not None:
        thresholds = [float(value.strip()) for value in args.score_thresholds.split(",") if value.strip()]
        model = _load_model(config, dataset_cfg, args.checkpoint)
        model.to(experiment_cfg["device"])
        dataloader = build_dataloader(dataset, batch_size=experiment_cfg["batch_size"], shuffle=False)
        payload["prediction_diagnostics"] = collect_detection_diagnostics(
            model=model,
            dataloader=dataloader,
            device=experiment_cfg["device"],
            class_names=dataset_cfg["class_names"],
            score_thresholds=thresholds,
            nms_iou_threshold=args.nms_iou_threshold,
            max_detections=args.max_detections,
            max_batches=args.max_batches,
        )

    if args.output:
        dump_json(payload, resolve_repo_path(args.output))
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
