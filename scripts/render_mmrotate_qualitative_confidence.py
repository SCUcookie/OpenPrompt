#!/usr/bin/env python
"""Render MMRotate qualitative examples ranked by prediction confidence."""

from __future__ import annotations

import argparse
from pathlib import Path

import mmcv
import numpy as np
from mmengine.config import Config
from mmengine.fileio import load
from mmengine.registry import init_default_scope
from mmengine.structures import InstanceData

from mmdet.registry import DATASETS
from mmdet.structures import DetDataSample
from mmrotate.structures.bbox import RotatedBoxes
from mmrotate.visualization import RotLocalVisualizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--score-thr", type=float, default=0.3)
    return parser.parse_args()


def to_instances(raw: dict) -> InstanceData:
    instances = InstanceData()
    bboxes = raw["bboxes"]
    if bboxes.shape[-1] == 5:
        instances.bboxes = RotatedBoxes(bboxes)
    else:
        instances.bboxes = bboxes
    instances.labels = raw["labels"]
    if "scores" in raw:
        instances.scores = raw["scores"]
    return instances


def mean_confidence(result: dict) -> float:
    scores = result["pred_instances"]["scores"]
    if len(scores) == 0:
        return 0.0
    return float(scores.float().mean())


def max_label_count(outputs: list) -> int:
    max_label = -1
    for result in outputs:
        for key in ("gt_instances", "pred_instances"):
            labels = result[key]["labels"]
            if len(labels) > 0:
                max_label = max(max_label, int(labels.max()))
    return max_label + 1


def normalize_palette(palette, min_colors: int):
    base = []
    if isinstance(palette, (list, tuple)):
        base = list(palette)

    generated = [
        (220, 20, 60),
        (119, 11, 32),
        (0, 0, 142),
        (0, 0, 230),
        (106, 0, 228),
        (0, 60, 100),
        (0, 80, 100),
        (0, 0, 70),
        (0, 0, 192),
        (250, 170, 30),
        (100, 170, 30),
        (220, 220, 0),
        (175, 116, 175),
        (250, 0, 30),
        (165, 42, 42),
        (255, 77, 255),
        (0, 226, 252),
        (182, 182, 255),
        (0, 82, 0),
        (120, 166, 157),
    ]
    index = 0
    while len(base) < min_colors:
        base.append(generated[index % len(generated)])
        index += 1
    return base


def render_sample(visualizer: RotLocalVisualizer, result: dict, out_file: Path,
                  score_thr: float) -> None:
    image = mmcv.imread(result["img_path"], channel_order="rgb")
    data_sample = DetDataSample()
    data_sample.pred_instances = to_instances(result["pred_instances"])
    data_sample.gt_instances = to_instances(result["gt_instances"])
    visualizer.add_datasample(
        "image",
        image,
        data_sample,
        show=False,
        draw_gt=True,
        draw_pred=True,
        pred_score_thr=score_thr,
        out_file=str(out_file),
    )


def main() -> None:
    args = parse_args()
    cfg = Config.fromfile(args.config)
    init_default_scope(cfg.get("default_scope", "mmdet"))

    outputs = load(args.predictions)
    ranked = sorted(enumerate(outputs), key=lambda item: mean_confidence(item[1]))
    bad = ranked[:args.topk]
    good = ranked[-args.topk:]

    cfg.test_dataloader.dataset.test_mode = True
    dataset = DATASETS.build(cfg.test_dataloader.dataset)
    classes = tuple(dataset.metainfo["classes"])
    palette = normalize_palette(
        dataset.metainfo.get("palette", None),
        max(len(classes), max_label_count(outputs)),
    )
    visualizer = RotLocalVisualizer()
    visualizer.dataset_meta = {"classes": classes, "palette": palette}

    for subset, items in (("bad", bad), ("good", good)):
        subset_dir = args.out_dir / subset
        subset_dir.mkdir(parents=True, exist_ok=True)
        for index, result in items:
            score = mean_confidence(result)
            stem = Path(result["img_path"]).stem
            out_file = subset_dir / f"{index:04d}_{stem}_{score:.3f}.png"
            render_sample(visualizer, result, out_file, args.score_thr)

    summary = args.out_dir / "ranking_summary.txt"
    lines = ["rank,index,split,img_path,mean_confidence"]
    for split, items in (("bad", bad), ("good", good)):
        for rank, (index, result) in enumerate(items, start=1):
            lines.append(
                f"{rank},{index},{split},{result['img_path']},{mean_confidence(result):.6f}"
            )
    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
