#!/usr/bin/env python3
"""Render qualitative detection comparisons for the TGRS manuscript (job A2).

Produces figure/geonexus_tgrs_qualitative.png: a two-row strip in which the
top row shows RoI Transformer baseline detections and the bottom row shows
GeoNexus-RSD detections on the same curated DIOR-R test images, drawn with a
shared class-color map so colors are comparable across rows.

Must run on the server inside the MMRotate environment, e.g.:

    CUDA_VISIBLE_DEVICES=0 python scripts/render_qualitative_detections_20260713.py \
        --baseline-config  <dior_r baseline config .py> \
        --baseline-ckpt    <baseline epoch_52.pth> \
        --model-config     <scene-adapter rep0 config .py> \
        --model-ckpt       <scene-adapter rep0 epoch_8.pth> \
        --image-dir        <DIOR_R test images dir> \
        --images 11726.png 12003.png 14830.png 17650.png \
        --score-thr 0.3 \
        --out-dir qualitative_20260713

Pick four scenes covering: a harbor with ship/harbor coexistence, a dense
vehicle lot, an overpass/bridge transition, and a storage-tank field. Inspect
candidates first and choose images where the baseline makes visible semantic
confusions; the exact image IDs above are placeholders.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def build_palette(class_names: list[str]) -> dict[str, tuple[int, int, int]]:
    """Deterministic, high-contrast BGR colors keyed by class name."""
    base = [
        (214, 120, 42), (0, 131, 0), (167, 58, 74), (52, 104, 235),
        (122, 175, 27), (72, 73, 227), (164, 123, 232), (38, 89, 217),
        (0, 168, 255), (128, 0, 128), (0, 200, 120), (30, 30, 200),
        (180, 130, 70), (90, 160, 40), (200, 80, 160), (40, 180, 220),
        (150, 60, 30), (60, 120, 90), (220, 40, 90), (100, 100, 240),
    ]
    return {name: base[i % len(base)] for i, name in enumerate(class_names)}


def draw_detections(img, result, class_names, palette, score_thr: float):
    import cv2

    pred = result.pred_instances
    keep = pred.scores >= score_thr
    bboxes = pred.bboxes[keep].cpu().numpy()  # (n, 5) cx, cy, w, h, theta
    labels = pred.labels[keep].cpu().numpy()
    for (cx, cy, w, h, theta), label in zip(bboxes, labels):
        name = class_names[int(label)]
        color = palette[name]
        pts = cv2.boxPoints(((float(cx), float(cy)), (float(w), float(h)), float(np.degrees(theta))))
        cv2.polylines(img, [pts.astype(np.int32)], isClosed=True, color=color, thickness=2)
        cv2.putText(img, name, (int(pts[0][0]), int(pts[0][1]) - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return img


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline-config", required=True)
    ap.add_argument("--baseline-ckpt", required=True)
    ap.add_argument("--model-config", required=True)
    ap.add_argument("--model-ckpt", required=True)
    ap.add_argument("--image-dir", required=True, type=Path)
    ap.add_argument("--images", nargs="+", required=True, help="Image filenames inside --image-dir")
    ap.add_argument("--score-thr", type=float, default=0.3)
    ap.add_argument("--out-dir", type=Path, default=Path("qualitative_20260713"))
    args = ap.parse_args()

    import cv2
    from mmdet.utils import register_all_modules as register_all_modules_mmdet
    from mmrotate.utils import register_all_modules
    from mmengine import DefaultScope
    register_all_modules_mmdet(init_default_scope=False)
    register_all_modules(init_default_scope=False)
    DefaultScope.get_instance("mmrotate", scope_name="mmrotate")
    import geonexus_mmrotate.prompt_bbox_head  # noqa: F401
    from mmdet.apis import inference_detector, init_detector

    args.out_dir.mkdir(parents=True, exist_ok=True)

    models = {
        "baseline": init_detector(args.baseline_config, args.baseline_ckpt, device="cuda:0"),
        "geonexus": init_detector(args.model_config, args.model_ckpt, device="cuda:0"),
    }
    class_names = list(models["baseline"].dataset_meta["classes"])
    palette = build_palette(class_names)

    rows = []
    for tag, model in models.items():
        panels = []
        for fname in args.images:
            path = args.image_dir / fname
            img = cv2.imread(str(path))
            if img is None:
                raise SystemExit(f"Could not read image {path}")
            result = inference_detector(model, str(path))
            drawn = draw_detections(img.copy(), result, class_names, palette, args.score_thr)
            panel_path = args.out_dir / f"{Path(fname).stem}_{tag}.png"
            cv2.imwrite(str(panel_path), drawn)
            panels.append(drawn)
            print(f"wrote {panel_path}")
        rows.append(cv2.hconcat([cv2.resize(p, (512, 512)) for p in panels]))

    strip = cv2.vconcat(rows)
    strip_path = args.out_dir / "geonexus_tgrs_qualitative.png"
    cv2.imwrite(str(strip_path), strip)
    print(f"\nwrote stitched strip {strip_path}")
    print("Copy it to _local_archive_20260601_pull_backup/docs/TGRS/figure/ and replace the")
    print("placeholder box in the Qualitative Results section of geonexus_tgrs.tex.")


if __name__ == "__main__":
    main()
