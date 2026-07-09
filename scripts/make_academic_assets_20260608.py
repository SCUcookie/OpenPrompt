"""Generate publication-style GeoNexus-RSD figures and tables.

The script only uses local project records for numeric evidence. Planned or
blocked rows are labelled as such in the generated tables and figures.
"""

from __future__ import annotations

import csv
import json
import math
import textwrap
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
PPT_DIR = ROOT / "artifacts" / "ppt_assets_20260608"
PAPER_DIR = ROOT / "artifacts" / "paper_assets_20260608"
DOCS = ROOT / "docs" / "experiments"
DOTAV2_IMAGES = ROOT / "DOTAv2" / "images" / "train"
DOTAV2_LABELS = ROOT / "DOTAv2" / "labels" / "train"
DOTA2_CLASSES = [
    "plane",
    "baseball-diamond",
    "bridge",
    "ground-track-field",
    "small-vehicle",
    "large-vehicle",
    "ship",
    "tennis-court",
    "basketball-court",
    "storage-tank",
    "soccer-ball-field",
    "roundabout",
    "harbor",
    "swimming-pool",
    "helicopter",
    "container-crane",
    "airport",
    "helipad",
]

PPT_SIZE = (13.333, 7.5)
PAPER_SIZE = (7.2, 4.55)
PPT_DPI = 240
PAPER_DPI = 240

BG = "#f7f8fb"
PAPER_BG = "#ffffff"
INK = "#17202a"
MUTED = "#667085"
FAINT = "#eef2f6"
LINE = "#d3dae6"
BLUE = "#2f6fed"
TEAL = "#159a8c"
GREEN = "#2e8b57"
AMBER = "#c98514"
RED = "#c53b3b"
PURPLE = "#6658d3"
SLATE = "#49566b"


@dataclass(frozen=True)
class MetricRow:
    name: str
    dataset: str
    status: str
    map: float | None
    ap50: float | None
    source: str
    note: str


@dataclass(frozen=True)
class CropSpec:
    image_id: str
    cls: str
    crop: tuple[int, int, int, int]


def read_json(rel_path: str) -> dict:
    with (ROOT / rel_path).open("r", encoding="utf-8") as f:
        return json.load(f)


def metric_from_json(rel_path: str, key: str = "metrics") -> tuple[float, float]:
    data = read_json(rel_path)
    metrics = data[key]
    return float(metrics["dota/mAP"]), float(metrics["dota/AP50"])


def assert_close(observed: float, expected: float, label: str) -> None:
    if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=5e-5):
        raise AssertionError(f"{label}: expected {expected}, observed {observed}")


def load_evidence() -> tuple[list[MetricRow], list[dict]]:
    roi_map, roi_ap50 = metric_from_json("docs/experiments/20260603_s0_dota2_roi_trans_validpng_metrics.json")
    openrsd_map, openrsd_ap50 = metric_from_json("docs/experiments/20260602_opensrd_dota2_epoch12_ssval_metrics.json")
    s1 = read_json("docs/experiments/20260605_geonexus_s1_retry2_metrics.json")
    s2 = read_json("docs/experiments/20260605_geonexus_s2_rerun_s1e32_metrics.json")
    s3 = read_json("docs/experiments/20260605_geonexus_s3_rerun_s2e4_metrics.json")
    s3_id = read_json("docs/experiments/20260605_geonexus_s3_identity_rerun_s2e4_metrics.json")
    s3_off = read_json("docs/experiments/20260606_geonexus_s3_adapter_off_rerun_s2e4_metrics.json")
    s2_refine = read_json("docs/experiments/20260606_geonexus_s2_refine_s2e4_lr1e4_metrics.json")

    assert_close(roi_map, 0.6088, "DOTA2 RoITrans mAP")
    assert_close(roi_ap50, 0.6090, "DOTA2 RoITrans AP50")
    assert_close(openrsd_map, 0.4202, "OpenRSD DOTA2 mAP")
    assert_close(openrsd_ap50, 0.4200, "OpenRSD DOTA2 AP50")
    assert_close(float(s1["best_metrics"]["dota/mAP"]), 0.3800, "DOTA v1.5 S1 best mAP")
    assert_close(float(s2["best_metrics"]["dota/mAP"]), 0.3858, "DOTA v1.5 S2 best mAP")
    assert_close(float(s3["best_metrics"]["dota/mAP"]), 0.3827, "DOTA v1.5 S3 best mAP")

    rows = [
        MetricRow(
            "GeoNexus S1 GPU-1",
            "DOTA2 ss_val",
            "Complete",
            0.6177,
            0.6180,
            "docs/experiments/20260608_dota2_s1_complete_and_s2_launch.md",
            "Completed S1; positive over S0",
        ),
        MetricRow(
            "RoI Transformer R50",
            "DOTA2 ss_val",
            "Complete",
            roi_map,
            roi_ap50,
            "docs/experiments/20260603_s0_dota2_roi_trans_validpng_metrics.json",
            "S0 gate; valid-PNG recovery",
        ),
        MetricRow(
            "GeoNexus S2 main epoch 4",
            "DOTA2 ss_val",
            "Active",
            0.6038,
            0.6040,
            "docs/experiments/20260608_dota2_s1_complete_and_s2_launch.md",
            "Active S2; not yet stronger than S1/S0",
        ),
        MetricRow(
            "GeoNexus S1 GPU-0 LR 5e-5",
            "DOTA2 ss_val",
            "Complete",
            0.6047,
            0.6050,
            "docs/experiments/20260608_dota2_s1_complete_and_s2_launch.md",
            "Low-LR replicate; below S0",
        ),
        MetricRow(
            "GeoNexus S1 GPU-6 LR 1e-4",
            "DOTA2 ss_val",
            "Complete",
            0.5997,
            0.6000,
            "docs/experiments/20260608_dota2_s1_complete_and_s2_launch.md",
            "Low-LR replicate; below S0",
        ),
        MetricRow(
            "Oriented R-CNN R50",
            "DOTA2 ss_val",
            "Complete",
            0.5973,
            0.5970,
            "docs/experiments/20260605_dota2_baseline_status.md",
            "Secondary baseline",
        ),
        MetricRow(
            "S2ANet",
            "DOTA2 ss_val",
            "Complete",
            0.5869,
            0.5870,
            "docs/experiments/20260605_dota2_baseline_status.md",
            "Secondary baseline",
        ),
        MetricRow(
            "R3Det-KFIoU",
            "DOTA2 ss_val",
            "Complete",
            0.5633,
            0.5630,
            "docs/experiments/20260605_dota2_baseline_status.md",
            "Secondary baseline",
        ),
        MetricRow(
            "OpenRSD epoch 12",
            "DOTA2 ss_val",
            "Reference",
            openrsd_map,
            openrsd_ap50,
            "docs/experiments/20260602_opensrd_dota2_epoch12_ssval_metrics.json",
            "Completed OpenRSD checkpoint; not GeoNexus",
        ),
        MetricRow(
            "RTMDet-M",
            "DOTA2 ss_val",
            "Complete",
            0.3312,
            0.3310,
            "docs/experiments/20260605_dota2_baseline_status.md",
            "Low secondary baseline",
        ),
        MetricRow(
            "RTMDet-L",
            "DOTA2 ss_val",
            "Complete",
            0.2779,
            0.2780,
            "docs/experiments/20260605_dota2_baseline_status.md",
            "Completed and deprioritized",
        ),
    ]

    route = [
        {
            "stage": "S0 RoITrans",
            "kind": "DOTA2 paper route",
            "map": roi_map,
            "ap50": roi_ap50,
            "source": "20260603_s0_dota2_roi_trans_validpng_metrics.json",
        },
        {
            "stage": "S1 prompt GPU-1",
            "kind": "DOTA2 paper route",
            "map": 0.6177,
            "ap50": 0.6180,
            "source": "20260608_dota2_s1_complete_and_s2_launch.md",
        },
        {
            "stage": "S2 hierarchy e4",
            "kind": "DOTA2 active",
            "map": 0.6038,
            "ap50": 0.6040,
            "source": "20260608_dota2_s1_complete_and_s2_launch.md",
        },
        {
            "stage": "DIOR-R S0",
            "kind": "cross-dataset required",
            "map": None,
            "ap50": None,
            "source": "20260607_dior_orcnn_nan_diag_and_roi_trans_launch.md",
        },
        {
            "stage": "DOTA v1.5 S1",
            "kind": "DOTA v1.5 diagnostic",
            "map": float(s1["best_metrics"]["dota/mAP"]),
            "ap50": float(s1["best_metrics"]["dota/AP50"]),
            "source": "20260605_geonexus_s1_retry2_metrics.json",
        },
        {
            "stage": "DOTA v1.5 S2",
            "kind": "DOTA v1.5 diagnostic",
            "map": float(s2["best_metrics"]["dota/mAP"]),
            "ap50": float(s2["best_metrics"]["dota/AP50"]),
            "source": "20260605_geonexus_s2_rerun_s1e32_metrics.json",
        },
        {
            "stage": "DOTA v1.5 S3",
            "kind": "DOTA v1.5 diagnostic",
            "map": float(s3["best_metrics"]["dota/mAP"]),
            "ap50": float(s3["best_metrics"]["dota/AP50"]),
            "source": "20260605_geonexus_s3_rerun_s2e4_metrics.json",
        },
        {
            "stage": "DOTA v1.5 S3 off",
            "kind": "DOTA v1.5 diagnostic",
            "map": float(s3_off["best_metrics"]["dota/mAP"]),
            "ap50": float(s3_off["best_metrics"]["dota/AP50"]),
            "source": "20260606_geonexus_s3_adapter_off_rerun_s2e4_metrics.json",
        },
    ]
    return rows, route


def setup_ax(size=PPT_SIZE, bg=BG):
    fig, ax = plt.subplots(figsize=size, dpi=PPT_DPI)
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    ax.axis("off")
    return fig, ax


def ensure_dirs() -> None:
    PPT_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)


def save_ppt(fig, name: str) -> None:
    fig.savefig(PPT_DIR / f"{name}_16x9.png", dpi=PPT_DPI, pad_inches=0.0)
    fig.savefig(PPT_DIR / f"{name}_16x9.svg", pad_inches=0.0)
    plt.close(fig)


def save_paper(fig, name: str) -> None:
    fig.savefig(PAPER_DIR / f"{name}.pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(PAPER_DIR / f"{name}.svg", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def rounded_box(ax, xy, wh, title, body="", color=BLUE, fc="#ffffff", lw=1.1, fontsize=10.5):
    x, y = xy
    w, h = wh
    patch = patches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.015,rounding_size=0.06",
        facecolor=fc,
        edgecolor=color,
        linewidth=lw,
    )
    ax.add_patch(patch)
    ax.text(x + 0.14, y + h - 0.16, title, va="top", ha="left", fontsize=fontsize, color=INK, weight="bold")
    if body:
        wrapped = "\n".join(textwrap.wrap(body, width=max(16, int(w * 7.2))))
        ax.text(x + 0.14, y + h - 0.50, wrapped, va="top", ha="left", fontsize=fontsize - 2.1, color=MUTED, linespacing=1.15)
    return patch


def arrow(ax, p0, p1, color=SLATE, lw=1.35, rad=0.0):
    ax.add_patch(
        patches.FancyArrowPatch(
            p0,
            p1,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=lw,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=4,
            shrinkB=4,
        )
    )


def parse_dota_labels(label_path: Path, img_w: int, img_h: int) -> list[tuple[str, list[float]]]:
    rows = []
    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) < 9:
            continue
        coords: list[float]
        cls_token: str
        first_is_class_id = False
        try:
            first_val = float(parts[0])
            first_is_class_id = first_val.is_integer() and 0 <= int(first_val) < len(DOTA2_CLASSES)
        except ValueError:
            first_is_class_id = False
        if first_is_class_id:
            try:
                # Local normalized format: class_id x1 y1 ... x4 y4.
                cls_idx = int(float(parts[0]))
                coords = [float(x) for x in parts[1:9]]
            except ValueError:
                continue
            cls_token = DOTA2_CLASSES[cls_idx] if 0 <= cls_idx < len(DOTA2_CLASSES) else f"class-{cls_idx}"
        else:
            try:
                # Standard DOTA text: x1 y1 ... x4 y4 class [difficulty].
                coords = [float(x) for x in parts[:8]]
                cls_token = parts[8]
            except ValueError:
                continue
        if max(coords) <= 1.5:
            coords = [coords[i] * (img_w if i % 2 == 0 else img_h) for i in range(8)]
        rows.append((cls_token, coords))
    return rows


def select_real_crops(limit: int = 4) -> list[CropSpec]:
    preferred = ["ship", "plane", "small-vehicle", "harbor", "large-vehicle", "storage-tank"]
    selected: list[CropSpec] = []
    seen: set[tuple[str, str]] = set()
    for label_path in sorted(DOTAV2_LABELS.glob("P*.txt")):
        image_path = DOTAV2_IMAGES / f"{label_path.stem}.jpg"
        if not image_path.exists():
            continue
        try:
            with Image.open(image_path) as im:
                img_w, img_h = im.size
        except OSError:
            continue
        rows = parse_dota_labels(label_path, img_w, img_h)
        for cls in preferred:
            matches = [coords for c, coords in rows if c == cls]
            if not matches or (label_path.stem, cls) in seen:
                continue
            coords = matches[0]
            xs = coords[0::2]
            ys = coords[1::2]
            cx = (min(xs) + max(xs)) / 2.0
            cy = (min(ys) + max(ys)) / 2.0
            side = max(max(xs) - min(xs), max(ys) - min(ys), 360.0)
            pad = side * 0.85
            x0 = max(0, int(cx - pad))
            y0 = max(0, int(cy - pad))
            x1 = min(img_w, int(cx + pad))
            y1 = min(img_h, int(cy + pad))
            if x1 - x0 < 96 or y1 - y0 < 96:
                continue
            selected.append(CropSpec(label_path.stem, cls, (x0, y0, x1, y1)))
            seen.add((label_path.stem, cls))
            if len(selected) >= limit:
                return selected
    if len(selected) < limit:
        raise AssertionError("Not enough DOTAv2 image/label crops for visual assets")
    return selected


def draw_real_crop(ax, spec: CropSpec, x: float, y: float, w: float, h: float, label: str | None = None, scale: float = 1.0) -> None:
    image_path = DOTAV2_IMAGES / f"{spec.image_id}.jpg"
    with Image.open(image_path) as im:
        crop = im.crop(spec.crop).resize((640, 360))
        arr = np.asarray(crop)
    ax.imshow(arr, extent=(x, x + w, y, y + h), aspect="auto", zorder=0)
    ax.add_patch(patches.Rectangle((x, y), w, h, facecolor="none", edgecolor="#ffffff", linewidth=1.5 * scale))
    ax.text(
        x + 0.05,
        y + 0.06,
        label or f"{spec.image_id} | {spec.cls}",
        fontsize=6.2 * scale,
        color="#ffffff",
        va="bottom",
        bbox=dict(facecolor=(0, 0, 0, 0.45), edgecolor="none", pad=1.4),
    )


def render_method_framework_ppt() -> None:
    crops = select_real_crops(4)
    fig, ax = setup_ax(PPT_SIZE, BG)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.text(0.38, 8.55, "GeoNexus-RSD Framework", fontsize=22, weight="bold", color=INK)
    ax.text(
        0.40,
        8.18,
        "Real DOTAv2 crops anchor the visual context; metrics remain tied to local project records.",
        fontsize=9.8,
        color=MUTED,
    )

    rounded_box(ax, (0.45, 5.78), (2.2, 1.86), "Input tile", "DOTA2 crop; valid PNGs", TEAL, "#ffffff", fontsize=9.5)
    draw_real_crop(ax, crops[0], 0.62, 5.88, 1.84, 0.98, f"{crops[0].image_id} | {crops[0].cls}", 0.95)
    rounded_box(ax, (3.15, 5.78), (2.25, 1.86), "Detector trunk", "R50 + FPN", BLUE, "#eef4ff", fontsize=9.5)
    for i, ht in enumerate([0.22, 0.32, 0.43, 0.54]):
        ax.add_patch(patches.Rectangle((3.56 + i * 0.36, 6.10), 0.25, ht, facecolor=BLUE, alpha=0.23 + 0.12 * i, edgecolor=BLUE, linewidth=0.7))
    rounded_box(ax, (5.88, 5.78), (2.25, 1.86), "RoI Transformer", "rotated RoI + cascade", PURPLE, "#f5f3ff", fontsize=9.5)
    ax.add_patch(patches.Rectangle((6.44, 6.16), 0.82, 0.25, angle=24, facecolor="#ffffff", edgecolor=PURPLE, linewidth=1.2))
    ax.text(6.18, 5.98, "(cx, cy, w, h, theta)", fontsize=6.4, color=MUTED)
    rounded_box(ax, (8.62, 5.78), (2.25, 1.86), "Prompt fusion", "RemoteCLIP embeddings", AMBER, "#fff8e8", fontsize=9.5)
    ax.add_patch(patches.FancyBboxPatch((9.05, 6.12), 1.10, 0.34, boxstyle="round,pad=0.03,rounding_size=0.05", facecolor="#ffffff", edgecolor=AMBER, linewidth=0.9))
    ax.text(9.14, 6.29, "ship near harbor", fontsize=6.4, color=INK, va="center")
    rounded_box(ax, (11.35, 5.78), (2.25, 1.86), "Pseudo-label filter", "score + VLM + hierarchy", GREEN, "#eff8f3", fontsize=9.5)
    ax.add_patch(patches.Polygon([(11.92, 6.55), (12.65, 6.55), (12.38, 6.16), (12.38, 5.98), (12.18, 5.98), (12.18, 6.16)], facecolor="#d9f0e2", edgecolor=GREEN, linewidth=1.0))
    ax.text(12.73, 6.42, "accept", fontsize=6.5, color=GREEN, weight="bold")
    ax.text(12.73, 6.15, "reject", fontsize=6.5, color=RED, weight="bold")
    rounded_box(ax, (14.05, 5.78), (1.55, 1.86), "Output", "OBBs", RED, "#fff2f2", fontsize=9.5)
    draw_real_crop(ax, crops[1], 14.20, 5.98, 1.20, 0.72, f"{crops[1].cls}", 0.75)

    rounded_box(ax, (1.05, 2.95), (3.15, 1.62), "RemoteCLIP prompt bank", "Class aliases and prompt templates produce 512-D text embeddings.", AMBER, "#ffffff", fontsize=9.3)
    rounded_box(ax, (4.70, 2.95), (3.15, 1.62), "Hierarchy taxonomy", "", BLUE, "#ffffff", fontsize=9.3)
    ax.text(4.96, 3.86, "object", fontsize=6.8, color=SLATE)
    ax.plot([5.16, 5.16], [3.78, 3.54], color=LINE, lw=0.8)
    ax.text(4.96, 3.42, "vehicle", fontsize=6.8, color=SLATE)
    ax.text(5.78, 3.42, "facility", fontsize=6.8, color=SLATE)
    ax.plot([5.16, 5.56], [3.54, 3.54], color=LINE, lw=0.8)
    ax.plot([5.56, 5.56], [3.54, 3.46], color=LINE, lw=0.8)
    ax.text(5.15, 3.14, "small-vehicle", fontsize=6.5, color=SLATE)
    ax.text(6.18, 3.14, "ship / harbor", fontsize=6.5, color=SLATE)
    rounded_box(ax, (8.35, 2.95), (3.15, 1.62), "Hierarchy regularizer", "S2 is active; epoch-4 is below the completed S1 gate.", PURPLE, "#ffffff", fontsize=9.3)
    rounded_box(ax, (12.00, 2.95), (3.15, 1.62), "Evidence guard", "DOTA2 S1 is positive; DIOR-R remains blocked by data/box diagnosis.", RED, "#ffffff", fontsize=9.3)

    for xs, xe in [(2.65, 3.15), (5.40, 5.88), (8.13, 8.62), (10.87, 11.35), (13.60, 14.05)]:
        arrow(ax, (xs, 6.72), (xe, 6.72), BLUE)
    arrow(ax, (2.65, 3.76), (8.75, 5.85), AMBER, rad=-0.10)
    arrow(ax, (6.35, 4.57), (9.05, 5.85), BLUE, rad=-0.08)
    arrow(ax, (9.95, 4.57), (10.10, 5.85), PURPLE, rad=0.10)
    arrow(ax, (13.35, 4.57), (12.35, 5.85), RED, rad=0.08)

    ax.text(0.55, 0.68, "Crop source: local DOTAv2/images/train and DOTAv2/labels/train; no inference output is implied.", fontsize=8.7, color=MUTED)
    save_ppt(fig, "method_framework")


def render_method_framework_paper() -> None:
    crops = select_real_crops(2)
    fig, ax = setup_ax(PAPER_SIZE, PAPER_BG)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.text(0.0, 6.72, "GeoNexus-RSD method schematic", fontsize=10.5, weight="bold", color=INK)
    blocks = [
        ((0.15, 4.75), (1.65, 1.05), "Tile", TEAL, "crop"),
        ((2.15, 4.75), (1.65, 1.05), "FPN", BLUE, "features"),
        ((4.15, 4.75), (1.65, 1.05), "RoITrans", PURPLE, "OBB head"),
        ((6.15, 4.75), (1.65, 1.05), "Prompt", AMBER, "RemoteCLIP"),
        ((8.15, 4.75), (1.65, 1.05), "Filter", GREEN, "pseudo labels"),
        ((10.15, 4.75), (1.65, 1.05), "Output", RED, "OBBs"),
    ]
    for xy, wh, title, color, body in blocks:
        rounded_box(ax, xy, wh, title, body, color=color, fc="#ffffff", fontsize=7.6)
    draw_real_crop(ax, crops[0], 0.25, 5.03, 1.45, 0.45, "DOTAv2", 0.55)
    for x0, x1 in [(1.80, 2.15), (3.80, 4.15), (5.80, 6.15), (7.80, 8.15), (9.80, 10.15)]:
        arrow(ax, (x0, 5.26), (x1, 5.26), SLATE, lw=0.8)
    aux = [
        ((1.10, 2.52), (2.30, 1.00), "Prompt bank", "class aliases + 512-D text embeddings", AMBER),
        ((4.85, 2.52), (2.30, 1.00), "Hierarchy", "taxonomy constraints and confusion groups", BLUE),
        ((8.60, 2.52), (2.30, 1.00), "Evidence gate", "S2 active; DIOR-R blocked", PURPLE),
    ]
    for xy, wh, title, body, color in aux:
        rounded_box(ax, xy, wh, title, body, color=color, fc="#ffffff", fontsize=7.2)
    arrow(ax, (2.25, 3.52), (6.65, 4.75), AMBER, lw=0.8, rad=-0.1)
    arrow(ax, (6.00, 3.52), (6.95, 4.75), BLUE, lw=0.8)
    arrow(ax, (9.75, 3.52), (7.25, 4.75), PURPLE, lw=0.8, rad=0.1)
    ax.text(0.0, 0.22, "Method schematic with a real DOTAv2 crop; numbers are reported in separate evidence tables.", fontsize=6.5, color=MUTED)
    save_paper(fig, "fig_method_framework")


def render_dota2_baseline_plot(rows: list[MetricRow], paper: bool = False) -> None:
    complete = [r for r in rows if r.map is not None]
    complete.sort(key=lambda r: r.map or 0.0)
    size = PAPER_SIZE if paper else PPT_SIZE
    fig, ax = plt.subplots(figsize=size, dpi=PAPER_DPI if paper else PPT_DPI)
    fig.patch.set_facecolor(PAPER_BG if paper else BG)
    ax.set_facecolor(PAPER_BG if paper else BG)
    names = [r.name for r in complete]
    vals = [r.map for r in complete]
    ap50s = [r.ap50 for r in complete]
    y = np.arange(len(complete))
    colors = [
        GREEN if "GeoNexus S1 GPU-1" in n else
        RED if "S2 main" in n else
        BLUE if "RoI Transformer" in n else
        TEAL if "OpenRSD" in n else
        AMBER if "GeoNexus S1" in n else
        SLATE if v < 0.45 else PURPLE
        for n, v in zip(names, vals)
    ]
    ax.hlines(y, 0, vals, color=LINE, lw=5 if paper else 8, zorder=1)
    ax.scatter(vals, y, s=85 if paper else 190, color=colors, edgecolor="#ffffff", linewidth=0.9, zorder=2)
    ax.axvline(0.6088, color=BLUE, linestyle="--", lw=1.0)
    ax.text(0.6088, len(y) - 0.55, "S0 gate", ha="center", va="bottom", fontsize=7.0 if paper else 9.0, color=BLUE)
    ax.axvline(0.6177, color=GREEN, linestyle=":", lw=1.2)
    ax.text(0.6177, len(y) - 1.25, "S1 best", ha="center", va="bottom", fontsize=7.0 if paper else 9.0, color=GREEN)
    for yi, v, ap50 in zip(y, vals, ap50s):
        ax.text(v + 0.014, yi, f"{v:.4f}/{ap50:.4f}", va="center", fontsize=7.1 if paper else 10.0, color=INK)
    ax.set_yticks(y, names, fontsize=7.2 if paper else 10.5, color=INK)
    ax.set_xlim(0, 0.67)
    ax.set_xlabel("DOTA2 ss_val mAP / AP50", fontsize=7.3 if paper else 10.0, color=MUTED)
    ax.tick_params(axis="x", labelsize=6.8 if paper else 9.2, colors=MUTED)
    ax.grid(axis="x", color=FAINT, lw=0.8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("DOTA2 ranked evidence chart", loc="left", fontsize=10.5 if paper else 21, weight="bold", color=INK, pad=24 if paper else 34)
    ax.text(
        0,
        1.015,
        "S1 GPU-1 is the only GeoNexus result currently above S0; S2 epoch-4 is active but not stronger.",
        transform=ax.transAxes,
        fontsize=6.6 if paper else 9.4,
        color=MUTED,
    )
    fig.tight_layout()
    if paper:
        save_paper(fig, "fig_dota2_baseline_lollipop")
    else:
        save_ppt(fig, "dota2_baseline_lollipop")


def render_class_heatmap(paper: bool = False) -> None:
    data = read_json("docs/experiments/20260603_s0_dota2_roi_trans_validpng_metrics.json")
    per_class = data["per_class"]
    records = [
        {
            "class": d["class"],
            "ap": float(d["ap"]),
            "recall": float(d["recall"]),
            "gts": int(d["gts"]),
        }
        for d in per_class
    ]
    high = sorted(records, key=lambda d: d["ap"], reverse=True)[:5]
    weak = sorted(records, key=lambda d: d["ap"])[:5]
    high_count_low_recall = sorted(records, key=lambda d: (d["gts"], -d["recall"]), reverse=True)[:8]
    high_count_low_recall = sorted(high_count_low_recall, key=lambda d: d["recall"])[:5]

    fig, axes = plt.subplots(1, 3, figsize=PAPER_SIZE if paper else PPT_SIZE, dpi=PAPER_DPI if paper else PPT_DPI)
    fig.patch.set_facecolor(PAPER_BG if paper else BG)
    groups = [
        ("High-performing AP", high, "ap", BLUE),
        ("Weak AP classes", weak, "ap", RED),
        ("High count, lower recall", high_count_low_recall, "recall", AMBER),
    ]
    for ax, (title, group, metric, color) in zip(axes, groups):
        ax.set_facecolor(PAPER_BG if paper else BG)
        labels = [d["class"] for d in group]
        vals = [d[metric] for d in group]
        y = np.arange(len(group))
        ax.barh(y, vals, color=color, alpha=0.90)
        ax.set_yticks(y, labels, fontsize=5.9 if paper else 8.5)
        ax.set_xlim(0, max(1.0, max(vals) * 1.10))
        ax.invert_yaxis()
        for yi, d, v in zip(y, group, vals):
            detail = f"{v:.2f}" if metric == "ap" else f"R {v:.2f}; n={d['gts']}"
            ax.text(v + 0.015, yi, detail, va="center", fontsize=5.4 if paper else 7.8, color=INK)
        ax.set_title(title, loc="left", fontsize=7.7 if paper else 13.0, weight="bold", color=INK)
        ax.tick_params(axis="x", labelsize=5.6 if paper else 8.0, colors=MUTED)
        ax.grid(axis="x", color=FAINT, lw=0.7)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle("RoI Transformer DOTA2 S0 class evidence groups", x=0.02, ha="left", fontsize=10.5 if paper else 21, weight="bold", color=INK)
    fig.text(0.02, 0.015, "Source: 20260603_s0_dota2_roi_trans_validpng_metrics.json; grouped for readability.", fontsize=6.2 if paper else 8.8, color=MUTED)
    fig.tight_layout()
    if paper:
        save_paper(fig, "fig_dota2_class_heatmap")
    else:
        save_ppt(fig, "dota2_class_heatmap")


def render_detector_family_plot(rows: list[MetricRow], paper: bool = False) -> None:
    complete = [r for r in rows if r.map is not None]
    labels = [r.name.replace(" R50", "") for r in complete]
    map_vals = [r.map for r in complete]
    ap50_vals = [r.ap50 for r in complete]
    status = [1.0 if r.status == "Complete" else 0.65 for r in complete]
    x = np.arange(len(labels))
    width = 0.28
    fig, ax = plt.subplots(figsize=PAPER_SIZE if paper else PPT_SIZE, dpi=PAPER_DPI if paper else PPT_DPI)
    fig.patch.set_facecolor(PAPER_BG if paper else BG)
    ax.set_facecolor(PAPER_BG if paper else BG)
    ax.bar(x - width, map_vals, width=width, color=BLUE, label="mAP")
    ax.bar(x, ap50_vals, width=width, color=TEAL, label="AP50")
    ax.bar(x + width, status, width=width, color="#cfd6e2", label="completion flag")
    ax.axhline(0.6088, color=BLUE, ls="--", lw=1.0)
    ax.set_ylim(0, 1.03)
    ax.set_xticks(x, labels, rotation=28, ha="right", fontsize=6.2 if paper else 9.3)
    ax.tick_params(axis="y", labelsize=6.5 if paper else 9.0, colors=MUTED)
    ax.grid(axis="y", color=FAINT, lw=0.8)
    ax.legend(frameon=False, fontsize=6.5 if paper else 9.2, ncol=3, loc="upper right")
    ax.set_ylabel("Score / status", fontsize=7.0 if paper else 9.8, color=MUTED)
    ax.set_title("Detector-family score/status comparison", loc="left", fontsize=10.5 if paper else 21, weight="bold", color=INK, pad=10)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    if paper:
        save_paper(fig, "fig_detector_family_grouped")
    else:
        save_ppt(fig, "detector_family_grouped")


def render_route_progression(route: list[dict], paper: bool = False) -> None:
    dota2 = [r for r in route if r["kind"].startswith("DOTA2")]
    archive = [r for r in route if r["kind"].startswith("DOTA v1.5")]
    fig, ax = plt.subplots(figsize=PAPER_SIZE if paper else PPT_SIZE, dpi=PAPER_DPI if paper else PPT_DPI)
    fig.patch.set_facecolor(PAPER_BG if paper else BG)
    ax.set_facecolor(PAPER_BG if paper else BG)
    x = np.arange(len(dota2))
    vals = [r["map"] for r in dota2]
    colors = [BLUE, GREEN, RED]
    ax.plot(x, vals, color=SLATE, lw=1.3, zorder=1)
    ax.scatter(x, vals, s=100 if paper else 210, color=colors, edgecolor="#ffffff", linewidth=0.9, zorder=2)
    for xi, v, r in zip(x, vals, dota2):
        dy = 0.008 if "S2" not in r["stage"] else -0.018
        va = "bottom" if dy > 0 else "top"
        ax.text(xi, v + dy, f"{v:.4f}/{r['ap50']:.4f}", ha="center", va=va, fontsize=6.1 if paper else 8.8, color=INK)
    ax.axhline(0.6088, color=BLUE, lw=1.0, ls="--", label="DOTA2 RoITrans S0 gate")
    ax.axhline(0.6177, color=GREEN, lw=1.0, ls=":", label="Completed S1")
    archive_text = "Archive only: " + ", ".join(f"{r['stage'].replace('DOTA v1.5 ', '')} {r['map']:.4f}" for r in archive)
    ax.text(0.02, 0.08, archive_text, transform=ax.transAxes, fontsize=5.9 if paper else 8.4, color=AMBER)
    ax.text(0.02, 0.015, "DIOR-R: blocked by NaN/Inf/data-box diagnosis; not baseline evidence.", transform=ax.transAxes, fontsize=5.9 if paper else 8.4, color=RED)
    ax.set_xticks(x, [r["stage"] for r in dota2], rotation=0, ha="center", fontsize=6.3 if paper else 9.2)
    ax.set_ylim(0.58, 0.628)
    ax.set_xlim(-0.45, len(dota2) - 0.55)
    ax.tick_params(axis="y", labelsize=6.3 if paper else 9.0, colors=MUTED)
    ax.set_ylabel("mAP", fontsize=7.0 if paper else 10.0, color=MUTED)
    ax.grid(axis="y", color=FAINT, lw=0.8)
    ax.legend(frameon=False, fontsize=6.1 if paper else 8.6, loc="upper right")
    ax.set_title("DOTA2-first route gate", loc="left", fontsize=10.2 if paper else 20.5, weight="bold", color=INK, pad=10)
    ax.text(
        0.0,
        1.02,
        "DOTA2 progression is shown separately from archive diagnostics to avoid mixed-dataset trend claims.",
        transform=ax.transAxes,
        fontsize=6.2 if paper else 8.9,
        color=MUTED,
    )
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    if paper:
        save_paper(fig, "fig_route_progression")
    else:
        save_ppt(fig, "route_progression")


def render_dior_timeline(paper: bool = False) -> None:
    events = [
        ("2026-06-06", "ORCNN startup", "finite early loss", GREEN, 0.24),
        ("2026-06-07", "ORCNN diagnostic", "NaN loss/grad; invalid validation", RED, 0.60),
        ("2026-06-07", "RoITrans startup", "finite at [1][200]", GREEN, 0.35),
        ("2026-06-07", "RoITrans stop", "RPN/cascade losses became NaN", RED, 0.78),
        ("2026-06-07", "RetinaNet probe", "loss_bbox=inf at [1][1200]", RED, 0.49),
        ("Next", "Data/box diagnosis", "records, class map, rbox conversion, loss targets", AMBER, 0.88),
    ]
    fig, ax = setup_ax(PAPER_SIZE if paper else PPT_SIZE, PAPER_BG if paper else BG)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.text(0.05, 3.72, "DIOR-R diagnostic timeline", fontsize=10.5 if paper else 21, weight="bold", color=INK)
    ax.text(0.06, 3.45, "Invalid runs are shown as failure signatures, not as baseline evidence.", fontsize=6.6 if paper else 9.4, color=MUTED)
    ax.plot([0.8, 9.2], [1.85, 1.85], color=LINE, lw=2)
    xs = np.linspace(0.8, 9.2, len(events))
    for x, (date, title, body, color, yoff) in zip(xs, events):
        ax.scatter([x], [1.85], s=85 if paper else 170, color=color, edgecolor="#ffffff", linewidth=1.0, zorder=2)
        y = 2.05 + yoff if x < 8.8 else 0.42
        va = "bottom" if y > 2 else "top"
        ax.plot([x, x], [1.85, y - 0.05 if y > 2 else y + 0.05], color=color, lw=0.8)
        ax.text(x, y + (0.05 if y > 2 else -0.05), title, ha="center", va=va, fontsize=6.7 if paper else 9.6, weight="bold", color=INK)
        wrapped = "\n".join(textwrap.wrap(body, width=18 if paper else 22))
        ax.text(x, y - (0.18 if y > 2 else 0.30), wrapped, ha="center", va=va, fontsize=5.7 if paper else 8.1, color=MUTED, linespacing=1.1)
        ax.text(x, 1.43, date, ha="center", fontsize=5.7 if paper else 8.1, color=SLATE)
    if paper:
        save_paper(fig, "fig_dior_diagnostic_timeline")
    else:
        save_ppt(fig, "dior_diagnostic_timeline")


def render_status_table_ppt(rows: list[MetricRow]) -> None:
    fig, ax = setup_ax(PPT_SIZE, BG)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.text(0.45, 8.42, "2026-06-08 Route Status Table", fontsize=22, weight="bold", color=INK)
    ax.text(0.47, 8.04, "Paper-facing claims stay limited to completed or explicitly active evidence.", fontsize=9.5, color=MUTED)
    headers = ["Track", "State", "mAP/AP50", "Use in paper"]
    table_rows = [
        ["DOTA2 S0 RoITrans", "complete", "0.6088 / 0.6090", "baseline gate"],
        ["DOTA2 GeoNexus S1 GPU-1", "complete", "0.6177 / 0.6180", "positive evidence"],
        ["DOTA2 S1 LR replicates", "complete", "0.5997 / 0.6000; 0.6047 / 0.6050", "stability context"],
        ["DOTA2 GeoNexus S2", "active", "epoch-4 0.6038 / 0.6040", "not yet stronger"],
        ["DIOR-R detector path", "blocked", "NaN/Inf", "diagnosis only"],
        ["DOTA v1.5 GeoNexus", "archive", "0.38 range", "diagnostic only"],
    ]
    x0, y0 = 0.55, 7.25
    widths = [4.0, 2.1, 4.2, 4.2]
    row_h = 0.88
    ax.add_patch(patches.Rectangle((x0, y0), sum(widths), row_h, facecolor=INK, edgecolor=INK, linewidth=0.8))
    xpos = x0
    for header, width in zip(headers, widths):
        ax.text(xpos + 0.16, y0 + row_h / 2, header, va="center", fontsize=10.2, color="#ffffff", weight="bold")
        xpos += width
    for i, row in enumerate(table_rows):
        y = y0 - (i + 1) * row_h
        fc = "#ffffff" if i % 2 == 0 else "#f1f4f8"
        ax.add_patch(patches.Rectangle((x0, y), sum(widths), row_h, facecolor=fc, edgecolor=LINE, linewidth=0.8))
        xpos = x0
        for j, (cell, width) in enumerate(zip(row, widths)):
            color = GREEN if cell == "complete" else RED if cell == "blocked" else AMBER if cell in {"active", "archive"} else INK
            ax.text(xpos + 0.16, y + row_h / 2, cell, va="center", fontsize=9.2, color=color if j == 1 else INK)
            xpos += width
    ax.text(0.55, 0.65, "Generated table figure; source map is emitted in paper_assets_20260608.", fontsize=8.8, color=MUTED)
    save_ppt(fig, "status_table")


def render_dotav2_contact_sheet(paper: bool = False) -> None:
    crops = select_real_crops(6)
    fig, axes = plt.subplots(2, 3, figsize=PAPER_SIZE if paper else PPT_SIZE, dpi=PAPER_DPI if paper else PPT_DPI)
    fig.patch.set_facecolor(PAPER_BG if paper else BG)
    for ax, spec in zip(axes.ravel(), crops):
        ax.set_axis_off()
        ax.set_facecolor(PAPER_BG if paper else BG)
        image_path = DOTAV2_IMAGES / f"{spec.image_id}.jpg"
        with Image.open(image_path) as im:
            crop = im.crop(spec.crop).resize((720, 405))
            arr = np.asarray(crop)
        ax.imshow(arr)
        ax.text(
            0.02,
            0.08,
            f"{spec.image_id} | {spec.cls}",
            transform=ax.transAxes,
            fontsize=6.2 if paper else 10.2,
            color="#ffffff",
            bbox=dict(facecolor=(0, 0, 0, 0.55), edgecolor="none", pad=2.2),
        )
    fig.suptitle("Deterministic DOTAv2 crop contact sheet", x=0.02, ha="left", fontsize=10.5 if paper else 21, weight="bold", color=INK)
    fig.text(0.02, 0.015, "Source: local DOTAv2/images/train and DOTAv2/labels/train; crops are deterministic visualization context, not model predictions.", fontsize=6.2 if paper else 8.8, color=MUTED)
    fig.tight_layout(rect=(0, 0.035, 1, 0.94))
    if paper:
        save_paper(fig, "fig_dotav2_contact_sheet")
    else:
        save_ppt(fig, "dotav2_contact_sheet")


def latex_escape(s: object) -> str:
    text = str(s)
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(repl.get(ch, ch) for ch in text)


def fmt_metric(v: float | None) -> str:
    return "Pending" if v is None else f"{v:.4f}"


def write_csv(path: Path, headers: list[str], rows: Iterable[Iterable[object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


def write_latex_table(path: Path, caption: str, label: str, headers: list[str], rows: list[list[object]], align: str | None = None, resize: bool = False) -> None:
    align = align or ("l" * len(headers))
    with path.open("w", encoding="utf-8") as f:
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write("\\small\n")
        f.write(f"\\caption{{{latex_escape(caption)}}}\n")
        f.write(f"\\label{{{label}}}\n")
        if resize:
            f.write("\\resizebox{\\linewidth}{!}{%\n")
        f.write(f"\\begin{{tabular}}{{{align}}}\n")
        f.write("\\toprule\n")
        f.write(" & ".join(latex_escape(h) for h in headers) + " \\\\\n")
        f.write("\\midrule\n")
        for row in rows:
            f.write(" & ".join(latex_escape(x) for x in row) + " \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        if resize:
            f.write("}\n")
        f.write("\\end{table}\n")


def render_tables(rows: list[MetricRow], route: list[dict]) -> None:
    main_headers = ["Detector", "Dataset", "Status", "mAP", "AP50", "Note"]
    main_rows = [[r.name, r.dataset, r.status, fmt_metric(r.map), fmt_metric(r.ap50), r.note] for r in rows]
    write_csv(PAPER_DIR / "table_dota2_baselines.csv", main_headers, main_rows)
    write_latex_table(
        PAPER_DIR / "table_dota2_baselines.tex",
        "DOTA2 baseline, S1, and active S2 gate evidence used for the current GeoNexus-RSD route.",
        "tab:dota2_baselines",
        main_headers,
        main_rows,
        align="lllrrl",
    )

    stage_headers = ["Stage", "Dataset role", "Best mAP", "Best AP50", "Status"]
    stage_rows = []
    for r in route:
        stage_rows.append([r["stage"], r["kind"], fmt_metric(r["map"]), fmt_metric(r["ap50"]), "Pending" if r["map"] is None else "Measured"])
    stage_rows.extend(
        [
            ["FAIR1M", "stretch evidence", "TBD", "TBD", "Paused"],
        ]
    )
    write_csv(PAPER_DIR / "table_geonexus_stage_ablation.csv", stage_headers, stage_rows)
    write_latex_table(
        PAPER_DIR / "table_geonexus_stage_ablation.tex",
        "GeoNexus stage table with DOTA2 S1/S2 gate evidence separated from archive diagnostics and blocked DIOR-R evidence.",
        "tab:geonexus_stage_ablation",
        stage_headers,
        stage_rows,
        align="llrrl",
    )

    dior_headers = ["Run", "Failure signature", "Evidence status", "Next action"]
    dior_rows = [
        ["ORCNN R50 initial", "accepted startup; later unstable/invalid", "Not cited", "Inspect DIOR-R records and rotated boxes"],
        ["ORCNN R50 low-LR diagnostic", "grad_norm/loss NaN; final validation 0 dets", "Invalid", "Check class mapping and loss targets"],
        ["RoI Transformer R50", "loss_rpn and cascade losses NaN before epoch-1 validation", "Invalid", "Diagnose data/box-coder path before relaunch"],
        ["Rotated RetinaNet R50", "loss_bbox=inf at epoch 1 iteration 1200", "Invalid", "Do not relaunch unchanged detector"],
    ]
    write_csv(PAPER_DIR / "table_dior_r_diagnostics.csv", dior_headers, dior_rows)
    write_latex_table(
        PAPER_DIR / "table_dior_r_diagnostics.tex",
        "DIOR-R diagnostic failures; invalid runs are not baseline evidence.",
        "tab:dior_r_diagnostics",
        dior_headers,
        dior_rows,
        align="llll",
        resize=True,
    )

    prompt_headers = ["Design", "Mechanism", "DOTA2 status", "DIOR-R status", "Use"]
    prompt_rows = [
        ["Flat class prompts", "RemoteCLIP text embedding per class name", "S1 complete", "Blocked", "Measured DOTA2 S1 evidence"],
        ["Hierarchy prompts", "parent/alias/confusion descriptors plus relation matrix", "S2 active", "Pending", "Do not claim final gain yet"],
        ["Context adapter", "tile/region-conditioned prompt modulation", "Paused", "Pending", "Future ablation after S2/DIOR-R"],
        ["Pseudo-label purification", "detector score plus hierarchy plus crop-text agreement", "Paused", "Pending", "Planned; no numeric claim"],
        ["Routing", "optional class/module route selector", "Paused", "Paused", "Secondary future work"],
    ]
    write_csv(PAPER_DIR / "table_prompt_design.csv", prompt_headers, prompt_rows)
    write_latex_table(
        PAPER_DIR / "table_prompt_design.tex",
        "Prompt and module design table. Pending rows are not reported as completed results.",
        "tab:prompt_design",
        prompt_headers,
        prompt_rows,
        align="lllll",
        resize=True,
    )

    efficiency_headers = ["Model", "Params", "Peak memory", "Latency/tile", "FLOPs", "Status"]
    efficiency_rows = [
        ["RoI Transformer S0", "TBD", "TBD", "TBD", "TBD", "Pending measurement"],
        ["GeoNexus S1 RemoteCLIP prompts", "TBD", "TBD", "TBD", "TBD", "Pending measurement"],
        ["GeoNexus S2 hierarchy regularizer", "TBD", "TBD", "TBD", "TBD", "Pending final run"],
        ["Context adapter", "TBD", "TBD", "TBD", "TBD", "Paused"],
        ["Pseudo-label purification", "TBD", "TBD", "TBD", "TBD", "Planned"],
    ]
    write_csv(PAPER_DIR / "table_efficiency_reporting.csv", efficiency_headers, efficiency_rows)
    write_latex_table(
        PAPER_DIR / "table_efficiency_reporting.tex",
        "Efficiency and reporting shell. Values must be measured before final submission.",
        "tab:efficiency_reporting",
        efficiency_headers,
        efficiency_rows,
        align="llllll",
    )

    source_headers = ["Artifact", "Number or claim", "Source"]
    source_rows = [[r.name, f"{fmt_metric(r.map)} / {fmt_metric(r.ap50)}", r.source] for r in rows]
    source_rows.extend([[r["stage"], f"{fmt_metric(r['map'])} / {fmt_metric(r['ap50'])}", r["source"]] for r in route])
    source_rows.extend(
        [
            ["DIOR-R NaN diagnosis", "invalid; do not cite", "docs/experiments/20260607_dior_orcnn_nan_diag_and_roi_trans_launch.md"],
            ["DOTA2 S1/S2 launch note", "S1 complete; S2 epoch-4 active", "docs/experiments/20260608_dota2_s1_complete_and_s2_launch.md"],
            ["Route gate", "DOTA2 before DIOR-R before FAIR1M", "PROJECT_INSTRUCTIONS.md"],
        ]
    )
    write_csv(PAPER_DIR / "table_artifact_sources.csv", source_headers, source_rows)
    write_latex_table(
        PAPER_DIR / "table_artifact_sources.tex",
        "Metric and claim source map for generated figures and tables.",
        "tab:artifact_sources",
        source_headers,
        source_rows,
        align="lll",
    )

    index_headers = ["Asset", "Format", "Use"]
    index_rows = [
        ["method_framework", "PPT PNG/SVG; paper PDF/SVG", "method overview with real DOTAv2 crop"],
        ["dota2_baseline_lollipop", "PPT PNG/SVG; paper PDF/SVG", "ranked DOTA2 evidence including S1/S2"],
        ["dota2_class_heatmap", "PPT PNG/SVG; paper PDF/SVG", "grouped class evidence panels"],
        ["detector_family_grouped", "PPT PNG/SVG; paper PDF/SVG", "detector-family status comparison"],
        ["route_progression", "PPT PNG/SVG; paper PDF/SVG", "DOTA2-first route gate"],
        ["dior_diagnostic_timeline", "PPT PNG/SVG; paper PDF/SVG", "DIOR-R diagnostic flow"],
        ["status_table", "PPT PNG/SVG", "slide-friendly current status table"],
        ["dotav2_contact_sheet", "PPT PNG/SVG; paper PDF/SVG", "deterministic real DOTAv2 crop/contact sheet"],
        ["prompt_design", "CSV/TEX", "prompt/module design and status"],
        ["efficiency_reporting", "CSV/TEX", "efficiency shell with TBD measurements"],
    ]
    write_csv(PAPER_DIR / "table_visual_asset_index.csv", index_headers, index_rows)
    write_latex_table(
        PAPER_DIR / "table_visual_asset_index.tex",
        "Current visualization asset index for the 2026-06-08 update.",
        "tab:visual_asset_index",
        index_headers,
        index_rows,
        align="lll",
    )


def write_readmes() -> None:
    ppt_files = sorted(p.name for p in PPT_DIR.iterdir() if p.is_file())
    paper_files = sorted(p.name for p in PAPER_DIR.iterdir() if p.is_file())
    (PPT_DIR / "README.md").write_text(
        "\n".join(
            [
                "# GeoNexus-RSD Academic PowerPoint Assets - 2026-06-08",
                "",
                "Generated by `scripts/make_academic_assets_20260608.py`.",
                "",
                "All PNG figures are 16:9 slide assets. Numeric labels are copied from local project records; active, archive, and blocked rows are explicitly marked.",
                "",
                "Files:",
                *[f"- `{name}`" for name in ppt_files if name != "README.md"],
                "",
            ]
        ),
        encoding="utf-8",
    )
    (PAPER_DIR / "README.md").write_text(
        "\n".join(
            [
                "# GeoNexus-RSD Paper Assets - 2026-06-08",
                "",
                "Generated by `scripts/make_academic_assets_20260608.py`.",
                "",
                "Figures are emitted as SVG/PDF. Tables are emitted as LaTeX booktabs fragments plus CSV source copies.",
                "",
                "Files:",
                *[f"- `{name}`" for name in paper_files if name != "README.md"],
                "",
            ]
        ),
        encoding="utf-8",
    )


def validate_outputs() -> None:
    pngs = sorted(PPT_DIR.glob("*.png"))
    if not pngs:
        raise AssertionError("No PPT PNGs were generated")
    for path in pngs:
        with Image.open(path) as im:
            w, h = im.size
        if w < 2500:
            raise AssertionError(f"{path} is only {w}px wide")
        ratio = w / h
        if not math.isclose(ratio, 16 / 9, rel_tol=0.0, abs_tol=0.04):
            raise AssertionError(f"{path} is not close to 16:9: {w}x{h}")
    for tex in PAPER_DIR.glob("*.tex"):
        text = tex.read_text(encoding="utf-8")
        if "\\toprule" not in text or "\\midrule" not in text or "\\bottomrule" not in text:
            raise AssertionError(f"{tex} is missing booktabs rules")
        if "|" in text.split("\\begin{tabular}", 1)[1].split("\n", 1)[0]:
            raise AssertionError(f"{tex} appears to use vertical table rules")
    for pdf in PAPER_DIR.glob("*.pdf"):
        if pdf.stat().st_size <= 1000:
            raise AssertionError(f"{pdf} is unexpectedly small")
    for directory in [PPT_DIR, PAPER_DIR]:
        readme = directory / "README.md"
        text = readme.read_text(encoding="utf-8")
        files = sorted(p.name for p in directory.iterdir() if p.is_file() and p.name != "README.md")
        missing = [name for name in files if f"`{name}`" not in text]
        if missing:
            raise AssertionError(f"{readme} does not list: {missing}")
    stale = [
        "DOTA2 S1 " + "pending",
        "S2 not " + "launched",
        "DOTA2 " + "pending",
        "DOTA2 gate still " + "pending",
    ]
    for directory in [PPT_DIR, PAPER_DIR]:
        for path in directory.glob("*"):
            if path.suffix.lower() not in {".md", ".tex", ".csv", ".svg"}:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for phrase in stale:
                if phrase in text:
                    raise AssertionError(f"Stale wording {phrase!r} remains in {path}")


def render_all() -> None:
    ensure_dirs()
    rows, route = load_evidence()
    render_method_framework_ppt()
    render_method_framework_paper()
    render_dotav2_contact_sheet(paper=False)
    render_dotav2_contact_sheet(paper=True)
    render_dota2_baseline_plot(rows, paper=False)
    render_dota2_baseline_plot(rows, paper=True)
    render_class_heatmap(paper=False)
    render_class_heatmap(paper=True)
    render_detector_family_plot(rows, paper=False)
    render_detector_family_plot(rows, paper=True)
    render_route_progression(route, paper=False)
    render_route_progression(route, paper=True)
    render_dior_timeline(paper=False)
    render_dior_timeline(paper=True)
    render_status_table_ppt(rows)
    render_tables(rows, route)
    write_readmes()
    validate_outputs()


if __name__ == "__main__":
    render_all()
