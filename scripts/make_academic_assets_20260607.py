"""Generate publication-style GeoNexus-RSD figures and tables.

The script only uses local project records for numeric evidence. Planned or
blocked rows are labelled as such in the generated tables and figures.
"""

from __future__ import annotations

import csv
import json
import math
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from matplotlib.colors import LinearSegmentedColormap


ROOT = Path(__file__).resolve().parents[1]
PPT_DIR = ROOT / "artifacts" / "ppt_assets_20260607_v2"
PAPER_DIR = ROOT / "artifacts" / "paper_assets_20260607_v2"
DOCS = ROOT / "docs" / "experiments"

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
            "RoI Transformer R50",
            "DOTA2 ss_val",
            "Complete",
            roi_map,
            roi_ap50,
            "docs/experiments/20260603_s0_dota2_roi_trans_validpng_metrics.json",
            "S0 gate; valid-PNG recovery",
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
            "Active/Pending",
            None,
            None,
            "docs/experiments/20260605_dota2_baseline_status.md",
            "Epoch-12 metric pending in local status",
        ),
    ]

    route = [
        {
            "stage": "S0 RoITrans",
            "kind": "DOTA2 formal",
            "map": roi_map,
            "ap50": roi_ap50,
            "source": "20260603_s0_dota2_roi_trans_validpng_metrics.json",
        },
        {
            "stage": "S1 prompt",
            "kind": "DOTA v1.5 diagnostic",
            "map": float(s1["best_metrics"]["dota/mAP"]),
            "ap50": float(s1["best_metrics"]["dota/AP50"]),
            "source": "20260605_geonexus_s1_retry2_metrics.json",
        },
        {
            "stage": "S2 hierarchy",
            "kind": "DOTA v1.5 diagnostic",
            "map": float(s2["best_metrics"]["dota/mAP"]),
            "ap50": float(s2["best_metrics"]["dota/AP50"]),
            "source": "20260605_geonexus_s2_rerun_s1e32_metrics.json",
        },
        {
            "stage": "S3 scene",
            "kind": "DOTA v1.5 diagnostic",
            "map": float(s3["best_metrics"]["dota/mAP"]),
            "ap50": float(s3["best_metrics"]["dota/AP50"]),
            "source": "20260605_geonexus_s3_rerun_s2e4_metrics.json",
        },
        {
            "stage": "S3 identity",
            "kind": "DOTA v1.5 diagnostic",
            "map": float(s3_id["best_metrics"]["dota/mAP"]),
            "ap50": float(s3_id["best_metrics"]["dota/AP50"]),
            "source": "20260605_geonexus_s3_identity_rerun_s2e4_metrics.json",
        },
        {
            "stage": "S3 off",
            "kind": "DOTA v1.5 diagnostic",
            "map": float(s3_off["best_metrics"]["dota/mAP"]),
            "ap50": float(s3_off["best_metrics"]["dota/AP50"]),
            "source": "20260606_geonexus_s3_adapter_off_rerun_s2e4_metrics.json",
        },
        {
            "stage": "S2 refine",
            "kind": "DOTA v1.5 diagnostic",
            "map": float(s2_refine["best_metrics"]["dota/mAP"]),
            "ap50": float(s2_refine["best_metrics"]["dota/AP50"]),
            "source": "20260606_geonexus_s2_refine_s2e4_lr1e4_metrics.json",
        },
        {
            "stage": "DOTA2 S1/S2",
            "kind": "DOTA2 pending",
            "map": None,
            "ap50": None,
            "source": "PROJECT_INSTRUCTIONS.md route gate",
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


def draw_aerial_tile(ax, x, y, w, h, label="schematic tile", scale=1.0):
    ax.add_patch(patches.Rectangle((x, y), w, h, facecolor="#dfeadf", edgecolor=LINE, linewidth=1.0))
    ax.add_patch(patches.Polygon([(x, y + h * 0.55), (x + w, y + h * 0.82), (x + w, y + h), (x, y + h)], color="#b7cdbd", alpha=0.95))
    ax.add_patch(patches.Polygon([(x, y), (x + w * 0.45, y), (x + w, y + h * 0.34), (x + w, y + h * 0.50), (x, y + h * 0.22)], color="#b9c8d6", alpha=0.8))
    for i, (cx, cy, rw, rh, ang, col) in enumerate(
        [
            (0.27, 0.32, 0.20, 0.045, 20, "#ffffff"),
            (0.44, 0.40, 0.16, 0.040, -17, "#f4f6fa"),
            (0.64, 0.63, 0.26, 0.060, 15, "#fdfdfd"),
            (0.72, 0.25, 0.13, 0.035, -24, "#fbfbfb"),
        ]
    ):
        rect = patches.Rectangle(
            (x + w * cx - w * rw / 2, y + h * cy - h * rh / 2),
            w * rw,
            h * rh,
            angle=ang,
            facecolor=col,
            edgecolor="#7a8796",
            linewidth=0.55 * scale,
        )
        ax.add_patch(rect)
    ax.text(x + 0.06, y + 0.07, label, fontsize=6.6 * scale, color=SLATE, va="bottom")


def render_method_framework_ppt() -> None:
    fig, ax = setup_ax(PPT_SIZE, BG)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.text(0.38, 8.55, "GeoNexus-RSD Framework", fontsize=22, weight="bold", color=INK)
    ax.text(
        0.40,
        8.18,
        "Original schematic inspired by remote-sensing oriented detection and VLM-prompt papers; metrics remain tied to local records.",
        fontsize=9.8,
        color=MUTED,
    )

    rounded_box(ax, (0.45, 5.78), (2.2, 1.86), "Input tile", "DOTA2 crop; valid PNGs", TEAL, "#ffffff", fontsize=9.5)
    draw_aerial_tile(ax, 0.71, 5.98, 1.62, 0.72, "schematic", 0.85)
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
    draw_aerial_tile(ax, 14.26, 6.05, 1.05, 0.62, "OBBs", 0.70)

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
    rounded_box(ax, (8.35, 2.95), (3.15, 1.62), "Scene adapter", "Scene context modulates prompt weights; DOTA2 gate still pending.", PURPLE, "#ffffff", fontsize=9.3)
    rounded_box(ax, (12.00, 2.95), (3.15, 1.62), "Evidence guard", "DOTA v1.5 diagnostics separated from DOTA2 formal route.", RED, "#ffffff", fontsize=9.3)

    for xs, xe in [(2.65, 3.15), (5.40, 5.88), (8.13, 8.62), (10.87, 11.35), (13.60, 14.05)]:
        arrow(ax, (xs, 6.72), (xe, 6.72), BLUE)
    arrow(ax, (2.65, 3.76), (8.75, 5.85), AMBER, rad=-0.10)
    arrow(ax, (6.35, 4.57), (9.05, 5.85), BLUE, rad=-0.08)
    arrow(ax, (9.95, 4.57), (10.10, 5.85), PURPLE, rad=0.10)
    arrow(ax, (13.35, 4.57), (12.35, 5.85), RED, rad=0.08)

    ax.text(0.55, 0.68, "Note: tile content is schematic; no unpublished detection image is implied.", fontsize=8.7, color=MUTED)
    save_ppt(fig, "method_framework")


def render_method_framework_paper() -> None:
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
    for x0, x1 in [(1.80, 2.15), (3.80, 4.15), (5.80, 6.15), (7.80, 8.15), (9.80, 10.15)]:
        arrow(ax, (x0, 5.26), (x1, 5.26), SLATE, lw=0.8)
    aux = [
        ((1.10, 2.52), (2.30, 1.00), "Prompt bank", "class aliases + 512-D text embeddings", AMBER),
        ((4.85, 2.52), (2.30, 1.00), "Hierarchy", "taxonomy constraints and confusion groups", BLUE),
        ((8.60, 2.52), (2.30, 1.00), "Scene context", "adapter gated by experiment route", PURPLE),
    ]
    for xy, wh, title, body, color in aux:
        rounded_box(ax, xy, wh, title, body, color=color, fc="#ffffff", fontsize=7.2)
    arrow(ax, (2.25, 3.52), (6.65, 4.75), AMBER, lw=0.8, rad=-0.1)
    arrow(ax, (6.00, 3.52), (6.95, 4.75), BLUE, lw=0.8)
    arrow(ax, (9.75, 3.52), (7.25, 4.75), PURPLE, lw=0.8, rad=0.1)
    ax.text(0.0, 0.22, "Schematic only; numbers are reported in separate evidence tables.", fontsize=6.5, color=MUTED)
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
    colors = [BLUE if "RoI Transformer" in n else TEAL if "OpenRSD" in n else GREEN if v > 0.55 else AMBER for n, v in zip(names, vals)]
    ax.hlines(y, 0, vals, color=LINE, lw=5 if paper else 8, zorder=1)
    ax.scatter(vals, y, s=85 if paper else 190, color=colors, edgecolor="#ffffff", linewidth=0.9, zorder=2)
    ax.axvline(0.6088, color=BLUE, linestyle="--", lw=1.0)
    ax.text(0.6088, len(y) - 0.55, "S0 gate", ha="center", va="bottom", fontsize=7.0 if paper else 9.0, color=BLUE)
    for yi, v, ap50 in zip(y, vals, ap50s):
        ax.text(v + 0.014, yi, f"{v:.4f}/{ap50:.4f}", va="center", fontsize=7.1 if paper else 10.0, color=INK)
    ax.set_yticks(y, names, fontsize=7.2 if paper else 10.5, color=INK)
    ax.set_xlim(0, 0.69)
    ax.set_xlabel("DOTA2 ss_val mAP / AP50", fontsize=7.3 if paper else 10.0, color=MUTED)
    ax.tick_params(axis="x", labelsize=6.8 if paper else 9.2, colors=MUTED)
    ax.grid(axis="x", color=FAINT, lw=0.8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("DOTA2 baseline comparison", loc="left", fontsize=10.5 if paper else 21, weight="bold", color=INK, pad=24 if paper else 34)
    ax.text(
        0,
        1.015,
        "Measured completed runs only; RTMDet-L remains pending in the source status note.",
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
    classes = [d["class"] for d in per_class]
    ap = np.array([float(d["ap"]) for d in per_class])
    recall = np.array([float(d["recall"]) for d in per_class])
    gts = np.array([float(d["gts"]) for d in per_class])
    dets = np.array([float(d["dets"]) for d in per_class])
    density = np.log10(gts + 1.0) / np.log10(gts.max() + 1.0)
    values = np.vstack([ap, recall, density])
    labels = ["AP", "Recall", "log GT count"]

    fig, ax = plt.subplots(figsize=PAPER_SIZE if paper else PPT_SIZE, dpi=PAPER_DPI if paper else PPT_DPI)
    fig.patch.set_facecolor(PAPER_BG if paper else BG)
    ax.set_facecolor(PAPER_BG if paper else BG)
    cmap = LinearSegmentedColormap.from_list("geo_heat", ["#f5f7fb", "#b9d6ee", "#2f6fed"])
    im = ax.imshow(values, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(classes)), classes, rotation=55, ha="right", fontsize=5.9 if paper else 8.1)
    ax.set_yticks(np.arange(3), labels, fontsize=7.2 if paper else 10.2)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            label = f"{values[i, j]:.2f}" if i < 2 else f"{int(gts[j])}"
            ax.text(j, i, label, ha="center", va="center", fontsize=4.5 if paper else 6.7, color="#111827" if values[i, j] < 0.62 else "white")
    cbar = fig.colorbar(im, ax=ax, fraction=0.030, pad=0.018)
    cbar.ax.tick_params(labelsize=6.2 if paper else 8.2, colors=MUTED)
    ax.set_title("RoI Transformer DOTA2 S0 class evidence", loc="left", fontsize=10.5 if paper else 21, weight="bold", color=INK, pad=10)
    ax.text(
        0.0,
        -0.30 if paper else -0.24,
        "Source: 20260603_s0_dota2_roi_trans_validpng_metrics.json; AP and recall are DOTAMetric per-class summaries.",
        transform=ax.transAxes,
        fontsize=6.2 if paper else 8.8,
        color=MUTED,
        va="top",
    )
    for spine in ax.spines.values():
        spine.set_visible(False)
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
    measured = [r for r in route if r["map"] is not None]
    x = np.arange(len(measured))
    vals = [r["map"] for r in measured]
    colors = [BLUE if "DOTA2" in r["kind"] else AMBER for r in measured]
    fig, ax = plt.subplots(figsize=PAPER_SIZE if paper else PPT_SIZE, dpi=PAPER_DPI if paper else PPT_DPI)
    fig.patch.set_facecolor(PAPER_BG if paper else BG)
    ax.set_facecolor(PAPER_BG if paper else BG)
    ax.plot(x, vals, color=SLATE, lw=1.2, zorder=1)
    ax.scatter(x, vals, s=90 if paper else 180, color=colors, edgecolor="#ffffff", linewidth=0.9, zorder=2)
    for xi, v, r in zip(x, vals, measured):
        ax.text(xi, v + 0.010, f"{v:.4f}", ha="center", fontsize=6.1 if paper else 8.8, color=INK)
    ax.axhspan(0.35, 0.41, color="#fff3d7", alpha=0.55, label="DOTA v1.5 diagnostic band")
    ax.axhline(0.6088, color=BLUE, lw=1.0, ls="--", label="DOTA2 RoITrans S0 gate")
    ax.set_xticks(x, [r["stage"] for r in measured], rotation=25, ha="right", fontsize=6.0 if paper else 9.0)
    ax.set_ylim(0.30, 0.64)
    ax.tick_params(axis="y", labelsize=6.3 if paper else 9.0, colors=MUTED)
    ax.set_ylabel("mAP", fontsize=7.0 if paper else 10.0, color=MUTED)
    ax.grid(axis="y", color=FAINT, lw=0.8)
    ax.legend(frameon=False, fontsize=6.1 if paper else 8.6, loc="lower right")
    ax.set_title("GeoNexus route progression with evidence separation", loc="left", fontsize=10.2 if paper else 20.5, weight="bold", color=INK, pad=10)
    ax.text(
        0.0,
        1.02,
        "DOTA v1.5 values are diagnostic/archive-only; DOTA2 GeoNexus entries remain pending until clean validation.",
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
        ("2026-06-06", "ORCNN launch", "accepted startup", GREEN, 0.18),
        ("2026-06-07", "ORCNN low-LR diag", "loss/grad_norm NaN; 0 dets", RED, 0.48),
        ("2026-06-07", "RoITrans launch", "finite startup", GREEN, 0.35),
        ("2026-06-07", "RoITrans stop", "loss_rpn and cascade losses NaN", RED, 0.68),
        ("Next", "Data/box diagnosis", "records, class map, box coder, loss targets", AMBER, 0.86),
    ]
    fig, ax = setup_ax(PAPER_SIZE if paper else PPT_SIZE, PAPER_BG if paper else BG)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.text(0.05, 3.72, "DIOR-R diagnostic timeline", fontsize=10.5 if paper else 21, weight="bold", color=INK)
    ax.text(0.06, 3.45, "Invalid runs are shown as failure signatures, not as baseline evidence.", fontsize=6.6 if paper else 9.4, color=MUTED)
    ax.plot([0.8, 9.2], [1.85, 1.85], color=LINE, lw=2)
    xs = np.linspace(1.0, 9.0, len(events))
    for x, (date, title, body, color, yoff) in zip(xs, events):
        ax.scatter([x], [1.85], s=85 if paper else 170, color=color, edgecolor="#ffffff", linewidth=1.0, zorder=2)
        y = 2.15 + yoff if x < 8.8 else 0.45
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


def write_latex_table(path: Path, caption: str, label: str, headers: list[str], rows: list[list[object]], align: str | None = None) -> None:
    align = align or ("l" * len(headers))
    with path.open("w", encoding="utf-8") as f:
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write(f"\\caption{{{latex_escape(caption)}}}\n")
        f.write(f"\\label{{{latex_escape(label)}}}\n")
        f.write(f"\\begin{{tabular}}{{{align}}}\n")
        f.write("\\toprule\n")
        f.write(" & ".join(latex_escape(h) for h in headers) + " \\\\\n")
        f.write("\\midrule\n")
        for row in rows:
            f.write(" & ".join(latex_escape(x) for x in row) + " \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")


def render_tables(rows: list[MetricRow], route: list[dict]) -> None:
    main_headers = ["Detector", "Dataset", "Status", "mAP", "AP50", "Note"]
    main_rows = [[r.name, r.dataset, r.status, fmt_metric(r.map), fmt_metric(r.ap50), r.note] for r in rows]
    write_csv(PAPER_DIR / "table_dota2_baselines.csv", main_headers, main_rows)
    write_latex_table(
        PAPER_DIR / "table_dota2_baselines.tex",
        "DOTA2 baseline evidence used for the current GeoNexus-RSD route gate.",
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
            ["DIOR-R S0", "cross-dataset required", "Pending", "Pending", "Blocked by NaN diagnosis"],
            ["FAIR1M", "stretch evidence", "TBD", "TBD", "Paused"],
        ]
    )
    write_csv(PAPER_DIR / "table_geonexus_stage_ablation.csv", stage_headers, stage_rows)
    write_latex_table(
        PAPER_DIR / "table_geonexus_stage_ablation.tex",
        "GeoNexus stage table with diagnostic DOTA v1.5 values separated from pending formal DOTA2 and DIOR-R evidence.",
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
        ["RetinaNet prior diagnostic", "loss=inf noted in project route gate", "Invalid", "Do not relaunch unchanged detector"],
    ]
    write_csv(PAPER_DIR / "table_dior_r_diagnostics.csv", dior_headers, dior_rows)
    write_latex_table(
        PAPER_DIR / "table_dior_r_diagnostics.tex",
        "DIOR-R diagnostic failures; invalid runs are not baseline evidence.",
        "tab:dior_r_diagnostics",
        dior_headers,
        dior_rows,
        align="llll",
    )

    source_headers = ["Artifact", "Number or claim", "Source"]
    source_rows = [[r.name, f"{fmt_metric(r.map)} / {fmt_metric(r.ap50)}", r.source] for r in rows]
    source_rows.extend([[r["stage"], f"{fmt_metric(r['map'])} / {fmt_metric(r['ap50'])}", r["source"]] for r in route])
    source_rows.extend(
        [
            ["DIOR-R NaN diagnosis", "invalid; do not cite", "docs/experiments/20260607_dior_orcnn_nan_diag_and_roi_trans_launch.md"],
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


def write_readmes() -> None:
    ppt_files = sorted(p.name for p in PPT_DIR.iterdir() if p.is_file())
    paper_files = sorted(p.name for p in PAPER_DIR.iterdir() if p.is_file())
    (PPT_DIR / "README.md").write_text(
        "\n".join(
            [
                "# GeoNexus-RSD Academic PowerPoint Assets - 2026-06-07 v2",
                "",
                "Generated by `scripts/make_academic_assets_20260607.py`.",
                "",
                "All PNG figures are 16:9 slide assets. Numeric labels are copied from local project records; pending and blocked rows are explicitly marked.",
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
                "# GeoNexus-RSD Paper Assets - 2026-06-07 v2",
                "",
                "Generated by `scripts/make_academic_assets_20260607.py`.",
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
    from PIL import Image

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


def render_all() -> None:
    ensure_dirs()
    rows, route = load_evidence()
    render_method_framework_ppt()
    render_method_framework_paper()
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
    render_tables(rows, route)
    write_readmes()
    validate_outputs()


if __name__ == "__main__":
    render_all()
