"""Render PowerPoint-ready GeoNexus-RSD figures for the 2026-06-07 update."""

from __future__ import annotations

from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import pandas as pd


OUT_DIR = Path("artifacts/ppt_assets_20260607")
W, H = 13.333, 7.5

BG = "#f7f8fb"
INK = "#17202a"
MUTED = "#687385"
BLUE = "#2f6fed"
TEAL = "#159a8c"
AMBER = "#c98514"
RED = "#c53b3b"
GREEN = "#2e8b57"
PURPLE = "#6b5bd6"
LINE = "#d4dae6"


def setup_fig():
    fig, ax = plt.subplots(figsize=(W, H), dpi=220)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")
    return fig, ax


def save(fig, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(OUT_DIR / f"{name}.{ext}", pad_inches=0.0)
    plt.close(fig)


def box(ax, xy, wh, title, body="", fc="#ffffff", ec=LINE, title_color=INK, lw=1.4):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.018,rounding_size=0.08",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(x + 0.18, y + h - 0.28, title, fontsize=13, weight="bold", color=title_color, va="top")
    if body:
        wrap_width = max(14, int(w * 8.1))
        ax.text(
            x + 0.18,
            y + h - 0.82,
            "\n".join(textwrap.wrap(body, width=wrap_width)),
            fontsize=8.9,
            color=MUTED,
            va="top",
            linespacing=1.25,
        )
    return patch


def arrow(ax, start, end, color=MUTED, rad=0.0, lw=1.7):
    arr = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=12,
        lw=lw,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=4,
        shrinkB=4,
    )
    ax.add_patch(arr)


def render_structure():
    fig, ax = setup_fig()
    ax.text(0.35, 8.55, "GeoNexus-RSD Updated Structure", fontsize=25, weight="bold", color=INK)
    ax.text(
        0.38,
        8.16,
        "DOTA2-centered oriented detection route: strong detector + hierarchy/context VLM prompting; DIOR-R held for data/loss diagnosis.",
        fontsize=11.5,
        color=MUTED,
    )

    box(ax, (0.45, 6.05), (2.75, 1.35), "Input Tiles", "DOTA2 1024 tiles; valid-PNG train filtering", fc="#ffffff")
    box(ax, (3.8, 6.05), (2.85, 1.35), "RoITrans S0", "Strong anchor: mAP 0.6088; AP50 0.6090", fc="#eef4ff", ec="#a9bff7")
    box(ax, (7.25, 6.05), (2.95, 1.35), "Proposal Features", "FPN + rotated RoI features", fc="#ffffff")
    box(ax, (10.9, 6.05), (2.55, 1.35), "OBB Output", "Class score + rotated box", fc="#edf8f6", ec="#9bd2cc")

    box(ax, (1.0, 3.35), (3.4, 1.55), "S1 Prompt Head", "RemoteCLIP ViT-B/32 prompt embeddings; DOTA2: 18 classes, 512-D", fc="#fffaf0", ec="#e8c277", title_color=AMBER)
    box(ax, (4.95, 3.35), (3.4, 1.55), "S2 Hierarchy Bank", "Taxonomy relations, aliases, confusion groups, hierarchy consistency", fc="#f2f7ff", ec="#9cb8eb", title_color=BLUE)
    box(ax, (8.9, 3.35), (3.4, 1.55), "S3 Scene Adapter", "Scene-conditioned prompt modulation; paused until S1/S2 stabilize", fc="#f5f2ff", ec="#b7acec", title_color=PURPLE)
    box(ax, (12.85, 3.35), (2.65, 1.55), "S4 Pseudo Labels", "VLM-assisted purification; paused for now", fc="#fff2f2", ec="#e4a4a4", title_color=RED)

    box(ax, (0.75, 1.08), (4.3, 1.36), "Current Gate", "Compare both DOTA2 S1 validations against RoITrans S0 before S2 launch.", fc="#ffffff")
    box(ax, (5.85, 1.08), (4.3, 1.36), "Cross-Dataset Gate", "DIOR-R ORCNN/RoITrans/RetinaNet are invalid due NaN/Inf or zero evidence.", fc="#ffffff")
    box(ax, (10.95, 1.08), (4.3, 1.36), "Paper Safety", "DOTA v1.5 GeoNexus ~0.38 mAP is diagnostic/archive-only, not headline evidence.", fc="#ffffff")

    arrow(ax, (3.2, 6.72), (3.8, 6.72), BLUE)
    arrow(ax, (6.65, 6.72), (7.25, 6.72), BLUE)
    arrow(ax, (10.2, 6.72), (10.9, 6.72), BLUE)
    arrow(ax, (2.7, 4.9), (4.05, 6.05), AMBER, rad=-0.08)
    arrow(ax, (6.65, 4.9), (7.95, 6.05), BLUE, rad=-0.08)
    arrow(ax, (10.6, 4.9), (8.75, 6.05), PURPLE, rad=0.08)
    arrow(ax, (13.95, 4.9), (12.0, 6.05), RED, rad=0.08)

    ax.text(0.45, 0.45, "Generated from project records on 2026-06-07. Formal benchmark order: DOTA2 -> DIOR-R -> FAIR1M.", fontsize=9.5, color=MUTED)
    save(fig, "geonexus_structure_16x9")


def render_baseline_plot():
    data = [
        ("RoI Transformer", 0.6088, 0.6090, "S0 anchor"),
        ("Oriented R-CNN", 0.5973, 0.5970, "complete"),
        ("S2ANet", 0.5869, 0.5870, "complete"),
        ("R3Det-KFIoU", 0.5633, 0.5630, "complete"),
        ("OpenRSD formal", 0.4202, 0.4200, "reference"),
        ("RTMDet-M", 0.3312, 0.3310, "low"),
        ("RTMDet-L", 0.2779, 0.2780, "degraded"),
    ]
    df = pd.DataFrame(data, columns=["Detector", "mAP", "AP50", "Note"])
    fig, ax = plt.subplots(figsize=(W, H), dpi=220)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    colors = [BLUE, GREEN, GREEN, GREEN, TEAL, AMBER, RED]
    bars = ax.barh(df["Detector"], df["mAP"], color=colors, height=0.62)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.70)
    ax.set_xlabel("DOTA2 ss_val mAP / AP50 at IoU 0.5", color=MUTED, labelpad=10)
    ax.grid(axis="x", color="#e1e6ef", linewidth=1)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="y", labelsize=12, colors=INK)
    ax.tick_params(axis="x", labelsize=10, colors=MUTED)

    for bar, ap50, note in zip(bars, df["AP50"], df["Note"]):
        x = bar.get_width()
        y = bar.get_y() + bar.get_height() / 2
        if x > 0.50:
            ax.text(x - 0.012, y, f"{x:.4f} / {ap50:.4f}", va="center", ha="right", fontsize=10.5, weight="bold", color="white")
        else:
            ax.text(x + 0.012, y, f"{x:.4f} / {ap50:.4f}", va="center", ha="left", fontsize=10.5, weight="bold", color=INK)
        ax.text(0.685, y, note, va="center", ha="right", fontsize=9.5, color=MUTED)

    ax.axvline(0.6088, color=BLUE, linestyle="--", linewidth=1.2, alpha=0.8)
    ax.text(0.6088, -0.68, "RoITrans S0 gate", ha="center", va="bottom", fontsize=9.5, color=BLUE)
    ax.set_title("Updated DOTA2 Baseline Evidence", loc="left", fontsize=24, weight="bold", color=INK, pad=22)
    ax.text(
        0,
        1.02,
        "GeoNexus DOTA2 S1 runs are launched/pending first validation; S2 waits for the better clean S1 checkpoint.",
        transform=ax.transAxes,
        fontsize=11,
        color=MUTED,
    )
    fig.tight_layout(rect=(0.06, 0.05, 0.98, 0.92))
    save(fig, "dota2_baseline_plot_16x9")


def render_status_table():
    rows = [
        ["DOTA2 S0", "RoITrans", "complete", "0.6088 / 0.6090", "formal anchor"],
        ["DOTA2 S0", "ORCNN, S2ANet, R3Det", "complete", "0.5973 / 0.5970; 0.5869 / 0.5870; 0.5633 / 0.5630", "secondary baselines"],
        ["DOTA2\nGeoNexus S1", "RemoteCLIP prompt head", "running", "pending first validation", "compare to 0.6088 gate"],
        ["DOTA2\nGeoNexus S1", "LR 1e-4 replicate", "running", "pending first validation", "candidate for S2 init"],
        ["DOTA2\nGeoNexus S2", "hierarchy regularizer", "paused", "not launched", "launch only from better clean S1"],
        ["DIOR-R", "ORCNN / RoITrans / RetinaNet", "blocked", "invalid: NaN, zero, or Inf evidence", "diagnose data and rotated boxes"],
        ["DOTA v1.5", "GeoNexus S1/S2/S3", "archive", "best diagnostic near 0.38 mAP", "not headline evidence"],
        ["FAIR1M", "fine-grained stretch", "paused", "not started", "after DOTA2 + DIOR-R stabilize"],
    ]
    cols = ["Track", "Run / Module", "State", "Metric", "Decision"]

    fig, ax = setup_fig()
    ax.text(0.35, 8.55, "Experiment Gate Table", fontsize=25, weight="bold", color=INK)
    ax.text(0.38, 8.16, "Use this as the presentation status slide for the 2026-06-07 experiment update.", fontsize=11.5, color=MUTED)

    x0, y0 = 0.35, 0.65
    table_w, row_h = 15.3, 0.76
    col_w = [2.75, 3.25, 1.6, 4.35, 3.35]
    header_y = 7.34

    ax.add_patch(Rectangle((x0, header_y), table_w, row_h, facecolor=INK, edgecolor=INK))
    x = x0
    for c, w in zip(cols, col_w):
        ax.text(x + 0.12, header_y + row_h / 2, c, va="center", fontsize=10.5, weight="bold", color="white")
        x += w

    state_colors = {
        "complete": GREEN,
        "running": BLUE,
        "paused": AMBER,
        "blocked": RED,
        "archive": MUTED,
    }

    for i, row in enumerate(rows):
        y = header_y - (i + 1) * row_h
        fc = "#ffffff" if i % 2 == 0 else "#f0f3f8"
        ax.add_patch(Rectangle((x0, y), table_w, row_h, facecolor=fc, edgecolor=LINE, linewidth=0.8))
        x = x0
        for j, (text, w) in enumerate(zip(row, col_w)):
            color = state_colors.get(text, INK) if j == 2 else INK
            weight = "bold" if j in (0, 2) else "normal"
            wrapped = "\n".join(
                "\n".join(textwrap.wrap(part, width=max(8, int(w * 7.1))))
                for part in text.split("\n")
            )
            ax.text(x + 0.12, y + row_h / 2, wrapped, va="center", fontsize=9.2, color=color, weight=weight, linespacing=1.12)
            x += w

    for xx in [x0 + sum(col_w[:i]) for i in range(1, len(col_w))]:
        ax.plot([xx, xx], [header_y - len(rows) * row_h, header_y + row_h], color=LINE, linewidth=0.8)

    ax.text(x0, 0.28, "Rule: do not cite DIOR-R detector runs or DOTA v1.5 GeoNexus as formal evidence under the current route.", fontsize=9.5, color=MUTED)
    save(fig, "experiment_gate_table_16x9")


def render_all():
    render_structure()
    render_baseline_plot()
    render_status_table()
    (OUT_DIR / "README.md").write_text(
        "\n".join(
            [
                "# GeoNexus-RSD PowerPoint Assets - 2026-06-07",
                "",
                "Generated slide-ready assets:",
                "",
                "- `geonexus_structure_16x9.png` / `.svg`: updated method and experiment-gate structure.",
                "- `dota2_baseline_plot_16x9.png` / `.svg`: current DOTA2 baseline metric plot.",
                "- `experiment_gate_table_16x9.png` / `.svg`: presentation status table.",
                "",
                "Use PNG for direct PowerPoint insertion. Use SVG if you want editable vector shapes/text.",
                "The figures intentionally mark DOTA2 GeoNexus S1 as pending and DIOR-R as blocked, matching the 2026-06-07 project records.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    render_all()
