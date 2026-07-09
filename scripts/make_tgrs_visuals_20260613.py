"""Regenerate manuscript-local TGRS visual assets for the June 13 draft.

The outputs are written directly under
_local_archive_20260601_pull_backup/docs/TGRS/figure.  The script uses only
local evidence records and the archived real DOTAv2 crop; no predictions or
synthetic bitmap imagery are generated.
"""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[1]
TGRS = ROOT / "_local_archive_20260601_pull_backup" / "docs" / "TGRS"
FIG = TGRS / "figure"
DOCS = ROOT / "docs" / "experiments"


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(read_text(path))


def require(path: Path, needle: str) -> None:
    text = read_text(path)
    if needle not in text:
        raise AssertionError(f"{path} is missing required evidence: {needle}")


def assert_close(value: float, expected: float, label: str, tol: float = 5e-5) -> None:
    if not math.isclose(float(value), expected, rel_tol=0.0, abs_tol=tol):
        raise AssertionError(f"{label}: expected {expected}, found {value}")


def assert_evidence() -> None:
    s0 = read_json(DOCS / "20260603_s0_dota2_roi_trans_validpng_metrics.json")
    assert_close(s0["metrics"]["dota/mAP"], 0.6088, "DOTA2 S0 mAP")
    assert_close(s0["metrics"]["dota/AP50"], 0.6090, "DOTA2 S0 AP50")

    require(DOCS / "20260608_dota2_s1_complete_and_s2_launch.md", "`0.6177 / 0.6180`")
    require(DOCS / "20260611_dota2_s2_loss0_replicates_analysis.md", "All 7 runs: best mean `0.620606`")
    require(DOCS / "20260611_dota2_s2_loss0_replicates_analysis.md", "final mean `0.616655`")
    require(DOCS / "20260613_dior_r_s0_sanitized_long_interim.md", "epoch 48: `dota/mAP=0.6531`, `dota/AP50=0.6530`")

    s1 = read_json(DOCS / "20260613_dior_r_geonexus_s1_s0e52_replicas_metrics.json")
    assert_close(s1["replicas"][0]["metrics"][-1]["dota_mAP"], 0.6750815511, "DIOR-R S1 rep0 mAP")
    assert_close(s1["replicas"][0]["metrics"][-1]["dota_AP50"], 0.675, "DIOR-R S1 rep0 AP50")
    assert_close(s1["replicas"][1]["metrics"][-1]["dota_mAP"], 0.6689543724, "DIOR-R S1 rep1 mAP")
    assert_close(s1["replicas"][1]["metrics"][-1]["dota_AP50"], 0.669, "DIOR-R S1 rep1 AP50")
    require(DOCS / "20260613_dior_r_geonexus_s2_hierarchy_replicas_launch.md", "Startup acceptance: reached `Epoch(train) [1][200/5862]`")


def make_crop_strip() -> None:
    src = FIG / "geonexus_tgrs_input_crop.png"
    meta = FIG / "geonexus_tgrs_input_crop.md"
    if not src.exists() or "real DOTAv2 train image crop" not in read_text(meta):
        raise FileNotFoundError("Expected archived real DOTAv2 crop and metadata are missing")

    img = Image.open(src).convert("RGB")
    w, h = img.size
    boxes = [
        (0, 0, int(w * 0.55), int(h * 0.55)),
        (int(w * 0.35), 0, w, int(h * 0.58)),
        (0, int(h * 0.38), int(w * 0.62), h),
        (int(w * 0.44), int(h * 0.34), w, h),
    ]
    thumbs = []
    for box in boxes:
        thumb = img.crop(box)
        thumb = ImageOps.fit(thumb, (300, 180), method=Image.Resampling.LANCZOS)
        thumbs.append(thumb)

    gap = 16
    canvas = Image.new("RGB", (4 * 300 + 3 * gap, 180), (248, 250, 252))
    for i, thumb in enumerate(thumbs):
        canvas.paste(thumb, (i * (300 + gap), 0))
    canvas.save(FIG / "geonexus_tgrs_qual_crop_strip.png")


def draw_architecture_scene() -> None:
    colors = {
        "ink": "#1d2530",
        "muted": "#667085",
        "line": "#cbd5e1",
        "paper": "#f8fafc",
        "blue": "#2563eb",
        "blue_fill": "#e8f1ff",
        "cyan": "#0891b2",
        "cyan_fill": "#e0f7ff",
        "teal": "#0f766e",
        "teal_fill": "#e6f7f4",
        "green": "#15803d",
        "green_fill": "#e9f8ee",
        "amber": "#b45309",
        "amber_fill": "#fff2cc",
        "orange": "#ea580c",
        "orange_fill": "#ffedd5",
        "violet": "#4f46e5",
        "violet_fill": "#eef2ff",
        "red": "#b91c1c",
        "red_fill": "#fee2e2",
        "rose": "#be123c",
        "gray_fill": "#f1f5f9",
    }

    fig, ax = plt.subplots(figsize=(14.4, 8.25))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8.8)
    ax.axis("off")
    boxes: list[tuple[str, float, float, float, float]] = []

    def panel(x: float, y: float, w: float, h: float, title: str, color: str) -> None:
        ax.add_patch(
            FancyBboxPatch(
                (x + 0.045, y - 0.045),
                w,
                h,
                boxstyle="round,pad=0.03,rounding_size=0.07",
                facecolor="#d9e2ef",
                edgecolor="none",
                alpha=0.42,
                zorder=2,
            )
        )
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.03,rounding_size=0.08",
                facecolor=colors["paper"],
                edgecolor=colors["line"],
                linewidth=1.0,
                zorder=0,
            )
        )
        ax.text(x + 0.18, y + h - 0.26, title, ha="left", va="top", fontsize=9.2, fontweight="bold", color=color, zorder=5)

    def node(
        name: str,
        x: float,
        y: float,
        w: float,
        h: float,
        label: str,
        edge: str,
        fill: str,
        fontsize: float = 8.6,
        dashed: bool = False,
    ) -> tuple[float, float]:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.03,rounding_size=0.07",
                facecolor=fill,
                edgecolor=edge,
                linewidth=1.15,
                linestyle=(0, (4, 2)) if dashed else "solid",
                zorder=3,
            )
        )
        ax.text(x + w / 2, y + h * 0.18, label, ha="center", va="bottom", fontsize=fontsize, color=colors["ink"], linespacing=1.05, zorder=5)
        boxes.append((name, x, y, w, h))
        return (x + w / 2, y + h / 2)

    def arrow(start: tuple[float, float], end: tuple[float, float], color: str, dashed: bool = False, rad: float = 0.0) -> None:
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=10,
                linewidth=1.25,
                color=color,
                linestyle=(0, (4, 2)) if dashed else "solid",
                connectionstyle=f"arc3,rad={rad}",
                shrinkA=7,
                shrinkB=7,
                zorder=1,
            )
        )

    def label(x: float, y: float, text: str, color: str = colors["muted"], size: float = 7.8) -> None:
        ax.text(x, y, text, ha="center", va="center", fontsize=size, color=color, zorder=5)

    def image_box(name: str, x: float, y: float, w: float, h: float, title: str, zoom: float) -> tuple[float, float]:
        ax.add_patch(
            FancyBboxPatch(
                (x + 0.045, y - 0.045),
                w,
                h,
                boxstyle="round,pad=0.03,rounding_size=0.08",
                facecolor="#d9e2ef",
                edgecolor="none",
                alpha=0.38,
                zorder=2,
            )
        )
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.03,rounding_size=0.08",
                facecolor="white",
                edgecolor=colors["blue"],
                linewidth=1.2,
                zorder=3,
            )
        )
        img = Image.open(FIG / "geonexus_tgrs_input_crop.png").convert("RGB")
        ab = AnnotationBbox(
            OffsetImage(img, zoom=zoom),
            (x + w / 2, y + h / 2 + 0.08),
            frameon=False,
            box_alignment=(0.5, 0.5),
            clip_on=True,
            zorder=4,
        )
        ax.add_artist(ab)
        ax.add_patch(Rectangle((x + 0.07, y + 0.07), w - 0.14, h - 0.14, facecolor="none", edgecolor="white", linewidth=2.0, zorder=4))
        ax.text(x + w / 2, y + 0.13, title, ha="center", va="bottom", fontsize=8.2, color=colors["ink"], fontweight="bold", zorder=5)
        boxes.append((name, x, y, w, h))
        return (x + w / 2, y + h / 2)

    def op_circle(x: float, y: float, text: str, edge: str, fill: str, size: float = 0.17) -> tuple[float, float]:
        ax.add_patch(Circle((x + 0.025, y - 0.025), size, facecolor="#d9e2ef", edgecolor="none", alpha=0.35, zorder=2))
        ax.add_patch(Circle((x, y), size, facecolor=fill, edgecolor=edge, linewidth=1.0, zorder=4))
        ax.text(x, y, text, ha="center", va="center", fontsize=7.2, color=edge, fontweight="bold", zorder=5)
        return (x, y)

    def mini_chip(x: float, y: float, w: float | str, text: str | None = None, edge: str | None = None, fill: str | None = None, size: float = 6.5) -> None:
        if isinstance(w, str):
            fill = edge
            edge = text
            text = w
            w = max(0.38, 0.085 * len(text) + 0.16)
        if text is None or edge is None or fill is None:
            raise AssertionError("mini_chip requires text, edge, and fill")
        ax.add_patch(FancyBboxPatch((x, y), w, 0.24, boxstyle="round,pad=0.018,rounding_size=0.05", facecolor=fill, edgecolor=edge, linewidth=0.75, zorder=4))
        ax.text(x + w / 2, y + 0.12, text, ha="center", va="center", fontsize=size, color=edge, fontweight="bold", zorder=5)

    def module_links(points: list[tuple[float, float]], color: str, dashed: bool = False) -> None:
        for start, end in zip(points, points[1:]):
            arrow(start, end, color, dashed=dashed)

    def clean_card(name: str, x: float, y: float, w: float, h: float, label_text: str, edge: str, fill: str, dashed: bool = False) -> tuple[float, float]:
        ax.add_patch(
            FancyBboxPatch(
                (x + 0.045, y - 0.045),
                w,
                h,
                boxstyle="round,pad=0.03,rounding_size=0.07",
                facecolor="#d9e2ef",
                edgecolor="none",
                alpha=0.34,
                zorder=2,
            )
        )
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.03,rounding_size=0.07",
                facecolor=fill,
                edgecolor=edge,
                linewidth=1.15,
                linestyle=(0, (4, 2)) if dashed else "solid",
                zorder=3,
            )
        )
        ax.text(x + w / 2, y + 0.21 * h, label_text, ha="center", va="center", fontsize=8.1, color=colors["ink"], linespacing=1.0, zorder=5)
        boxes.append((name, x, y, w, h))
        return (x + w / 2, y + h / 2)

    def icon_fpn(x: float, y: float, edge: str) -> None:
        for i, width in enumerate([0.5, 0.68, 0.86]):
            yy = y + 0.52 - i * 0.17
            ax.add_patch(Polygon([(x + 0.18, yy), (x + 0.18 + width, yy), (x + 0.18 + width - 0.13, yy + 0.11), (x + 0.18, yy + 0.11)], closed=True, facecolor="white", edgecolor=edge, linewidth=0.8, zorder=4))

    def icon_rot_boxes(cx: float, cy: float, edge: str) -> None:
        ax.add_patch(Rectangle((cx - 0.25, cy + 0.08), 0.5, 0.18, angle=16, facecolor="none", edgecolor=edge, linewidth=0.9, zorder=4))
        ax.add_patch(Rectangle((cx - 0.22, cy - 0.16), 0.46, 0.17, angle=-18, facecolor="none", edgecolor=edge, linewidth=0.9, zorder=4))

    def icon_matrix(x: float, y: float, edge: str) -> None:
        for i in range(5):
            for j in range(5):
                ax.add_patch(Rectangle((x + i * 0.07, y + j * 0.055), 0.048, 0.036, facecolor=edge, edgecolor="none", alpha=0.2 + 0.05 * (i + j), zorder=4))

    def icon_bars(x: float, y: float, edge: str) -> None:
        for i, h in enumerate([0.18, 0.34, 0.5]):
            ax.add_patch(Rectangle((x + i * 0.16, y), 0.09, h, facecolor=edge, edgecolor="none", alpha=0.55, zorder=4))

    def icon_tree(x: float, y: float, edge: str) -> None:
        ax.plot([x, x, x - 0.22, x + 0.22], [y + 0.38, y + 0.18, y, y], color=edge, linewidth=1.0, zorder=4)
        for px, py in [(x, y + 0.38), (x - 0.22, y), (x + 0.22, y)]:
            ax.add_patch(Circle((px, py), 0.045, facecolor="white", edgecolor=edge, linewidth=0.8, zorder=5))

    ax.add_patch(
        FancyBboxPatch(
            (0.15, 0.15),
            15.7,
            8.45,
            boxstyle="round,pad=0.04,rounding_size=0.08",
            facecolor="white",
            edgecolor=colors["line"],
            linewidth=1.0,
            zorder=0,
        )
    )
    ax.text(0.35, 8.35, "GeoNexus-RSD framework: geometry, prompts, and context", fontsize=12, fontweight="bold", color=colors["ink"], va="top", zorder=5)
    ax.text(0.35, 8.08, "A connected detector-language-context pipeline for prompt-aware oriented object detection.", fontsize=8.6, color=colors["muted"], va="top", zorder=5)

    panel(0.35, 5.05, 15.1, 2.75, "A. Oriented detector stream", colors["blue"])
    panel(0.35, 2.55, 15.1, 2.05, "B. Prompt hierarchy and multimodal fusion", colors["teal"])
    panel(0.35, 0.45, 15.1, 1.55, "C. Scene context and paused pseudo-label route", colors["amber"])

    crop = image_box("crop_final", 0.75, 5.5, 1.72, 1.5, "DOTAv2", 0.073)
    ax.add_patch(Rectangle((1.08, 6.03), 0.55, 0.25, angle=14, facecolor="none", edgecolor=colors["orange"], linewidth=1.0, zorder=5))
    ax.add_patch(Rectangle((1.45, 6.22), 0.32, 0.15, angle=-10, facecolor="none", edgecolor=colors["rose"], linewidth=1.0, zorder=5))
    fpn = clean_card("fpn_final", 3.0, 5.58, 1.18, 1.2, "FPN\npyramid", colors["blue"], colors["blue_fill"])
    icon_fpn(3.12, 6.03, colors["blue"])
    rpn = clean_card("rpn_final", 4.75, 5.58, 1.18, 1.2, "rotated\nRPN", colors["blue"], colors["blue_fill"])
    icon_rot_boxes(5.34, 6.35, colors["blue"])
    roi = clean_card("roi_final", 6.5, 5.58, 1.18, 1.2, "rotated\nRoI", colors["blue"], colors["blue_fill"])
    ax.add_patch(Rectangle((6.83, 6.36), 0.55, 0.2, angle=-25, facecolor="none", edgecolor=colors["blue"], linewidth=1.0, zorder=4))
    feat = clean_card("feat_final", 8.25, 5.58, 1.25, 1.2, "RoI\nfeatures", colors["blue"], colors["blue_fill"])
    for i, c in enumerate([colors["blue"], colors["cyan"], colors["violet"]]):
        ax.add_patch(Rectangle((8.58 + i * 0.15, 6.33 - i * 0.06), 0.42, 0.2, facecolor="white", edgecolor=c, linewidth=0.8, zorder=4))
    box_head = clean_card("box_final", 10.45, 6.12, 1.12, 0.72, "box\nhead", colors["blue"], colors["blue_fill"])
    op_circle(11.01, 6.65, "d", colors["blue"], "white", 0.12)
    prompt_head = clean_card("prompt_final", 10.45, 5.2, 1.28, 0.72, "prompt\nhead", colors["teal"], colors["teal_fill"])
    op_circle(11.09, 5.72, "s", colors["teal"], "white", 0.12)
    out = clean_card("out_final", 13.25, 5.65, 1.18, 1.0, "oriented\nboxes", colors["red"], colors["red_fill"])
    icon_rot_boxes(13.86, 6.38, colors["red"])
    module_links([crop, fpn, rpn, roi, feat], colors["blue"])
    arrow(feat, box_head, colors["blue"], rad=0.1)
    arrow(feat, prompt_head, colors["teal"], rad=-0.1)
    arrow(box_head, out, colors["blue"])
    arrow(prompt_head, out, colors["teal"])

    doc = clean_card("doc_final", 0.75, 2.92, 1.25, 0.98, "taxonomy\nprompts", colors["teal"], colors["teal_fill"])
    ax.add_patch(Polygon([(1.12, 3.67), (1.4, 3.67), (1.52, 3.55), (1.52, 3.36), (1.12, 3.36)], closed=True, facecolor="white", edgecolor=colors["teal"], linewidth=0.8, zorder=4))
    enc = clean_card("enc_final", 2.72, 2.92, 1.34, 0.98, "RemoteCLIP\nencoder", colors["teal"], colors["teal_fill"])
    ax.add_patch(Circle((3.08, 3.6), 0.1, facecolor=colors["violet_fill"], edgecolor=colors["violet"], linewidth=0.8, zorder=4))
    ax.add_patch(Circle((3.32, 3.6), 0.1, facecolor=colors["cyan_fill"], edgecolor=colors["cyan"], linewidth=0.8, zorder=4))
    tree = clean_card("tree_final", 4.72, 2.92, 1.22, 0.98, "hierarchy\ntree", colors["teal"], colors["teal_fill"])
    icon_tree(5.33, 3.35, colors["teal"])
    mat = clean_card("mat_final", 6.72, 2.92, 1.22, 0.98, "relation\nmatrix", colors["teal"], colors["teal_fill"])
    icon_matrix(7.06, 3.4, colors["teal"])
    proto = clean_card("proto_final", 8.74, 2.92, 1.36, 0.98, "prototype\nalignment", colors["teal"], colors["teal_fill"])
    op_circle(9.18, 3.62, "cos", colors["teal"], "white", 0.14)
    logits = clean_card("logits_final", 10.8, 2.92, 1.26, 0.98, "prompt\nlogits", colors["teal"], colors["teal_fill"])
    icon_bars(11.18, 3.4, colors["teal"])
    fusion = Circle((13.25, 3.44), 0.58, facecolor=colors["violet_fill"], edgecolor=colors["violet"], linewidth=1.4, zorder=3)
    ax.add_patch(fusion)
    ax.add_patch(Circle((13.25, 3.44), 0.38, facecolor="none", edgecolor=colors["cyan"], linewidth=1.0, zorder=4))
    ax.add_patch(Circle((13.25, 3.44), 0.2, facecolor="none", edgecolor=colors["teal"], linewidth=1.0, zorder=4))
    ax.text(13.25, 3.44, "fusion\ncore", ha="center", va="center", fontsize=8.1, color=colors["ink"], fontweight="bold", zorder=5)
    boxes.append(("fusion_final", 12.67, 2.86, 1.16, 1.16))
    module_links([doc, enc, tree, mat, proto, logits], colors["teal"])
    arrow(logits, (12.68, 3.44), colors["teal"])
    arrow((13.08, 4.0), (11.16, 5.2), colors["teal"], rad=0.12)
    arrow((12.85, 3.95), (9.0, 5.58), colors["violet"], rad=0.08)
    label(3.38, 2.68, "E = 18 x 512")
    label(7.34, 2.68, "R = 18 x 18")

    scene = image_box("scene_final", 0.78, 0.7, 1.12, 0.82, "scene", 0.052)
    gate = clean_card("gate_final", 2.55, 0.72, 1.12, 0.74, "scene\ngate", colors["amber"], colors["amber_fill"])
    ax.add_patch(Polygon([(3.11, 1.35), (3.28, 1.2), (3.11, 1.05), (2.94, 1.2)], closed=True, facecolor="white", edgecolor=colors["amber"], linewidth=0.9, zorder=4))
    ctx = clean_card("ctx_final", 4.48, 0.72, 1.26, 0.74, "context\ntokens", colors["orange"], colors["orange_fill"])
    for i, c in enumerate([colors["orange"], colors["amber"], colors["red"]]):
        ax.add_patch(Circle((4.83 + 0.2 * i, 1.3), 0.06, facecolor=c, edgecolor="white", linewidth=0.5, zorder=4))
    teacher = clean_card("teacher_final", 6.5, 0.72, 1.04, 0.74, "teacher", colors["amber"], "#fff7e6", dashed=True)
    ax.add_patch(Circle((7.02, 1.3), 0.12, facecolor="white", edgecolor=colors["amber"], linewidth=0.9, zorder=4))
    cand = clean_card("cand_final", 8.2, 0.72, 1.25, 0.74, "candidate\nboxes", colors["orange"], colors["orange_fill"], dashed=True)
    icon_rot_boxes(8.83, 1.3, colors["orange"])
    sieve = clean_card("sieve_final", 10.05, 0.72, 1.08, 0.74, "quality\nsieve", colors["amber"], "#fff7e6", dashed=True)
    ax.plot([10.34, 10.55, 10.76], [1.36, 1.12, 1.36], color=colors["amber"], linewidth=1.1, zorder=4)
    student = clean_card("student_final", 11.75, 0.72, 1.06, 0.74, "student\nupdate", colors["orange"], colors["orange_fill"], dashed=True)
    ax.add_patch(Polygon([(12.17, 1.36), (12.37, 1.2), (12.17, 1.04)], closed=True, facecolor="white", edgecolor=colors["orange"], linewidth=0.9, zorder=4))
    paused = clean_card("paused_final", 13.42, 0.72, 1.45, 0.74, "paused\nno PL gain", colors["red"], colors["red_fill"], dashed=True)
    module_links([scene, gate, ctx], colors["amber"])
    module_links([ctx, teacher, cand, sieve, student, paused], colors["amber"], dashed=True)
    arrow((5.35, 1.46), (12.82, 2.98), colors["orange"], rad=-0.12)
    arrow((3.35, 1.46), (10.62, 5.22), colors["amber"], rad=0.32)

    for suffix in ("svg", "pdf"):
        fig.savefig(FIG / f"geonexus_tgrs_architecture.{suffix}", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return

    # Clean full-width architecture scene. The previous evidence-strip scene is
    # intentionally bypassed because this figure should focus on the framework.
    ax.add_patch(
        FancyBboxPatch(
            (0.15, 0.15),
            15.7,
            8.45,
            boxstyle="round,pad=0.04,rounding_size=0.08",
            facecolor="white",
            edgecolor=colors["line"],
            linewidth=1.0,
            zorder=0,
        )
    )
    ax.text(0.35, 8.35, "GeoNexus-RSD framework: geometry, prompts, and context", fontsize=12, fontweight="bold", color=colors["ink"], va="top", zorder=5)
    ax.text(0.35, 8.08, "Detector geometry, taxonomy prompts, and scene context are connected into prompt-aware rotated-box prediction.", fontsize=8.6, color=colors["muted"], va="top", zorder=5)

    panel(0.35, 5.0, 15.1, 2.85, "A. Oriented detector stream", colors["blue"])
    panel(0.35, 2.28, 15.1, 2.3, "B. Prompt hierarchy and multimodal fusion", colors["teal"])
    panel(0.35, 0.45, 15.1, 1.4, "C. Scene context and paused pseudo-label route", colors["amber"])
    ax.add_patch(Rectangle((0.55, 5.13), 14.65, 0.08, facecolor=colors["blue"], edgecolor="none", alpha=0.13, zorder=1))
    ax.add_patch(Rectangle((0.55, 2.41), 14.65, 0.08, facecolor=colors["teal"], edgecolor="none", alpha=0.13, zorder=1))
    ax.add_patch(Rectangle((0.55, 0.56), 14.65, 0.06, facecolor=colors["amber"], edgecolor="none", alpha=0.17, zorder=1))

    crop = image_box("crop_new", 0.78, 5.48, 1.7, 1.55, "DOTAv2", 0.074)
    mini_chip(0.95, 6.76, 0.68, "real", colors["blue"], "white", 5.6)
    mini_chip(1.72, 6.76, 0.62, "tile", colors["cyan"], colors["cyan_fill"], 5.6)
    ax.add_patch(Rectangle((1.1, 6.02), 0.55, 0.25, angle=14, facecolor="none", edgecolor=colors["orange"], linewidth=1.0, zorder=5))
    ax.add_patch(Rectangle((1.45, 6.23), 0.32, 0.15, angle=-10, facecolor="none", edgecolor=colors["rose"], linewidth=1.0, zorder=5))

    fpn = node("fpn_new", 3.02, 5.58, 1.2, 1.28, "FPN\npyramid", colors["blue"], colors["blue_fill"])
    for i, width in enumerate([0.55, 0.72, 0.9]):
        y0 = 6.55 - 0.2 * i
        ax.add_patch(Polygon([(3.23, y0), (3.23 + width, y0), (3.23 + width - 0.15, y0 + 0.12), (3.23, y0 + 0.12)], closed=True, facecolor="white", edgecolor=colors["blue"], linewidth=0.8, zorder=4))
    for i, c in enumerate([colors["blue"], colors["cyan"], colors["violet"]]):
        ax.add_patch(Circle((3.42 + 0.2 * i, 5.84), 0.045, facecolor=c, edgecolor="white", linewidth=0.5, zorder=5))

    rpn = node("rpn_new", 4.82, 5.58, 1.22, 1.28, "rotated\nRPN", colors["blue"], colors["blue_fill"])
    for cx, cy, ang in [(5.38, 6.42, 15), (5.32, 6.2, -18)]:
        ax.add_patch(Rectangle((cx - 0.25, cy - 0.09), 0.5, 0.18, angle=ang, facecolor="none", edgecolor=colors["blue"], linewidth=0.9, zorder=4))
    op_circle(5.42, 5.92, "th", colors["orange"], colors["orange_fill"], 0.15)

    roi = node("roi_new", 6.72, 5.58, 1.24, 1.28, "rotated\nRoI", colors["blue"], colors["blue_fill"])
    ax.add_patch(Rectangle((7.08, 6.35), 0.55, 0.2, angle=-25, facecolor="none", edgecolor=colors["blue"], linewidth=1.0, zorder=4))
    op_circle(7.35, 5.96, "T", colors["violet"], colors["violet_fill"], 0.15)

    feat = node("feat_new", 8.7, 5.58, 1.3, 1.28, "RoI\nfeatures", colors["blue"], colors["blue_fill"])
    for i, c in enumerate([colors["blue"], colors["cyan"], colors["violet"]]):
        ax.add_patch(Rectangle((9.05 + i * 0.13, 6.36 - i * 0.06), 0.42, 0.2, facecolor="white", edgecolor=c, linewidth=0.8, zorder=4))
    for i, c in enumerate([colors["blue"], colors["teal"], colors["violet"]]):
        ax.add_patch(Circle((9.02 + 0.16 * i, 5.92), 0.045, facecolor=c, edgecolor="white", linewidth=0.5, zorder=5))

    box_head = node("box_new", 11.2, 6.2, 1.05, 0.76, "box\nhead", colors["blue"], colors["blue_fill"], fontsize=8.0)
    op_circle(11.72, 6.78, "d", colors["blue"], "white", 0.13)
    cls_head = node("cls_new", 11.2, 5.26, 1.28, 0.76, "prompt\nhead", colors["teal"], colors["teal_fill"], fontsize=8.0)
    op_circle(11.84, 5.82, "s", colors["teal"], "white", 0.13)
    out = node("out_new", 13.78, 5.72, 1.0, 0.96, "oriented\nboxes", colors["red"], colors["red_fill"], fontsize=7.8)
    for x, y, angle in [(14.1, 6.33, 22), (14.32, 6.14, -12), (14.04, 5.98, 35)]:
        ax.add_patch(Rectangle((x - 0.14, y - 0.055), 0.28, 0.11, angle=angle, facecolor="none", edgecolor=colors["red"], linewidth=0.9, zorder=4))
    module_links([crop, fpn, rpn, roi, feat], colors["blue"])
    arrow(feat, box_head, colors["blue"], rad=0.12)
    arrow(feat, cls_head, colors["teal"], rad=-0.14)
    arrow(box_head, out, colors["blue"])
    arrow(cls_head, out, colors["teal"])

    doc = node("doc_new", 0.8, 2.84, 1.2, 0.98, "taxonomy\nprompts", colors["teal"], colors["teal_fill"], fontsize=8.0)
    ax.add_patch(Polygon([(1.12, 3.66), (1.42, 3.66), (1.54, 3.54), (1.54, 3.33), (1.12, 3.33)], closed=True, facecolor="white", edgecolor=colors["teal"], linewidth=0.8, zorder=4))
    enc = node("enc_new", 2.7, 2.84, 1.38, 0.98, "RemoteCLIP\nencoder", colors["teal"], colors["teal_fill"], fontsize=7.9)
    tree = node("tree_new", 4.78, 2.84, 1.2, 0.98, "hierarchy\ntree", colors["teal"], colors["teal_fill"], fontsize=8.0)
    ax.plot([5.38, 5.38, 5.12, 5.64], [3.62, 3.43, 3.3, 3.3], color=colors["teal"], linewidth=1.0, zorder=4)
    mat = node("mat_new", 6.8, 2.84, 1.18, 0.98, "relation\nmatrix", colors["teal"], colors["teal_fill"], fontsize=8.0)
    for i in range(5):
        for j in range(5):
            ax.add_patch(Rectangle((7.04 + i * 0.08, 3.34 + j * 0.06), 0.055, 0.038, facecolor=colors["teal"], edgecolor="none", alpha=0.18 + 0.06 * (i + j), zorder=4))
    proto = node("proto_new", 8.8, 2.84, 1.35, 0.98, "prototype\nalignment", colors["teal"], colors["teal_fill"], fontsize=7.9)
    op_circle(9.2, 3.52, "cos", colors["teal"], "white", 0.14)
    logits = node("logits_new", 10.9, 2.84, 1.26, 0.98, "prompt\nlogits", colors["teal"], colors["teal_fill"], fontsize=8.0)
    for i, h in enumerate([0.18, 0.32, 0.48]):
        ax.add_patch(Rectangle((11.26 + i * 0.18, 3.3), 0.1, h, facecolor=colors["teal"], edgecolor="none", alpha=0.55, zorder=4))
    fusion = Circle((13.25, 3.33), 0.58, facecolor=colors["violet_fill"], edgecolor=colors["violet"], linewidth=1.4, zorder=3)
    ax.add_patch(fusion)
    ax.add_patch(Circle((13.25, 3.33), 0.39, facecolor="none", edgecolor=colors["cyan"], linewidth=1.0, alpha=0.9, zorder=4))
    ax.add_patch(Circle((13.25, 3.33), 0.22, facecolor="none", edgecolor=colors["teal"], linewidth=1.0, alpha=0.9, zorder=4))
    ax.text(13.25, 3.33, "fusion\ncore", ha="center", va="center", fontsize=8.3, color=colors["ink"], fontweight="bold", zorder=5)
    boxes.append(("fusion_new", 12.67, 2.75, 1.16, 1.16))
    module_links([doc, enc, tree, mat, proto, logits], colors["teal"])
    arrow(logits, (12.68, 3.33), colors["teal"])
    arrow((13.25, 3.91), (11.84, 5.26), colors["teal"], rad=0.1)
    arrow((13.0, 3.88), (9.35, 5.58), colors["violet"], rad=0.12)
    label(3.38, 2.58, "E = 18 x 512")
    label(7.4, 2.58, "R = 18 x 18")

    scene = image_box("scene_new", 0.8, 0.58, 1.2, 0.72, "context", 0.052)
    gate = node("gate_new", 2.7, 0.62, 1.12, 0.66, "scene\ngate", colors["amber"], colors["amber_fill"], fontsize=7.6)
    ctx = node("ctx_new", 4.75, 0.62, 1.25, 0.66, "context\ntokens", colors["orange"], colors["orange_fill"], fontsize=7.5)
    teacher = node("teacher_new", 7.02, 0.62, 1.05, 0.66, "teacher", colors["amber"], colors["amber_fill"], fontsize=8.0, dashed=True)
    sieve = node("sieve_new", 8.95, 0.62, 1.18, 0.66, "label\nsieve", colors["amber"], colors["amber_fill"], fontsize=7.6, dashed=True)
    student = node("student_new", 11.2, 0.62, 1.05, 0.66, "student", colors["amber"], colors["amber_fill"], fontsize=8.0, dashed=True)
    paused = node("paused_new", 13.22, 0.62, 1.65, 0.66, "paused\nno PL gain", colors["amber"], colors["amber_fill"], fontsize=7.3, dashed=True)
    module_links([scene, gate, ctx], colors["amber"])
    module_links([ctx, teacher, sieve, student, paused], colors["amber"], dashed=True)
    arrow((5.8, 1.28), (13.25, 2.76), colors["orange"], rad=-0.12)
    arrow(gate, (11.28, 5.36), colors["amber"], rad=0.34)

    for suffix in ("svg", "pdf"):
        fig.savefig(FIG / f"geonexus_tgrs_architecture.{suffix}", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return

    ax.add_patch(
        FancyBboxPatch(
            (0.15, 0.15),
            15.7,
            8.45,
            boxstyle="round,pad=0.04,rounding_size=0.08",
            facecolor="white",
            edgecolor=colors["line"],
            linewidth=1.0,
            zorder=0,
        )
    )
    ax.text(0.35, 8.35, "GeoNexus-RSD visual framework and evidence status", fontsize=12, fontweight="bold", color=colors["ink"], va="top", zorder=5)
    ax.text(0.35, 8.08, "Real DOTAv2 imagery anchors the method view; metric chips are separated from the model flow.", fontsize=8.6, color=colors["muted"], va="top", zorder=5)

    panel(0.35, 5.0, 11.2, 2.85, "A. Oriented detector geometry", colors["blue"])
    panel(0.35, 2.25, 11.2, 2.35, "B. RemoteCLIP prompt hierarchy", colors["teal"])
    panel(0.35, 0.45, 11.2, 1.35, "C. Context and pseudo-label route: paused/planned", colors["amber"])
    panel(11.85, 0.45, 3.75, 7.4, "Evidence strip", colors["red"])
    ax.add_patch(Rectangle((0.5, 5.12), 10.86, 0.08, facecolor=colors["blue"], edgecolor="none", alpha=0.13, zorder=1))
    ax.add_patch(Rectangle((0.5, 2.38), 10.86, 0.08, facecolor=colors["teal"], edgecolor="none", alpha=0.13, zorder=1))
    ax.add_patch(Rectangle((0.5, 0.54), 10.86, 0.06, facecolor=colors["amber"], edgecolor="none", alpha=0.17, zorder=1))

    crop = image_box("crop", 0.75, 5.45, 1.65, 1.55, "DOTAv2", 0.075)
    mini_chip(0.86, 6.74, "real crop", colors["blue"], "white")
    mini_chip(1.62, 6.74, "1024 tile", colors["cyan"], colors["cyan_fill"])
    ax.add_patch(Rectangle((1.08, 6.0), 0.54, 0.26, angle=14, facecolor="none", edgecolor=colors["orange"], linewidth=1.0, zorder=5))
    ax.add_patch(Rectangle((1.42, 6.22), 0.34, 0.16, angle=-10, facecolor="none", edgecolor=colors["rose"], linewidth=1.0, zorder=5))
    fpn = node("fpn", 2.85, 5.58, 1.05, 1.28, "FPN\npyramid", colors["blue"], colors["blue_fill"])
    for i, width in enumerate([0.52, 0.66, 0.8]):
        y0 = 6.56 - 0.19 * i
        ax.add_patch(Polygon([(3.06, y0), (3.06 + width, y0), (3.06 + width - 0.15, y0 + 0.12), (3.06, y0 + 0.12)], closed=True, facecolor="white", edgecolor=colors["blue"], linewidth=0.8, zorder=4))
    for i, c in enumerate([colors["blue"], colors["cyan"], colors["violet"]]):
        ax.add_patch(Circle((3.12 + 0.18 * i, 5.82), 0.045, facecolor=c, edgecolor="white", linewidth=0.5, zorder=5))
    rpn = node("rpn", 4.25, 5.58, 1.1, 1.28, "rotated\nRPN", colors["blue"], colors["blue_fill"])
    for cx, cy, ang in [(4.82, 6.43, 15), (4.78, 6.21, -18)]:
        rect = Rectangle((cx - 0.25, cy - 0.09), 0.5, 0.18, angle=ang, facecolor="none", edgecolor=colors["blue"], linewidth=0.9, zorder=4)
        ax.add_patch(rect)
    op_circle(4.8, 5.92, "θ", colors["orange"], colors["orange_fill"], 0.15)
    roi = node("roi", 5.7, 5.58, 1.15, 1.28, "rotated\nRoI", colors["blue"], colors["blue_fill"])
    ax.add_patch(Rectangle((6.02, 6.35), 0.55, 0.2, angle=-25, facecolor="none", edgecolor=colors["blue"], linewidth=1.0, zorder=4))
    op_circle(6.28, 5.95, "T", colors["violet"], colors["violet_fill"], 0.15)
    feat = node("feat", 7.25, 5.58, 1.15, 1.28, "RoI\nfeatures", colors["blue"], colors["blue_fill"])
    for i, c in enumerate([colors["blue"], colors["cyan"], colors["violet"]]):
        ax.add_patch(Rectangle((7.55 + i * 0.12, 6.36 - i * 0.06), 0.42, 0.2, facecolor="white", edgecolor=c, linewidth=0.8, zorder=4))
    mini_chip(7.46, 5.85, "roi tokens", colors["violet"], colors["violet_fill"], 5.8)
    box_head = node("box", 8.9, 6.22, 1.0, 0.72, "box\nhead", colors["blue"], colors["blue_fill"], fontsize=8.0)
    op_circle(9.4, 6.77, "Δ", colors["blue"], "white", 0.13)
    cls_head = node("cls", 8.9, 5.28, 1.22, 0.72, "prompt\nhead", colors["teal"], colors["teal_fill"], fontsize=8.0)
    op_circle(9.52, 5.82, "σ", colors["teal"], "white", 0.13)
    out = node("out", 10.45, 5.72, 0.82, 0.92, "oriented\nboxes", colors["red"], colors["red_fill"], fontsize=7.8)
    for x, y, angle in [(10.72, 6.24, 22), (10.9, 6.02, -12), (10.68, 5.85, 35)]:
        ax.add_patch(Rectangle((x - 0.14, y - 0.055), 0.28, 0.11, angle=angle, facecolor="none", edgecolor=colors["red"], linewidth=0.9, zorder=4))
    for s, e in [(crop, fpn), (fpn, rpn), (rpn, roi), (roi, feat), (box_head, out), (cls_head, out)]:
        arrow(s, e, colors["blue"] if e != out or s == box_head else colors["teal"])
    arrow(feat, box_head, colors["blue"], rad=0.12)
    arrow(feat, cls_head, colors["teal"], rad=-0.15)
    for x, text, edge in [(2.8, "C", colors["blue"]), (4.18, "N", colors["orange"]), (5.7, "A", colors["violet"]), (8.78, "S", colors["teal"])]:
        op_circle(x, 5.2, text, edge, "white", 0.13)

    doc = node("doc", 0.78, 2.8, 1.05, 0.92, "taxonomy\nprompts", colors["teal"], colors["teal_fill"], fontsize=8.0)
    ax.add_patch(Polygon([(1.11, 3.62), (1.4, 3.62), (1.51, 3.5), (1.51, 3.31), (1.11, 3.31)], closed=True, facecolor="white", edgecolor=colors["teal"], linewidth=0.8, zorder=4))
    for i, txt in enumerate(["ship", "plane", "vehicle"]):
        mini_chip(1.57, 3.47 - i * 0.18, 0.52, txt, colors["green"], "white", 4.4)
    enc = node("enc", 2.28, 2.8, 1.28, 0.92, "RemoteCLIP\nencoder", colors["teal"], colors["teal_fill"], fontsize=7.9)
    ax.add_patch(Circle((2.62, 3.47), 0.1, facecolor=colors["violet_fill"], edgecolor=colors["violet"], linewidth=0.8, zorder=4))
    ax.add_patch(Circle((2.86, 3.47), 0.1, facecolor=colors["cyan_fill"], edgecolor=colors["cyan"], linewidth=0.8, zorder=4))
    ax.plot([2.72, 2.78], [3.47, 3.47], color=colors["teal"], linewidth=1.0, zorder=4)
    tree = node("tree", 4.0, 2.8, 1.08, 0.92, "hierarchy\ntree", colors["teal"], colors["teal_fill"], fontsize=8.0)
    ax.plot([4.54, 4.54, 4.32, 4.76], [3.58, 3.42, 3.3, 3.3], color=colors["teal"], linewidth=1.0, zorder=4)
    for x, y in [(4.54, 3.58), (4.32, 3.3), (4.76, 3.3)]:
        ax.add_patch(Circle((x, y), 0.045, facecolor="white", edgecolor=colors["teal"], linewidth=0.8, zorder=5))
    mat = node("mat", 5.55, 2.8, 1.08, 0.92, "relation\nmatrix", colors["teal"], colors["teal_fill"], fontsize=8.0)
    for i in range(5):
        for j in range(5):
            alpha = 0.18 + 0.06 * (i + j)
            ax.add_patch(Rectangle((5.78 + i * 0.075, 3.3 + j * 0.055), 0.055, 0.038, facecolor=colors["teal"], edgecolor="none", alpha=alpha, zorder=4))
    cos = node("cos", 7.02, 2.8, 1.18, 0.92, "prototype\nalignment", colors["teal"], colors["teal_fill"], fontsize=7.9)
    op_circle(7.34, 3.47, "cos", colors["teal"], "white", 0.14)
    op_circle(7.78, 3.47, "τ", colors["orange"], colors["orange_fill"], 0.13)
    logits = node("logits", 8.68, 2.8, 1.18, 0.92, "prompt\nlogits", colors["teal"], colors["teal_fill"], fontsize=8.0)
    for i, h in enumerate([0.18, 0.32, 0.48]):
        ax.add_patch(Rectangle((9.0 + i * 0.18, 3.28), 0.1, h, facecolor=colors["teal"], edgecolor="none", alpha=0.55, zorder=4))
    fusion = Circle((10.55, 3.25), 0.58, facecolor=colors["violet_fill"], edgecolor=colors["violet"], linewidth=1.4, zorder=3)
    ax.add_patch(fusion)
    ax.add_patch(Circle((10.55, 3.25), 0.39, facecolor="none", edgecolor=colors["cyan"], linewidth=1.0, alpha=0.9, zorder=4))
    ax.add_patch(Circle((10.55, 3.25), 0.22, facecolor="none", edgecolor=colors["teal"], linewidth=1.0, alpha=0.9, zorder=4))
    for px, py, pc in [(10.23, 3.48, colors["blue"]), (10.87, 3.48, colors["teal"]), (10.24, 3.04, colors["orange"]), (10.86, 3.04, colors["red"])]:
        ax.add_patch(Circle((px, py), 0.045, facecolor=pc, edgecolor="white", linewidth=0.5, zorder=5))
    ax.text(10.55, 3.25, "fusion\ncore", ha="center", va="center", fontsize=8.3, color=colors["ink"], fontweight="bold", zorder=5)
    boxes.append(("fusion", 9.97, 2.67, 1.16, 1.16))
    for s, e in [(doc, enc), (enc, tree), (tree, mat), (mat, cos), (cos, logits)]:
        arrow(s, e, colors["teal"])
    arrow(logits, (10.05, 3.25), colors["teal"])
    arrow((10.55, 3.83), (9.52, 5.28), colors["teal"], rad=0.12)
    arrow((10.55, 3.83), (8.0, 5.58), "#4f46e5", rad=0.12)
    label(2.92, 2.58, "E = 18 x 512")
    label(6.09, 2.58, "R = 18 x 18")
    for x, text, edge in [(2.1, "tok", colors["cyan"]), (3.78, "E", colors["teal"]), (5.32, "R", colors["green"]), (6.92, "cos", colors["teal"]), (8.48, "z", colors["violet"])]:
        op_circle(x, 2.42, text, edge, "white", 0.13)

    scene = image_box("scene", 0.8, 0.58, 1.2, 0.72, "context", 0.052)
    mini_chip(0.9, 1.16, "scene", colors["amber"], "white", 5.5)
    gate = node("gate", 2.45, 0.62, 1.0, 0.65, "scene\ngate", colors["amber"], colors["amber_fill"], fontsize=7.6)
    op_circle(2.95, 1.12, "g", colors["amber"], "white", 0.12)
    teacher = node("teacher", 4.0, 0.62, 1.0, 0.65, "teacher", colors["amber"], colors["amber_fill"], fontsize=8.0, dashed=True)
    ax.add_patch(Circle((4.5, 1.11), 0.11, facecolor="white", edgecolor=colors["amber"], linewidth=0.8, zorder=4))
    sieve = node("sieve", 5.55, 0.62, 1.18, 0.65, "label\nsieve", colors["amber"], colors["amber_fill"], fontsize=7.6, dashed=True)
    ax.plot([5.85, 5.98, 6.2, 6.38], [1.12, 0.99, 1.16, 0.92], color=colors["amber"], linewidth=1.0, zorder=4)
    student = node("student", 7.28, 0.62, 1.0, 0.65, "student", colors["amber"], colors["amber_fill"], fontsize=8.0, dashed=True)
    paused = node("paused", 8.92, 0.62, 1.8, 0.65, "paused/planned\nno PL gain claimed", colors["amber"], colors["amber_fill"], fontsize=7.3, dashed=True)
    for s, e in [(scene, gate), (gate, teacher), (teacher, sieve), (sieve, student), (student, paused)]:
        arrow(s, e, colors["amber"], dashed=e != gate)

    chips = [
        ("DOTA2 S0", "0.6088 / 0.6090", colors["gray_fill"], colors["line"], "ref"),
        ("DOTA2 S1", "0.6177 / 0.6180", colors["teal_fill"], colors["teal"], "done"),
        ("DOTA2 S2", "best mean 0.620606\nfinal mean 0.616655", colors["amber_fill"], colors["amber"], "unstable"),
        ("DIOR-R S0", "0.6531 / 0.6530", colors["gray_fill"], colors["line"], "ref"),
        ("DIOR-R S1", "0.6751 / 0.6750\n0.6690 / 0.6690", colors["teal_fill"], colors["teal"], "done"),
        ("DIOR-R S2", "metrics pending", colors["amber_fill"], colors["amber"], "pending"),
    ]
    y = 6.86
    ax.plot([12.36, 12.36], [1.62, 6.98], color=colors["line"], linewidth=1.0, zorder=1)
    for title, value, fill, edge, status in chips:
        ax.add_patch(FancyBboxPatch((12.18, y - 0.65), 3.12, 0.54, boxstyle="round,pad=0.035,rounding_size=0.08", facecolor="#d9e2ef", edgecolor="none", alpha=0.32, zorder=2))
        ax.add_patch(FancyBboxPatch((12.15, y - 0.62), 3.12, 0.54, boxstyle="round,pad=0.035,rounding_size=0.08", facecolor=fill, edgecolor=edge, linewidth=1.0, zorder=3))
        ax.add_patch(Rectangle((12.15, y - 0.62), 0.08, 0.54, facecolor=edge, edgecolor="none", zorder=4))
        ax.add_patch(Circle((12.36, y - 0.35), 0.08, facecolor=fill, edgecolor=edge, linewidth=0.9, zorder=5))
        ax.text(12.52, y - 0.23, title, ha="left", va="center", fontsize=7.9, color=colors["ink"], fontweight="bold", zorder=5)
        value_size = 7.15 if title == "DIOR-R S2" else 7.55
        ax.text(15.14, y - 0.23, value, ha="right", va="center", fontsize=value_size, color=colors["ink"], linespacing=1.05, zorder=5)
        mini_chip(12.49, y - 0.54, 0.62, status, edge, "white", 5.3)
        boxes.append((title, 12.15, y - 0.62, 3.12, 0.54))
        y -= 0.86 if "\n" not in value else 1.08
    ax.text(12.22, 1.05, "Chips summarize records only;\nthey are not qualitative detections.", ha="left", va="center", fontsize=7.7, color=colors["muted"], linespacing=1.1)

    # Guard against accidental same-panel node collisions in future edits.
    for i, (a, ax0, ay0, aw, ah) in enumerate(boxes):
        for b, bx0, by0, bw, bh in boxes[i + 1 :]:
            if a in {"fusion"} or b in {"fusion"}:
                continue
            overlap = ax0 < bx0 + bw and ax0 + aw > bx0 and ay0 < by0 + bh and ay0 + ah > by0
            if overlap:
                raise AssertionError(f"Architecture nodes overlap: {a} and {b}")

    for suffix in ("svg", "pdf"):
        fig.savefig(FIG / f"geonexus_tgrs_architecture.{suffix}", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def write_qual_tex() -> None:
    tex = r"""\documentclass[tikz,border=4pt]{standalone}
\usepackage{graphicx}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,calc,fit,positioning}
\definecolor{ink}{RGB}{26,32,44}
\definecolor{muted}{RGB}{87,99,114}
\definecolor{line}{RGB}{199,210,221}
\definecolor{paper}{RGB}{250,252,255}
\definecolor{green}{RGB}{21,128,61}
\definecolor{amber}{RGB}{180,83,9}
\definecolor{red}{RGB}{185,28,28}
\definecolor{blue}{RGB}{37,99,235}
\definecolor{gray}{RGB}{102,112,133}
\tikzset{
  font=\sffamily,
  panel/.style={draw=line,fill=paper,rounded corners=2pt,line width=.65pt},
  title/.style={font=\sffamily\bfseries\scriptsize,text=ink,anchor=west},
  chip/.style={draw=line,fill=white,rounded corners=1.4pt,line width=.55pt,align=center,font=\sffamily\tiny,text=ink,inner xsep=2.5pt,inner ysep=1.7pt},
  ok/.style={chip,draw=green,text=green,fill=green!7},
  warn/.style={chip,draw=amber,text=amber,fill=amber!9},
  stop/.style={chip,draw=red,text=red,fill=red!7},
  arch/.style={chip,draw=gray,text=gray,fill=gray!7},
  flow/.style={-{Latex[length=1.8mm,width=1.1mm]},line width=.7pt,draw=blue,shorten >=1pt,shorten <=1pt}
}
\begin{document}
\begin{tikzpicture}
\node[panel,minimum width=142mm,minimum height=63mm,anchor=south west] (frame) at (0,0) {};
\node[title] at (2mm,60mm) {Qualitative context and experiment status board};
\node[font=\sffamily\tiny,text=muted,anchor=west] at (2mm,56.8mm) {Image crops are deterministic subcrops of the archived real DOTAv2 crop and are visual context only, not model predictions.};

\node[panel,minimum width=88mm,minimum height=23mm,anchor=north west] (stripbox) at (2mm,54mm) {};
\node[inner sep=0pt,anchor=north west] at ($(stripbox.north west)+(2mm,-4mm)$) {\includegraphics[width=84mm]{geonexus_tgrs_qual_crop_strip.png}};
\node[font=\sffamily\tiny,text=muted,anchor=west] at ($(stripbox.south west)+(2mm,2mm)$) {DOTAv2 crop strip: no detection boxes, no pseudo labels, no fabricated outputs.};

\node[panel,minimum width=47mm,minimum height=23mm,anchor=north east] (status) at ($(frame.north east)+(-2mm,-9mm)$) {};
\node[title,text=blue] at ($(status.north west)+(2mm,-2.8mm)$) {measured evidence chips};
\node[ok,anchor=north west,minimum width=20mm] at ($(status.north west)+(2mm,-6.5mm)$) {DOTA2 S1\\complete};
\node[warn,anchor=north west,minimum width=20mm] at ($(status.north west)+(25mm,-6.5mm)$) {DOTA2 S2\\unstable};
\node[ok,anchor=north west,minimum width=20mm] at ($(status.north west)+(2mm,-15mm)$) {DIOR-R S1\\measured};
\node[warn,anchor=north west,minimum width=20mm] at ($(status.north west)+(25mm,-15mm)$) {DIOR-R S2\\pending};

\node[panel,minimum width=64mm,minimum height=25mm,anchor=south west] (hier) at (2mm,3mm) {};
\node[title,text=green] at ($(hier.north west)+(2mm,-2.8mm)$) {hierarchy/context examples};
\node[chip,anchor=north west,minimum width=15mm] (veh) at ($(hier.north west)+(4mm,-7mm)$) {vehicle};
\node[chip,anchor=north west,minimum width=17mm] (small) at ($(hier.north west)+(3mm,-16mm)$) {small\\vehicle};
\node[chip,anchor=north west,minimum width=17mm] (large) at ($(hier.north west)+(23mm,-16mm)$) {large\\vehicle};
\node[chip,anchor=north west,minimum width=15mm] (ship) at ($(hier.north west)+(43mm,-7mm)$) {ship};
\node[chip,anchor=north west,minimum width=15mm] (harbor) at ($(hier.north west)+(43mm,-16mm)$) {harbor\\context};
\draw[flow] (veh.south) -- (small.north); \draw[flow] (veh.south) -- (large.north); \draw[flow] (ship.south) -- (harbor.north);

\node[panel,minimum width=70mm,minimum height=25mm,anchor=south east] (route) at ($(frame.south east)+(-2mm,3mm)$) {};
\node[title,text=red] at ($(route.north west)+(2mm,-2.8mm)$) {route labels used in the paper};
\node[ok,anchor=north west,minimum width=16mm] at ($(route.north west)+(3mm,-7mm)$) {completed};
\node[warn,anchor=north west,minimum width=16mm] at ($(route.north west)+(22mm,-7mm)$) {unstable};
\node[warn,anchor=north west,minimum width=16mm] at ($(route.north west)+(41mm,-7mm)$) {pending};
\node[stop,anchor=north west,minimum width=16mm] at ($(route.north west)+(3mm,-16mm)$) {paused};
\node[arch,anchor=north west,minimum width=16mm] at ($(route.north west)+(22mm,-16mm)$) {archive};
\node[font=\sffamily\tiny,text=muted,align=left,anchor=north west] at ($(route.north west)+(41mm,-15.5mm)$) {DOTA v1.5 stays\\debug-only};
\end{tikzpicture}
\end{document}
"""
    (FIG / "geonexus_tgrs_qual_status.tex").write_text(tex, encoding="utf-8", newline="\n")


def compile_tex(name: str) -> None:
    subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", name],
        cwd=FIG,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    assert_evidence()
    make_crop_strip()
    draw_architecture_scene()
    write_qual_tex()
    compile_tex("geonexus_tgrs_qual_status.tex")
    print("Regenerated geonexus_tgrs_architecture.svg, geonexus_tgrs_architecture.pdf, and geonexus_tgrs_qual_status.pdf")


if __name__ == "__main__":
    main()
