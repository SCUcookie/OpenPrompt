"""Render 16:9 conference-ready result assets for the 2026-06-14 update.

The script computes headline values from recorded JSON/JSONL/markdown sources
before plotting. It intentionally keeps DOTA v1.5 on an appendix-only slide.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from textwrap import wrap

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "artifacts" / "ppt_assets_20260614"

W, H = 13.333, 7.5
BG = "#f7f8fb"
INK = "#17202a"
MUTED = "#647184"
GRID = "#d9e0ea"
BLUE = "#2f6fed"
TEAL = "#159a8c"
GREEN = "#2f8f5b"
AMBER = "#c98514"
RED = "#c74343"
PURPLE = "#6659cf"
LIGHT_BLUE = "#dbe8ff"
LIGHT_TEAL = "#dff3ef"
LIGHT_AMBER = "#fff0d5"
LIGHT_RED = "#fde1e1"


@dataclass(frozen=True)
class MetricPoint:
    epoch: int
    map: float
    ap50: float


def rel(path: str) -> Path:
    return (ROOT / path).resolve()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_scalar_points(path: Path) -> list[MetricPoint]:
    points: list[MetricPoint] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if "dota/mAP" in row:
                points.append(
                    MetricPoint(
                        epoch=int(row["step"]),
                        map=float(row["dota/mAP"]),
                        ap50=float(row.get("dota/AP50", row["dota/mAP"])),
                    )
                )
    if not points:
        raise ValueError(f"No validation metric points found in {path}")
    return points


def parse_metric_pair_from_md(path: Path, label: str) -> tuple[float, float]:
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(
        rf"{re.escape(label)}.*?dota/mAP=([0-9.]+).*?dota/AP50=([0-9.]+)",
        re.DOTALL,
    )
    m = pattern.search(text)
    if not m:
        raise ValueError(f"Could not parse {label!r} from {path}")
    return float(m.group(1)), float(m.group(2))


def rounded(x: float) -> float:
    return round(float(x), 4)


def assert_close(name: str, actual: float, expected: float) -> None:
    if rounded(actual) != expected:
        raise AssertionError(f"{name}: got {actual:.8f}, expected rounded {expected:.4f}")


def collect_data() -> dict:
    s0_dota2 = load_json(rel("docs/experiments/20260603_s0_dota2_roi_trans_validpng_metrics.json"))
    s0_dota2_map = float(s0_dota2["metrics"]["dota/mAP"])

    s1_dota2_path = Path(
        "/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/"
        "roi_trans_remoteclip_s1_validpng_20260607/20260607_101146/vis_data/scalars.json"
    )
    s1_dota2 = load_scalar_points(s1_dota2_path)[-1]

    dota2_loss0_specs = [
        (
            "loss-0",
            Path(
                "/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/"
                "roi_trans_remoteclip_s2_hierarchy_ablate_loss0_s1e12_20260610/"
                "20260610_100253/vis_data/scalars.json"
            ),
        ),
        (
            "rep3407",
            Path(
                "/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/"
                "roi_trans_remoteclip_s2_hierarchy_ablate_loss0_s1e12_rep3407_20260610/"
                "20260610_191026/vis_data/scalars.json"
            ),
        ),
        (
            "rep4407",
            Path(
                "/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/"
                "roi_trans_remoteclip_s2_hierarchy_ablate_loss0_s1e12_rep4407_20260610/"
                "20260610_210021/vis_data/scalars.json"
            ),
        ),
        (
            "rep5407",
            Path(
                "/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/"
                "roi_trans_remoteclip_s2_hierarchy_ablate_loss0_s1e12_rep5407_20260610/"
                "20260610_210021/vis_data/scalars.json"
            ),
        ),
        (
            "rep6407",
            Path(
                "/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/"
                "roi_trans_remoteclip_s2_hierarchy_ablate_loss0_s1e12_rep6407_20260611/"
                "20260611_102732/vis_data/scalars.json"
            ),
        ),
        (
            "rep7407",
            Path(
                "/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/"
                "roi_trans_remoteclip_s2_hierarchy_ablate_loss0_s1e12_rep7407_20260611/"
                "20260611_102732/vis_data/scalars.json"
            ),
        ),
        (
            "rep8407",
            Path(
                "/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/"
                "roi_trans_remoteclip_s2_hierarchy_ablate_loss0_s1e12_rep8407_20260611/"
                "20260611_102732/vis_data/scalars.json"
            ),
        ),
    ]
    dota2_s2_runs = []
    for name, path in dota2_loss0_specs:
        points = load_scalar_points(path)
        best = max(points, key=lambda p: p.map)
        final = points[-1]
        dota2_s2_runs.append(
            {
                "name": name,
                "source": str(path),
                "points": points,
                "best_epoch": best.epoch,
                "best_map": best.map,
                "final_epoch": final.epoch,
                "final_map": final.map,
            }
        )
    dota2_s2_best_mean = mean(r["best_map"] for r in dota2_s2_runs)
    dota2_s2_final_mean = mean(r["final_map"] for r in dota2_s2_runs)
    assert_close("DOTA2 S2 loss-0 best mean", dota2_s2_best_mean, 0.6206)
    assert_close("DOTA2 S2 loss-0 final mean", dota2_s2_final_mean, 0.6167)

    s0_dior_path = Path(
        "/data5/2025/ldh/OpenRSD/work_dirs/dior_r_s0_roi_trans_sanitized_long_20260612_gpu1/"
        "20260612_232047/vis_data/scalars.json"
    )
    s0_dior = load_scalar_points(s0_dior_path)[-1]

    s1_dior = load_json(rel("docs/experiments/20260613_dior_r_geonexus_s1_s0e52_replicas_metrics.json"))
    s1_dior_finals = [
        next(m for m in rep["metrics"] if m["epoch"] == rep["final_epoch"])["dota_mAP"]
        for rep in s1_dior["replicas"]
    ]
    s1_dior_mean = mean(s1_dior_finals)

    s2_dior = load_json(rel("docs/experiments/20260614_dior_r_geonexus_s2_replicas_complete.json"))
    s2_dior_runs = []
    for rep in s2_dior["replicas"]:
        points = [
            MetricPoint(epoch=m["epoch"], map=m["dota_mAP"], ap50=m["dota_AP50"])
            for m in rep["metrics"]
        ]
        s2_dior_runs.append(
            {
                "name": f"rep{rep['replica']}",
                "seed": rep["seed"],
                "source": rep["metric_source"],
                "points": points,
                "best_epoch": rep["best_epoch"],
                "best_map": rep["best_mAP"],
                "final_epoch": rep["final_epoch"],
                "final_map": rep["final_mAP"],
            }
        )
    s2_dior_best_mean = mean(r["best_map"] for r in s2_dior_runs)
    s2_dior_final_mean = mean(r["final_map"] for r in s2_dior_runs)
    assert_close("DIOR-R S2 best mean", s2_dior_best_mean, 0.6884)
    assert_close("DIOR-R S2 final mean", s2_dior_final_mean, 0.6853)

    dota2_baselines_md = rel("docs/experiments/20260605_dota2_baseline_status.md")
    r3det_md = rel("docs/experiments/20260607_current_status_and_next_launch.md")
    detector_dota2 = [
        {"detector": "RoI Transformer", "mAP": s0_dota2_map, "AP50": 0.6090, "source": str(s0_dota2_path())},
        {
            "detector": "Oriented R-CNN",
            "mAP": parse_metric_pair_from_md(dota2_baselines_md, "Oriented R-CNN R50 bs1")[0],
            "AP50": parse_metric_pair_from_md(dota2_baselines_md, "Oriented R-CNN R50 bs1")[1],
            "source": str(dota2_baselines_md),
        },
        {
            "detector": "S2ANet",
            "mAP": parse_metric_pair_from_md(dota2_baselines_md, "S2ANet bs1")[0],
            "AP50": parse_metric_pair_from_md(dota2_baselines_md, "S2ANet bs1")[1],
            "source": str(dota2_baselines_md),
        },
        {
            "detector": "R3Det-KFIoU",
            "mAP": parse_metric_pair_from_md(dota2_baselines_md, "R3Det-KFIoU bs1")[0],
            "AP50": parse_metric_pair_from_md(dota2_baselines_md, "R3Det-KFIoU bs1")[1],
            "source": str(r3det_md),
        },
        {
            "detector": "RTMDet-M",
            "mAP": parse_metric_pair_from_md(dota2_baselines_md, "RTMDet-M bs1")[0],
            "AP50": parse_metric_pair_from_md(dota2_baselines_md, "RTMDet-M bs1")[1],
            "source": str(dota2_baselines_md),
        },
        {
            "detector": "RTMDet-L",
            "mAP": parse_metric_pair_from_md(r3det_md, "RTMDet-L"),
            "AP50": None,
            "source": str(r3det_md),
        },
        {
            "detector": "OpenRSD reference",
            "mAP": float(load_json(rel("docs/experiments/20260602_opensrd_dota2_epoch12_ssval_metrics.json"))["metrics"]["dota/mAP"]),
            "AP50": 0.4200,
            "source": str(rel("docs/experiments/20260602_opensrd_dota2_epoch12_ssval_metrics.json")),
        },
    ]
    for row in detector_dota2:
        if isinstance(row["mAP"], tuple):
            row["mAP"], row["AP50"] = row["mAP"]

    dior_orcnn = all_points(
        [
            Path(
                "/data5/2025/ldh/OpenRSD/work_dirs/dior_r_s0_orcnn_sanitized_long_20260612_gpu0/"
                "20260612_181155/vis_data/scalars.json"
            ),
            Path(
                "/data5/2025/ldh/OpenRSD/work_dirs/dior_r_s0_orcnn_sanitized_long_20260612_gpu0/"
                "20260612_235635/vis_data/scalars.json"
            ),
        ]
    )
    dior_retina = all_points(
        [
            Path(
                "/data5/2025/ldh/OpenRSD/work_dirs/dior_r_s0_retinanet_sanitized_long_20260612_gpu2/"
                "20260612_181213/vis_data/scalars.json"
            ),
            Path(
                "/data5/2025/ldh/OpenRSD/work_dirs/dior_r_s0_retinanet_sanitized_long_20260612_gpu2/"
                "20260612_235311/vis_data/scalars.json"
            ),
        ]
    )
    dior_roi = all_points(
        [
            Path(
                "/data5/2025/ldh/OpenRSD/work_dirs/dior_r_s0_roi_trans_sanitized_long_20260612_gpu1/"
                "20260612_181202/vis_data/scalars.json"
            ),
            s0_dior_path,
        ]
    )
    detector_dior = [
        detector_summary("RoI Transformer", dior_roi, str(s0_dior_path)),
        detector_summary("Oriented R-CNN", dior_orcnn, "dior_r_s0_orcnn_sanitized_long_20260612_gpu0 scalars"),
        detector_summary("Rotated RetinaNet", dior_retina, "dior_r_s0_retinanet_sanitized_long_20260612_gpu2 scalars"),
    ]

    dota15 = [
        (
            "S0 RoITrans",
            load_json(rel("docs/experiments/20260526_roi_transformer_3x_dota15_metrics.json"))["best_map"],
            load_json(rel("docs/experiments/20260526_roi_transformer_3x_dota15_metrics.json"))["final_map"],
        ),
        (
            "S1",
            load_json(rel("docs/experiments/20260605_geonexus_s1_retry2_metrics.json"))["best_metrics"]["dota/mAP"],
            load_json(rel("docs/experiments/20260605_geonexus_s1_retry2_metrics.json"))["final_metrics"]["dota/mAP"],
        ),
        (
            "S2",
            load_json(rel("docs/experiments/20260605_geonexus_s2_rerun_s1e32_metrics.json"))["best_metrics"]["dota/mAP"],
            load_json(rel("docs/experiments/20260605_geonexus_s2_rerun_s1e32_metrics.json"))["final_metrics"]["dota/mAP"],
        ),
        (
            "S3",
            load_json(rel("docs/experiments/20260605_geonexus_s3_rerun_s2e4_metrics.json"))["best_metrics"]["dota/mAP"],
            load_json(rel("docs/experiments/20260605_geonexus_s3_rerun_s2e4_metrics.json"))["final_metrics"]["dota/mAP"],
        ),
    ]

    data = {
        "main": {
            "dota2": {
                "s0_roi_trans": s0_dota2_map,
                "s1": s1_dota2.map,
                "s2_loss0_best_mean": dota2_s2_best_mean,
                "s2_loss0_final_mean": dota2_s2_final_mean,
            },
            "dior_r": {
                "s0_roi_trans_final": s0_dior.map,
                "s1_final_mean": s1_dior_mean,
                "s2_best_mean": s2_dior_best_mean,
                "s2_final_mean": s2_dior_final_mean,
            },
        },
        "dota2_s2_runs": dota2_s2_runs,
        "dior_s1_finals": s1_dior_finals,
        "dior_s2_runs": s2_dior_runs,
        "detector_dota2": detector_dota2,
        "detector_dior": detector_dior,
        "dota15": dota15,
        "sources": {
            "dota2_s1": str(s1_dota2_path),
            "dior_s0": str(s0_dior_path),
            "dior_s2_archive": str(rel("docs/experiments/20260614_dior_r_geonexus_s2_replicas_complete.json")),
        },
    }

    assert_close("DIOR-R S0 final", data["main"]["dior_r"]["s0_roi_trans_final"], 0.6544)
    assert_close("DIOR-R S1 final mean", data["main"]["dior_r"]["s1_final_mean"], 0.6720)
    assert_close("DOTA2 S1", data["main"]["dota2"]["s1"], 0.6177)
    return data


def s0_dota2_path() -> Path:
    return rel("docs/experiments/20260603_s0_dota2_roi_trans_validpng_metrics.json")


def all_points(paths: list[Path]) -> list[MetricPoint]:
    points: list[MetricPoint] = []
    for path in paths:
        points.extend(load_scalar_points(path))
    return sorted(points, key=lambda p: p.epoch)


def detector_summary(name: str, points: list[MetricPoint], source: str) -> dict:
    best = max(points, key=lambda p: p.map)
    final = points[-1]
    return {
        "detector": name,
        "best_epoch": best.epoch,
        "best_mAP": best.map,
        "best_AP50": best.ap50,
        "final_epoch": final.epoch,
        "final_mAP": final.map,
        "final_AP50": final.ap50,
        "source": source,
    }


def setup_ax(title: str, subtitle: str = ""):
    fig, ax = plt.subplots(figsize=(W, H), dpi=220)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")
    ax.text(0.45, 8.55, title, fontsize=25, weight="bold", color=INK, va="top")
    if subtitle:
        ax.text(0.47, 8.12, subtitle, fontsize=11.2, color=MUTED, va="top")
    return fig, ax


def save(fig, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(OUT_DIR / f"{name}.{ext}", pad_inches=0.0, facecolor=BG)
    plt.close(fig)


def add_footer(ax, text: str) -> None:
    ax.text(0.45, 0.25, text, fontsize=8.8, color=MUTED)


def draw_table(ax, x: float, y: float, col_w: list[float], row_h: float, headers: list[str], rows: list[list[str]], title: str | None = None):
    if title:
        ax.text(x, y + row_h * (len(rows) + 1) + 0.28, title, fontsize=14, weight="bold", color=INK)
    total_w = sum(col_w)
    ax.add_patch(Rectangle((x, y + row_h * len(rows)), total_w, row_h, facecolor=INK, edgecolor=INK))
    cx = x
    for header, w in zip(headers, col_w):
        ax.text(cx + 0.12, y + row_h * len(rows) + row_h / 2, header, fontsize=9.4, color="white", weight="bold", va="center")
        cx += w
    for r, row in enumerate(rows):
        yy = y + row_h * (len(rows) - 1 - r)
        fc = "white" if r % 2 == 0 else "#eef2f7"
        ax.add_patch(Rectangle((x, yy), total_w, row_h, facecolor=fc, edgecolor=GRID, linewidth=0.8))
        cx = x
        for txt, w in zip(row, col_w):
            wrapped = "\n".join(wrap(str(txt), width=max(8, int(w * 7.0))))
            ax.text(cx + 0.12, yy + row_h / 2, wrapped, fontsize=8.8, color=INK, va="center", linespacing=1.05)
            cx += w
    cx = x
    for w in col_w[:-1]:
        cx += w
        ax.plot([cx, cx], [y, y + row_h * (len(rows) + 1)], color=GRID, lw=0.8)


def render_main_result_table(data: dict) -> None:
    fig, ax = setup_ax(
        "Main Result Table",
        "Formal evidence only: DOTA2_1024_500/ss_val and sanitized DIOR_R_dota/test.",
    )
    rows = [
        [
            "DOTA2_1024_500/ss_val",
            "S0 RoITrans",
            f"{data['main']['dota2']['s0_roi_trans']:.4f}",
            "closed-set detector",
        ],
        [
            "DOTA2_1024_500/ss_val",
            "S1 RemoteCLIP",
            f"{data['main']['dota2']['s1']:.4f}",
            "+0.0089 vs S0",
        ],
        [
            "DOTA2_1024_500/ss_val",
            "S2 loss-0 best mean",
            f"{data['main']['dota2']['s2_loss0_best_mean']:.4f}",
            "7 runs, early checkpoints",
        ],
        [
            "DOTA2_1024_500/ss_val",
            "S2 loss-0 final mean",
            f"{data['main']['dota2']['s2_loss0_final_mean']:.4f}",
            "final-unstable",
        ],
        [
            "DIOR_R_dota/test",
            "S0 RoITrans final",
            f"{data['main']['dior_r']['s0_roi_trans_final']:.4f}",
            "sanitized labels",
        ],
        [
            "DIOR_R_dota/test",
            "S1 RemoteCLIP mean",
            f"{data['main']['dior_r']['s1_final_mean']:.4f}",
            "2 replicas, final",
        ],
        [
            "DIOR_R_dota/test",
            "S2 hierarchy best mean",
            f"{data['main']['dior_r']['s2_best_mean']:.4f}",
            "3 replicas, best",
        ],
        [
            "DIOR_R_dota/test",
            "S2 hierarchy final mean",
            f"{data['main']['dior_r']['s2_final_mean']:.4f}",
            "3 replicas, final",
        ],
    ]
    draw_table(ax, 0.55, 1.25, [3.6, 3.25, 1.55, 5.75], 0.72, ["Dataset / split", "Stage", "mAP", "Read"], rows)
    ax.text(0.65, 0.75, "Main story: DOTA2 has modest early S2 signal; DIOR-R has stronger S0 -> S1 -> S2 gains.", fontsize=12.5, weight="bold", color=INK)
    add_footer(ax, "Numbers are computed from recorded JSON/scalar sources by scripts/make_ppt_assets_20260614.py.")
    save(fig, "main_result_table_16x9")


def render_dior_progression(data: dict) -> None:
    fig, ax = plt.subplots(figsize=(W, H), dpi=220)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    labels = ["S0 RoITrans", "S1 RemoteCLIP", "S2 hierarchy\nbest", "S2 hierarchy\nfinal"]
    means = [
        data["main"]["dior_r"]["s0_roi_trans_final"],
        data["main"]["dior_r"]["s1_final_mean"],
        data["main"]["dior_r"]["s2_best_mean"],
        data["main"]["dior_r"]["s2_final_mean"],
    ]
    x = list(range(len(labels)))
    bars = ax.bar(x, means, color=[BLUE, TEAL, GREEN, AMBER], width=0.55, alpha=0.9)
    ax.plot(x, means, color=INK, lw=2.2, marker="o", markersize=8, zorder=4)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.0013, f"{val:.4f}", ha="center", va="bottom", fontsize=12, weight="bold", color=INK)

    s1_points = data["dior_s1_finals"]
    for j, val in enumerate(s1_points):
        ax.scatter([1 - 0.08 + j * 0.16], [val], s=48, color="white", edgecolor=TEAL, linewidth=1.8, zorder=5)
    for j, run in enumerate(data["dior_s2_runs"]):
        ax.scatter([2 - 0.12 + j * 0.12], [run["best_map"]], s=52, color="white", edgecolor=GREEN, linewidth=1.9, zorder=5)
        ax.scatter([3 - 0.12 + j * 0.12], [run["final_map"]], s=52, color="white", edgecolor=AMBER, linewidth=1.9, zorder=5)

    best_run = max(data["dior_s2_runs"], key=lambda r: r["best_map"])
    ax.annotate(
        f"best replica {best_run['best_map']:.4f}",
        xy=(2, best_run["best_map"]),
        xytext=(2.45, best_run["best_map"] + 0.010),
        arrowprops={"arrowstyle": "->", "color": GREEN, "lw": 1.4},
        fontsize=11,
        color=GREEN,
        weight="bold",
    )
    ax.set_title("DIOR-R Stage Progression", loc="left", fontsize=24, weight="bold", color=INK, pad=20)
    ax.text(0, 1.01, "Sanitized DIOR_R_dota/test; dots show replica values, bold markers show means.", transform=ax.transAxes, fontsize=11, color=MUTED)
    ax.set_xticks(x, labels, fontsize=11)
    ax.set_ylabel("mAP", fontsize=11, color=MUTED)
    ax.set_ylim(0.645, 0.696)
    ax.grid(axis="y", color=GRID, linewidth=1)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="y", labelsize=10, colors=MUTED)
    ax.tick_params(axis="x", colors=INK)
    fig.tight_layout(rect=(0.06, 0.07, 0.98, 0.92))
    save(fig, "dior_r_stage_progression_16x9")


def render_dota2_stability(data: dict) -> None:
    fig, ax = plt.subplots(figsize=(W, H), dpi=220)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    s0 = data["main"]["dota2"]["s0_roi_trans"]
    s1 = data["main"]["dota2"]["s1"]
    bests = [r["best_map"] for r in data["dota2_s2_runs"]]
    finals = [r["final_map"] for r in data["dota2_s2_runs"]]
    labels = [r["name"] for r in data["dota2_s2_runs"]]
    x = list(range(len(labels)))
    ax.axhspan(s1, max(bests) + 0.0008, color=LIGHT_TEAL, alpha=0.55, label="above S1")
    ax.axhline(s0, color=BLUE, lw=1.8, linestyle="--", label=f"S0 {s0:.4f}")
    ax.axhline(s1, color=INK, lw=2.0, label=f"S1 {s1:.4f}")
    ax.scatter(x, bests, s=80, color=GREEN, edgecolor="white", linewidth=1.3, zorder=4, label="S2 best checkpoint")
    ax.scatter(x, finals, s=80, color=RED, edgecolor="white", linewidth=1.3, zorder=4, label="S2 final checkpoint")
    for xi, b, f in zip(x, bests, finals):
        ax.plot([xi, xi], [f, b], color="#9ba7b8", lw=1.4, alpha=0.85)
    ax.set_xticks(x, labels, rotation=22, ha="right", fontsize=9.5)
    ax.set_ylim(0.611, 0.623)
    ax.set_ylabel("DOTA2 ss_val mAP", fontsize=11, color=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=1)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="y", labelsize=10, colors=MUTED)
    ax.tick_params(axis="x", colors=INK)
    ax.set_title("DOTA2 S2 Stability Check", loc="left", fontsize=24, weight="bold", color=INK, pad=20)
    ax.text(0, 1.01, "Seven loss-0 S2 runs: best checkpoints are repeatably above S1, finals are unstable.", transform=ax.transAxes, fontsize=11, color=MUTED)
    ax.text(0.02, 0.06, f"best mean {mean(bests):.4f}", transform=ax.transAxes, fontsize=12, color=GREEN, weight="bold")
    ax.text(0.22, 0.06, f"final mean {mean(finals):.4f}", transform=ax.transAxes, fontsize=12, color=RED, weight="bold")
    ax.legend(loc="lower right", frameon=False, fontsize=9.5)
    fig.tight_layout(rect=(0.06, 0.09, 0.98, 0.92))
    save(fig, "dota2_stability_loss0_16x9")


def render_detector_table(data: dict) -> None:
    fig, ax = setup_ax(
        "Detector Baseline Table",
        "Closed-set detector context for the formal DOTA2 and sanitized DIOR-R evidence.",
    )
    drows = [
        [r["detector"], f"{r['mAP']:.4f}", f"{r['AP50']:.4f}", "DOTA2_1024_500/ss_val"]
        for r in data["detector_dota2"]
    ]
    draw_table(ax, 0.5, 4.00, [3.35, 1.15, 1.15, 3.65], 0.42, ["DOTA2 detector", "mAP", "AP50", "Protocol"], drows, "DOTA2 Detector Family")
    rows = [
        [
            r["detector"],
            f"{r['best_mAP']:.4f} @ e{r['best_epoch']}",
            f"{r['final_mAP']:.4f} @ e{r['final_epoch']}",
            "DIOR_R_dota/test",
        ]
        for r in data["detector_dior"]
    ]
    draw_table(ax, 0.5, 1.00, [3.35, 2.0, 2.0, 3.95], 0.56, ["DIOR-R detector", "Best mAP", "Final mAP", "Protocol"], rows, "DIOR-R Detector Family")
    add_footer(ax, "Detector values are parsed from experiment JSON, scalar JSONL, and recorded status notes.")
    save(fig, "detector_baseline_table_16x9")


def render_appendix(data: dict) -> None:
    fig, ax = setup_ax(
        "Appendix: DOTA v1.5 Archive Evidence",
        "Diagnostic/archive evidence only. Do not use as the headline benchmark.",
    )
    labels = [r[0] for r in data["dota15"]]
    bests = [r[1] for r in data["dota15"]]
    finals = [r[2] for r in data["dota15"]]
    x = list(range(len(labels)))
    ax2 = fig.add_axes([0.10, 0.18, 0.84, 0.58])
    ax2.set_facecolor(BG)
    ax2.plot(x, bests, marker="o", color=PURPLE, lw=2.2, label="best observed")
    ax2.plot(x, finals, marker="s", color=AMBER, lw=2.0, label="final")
    for xi, b, f in zip(x, bests, finals):
        ax2.text(xi, b + 0.004, f"{b:.4f}", ha="center", fontsize=9.5, color=PURPLE, weight="bold")
        ax2.text(xi, f - 0.007, f"{f:.4f}", ha="center", fontsize=9.5, color=AMBER, weight="bold")
    ax2.set_xticks(x, labels, fontsize=11)
    ax2.set_ylim(0.24, 0.40)
    ax2.set_ylabel("DOTA v1.5 diagnostic mAP", fontsize=11, color=MUTED)
    ax2.grid(axis="y", color=GRID)
    for spine in ax2.spines.values():
        spine.set_visible(False)
    ax2.tick_params(axis="y", colors=MUTED)
    ax2.tick_params(axis="x", colors=INK)
    ax2.legend(frameon=False, fontsize=10, loc="lower right")
    add_footer(ax, "DOTA v1.5 route was retired by the 2026-06-06 DOTA2/DIOR-R pivot.")
    save(fig, "appendix_dota15_archive_ablation_16x9")


def write_summary(data: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    serializable = json.loads(
        json.dumps(
            data,
            default=lambda o: o.__dict__ if isinstance(o, MetricPoint) else str(o),
        )
    )
    (OUT_DIR / "result_summary.json").write_text(json.dumps(serializable, indent=2) + "\n", encoding="utf-8")

    with (OUT_DIR / "main_result_table.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "stage", "mAP"])
        for dataset, values in data["main"].items():
            for stage, value in values.items():
                writer.writerow([dataset, stage, f"{value:.6f}"])

    readme = """# GeoNexus-RSD PowerPoint Assets - 2026-06-14

Generated 16:9 slide assets:

- `main_result_table_16x9.png` / `.svg`
- `dior_r_stage_progression_16x9.png` / `.svg`
- `dota2_stability_loss0_16x9.png` / `.svg`
- `detector_baseline_table_16x9.png` / `.svg`
- `appendix_dota15_archive_ablation_16x9.png` / `.svg`

Use DOTA v1.5 only as appendix/archive evidence. The formal main evidence is
`DOTA2_1024_500/ss_val` and sanitized `DIOR_R_dota/test`.
"""
    (OUT_DIR / "README.md").write_text(readme, encoding="utf-8")


def render_all() -> None:
    data = collect_data()
    render_main_result_table(data)
    render_dior_progression(data)
    render_dota2_stability(data)
    render_detector_table(data)
    render_appendix(data)
    write_summary(data)


if __name__ == "__main__":
    render_all()
