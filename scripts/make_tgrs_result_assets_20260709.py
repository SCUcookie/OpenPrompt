"""Generate paper-ready TGRS result assets for the 2026-07-09 evidence state.

Extends the 2026-06-14 generator (`make_tgrs_result_assets_20260614.py`) with:

- DIOR-R S3 (scene adapter) and S4 (pseudo-label stabilization, closed/negative)
- A SOTA comparator table/figure (GeoNexus vs OrientedFormer / Strip R-CNN / AOPG)

Outputs are written under:

  _local_archive_20260601_pull_backup/docs/TGRS/figure
  _local_archive_20260601_pull_backup/docs/TGRS/tables

The generator uses repo-local experiment records only. It does not read server
scalar files directly, so the package can be rebuilt from this workspace.
"""

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
TGRS_DIR = ROOT / "_local_archive_20260601_pull_backup" / "docs" / "TGRS"
FIG_DIR = TGRS_DIR / "figure"
TAB_DIR = TGRS_DIR / "tables"

DOCS = ROOT / "docs" / "experiments"

INK = "#111111"
MUTED = "#555555"
GRID = "#c9c9c9"
BLUE = "#1f77b4"
ORANGE = "#d55e00"
GREEN = "#009e73"
PURPLE = "#6a51a3"
GRAY = "#666666"
RED = "#b02727"
LIGHT = "#f4f4f4"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


@dataclass(frozen=True)
class StageValue:
    dataset: str
    split: str
    stage: str
    metric: str
    value: float
    replicas: str
    source: str
    note: str


@dataclass(frozen=True)
class Dota2Run:
    name: str
    epochs: list[int]
    maps: list[float]
    ap50s: list[float]
    best_epoch: int
    best_map: float
    final_epoch: int
    final_map: float
    source: str


@dataclass(frozen=True)
class DiorRun:
    name: str
    seed: int
    epochs: list[int]
    maps: list[float]
    best_epoch: int
    best_map: float
    final_epoch: int
    final_map: float
    source: str


@dataclass(frozen=True)
class ComparatorRow:
    method: str
    dataset_note: str
    ap50: float
    kind: str  # "geonexus" | "confirmed_run" | "public_paper" | "blocked"
    source: str


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_pair(text: str, label: str) -> tuple[float, float]:
    pattern = re.compile(
        rf"{re.escape(label)}.*?dota/mAP=([0-9.]+).*?dota/AP50=([0-9.]+)",
        re.DOTALL,
    )
    match = pattern.search(text)
    if not match:
        raise ValueError(f"Could not parse metric for {label!r}")
    return float(match.group(1)), float(match.group(2))


def assert_round(name: str, actual: float, expected: float) -> None:
    if round(actual, 4) != expected:
        raise AssertionError(f"{name}: got {actual:.8f}, expected rounded {expected:.4f}")


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def fmt(x: float | str | int) -> str:
    if isinstance(x, float):
        return f"{x:.4f}"
    return str(x)


def tex_escape(text: str) -> str:
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


def short_source(text: str) -> str:
    text = text.replace("\\", "/")
    if text.startswith("docs/experiments/"):
        return text.removeprefix("docs/experiments/")
    if text.startswith("/data5/2025/ldh/OpenRSD/work_dirs/"):
        tail = text.removeprefix("/data5/2025/ldh/OpenRSD/work_dirs/")
        parts = tail.split("/")
        return "work_dir:" + "/".join(parts[:2]) if len(parts) >= 2 else "work_dir:" + tail
    return text


def write_csv(path: Path, headers: list[str], rows: list[list[object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


def latex_table(
    *,
    headers: list[str],
    rows: list[list[object]],
    caption: str,
    label: str,
    widths: str,
    starred: bool = False,
    footnote: str | None = None,
    scale_to_width: str | None = None,
) -> str:
    env = "table*" if starred else "table"
    body = [
        rf"\begin{{{env}}}[t]",
        r"\centering",
        r"\footnotesize",
        rf"\caption{{{tex_escape(caption)}}}",
        rf"\label{{{label}}}",
    ]
    if scale_to_width:
        body.append(rf"\resizebox{{{scale_to_width}}}{{!}}{{%")
    body.extend(
        [
        r"\begin{tabular}{" + widths + r"}",
        r"\toprule",
        " & ".join(tex_escape(h) for h in headers) + r" \\",
        r"\midrule",
        ]
    )
    for row in rows:
        body.append(" & ".join(tex_escape(fmt(cell)) for cell in row) + r" \\")
    body.extend([r"\bottomrule", r"\end{tabular}"])
    if scale_to_width:
        body.append(r"}")
    if footnote:
        body.append(r"\vspace{1mm}")
        body.append(r"\parbox{0.96\linewidth}{\scriptsize " + tex_escape(footnote) + "}")
    body.append(rf"\end{{{env}}}")
    return "\n".join(body) + "\n"


def figure_fragment(path: Path, caption: str, label: str, width: str = r"\textwidth", starred: bool = True) -> str:
    env = "figure*" if starred else "figure"
    stem = path.as_posix()
    return "\n".join(
        [
            rf"\begin{{{env}}}[t]",
            r"\centering",
            rf"\includegraphics[width={width}]{{{stem}}}",
            rf"\caption{{{tex_escape(caption)}}}",
            rf"\label{{{label}}}",
            rf"\end{{{env}}}",
            "",
        ]
    )


def collect_dota2_runs() -> list[Dota2Run]:
    md_path = DOCS / "20260611_dota2_s2_loss0_replicates_analysis.md"
    rows = [
        ("loss-0", [1, 2, 3, 4], [0.617559, 0.620374, 0.618740, 0.617885], [0.6180, 0.6200, 0.6190, 0.6180], md_path),
        ("rep3407", [1, 2, 3, 4], [0.621121, 0.617161, 0.612046, 0.620299], [0.6210, 0.6170, 0.6120, 0.6200], md_path),
        ("rep4407", [1, 2, 3, 4], [0.615142, 0.615141, 0.620637, 0.614786], [0.6150, 0.6150, 0.6210, 0.6150], md_path),
        ("rep5407", [1, 2, 3, 4], [0.621215, 0.620206, 0.616239, 0.614990], [0.6210, 0.6200, 0.6160, 0.6150], md_path),
        ("rep6407", [1, 2, 3], [0.620483, 0.618960, 0.618167], [0.6200, 0.6190, 0.6180], md_path),
        ("rep7407", [1, 2, 3], [0.620785, 0.614483, 0.618315], [0.6210, 0.6140, 0.6180], md_path),
        ("rep8407", [1, 2, 3], [0.616526, 0.619625, 0.612147], [0.6170, 0.6200, 0.6120], md_path),
    ]
    runs = []
    for name, epochs, maps, ap50s, src in rows:
        best_idx = max(range(len(maps)), key=lambda i: maps[i])
        runs.append(
            Dota2Run(
                name=name,
                epochs=epochs,
                maps=maps,
                ap50s=ap50s,
                best_epoch=epochs[best_idx],
                best_map=maps[best_idx],
                final_epoch=epochs[-1],
                final_map=maps[-1],
                source=rel(src),
            )
        )
    assert_round("DOTA2 S2 best mean", mean(r.best_map for r in runs), 0.6206)
    assert_round("DOTA2 S2 final mean", mean(r.final_map for r in runs), 0.6167)
    return runs


def collect_dior_s3() -> list[DiorRun]:
    path = DOCS / "20260615_dior_r_geonexus_s3_scene_adapter_replicas_complete.json"
    payload = load_json(path)
    runs = []
    for rep in payload["replicas"]:
        epochs = [int(m["epoch"]) for m in rep["metrics"]]
        maps = [float(m["dota_mAP"]) for m in rep["metrics"]]
        runs.append(
            DiorRun(
                name=f"rep{rep['replica']}",
                seed=int(rep["seed"]),
                epochs=epochs,
                maps=maps,
                best_epoch=int(rep["best_epoch"]),
                best_map=float(rep["best_mAP"]),
                final_epoch=int(rep["final_epoch"]),
                final_map=float(rep["final_mAP"]),
                source=rel(path),
            )
        )
    assert_round("DIOR-R S3 best mean", payload["aggregates"]["best_mean_mAP"], 0.6979)
    assert_round("DIOR-R S3 final mean", payload["aggregates"]["final_mean_mAP"], 0.6859)
    return runs


def collect_dior_s4() -> list[DiorRun]:
    path = DOCS / "20260701_dior_r_s4_lr5e6_complete.json"
    payload = load_json(path)
    runs = []
    for rep in payload["replicas"]:
        runs.append(
            DiorRun(
                name=rep["replica"],
                seed=int(rep["seed"]),
                epochs=[rep["best_epoch"], rep["final_epoch"]],
                maps=[float(rep["best_mAP"]), float(rep["final_mAP"])],
                best_epoch=int(rep["best_epoch"]),
                best_map=float(rep["best_mAP"]),
                final_epoch=int(rep["final_epoch"]),
                final_map=float(rep["final_mAP"]),
                source=rel(path),
            )
        )
    assert_round("DIOR-R S4 best mean", payload["aggregate"]["best_mean_mAP"], 0.6973)
    assert_round("DIOR-R S4 final mean", payload["aggregate"]["final_mean_mAP"], 0.6940)
    return runs, payload["aggregate"]["s3_mean_gate"]


def collect_comparators(dior_s3_best_mean: float) -> list[ComparatorRow]:
    # OrientedFormer Swin-T confirmed by our own reproducibility rerun (2026-07-04)
    # on the bridged sanitized DIOR-R protocol. Strip R-CNN / AOPG are public
    # paper rows only (2026-06-30 SOTA audit); we could not independently
    # evaluate them because no usable DIOR-R checkpoint was released for
    # Strip R-CNN-S, and AOPG was not re-run locally.
    return [
        ComparatorRow("GeoNexus-RSD S3 (best mean)", "sanitized DIOR-R test, our protocol", round(dior_s3_best_mean, 4), "geonexus", "20260615_dior_r_geonexus_s3_scene_adapter_replicas_complete.json"),
        ComparatorRow("GeoNexus-RSD S3 (final mean)", "sanitized DIOR-R test, our protocol", 0.6859, "geonexus", "20260615_dior_r_geonexus_s3_scene_adapter_replicas_complete.json"),
        ComparatorRow("OrientedFormer Swin-T", "our rerun, bridged sanitized DIOR-R", 0.6883, "confirmed_run", "20260704_orientedformer_dior_r_protocol_eval_rerun.md"),
        ComparatorRow("OrientedFormer Swin-T", "public paper row (TGRS 2024)", 0.6884, "public_paper", "20260630_sota_audit.md"),
        ComparatorRow("OrientedFormer R50", "public paper row (TGRS 2024)", 0.6728, "public_paper", "20260630_sota_audit.md"),
        ComparatorRow("Strip R-CNN-S", "public paper row; checkpoint unavailable for our eval", 0.6870, "blocked", "20260630_sota_audit.md; PROJECT_INSTRUCTIONS.md 2026-07-03/07-06"),
        ComparatorRow("AOPG", "public paper row (arXiv 2110.01931)", 0.6441, "public_paper", "20260630_sota_audit.md"),
    ]


def collect_data() -> dict:
    s0_dota2_path = DOCS / "20260603_s0_dota2_roi_trans_validpng_metrics.json"
    s0_dota2 = load_json(s0_dota2_path)["metrics"]
    s0_dota2_map = float(s0_dota2["dota/mAP"])
    s0_dota2_ap50 = float(s0_dota2["dota/AP50"])

    s1_dota2 = {"map": 0.6177, "ap50": 0.6180, "source": "docs/experiments/20260608_dota2_s1_complete_and_s2_launch.md"}
    dota2_runs = collect_dota2_runs()

    s1_dior_path = DOCS / "20260613_dior_r_geonexus_s1_s0e52_replicas_metrics.json"
    s1_dior = load_json(s1_dior_path)
    s1_finals = []
    for rep in s1_dior["replicas"]:
        final = next(m for m in rep["metrics"] if int(m["epoch"]) == int(rep["final_epoch"]))
        s1_finals.append(float(final["dota_mAP"]))
    s1_dior_mean = mean(s1_finals)

    s2_dior_path = DOCS / "20260614_dior_r_geonexus_s2_replicas_complete.json"
    s2_dior = load_json(s2_dior_path)
    dior_s2_runs = []
    for rep in s2_dior["replicas"]:
        epochs = [int(m["epoch"]) for m in rep["metrics"]]
        maps = [float(m["dota_mAP"]) for m in rep["metrics"]]
        dior_s2_runs.append(
            DiorRun(
                name=f"rep{rep['replica']}",
                seed=int(rep["seed"]),
                epochs=epochs,
                maps=maps,
                best_epoch=int(rep["best_epoch"]),
                best_map=float(rep["best_mAP"]),
                final_epoch=int(rep["final_epoch"]),
                final_map=float(rep["final_mAP"]),
                source=rep.get("metric_source", rel(s2_dior_path)),
            )
        )
    s2_dior_best_mean = mean(r.best_map for r in dior_s2_runs)
    s2_dior_final_mean = mean(r.final_map for r in dior_s2_runs)

    dior_s3_runs = collect_dior_s3()
    s3_dior_best_mean = mean(r.best_map for r in dior_s3_runs)
    s3_dior_final_mean = mean(r.final_map for r in dior_s3_runs)

    dior_s4_runs, s3_gate = collect_dior_s4()
    s4_dior_best_mean = mean(r.best_map for r in dior_s4_runs)
    s4_dior_final_mean = mean(r.final_map for r in dior_s4_runs)

    comparators = collect_comparators(s3_dior_best_mean)

    assert_round("DOTA2 S0", s0_dota2_map, 0.6088)
    assert_round("DOTA2 S1", s1_dota2["map"], 0.6177)
    assert_round("DIOR-R S1 final mean", s1_dior_mean, 0.6720)
    assert_round("DIOR-R S2 best mean", s2_dior_best_mean, 0.6887)
    assert_round("DIOR-R S2 final mean", s2_dior_final_mean, 0.6856)

    baselines_md = (DOCS / "20260605_dota2_baseline_status.md").read_text(encoding="utf-8")
    current_md = (DOCS / "20260607_current_status_and_next_launch.md").read_text(encoding="utf-8")
    opensrd = load_json(DOCS / "20260602_opensrd_dota2_epoch12_ssval_metrics.json")["metrics"]
    detector_dota2 = [
        ["RoI Transformer", s0_dota2_map, s0_dota2_ap50, "epoch 12 final", rel(s0_dota2_path)],
        ["Oriented R-CNN", *parse_pair(baselines_md, "Oriented R-CNN R50 bs1"), "epoch 12 final", "docs/experiments/20260605_dota2_baseline_status.md"],
        ["S2ANet", *parse_pair(baselines_md, "S2ANet bs1"), "epoch 12 final", "docs/experiments/20260605_dota2_baseline_status.md"],
        ["R3Det-KFIoU", *parse_pair(current_md, "DOTA2 R3Det-KFIoU valid-PNG bs1 completed epoch 12"), "epoch 12 final", "docs/experiments/20260607_current_status_and_next_launch.md"],
        ["RTMDet-M", *parse_pair(baselines_md, "RTMDet-M bs1"), "epoch 12 final", "docs/experiments/20260605_dota2_baseline_status.md"],
        ["RTMDet-L", *parse_pair(current_md, "DOTA2 RTMDet-L valid-PNG bs1 completed epoch 12"), "epoch 12 final", "docs/experiments/20260607_current_status_and_next_launch.md"],
        ["OpenRSD reference", float(opensrd["dota/mAP"]), float(opensrd["dota/AP50"]), "epoch 12 eval", "docs/experiments/20260602_opensrd_dota2_epoch12_ssval_metrics.json"],
    ]

    detector_dior = [
        ["RoI Transformer", 0.6544, 0.6540, "epoch 52 final", "PROJECT_INSTRUCTIONS.md"],
        ["Oriented R-CNN", 0.6341, 0.6340, "best epoch 28; final not locally archived", "docs/experiments/20260613_dior_r_s0_sanitized_long_interim.md"],
        ["Rotated RetinaNet", 0.5694, 0.5690, "epoch 96 final/best", "docs/experiments/20260613_dior_r_s0_retinanet_sanitized_long_complete.md"],
    ]

    main_rows = [
        StageValue("DOTA-v2.0", "DOTA2_1024_500/ss_val", "S0 RoI Transformer", "mAP", s0_dota2_map, "1", rel(s0_dota2_path), "closed-set detector reference"),
        StageValue("DOTA-v2.0", "DOTA2_1024_500/ss_val", "S1 RemoteCLIP prompts", "final mAP", s1_dota2["map"], "1", s1_dota2["source"], "completed positive result"),
        StageValue("DOTA-v2.0", "DOTA2_1024_500/ss_val", "S2 loss-0 best checkpoints", "best-checkpoint mean mAP", mean(r.best_map for r in dota2_runs), "7", "docs/experiments/20260611_dota2_s2_loss0_replicates_analysis.md", "early-checkpoint evidence; above S1"),
        StageValue("DOTA-v2.0", "DOTA2_1024_500/ss_val", "S2 loss-0 final checkpoints", "final mean mAP", mean(r.final_map for r in dota2_runs), "7", "docs/experiments/20260611_dota2_s2_loss0_replicates_analysis.md", "final checkpoints unstable; below S1"),
        StageValue("DOTA-v2.0", "DOTA2_1024_500/ss_val", "S3 scene adapter", "best mean mAP", 0.6199, "3", "docs/experiments/20260616_dota2_s3_scene_adapter_loss0_best_complete.md", "exploratory; below S2 best/final"),
        StageValue("DIOR-R", "DIOR_R_dota/test", "S0 RoI Transformer", "final mAP", 0.6544, "1", "PROJECT_INSTRUCTIONS.md", "sanitized-label detector reference"),
        StageValue("DIOR-R", "DIOR_R_dota/test", "S1 RemoteCLIP prompts", "final mean mAP", s1_dior_mean, "2", rel(s1_dior_path), "two-replica mean"),
        StageValue("DIOR-R", "DIOR_R_dota/test", "S2 hierarchy", "best-checkpoint mean mAP", s2_dior_best_mean, str(len(dior_s2_runs)), rel(s2_dior_path), "best checkpoint mean"),
        StageValue("DIOR-R", "DIOR_R_dota/test", "S2 hierarchy", "final mean mAP", s2_dior_final_mean, str(len(dior_s2_runs)), rel(s2_dior_path), "final checkpoint mean"),
        StageValue("DIOR-R", "DIOR_R_dota/test", "S3 scene adapter", "best-checkpoint mean mAP", s3_dior_best_mean, "3", rel(DOCS / "20260615_dior_r_geonexus_s3_scene_adapter_replicas_complete.json"), "strongest local row; best single 0.6992"),
        StageValue("DIOR-R", "DIOR_R_dota/test", "S3 scene adapter", "final mean mAP", s3_dior_final_mean, "3", rel(DOCS / "20260615_dior_r_geonexus_s3_scene_adapter_replicas_complete.json"), "roughly tied with S2 final"),
        StageValue("DIOR-R", "DIOR_R_dota/test", "S4 pseudo-label (closed)", "best-checkpoint mean mAP", s4_dior_best_mean, "3", rel(DOCS / "20260701_dior_r_s4_lr5e6_complete.json"), f"below S3 gate {s3_gate:.4f}; route paused"),
        StageValue("DIOR-R", "DIOR_R_dota/test", "S4 pseudo-label (closed)", "final-checkpoint mean mAP", s4_dior_final_mean, "3", rel(DOCS / "20260701_dior_r_s4_lr5e6_complete.json"), "stabilization only, not superiority"),
    ]
    return {
        "s0_dota2": s0_dota2_map,
        "s0_dota2_ap50": s0_dota2_ap50,
        "s1_dota2": s1_dota2["map"],
        "s1_dota2_ap50": s1_dota2["ap50"],
        "dota2_runs": dota2_runs,
        "s1_dior_finals": s1_finals,
        "dior_s2_runs": dior_s2_runs,
        "dior_s3_runs": dior_s3_runs,
        "dior_s4_runs": dior_s4_runs,
        "s3_gate": s3_gate,
        "s0_dior": 0.6544,
        "s1_dior_mean": s1_dior_mean,
        "s2_dior_best_mean": s2_dior_best_mean,
        "s2_dior_final_mean": s2_dior_final_mean,
        "s3_dior_best_mean": s3_dior_best_mean,
        "s3_dior_final_mean": s3_dior_final_mean,
        "s4_dior_best_mean": s4_dior_best_mean,
        "s4_dior_final_mean": s4_dior_final_mean,
        "comparators": comparators,
        "main_rows": main_rows,
        "detector_dota2": detector_dota2,
        "detector_dior": detector_dior,
    }


def save_figure(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg"):
        fig.savefig(FIG_DIR / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


def style_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y", color=GRID, linewidth=0.55, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=7)


def render_dior_progression(data: dict) -> None:
    labels = ["S0", "S1\nmean", "S2\nbest", "S2\nfinal", "S3\nbest", "S3\nfinal", "S4\nbest", "S4\nfinal"]
    means = [
        data["s0_dior"], data["s1_dior_mean"],
        data["s2_dior_best_mean"], data["s2_dior_final_mean"],
        data["s3_dior_best_mean"], data["s3_dior_final_mean"],
        data["s4_dior_best_mean"], data["s4_dior_final_mean"],
    ]
    colors = [GRAY, BLUE, GREEN, GREEN, PURPLE, PURPLE, ORANGE, ORANGE]
    fig, ax = plt.subplots(figsize=(7.05, 2.5))
    x = list(range(len(means)))
    ax.bar(x, means, width=0.6, color=colors, alpha=0.9, edgecolor=INK, linewidth=0.35)
    ax.plot(x, means, color=INK, marker="o", markersize=3.2, linewidth=1.0, zorder=3)
    ax.axhline(data["s3_dior_best_mean"], color=PURPLE, linestyle="--", linewidth=0.7, alpha=0.55)
    for xi, val in zip(x, means):
        ax.text(xi, val + 0.0013, f"{val:.4f}", ha="center", va="bottom", fontsize=6.3)
    for j, run in enumerate(data["dior_s3_runs"]):
        offset = -0.09 + 0.09 * j
        ax.scatter(4 + offset, run.best_map, s=18, marker="D", facecolor="white", edgecolor=PURPLE, linewidth=0.75, zorder=4)
        ax.scatter(5 + offset, run.final_map, s=18, marker="o", facecolor="white", edgecolor=PURPLE, linewidth=0.75, zorder=4)
    best_s3 = max(data["dior_s3_runs"], key=lambda r: r.best_map)
    ax.annotate(
        f"best single: {best_s3.name} e{best_s3.best_epoch}, {best_s3.best_map:.4f}",
        xy=(4, best_s3.best_map),
        xytext=(4.65, best_s3.best_map + 0.0075),
        arrowprops={"arrowstyle": "->", "lw": 0.6, "color": PURPLE},
        fontsize=6.0,
        color=INK,
    )
    ax.annotate(
        "S4 stabilization attempt:\nbelow S3 best-mean gate",
        xy=(6.3, data["s4_dior_best_mean"] - 0.001),
        xytext=(6.55, 0.6635),
        arrowprops={"arrowstyle": "->", "lw": 0.6, "color": ORANGE},
        fontsize=6.0,
        color=INK,
        ha="left",
    )
    ax.set_xticks(x, labels)
    ax.set_ylabel("DIOR-R mAP")
    ax.set_ylim(0.648, 0.706)
    ax.text(0.01, 0.985, "DIOR-R progression: S0 to S4", transform=ax.transAxes, fontsize=8.4, fontweight="bold", va="top")
    ax.text(0.01, 0.895, "S3 scene adapter is the strongest local row; S4 pseudo-label stabilization did not clear the S3 gate.", transform=ax.transAxes, fontsize=6.4, color=MUTED)
    style_axis(ax)
    fig.tight_layout(pad=0.2)
    save_figure(fig, "geonexus_tgrs_dior_progression")


def render_comparators(data: dict) -> None:
    rows = data["comparators"]
    fig, ax = plt.subplots(figsize=(7.05, 2.65))
    y = list(range(len(rows)))
    color_by_kind = {"geonexus": GREEN, "confirmed_run": BLUE, "public_paper": GRAY, "blocked": RED}
    colors = [color_by_kind[r.kind] for r in rows]
    hatches = ["" if r.kind != "public_paper" else "//" for r in rows]
    bars = ax.barh(y, [r.ap50 for r in rows], color=colors, edgecolor=INK, linewidth=0.35, height=0.58)
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    ax.set_yticks(y, [f"{r.method}\n{r.dataset_note}" for r in rows], fontsize=6.0)
    ax.invert_yaxis()
    ax.set_xlabel("DIOR-R AP50")
    ax.set_xlim(0.60, 0.72)
    for yi, r in zip(y, rows):
        ax.text(r.ap50 + 0.003, yi, f"{r.ap50:.4f}", va="center", fontsize=6.4)
    ax.axvline(data["s3_dior_best_mean"], color=GREEN, linestyle="--", linewidth=0.7, alpha=0.6)
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=GREEN, label="GeoNexus-RSD (ours)"),
        plt.Rectangle((0, 0), 1, 1, color=BLUE, label="our reproduction of released checkpoint"),
        plt.Rectangle((0, 0), 1, 1, color=GRAY, hatch="//", label="public paper row (protocol not verified identical)"),
        plt.Rectangle((0, 0), 1, 1, color=RED, label="blocked: no usable checkpoint for our eval"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        fontsize=6.2,
        frameon=False,
        handlelength=1.2,
        ncol=2,
        columnspacing=1.1,
    )
    ax.text(0.01, 1.06, "DIOR-R comparator context (not a claimed SOTA table)", transform=ax.transAxes, fontsize=8.0, fontweight="bold", va="top")
    style_axis(ax)
    fig.subplots_adjust(bottom=0.30, top=0.86)
    save_figure(fig, "geonexus_tgrs_sota_comparators")


def render_main_results(data: dict) -> None:
    labels = ["DOTA2\nS0", "DOTA2\nS1", "DOTA2\nS2 best", "DOTA2\nS2 final", "DIOR-R\nS0", "DIOR-R\nS1", "DIOR-R\nS2 best", "DIOR-R\nS3 best"]
    values = [
        data["s0_dota2"], data["s1_dota2"],
        mean(r.best_map for r in data["dota2_runs"]), mean(r.final_map for r in data["dota2_runs"]),
        data["s0_dior"], data["s1_dior_mean"], data["s2_dior_best_mean"], data["s3_dior_best_mean"],
    ]
    colors = [GRAY, BLUE, GREEN, ORANGE, GRAY, BLUE, GREEN, PURPLE]
    fig, ax = plt.subplots(figsize=(7.05, 2.35))
    x = range(len(values))
    ax.bar(x, values, color=colors, width=0.62, edgecolor=INK, linewidth=0.35)
    ax.axvline(3.5, color=INK, linewidth=0.7)
    for xi, val in zip(x, values):
        ax.text(xi, val + 0.0024, f"{val:.4f}", ha="center", va="bottom", fontsize=6.7)
    ax.set_xticks(list(x), labels)
    ax.set_ylabel("mAP")
    ax.set_ylim(0.585, 0.712)
    ax.text(0.02, 0.96, "Formal evidence summary", transform=ax.transAxes, fontsize=8.6, fontweight="bold", va="top")
    ax.text(0.02, 0.875, "DOTA2 S2 separates best-checkpoint and final means; DIOR-R S3 is the strongest local stage.", transform=ax.transAxes, fontsize=6.7, color=MUTED)
    style_axis(ax)
    fig.tight_layout(pad=0.2)
    save_figure(fig, "geonexus_tgrs_main_results")


def render_dota2_stability(data: dict) -> None:
    runs = data["dota2_runs"]
    x = list(range(len(runs)))
    bests = [r.best_map for r in runs]
    finals = [r.final_map for r in runs]
    fig, ax = plt.subplots(figsize=(3.45, 2.35))
    ax.axhline(data["s0_dota2"], color=GRAY, linestyle="--", linewidth=0.9, label=f"S0 {data['s0_dota2']:.4f}")
    ax.axhline(data["s1_dota2"], color=BLUE, linestyle="-", linewidth=0.9, label=f"S1 {data['s1_dota2']:.4f}")
    for xi, b, f in zip(x, bests, finals):
        ax.plot([xi, xi], [f, b], color="#9b9b9b", linewidth=0.7, zorder=1)
    ax.scatter(x, bests, color=GREEN, marker="D", s=22, label="S2 best", zorder=3)
    ax.scatter(x, finals, color=ORANGE, marker="o", s=22, label="S2 final", zorder=3)
    ax.set_xticks(x, [r.name for r in runs], rotation=32, ha="right")
    ax.set_ylabel("DOTA2 mAP")
    ax.set_ylim(0.611, 0.623)
    ax.text(0.01, 0.98, "DOTA2 S2 stability", transform=ax.transAxes, fontsize=8.3, fontweight="bold", va="top")
    ax.text(0.01, 0.08, f"best mean {mean(bests):.4f}; final mean {mean(finals):.4f}", transform=ax.transAxes, fontsize=6.2, color=MUTED)
    ax.legend(frameon=False, loc="lower right", fontsize=6.1, ncol=2, handlelength=1.1, columnspacing=0.8)
    style_axis(ax)
    fig.tight_layout(pad=0.2)
    save_figure(fig, "geonexus_tgrs_dota2_s2_stability")


def render_result_synthesis(data: dict) -> None:
    """Render the main Figure 6 replacement as a quantitative synthesis."""
    dior_labels = ["S0", "S1", "S2\nbest", "S3\nbest", "S4\nbest"]
    dior_values = [data["s0_dior"], data["s1_dior_mean"], data["s2_dior_best_mean"], data["s3_dior_best_mean"], data["s4_dior_best_mean"]]
    dota_labels = ["S0", "S1", "S2\nbest", "S2\nfinal"]
    dota_values = [
        data["s0_dota2"],
        data["s1_dota2"],
        mean(r.best_map for r in data["dota2_runs"]),
        mean(r.final_map for r in data["dota2_runs"]),
    ]

    fig = plt.figure(figsize=(7.05, 2.75))
    gs = fig.add_gridspec(2, 2, height_ratios=[4.2, 1.05], hspace=0.32, wspace=0.24)
    ax_dior = fig.add_subplot(gs[0, 0])
    ax_dota = fig.add_subplot(gs[0, 1])
    ax_note = fig.add_subplot(gs[1, :])

    def stage_panel(ax: plt.Axes, labels: list[str], values: list[float], colors: list[str], title: str, ylim: tuple[float, float]) -> None:
        x = list(range(len(values)))
        ax.bar(x, values, color=colors, width=0.58, edgecolor=INK, linewidth=0.35)
        ax.plot(x, values, color=INK, linewidth=1.0, marker="o", markersize=3.2)
        for xi, val in zip(x, values):
            ax.text(xi, val + (ylim[1] - ylim[0]) * 0.028, f"{val:.4f}", ha="center", va="bottom", fontsize=6.6)
        ax.set_xticks(x, labels)
        ax.set_ylim(*ylim)
        ax.set_ylabel("mAP", fontsize=7.2)
        ax.text(0.01, 0.98, title, transform=ax.transAxes, ha="left", va="top", fontsize=7.9, fontweight="bold")
        style_axis(ax)

    stage_panel(
        ax_dior,
        dior_labels,
        dior_values,
        [GRAY, BLUE, GREEN, PURPLE, ORANGE],
        "DIOR-R progression (best checkpoints)",
        (0.648, 0.706),
    )
    stage_panel(
        ax_dota,
        dota_labels,
        dota_values,
        [GRAY, BLUE, GREEN, ORANGE],
        "DOTA2 best/final reliability",
        (0.606, 0.6245),
    )
    ax_dota.axhline(data["s1_dota2"], color=BLUE, linewidth=0.75, linestyle="--", alpha=0.75)
    ax_dota.annotate(
        "final mean below S1",
        xy=(3, dota_values[3]),
        xytext=(1.65, 0.6080),
        arrowprops={"arrowstyle": "->", "lw": 0.65, "color": ORANGE},
        fontsize=6.2,
        color=INK,
    )

    ax_note.axis("off")
    ax_note.add_patch(
        plt.Rectangle(
            (0.0, 0.12),
            1.0,
            0.76,
            transform=ax_note.transAxes,
            facecolor=LIGHT,
            edgecolor=GRID,
            linewidth=0.6,
        )
    )
    ax_note.text(0.02, 0.67, "Claim boundary", transform=ax_note.transAxes, fontsize=7.2, fontweight="bold", color=INK, va="center")
    ax_note.text(
        0.18,
        0.67,
        "DIOR-R S3 is the strongest local row (best mean 0.6979); S4 pseudo-label stabilization did not clear the S3 gate.",
        transform=ax_note.transAxes,
        fontsize=6.7,
        color=INK,
        va="center",
    )
    ax_note.text(
        0.18,
        0.34,
        "GeoNexus DIOR-R S3 exceeds OrientedFormer Swin-T (our confirmed rerun, 0.6883) under a comparable but not protocol-identical setup; not yet a SOTA claim.",
        transform=ax_note.transAxes,
        fontsize=6.5,
        color=MUTED,
        va="center",
    )
    fig.tight_layout(pad=0.18)
    save_figure(fig, "geonexus_tgrs_result_synthesis")


def render_detector_baselines(data: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.45), gridspec_kw={"width_ratios": [1.35, 1.0]})
    for ax, rows, title in [
        (axes[0], data["detector_dota2"], "DOTA-v2.0 detector family"),
        (axes[1], data["detector_dior"], "DIOR-R detector family"),
    ]:
        names = [r[0] for r in rows]
        vals = [r[1] for r in rows]
        y = list(range(len(rows)))
        ax.barh(y, vals, color=BLUE, edgecolor=INK, linewidth=0.3, height=0.58)
        ax.set_yticks(y, names)
        ax.invert_yaxis()
        ax.set_xlabel("mAP")
        ax.set_xlim(0, max(vals) + 0.06)
        ax.set_title(title, loc="left", fontsize=8.0, fontweight="bold")
        for yi, val in zip(y, vals):
            ax.text(val + 0.007, yi, f"{val:.4f}", va="center", fontsize=6.3)
        style_axis(ax)
        ax.tick_params(axis="y", labelsize=6.4)
    fig.tight_layout(pad=0.25, w_pad=1.1)
    save_figure(fig, "geonexus_tgrs_detector_baselines")


def write_tables(data: dict) -> None:
    TAB_DIR.mkdir(parents=True, exist_ok=True)

    main_rows = [[r.dataset, r.split, r.stage, r.metric, r.value, r.replicas, r.note] for r in data["main_rows"]]
    write_csv(TAB_DIR / "table_tgrs_main_results.csv", ["Dataset", "Split", "Stage", "Metric", "Value", "Replicas", "Read"], main_rows)
    (TAB_DIR / "table_tgrs_main_results.tex").write_text(
        latex_table(
            headers=["Dataset", "Split", "Stage", "Metric", "mAP", "n", "Read"],
            rows=main_rows,
            caption="Main GeoNexus-RSD evidence on DOTA-v2.0 ss\\_val and sanitized DIOR-R test. DOTA-v2.0 S2/S3 and DIOR-R S2/S3/S4 report best-checkpoint and final-checkpoint means separately. DIOR-R S3 is the strongest local row; DIOR-R S4 is a closed, negative-to-neutral pseudo-label stabilization attempt.",
            label="tab:tgrs_main_results",
            widths="@{}p{0.10\\textwidth}p{0.16\\textwidth}p{0.17\\textwidth}p{0.15\\textwidth}cp{0.03\\textwidth}p{0.25\\textwidth}@{}",
            starred=True,
            scale_to_width=r"\textwidth",
        ),
        encoding="utf-8",
    )

    n_s2 = str(len(data["dior_s2_runs"]))
    n_s3 = str(len(data["dior_s3_runs"]))
    n_s4 = str(len(data["dior_s4_runs"]))
    dior_rows = [
        ["S0 RoI Transformer", "final", data["s0_dior"], "1", "Project instructions"],
        ["S1 RemoteCLIP prompts", "final mean", data["s1_dior_mean"], "2", "20260613 DIOR-R S1 JSON"],
        ["S2 hierarchy", "best mean", data["s2_dior_best_mean"], n_s2, "20260614 DIOR-R S2 JSON"],
        ["S2 hierarchy", "final mean", data["s2_dior_final_mean"], n_s2, "20260614 DIOR-R S2 JSON"],
        ["S3 scene adapter", "best mean", data["s3_dior_best_mean"], n_s3, "20260615 DIOR-R S3 JSON"],
        ["S3 scene adapter", "final mean", data["s3_dior_final_mean"], n_s3, "20260615 DIOR-R S3 JSON"],
        ["S4 pseudo-label (closed)", "best mean", data["s4_dior_best_mean"], n_s4, "20260701 DIOR-R S4 LR5e-6 JSON"],
        ["S4 pseudo-label (closed)", "final mean", data["s4_dior_final_mean"], n_s4, "20260701 DIOR-R S4 LR5e-6 JSON"],
    ]
    write_csv(TAB_DIR / "table_tgrs_dior_r.csv", ["Stage", "Reported statistic", "mAP", "Replicas", "Source"], dior_rows)
    (TAB_DIR / "table_tgrs_dior_r.tex").write_text(
        latex_table(
            headers=["Stage", "Statistic", "mAP", "n", "Source"],
            rows=dior_rows,
            caption="Sanitized DIOR-R results on DIOR\\_R\\_dota/test using MMRotate DOTAMetric mAP at IoU 0.5. S3 scene adapter is the strongest local row (best single replica 0.6992). S4 pseudo-label stabilization is a closed, negative-to-neutral route: its best mean 0.6973 remains below the S3 best-mean gate 0.6979.",
            label="tab:tgrs_dior_r",
            widths="@{}p{0.25\\linewidth}p{0.18\\linewidth}cp{0.05\\linewidth}p{0.36\\linewidth}@{}",
            starred=False,
            scale_to_width=r"\linewidth",
        ),
        encoding="utf-8",
    )

    dota_rows = [[r.name, r.best_epoch, r.best_map, r.final_epoch, r.final_map, r.source] for r in data["dota2_runs"]]
    write_csv(TAB_DIR / "table_tgrs_dota2_stability.csv", ["Run", "Best epoch", "Best mAP", "Final epoch", "Final mAP", "Source"], dota_rows)
    (TAB_DIR / "table_tgrs_dota2_stability.tex").write_text(
        latex_table(
            headers=["Run", "Best epoch", "Best mAP", "Final epoch", "Final mAP"],
            rows=[row[:5] for row in dota_rows],
            caption="DOTA-v2.0 S2 loss-0 stability on DOTA2\\_1024\\_500/ss\\_val. Seven controlled S2 runs have best-checkpoint mean 0.6206, above S1 0.6177, but final-checkpoint mean 0.6167, below S1.",
            label="tab:tgrs_dota2_stability",
            widths="@{}lcccc@{}",
            starred=False,
            footnote="All rows use the same S1 epoch-12 initialization and are recorded in docs/experiments/20260611_dota2_s2_loss0_replicates_analysis.md.",
        ),
        encoding="utf-8",
    )

    det_rows = []
    for r in data["detector_dota2"]:
        det_rows.append(["DOTA-v2.0", r[0], r[1], r[2], r[3], r[4]])
    for r in data["detector_dior"]:
        det_rows.append(["DIOR-R", r[0], r[1], r[2], r[3], r[4]])
    write_csv(TAB_DIR / "table_tgrs_detector_baselines.csv", ["Dataset", "Detector", "mAP", "AP50", "Checkpoint read", "Source"], det_rows)
    (TAB_DIR / "table_tgrs_detector_baselines.tex").write_text(
        latex_table(
            headers=["Dataset", "Detector", "mAP", "AP50", "Read"],
            rows=[r[:5] for r in det_rows],
            caption="Detector-family baselines used to contextualize the GeoNexus-RSD results. DOTA-v2.0 values use the valid-PNG DOTA2\\_1024\\_500/ss\\_val protocol; DIOR-R values use the sanitized DIOR\\_R\\_dota/test protocol.",
            label="tab:tgrs_detector_baselines",
            widths="@{}p{0.14\\textwidth}p{0.22\\textwidth}ccp{0.47\\textwidth}@{}",
            starred=True,
            scale_to_width=r"\textwidth",
        ),
        encoding="utf-8",
    )

    comparator_rows = [[r.method, r.dataset_note, r.ap50, r.kind, r.source] for r in data["comparators"]]
    write_csv(TAB_DIR / "table_tgrs_sota_comparators.csv", ["Method", "Note", "AP50", "Kind", "Source"], comparator_rows)
    (TAB_DIR / "table_tgrs_sota_comparators.tex").write_text(
        latex_table(
            headers=["Method", "Note", "AP50", "Kind", "Source"],
            rows=[[r.method, r.dataset_note, r.ap50, r.kind.replace("_", " "), short_source(r.source)] for r in data["comparators"]],
            caption="DIOR-R comparator context, not a claimed SOTA table. GeoNexus-RSD S3 exceeds the confirmed OrientedFormer Swin-T rerun and the public Strip R-CNN-S and AOPG rows, but exact split/label/evaluator equivalence has not been proven for the public-paper rows, and Strip R-CNN-S could not be independently re-evaluated because no DIOR-R-trained checkpoint was released.",
            label="tab:tgrs_sota_comparators",
            widths="@{}p{0.20\\textwidth}p{0.30\\textwidth}cp{0.14\\textwidth}p{0.20\\textwidth}@{}",
            starred=True,
            scale_to_width=r"\textwidth",
        ),
        encoding="utf-8",
    )

    source_csv_rows = [[r.dataset, r.stage, r.metric, r.value, r.source] for r in data["main_rows"]]
    source_csv_rows.extend(
        [[f"DOTA2 S2 {r.name}", f"best e{r.best_epoch}; final e{r.final_epoch}", "mAP", f"{r.best_map:.4f}/{r.final_map:.4f}", r.source] for r in data["dota2_runs"]]
    )
    source_csv_rows.extend(
        [[f"DIOR-R S2 {r.name}", f"seed {r.seed}", "mAP", f"{r.best_map:.4f}/{r.final_map:.4f}", r.source] for r in data["dior_s2_runs"]]
    )
    source_csv_rows.extend(
        [[f"DIOR-R S3 {r.name}", f"seed {r.seed}", "mAP", f"{r.best_map:.4f}/{r.final_map:.4f}", r.source] for r in data["dior_s3_runs"]]
    )
    source_csv_rows.extend(
        [[f"DIOR-R S4 {r.name}", f"seed {r.seed}", "mAP", f"{r.best_map:.4f}/{r.final_map:.4f}", r.source] for r in data["dior_s4_runs"]]
    )
    write_csv(TAB_DIR / "table_tgrs_sources.csv", ["Item", "Stage/read", "Metric", "Value", "Source"], source_csv_rows)
    source_tex_rows = [[r.dataset, r.stage, r.metric, r.value, short_source(r.source)] for r in data["main_rows"]]
    source_tex_rows.extend(
        [[f"DOTA2 S2 {r.name}", f"best e{r.best_epoch}; final e{r.final_epoch}", "mAP", f"{r.best_map:.4f}/{r.final_map:.4f}", "20260611 DOTA2 S2 analysis note"] for r in data["dota2_runs"]]
    )
    source_tex_rows.extend(
        [[f"DIOR-R S2 {r.name}", f"seed {r.seed}", "mAP", f"{r.best_map:.4f}/{r.final_map:.4f}", "20260614 DIOR-R S2 JSON; scalar record"] for r in data["dior_s2_runs"]]
    )
    source_tex_rows.extend(
        [[f"DIOR-R S3 {r.name}", f"seed {r.seed}", "mAP", f"{r.best_map:.4f}/{r.final_map:.4f}", "20260615 DIOR-R S3 JSON; scalar record"] for r in data["dior_s3_runs"]]
    )
    source_tex_rows.extend(
        [[f"DIOR-R S4 {r.name}", f"seed {r.seed}", "mAP", f"{r.best_map:.4f}/{r.final_map:.4f}", "20260701 DIOR-R S4 LR5e-6 JSON"] for r in data["dior_s4_runs"]]
    )
    (TAB_DIR / "table_tgrs_sources.tex").write_text(
        latex_table(
            headers=["Item", "Stage/read", "Metric", "Value", "Source"],
            rows=source_tex_rows,
            caption="Provenance for reported GeoNexus-RSD values. Best/final pairs are shown explicitly where checkpoint selection affects the interpretation.",
            label="tab:tgrs_sources",
            widths="@{}p{0.18\\textwidth}p{0.18\\textwidth}p{0.08\\textwidth}p{0.10\\textwidth}p{0.39\\textwidth}@{}",
            starred=True,
            scale_to_width=r"\textwidth",
        ),
        encoding="utf-8",
    )


def write_fragments() -> None:
    fragments = {
        "fragment_fig_tgrs_main_results.tex": figure_fragment(
            Path("figure/geonexus_tgrs_main_results.pdf"),
            "Formal GeoNexus-RSD result summary on DOTA-v2.0 ss_val and sanitized DIOR-R test. Bars report mAP; DOTA-v2.0 S2 separates the seven-run best-checkpoint mean from the final-checkpoint mean, and DIOR-R S3 (best mean) is shown as the strongest local stage.",
            "fig:tgrs_main_results",
        ),
        "fragment_fig_tgrs_dior_progression.tex": figure_fragment(
            Path("figure/geonexus_tgrs_dior_progression.pdf"),
            "DIOR-R stage progression from S0 through S4 on sanitized DIOR_R_dota/test using MMRotate DOTAMetric mAP at IoU 0.5. S3 scene adapter is the strongest local row; S4 pseudo-label stabilization is shown as a closed, negative-to-neutral attempt that did not clear the S3 best-mean gate.",
            "fig:tgrs_dior_progression",
            width=r"\textwidth",
            starred=True,
        ),
        "fragment_fig_tgrs_dota2_stability.tex": figure_fragment(
            Path("figure/geonexus_tgrs_dota2_s2_stability.pdf"),
            "DOTA-v2.0 S2 loss-0 stability on DOTA2_1024_500/ss_val. Each run shows the best checkpoint and final checkpoint; best checkpoints repeatably exceed S1, while the final-checkpoint mean is below S1.",
            "fig:tgrs_dota2_stability",
            width=r"\columnwidth",
            starred=False,
        ),
        "fragment_fig_tgrs_detector_baselines.tex": figure_fragment(
            Path("figure/geonexus_tgrs_detector_baselines.pdf"),
            "Closed-set detector baselines for the DOTA-v2.0 and sanitized DIOR-R protocols. These values contextualize the RoI Transformer route used for GeoNexus-RSD and are not pseudo-label or efficiency claims.",
            "fig:tgrs_detector_baselines",
        ),
        "fragment_fig_tgrs_result_synthesis.tex": figure_fragment(
            Path("figure/geonexus_tgrs_result_synthesis.pdf"),
            "Result synthesis for the July 9 evidence state. DIOR-R uses sanitized DIOR_R_dota/test with MMRotate DOTAMetric mAP at IoU 0.5 and shows the completed S0 through S4 best-checkpoint progression. DOTA-v2.0 uses DOTA2_1024_500/ss_val and reports S0, S1, and seven-run S2 loss-0 best/final means. No pseudo-label superiority, efficiency, or qualitative detection claims are included; S4 is shown as a closed stabilization attempt.",
            "fig:tgrs_result_synthesis",
        ),
        "fragment_fig_tgrs_sota_comparators.tex": figure_fragment(
            Path("figure/geonexus_tgrs_sota_comparators.pdf"),
            "DIOR-R comparator context. GeoNexus-RSD S3 is shown against our own confirmed reproduction of the released OrientedFormer Swin-T checkpoint and against public paper rows for OrientedFormer variants, Strip R-CNN-S, and AOPG. This is comparator context, not a SOTA claim: public-paper rows are not protocol-verified identical, and Strip R-CNN-S could not be independently re-evaluated because no DIOR-R-trained checkpoint was released.",
            "fig:tgrs_sota_comparators",
            width=r"\textwidth",
            starred=True,
        ),
    }
    for name, text in fragments.items():
        (TAB_DIR / name).write_text(text, encoding="utf-8")


def write_readme(data: dict) -> None:
    text = f"""# TGRS Result Assets - 2026-07-09

Generated by `scripts/make_tgrs_result_assets_20260709.py`, extending the
2026-06-14 package with DIOR-R S3/S4 and a SOTA comparator table/figure.

Main checked values:

- DOTA-v2.0: S0 `{data['s0_dota2']:.4f}`, S1 `{data['s1_dota2']:.4f}`, S2 best mean `{mean(r.best_map for r in data['dota2_runs']):.4f}`, S2 final mean `{mean(r.final_map for r in data['dota2_runs']):.4f}`.
- DIOR-R: S0 `{data['s0_dior']:.4f}`, S1 final mean `{data['s1_dior_mean']:.4f}`, S2 best mean `{data['s2_dior_best_mean']:.4f}`, S2 final mean `{data['s2_dior_final_mean']:.4f}`, S3 best mean `{data['s3_dior_best_mean']:.4f}`, S3 final mean `{data['s3_dior_final_mean']:.4f}`, S4 best mean `{data['s4_dior_best_mean']:.4f}` (closed, below S3 gate `{data['s3_gate']:.4f}`), S4 final mean `{data['s4_dior_final_mean']:.4f}`.
- Comparators: OrientedFormer Swin-T our confirmed rerun `0.6883`; public paper rows OrientedFormer Swin-T `0.6884`, OrientedFormer R50 `0.6728`, Strip R-CNN-S `0.6870` (checkpoint unavailable for independent eval), AOPG `0.6441`.

DOTA v1.5 is intentionally excluded from the generated main result assets.
No pseudo-label superiority, efficiency, or qualitative detection result is
fabricated; S4 is reported as a closed, negative-to-neutral stabilization
attempt. PKINet, PKINet-v2, and LSKNet DIOR-R rows are not included because no
verified public AP50 value was available for them at generation time.
"""
    (TAB_DIR / "README_tgrs_result_assets_20260709.md").write_text(text, encoding="utf-8")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    data = collect_data()
    render_main_results(data)
    render_dior_progression(data)
    render_dota2_stability(data)
    render_result_synthesis(data)
    render_detector_baselines(data)
    render_comparators(data)
    write_tables(data)
    write_fragments()
    write_readme(data)
    print(f"Wrote TGRS result assets to {TGRS_DIR}")


if __name__ == "__main__":
    main()
