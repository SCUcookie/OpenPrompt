"""Generate paper-only GeoNexus-RSD visual assets for the 2026-06-13 draft.

Outputs live under artifacts/paper_assets_20260613.  The script intentionally
does not create PPT assets.  Numeric labels are asserted against local
experiment records before figures and tables are written.
"""

from __future__ import annotations

import csv
import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "paper_assets_20260613"
DOCS = ROOT / "docs" / "experiments"

INK = "#1f2933"
MUTED = "#667085"
LINE = "#cfd8e3"
BLUE = "#2563eb"
TEAL = "#0f766e"
GREEN = "#15803d"
AMBER = "#b7791f"
RED = "#b91c1c"
PURPLE = "#6d28d9"
GRAY = "#6b7280"


def read_text(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def read_json(rel: str) -> dict:
    return json.loads(read_text(rel))


def assert_close(value: float, expected: float, label: str, tol: float = 5e-5) -> None:
    if not math.isclose(float(value), expected, rel_tol=0.0, abs_tol=tol):
        raise AssertionError(f"{label}: expected {expected}, found {value}")


def require_substring(rel: str, needle: str) -> None:
    text = read_text(rel)
    if needle not in text:
        raise AssertionError(f"{rel} does not contain required evidence: {needle}")


def load_evidence() -> dict:
    dota_s0 = read_json("docs/experiments/20260603_s0_dota2_roi_trans_validpng_metrics.json")
    assert_close(dota_s0["metrics"]["dota/mAP"], 0.6088, "DOTA2 S0 mAP")
    assert_close(dota_s0["metrics"]["dota/AP50"], 0.6090, "DOTA2 S0 AP50")

    require_substring("docs/experiments/20260608_dota2_s1_complete_and_s2_launch.md", "`0.6177 / 0.6180`")
    s2_record = "docs/experiments/20260611_dota2_s2_loss0_replicates_analysis.md"
    for needle in [
        "All 7 runs: best mean `0.620606`, `+0.002906` over S1",
        "final mean `0.616655`, `-0.001045` below S1",
    ]:
        require_substring(s2_record, needle)

    require_substring(
        "docs/experiments/20260613_dior_r_s0_sanitized_long_interim.md",
        "epoch 48: `dota/mAP=0.6531`, `dota/AP50=0.6530`",
    )

    dior_s1 = read_json("docs/experiments/20260613_dior_r_geonexus_s1_s0e52_replicas_metrics.json")
    rep0 = dior_s1["replicas"][0]["metrics"][-1]
    rep1 = dior_s1["replicas"][1]["metrics"][-1]
    assert_close(rep0["dota_mAP"], 0.6750815511, "DIOR-R S1 rep0 mAP")
    assert_close(rep0["dota_AP50"], 0.675, "DIOR-R S1 rep0 AP50")
    assert_close(rep1["dota_mAP"], 0.6689543724, "DIOR-R S1 rep1 mAP")
    assert_close(rep1["dota_AP50"], 0.669, "DIOR-R S1 rep1 AP50")

    require_substring(
        "docs/experiments/20260613_dior_r_geonexus_s2_hierarchy_replicas_launch.md",
        "Startup acceptance: reached `Epoch(train) [1][200/5862]`",
    )

    return {
        "dota_s0": (0.6088, 0.6090),
        "dota_s1": (0.6177, 0.6180),
        "dota_s2_best_mean": (0.620606, None),
        "dota_s2_final_mean": (0.616655, None),
        "dior_s0": (0.6531, 0.6530),
        "dior_s1_rep0": (0.6750815511, 0.675),
        "dior_s1_rep1": (0.6689543724, 0.669),
        "dota_s0_per_class": dota_s0["per_class"],
    }


def ensure_out() -> None:
    OUT.mkdir(parents=True, exist_ok=True)


def latex_escape(value: object) -> str:
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
    return "".join(repl.get(ch, ch) for ch in str(value))


def write_csv(path: Path, headers: list[str], rows: list[list[object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


def write_table(path: Path, caption: str, label: str, headers: list[str], rows: list[list[object]], align: str) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("\\begin{table}[t]\n\\centering\n\\small\n")
        f.write(f"\\caption{{{latex_escape(caption)}}}\n")
        f.write(f"\\label{{{label}}}\n")
        f.write(f"\\begin{{tabular}}{{{align}}}\n\\toprule\n")
        f.write(" & ".join(latex_escape(h) for h in headers) + " \\\\\n\\midrule\n")
        for row in rows:
            f.write(" & ".join(latex_escape(v) for v in row) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def set_axes(ax) -> None:
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="y", color="#edf2f7", lw=0.8)
    ax.tick_params(axis="y", colors=MUTED, labelsize=7)
    ax.tick_params(axis="x", colors=INK, labelsize=7)


def save_fig(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def render_dota2_stability(e: dict) -> None:
    labels = ["S0\nRoITrans", "S1\nRemoteCLIP", "S2\nbest mean", "S2\nfinal mean"]
    values = [e["dota_s0"][0], e["dota_s1"][0], e["dota_s2_best_mean"][0], e["dota_s2_final_mean"][0]]
    colors = [GRAY, BLUE, GREEN, RED]
    fig, ax = plt.subplots(figsize=(6.9, 2.55), dpi=220)
    bars = ax.bar(labels, values, color=colors, width=0.58)
    ax.axhline(e["dota_s1"][0], color=BLUE, lw=1.0, ls="--", label="S1 comparator")
    ax.set_ylim(0.604, 0.623)
    ax.set_ylabel("mAP", fontsize=8, color=MUTED)
    ax.set_title("DOTA2 S2 early-checkpoint signal vs. final instability", loc="left", fontsize=10, weight="bold", color=INK)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.00045, f"{value:.4f}", ha="center", va="bottom", fontsize=7, color=INK)
    ax.text(2.5, 0.6052, "All seven S2 loss-0 runs: best mean +0.002906 over S1; final mean -0.001045 below S1.", ha="center", fontsize=7, color=RED)
    ax.legend(frameon=False, loc="upper left", fontsize=7)
    set_axes(ax)
    save_fig(fig, "fig_dota2_s2_stability")


def render_dior_transfer(e: dict) -> None:
    labels = ["S0\nRoITrans", "S1 rep0\nRemoteCLIP", "S1 rep1\nRemoteCLIP", "S2 reps\nlaunched"]
    values = [e["dior_s0"][0], e["dior_s1_rep0"][0], e["dior_s1_rep1"][0], np.nan]
    fig, ax = plt.subplots(figsize=(6.9, 2.55), dpi=220)
    xs = np.arange(len(labels))
    ax.bar(xs[:3], values[:3], color=[GRAY, BLUE, BLUE], width=0.55)
    ax.scatter([3], [0.652], s=90, facecolors="none", edgecolors=AMBER, linewidths=1.5)
    ax.text(3, 0.654, "startup clean\nmetrics pending", ha="center", va="bottom", fontsize=7, color=AMBER)
    for x, value in zip(xs[:3], values[:3]):
        ax.text(x, value + 0.0011, f"{value:.4f}", ha="center", va="bottom", fontsize=7, color=INK)
    ax.set_xticks(xs, labels)
    ax.set_ylim(0.648, 0.679)
    ax.set_ylabel("mAP", fontsize=8, color=MUTED)
    ax.set_title("DIOR-R sanitized validation: S1 measured, S2 pending", loc="left", fontsize=10, weight="bold", color=INK)
    set_axes(ax)
    save_fig(fig, "fig_dior_r_transfer")


def render_route_status() -> None:
    rows = [
        ("DOTA2 S0/S1", "measured", GREEN, "S0 0.6088/0.6090; S1 0.6177/0.6180"),
        ("DOTA2 S2", "unstable", AMBER, "best mean 0.620606; final mean 0.616655"),
        ("DIOR-R S0/S1", "measured", GREEN, "S0 0.6531/0.6530; S1 reps 0.6751 and 0.6690"),
        ("DIOR-R S2", "running", AMBER, "three hierarchy replicas launched; metrics pending"),
        ("DOTA v1.5", "archive", GRAY, "diagnostic only after pivot"),
        ("S3/S4/pseudo/FAIR1M", "paused", RED, "not claimed until gates close"),
    ]
    fig, ax = plt.subplots(figsize=(6.9, 3.0), dpi=220)
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(rows) + 0.4)
    ax.text(0, len(rows) + 0.12, "Experiment route status on 2026-06-13", fontsize=10, weight="bold", color=INK)
    for idx, (name, status, color, note) in enumerate(rows):
        y = len(rows) - idx - 0.45
        ax.plot([0.3, 9.7], [y, y], color="#edf2f7", lw=0.8)
        ax.scatter([0.55], [y], s=75, color=color)
        ax.text(0.82, y + 0.08, name, ha="left", va="center", fontsize=8.2, weight="bold", color=INK)
        ax.text(2.95, y + 0.08, status, ha="left", va="center", fontsize=7.3, color=color)
        ax.text(4.15, y + 0.08, note, ha="left", va="center", fontsize=7.1, color=MUTED)
    save_fig(fig, "fig_route_status")


def render_class_heatmap(e: dict) -> None:
    rows = sorted(e["dota_s0_per_class"], key=lambda r: r["ap"], reverse=True)
    classes = [r["class"].replace("-", "\n") for r in rows]
    data = np.array([[r["ap"], r["recall"]] for r in rows])
    fig, ax = plt.subplots(figsize=(6.9, 3.2), dpi=220)
    im = ax.imshow(data.T, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_yticks([0, 1], ["AP", "Recall"], fontsize=7, color=INK)
    ax.set_xticks(np.arange(len(classes)), classes, rotation=65, ha="right", fontsize=5.8)
    ax.set_title("DOTA2 S0 class-level diagnostic evidence", loc="left", fontsize=10, weight="bold", color=INK)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(i, j, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=4.8, color="white" if data[i, j] < 0.55 else "black")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.012)
    cbar.ax.tick_params(labelsize=6)
    save_fig(fig, "fig_dota2_class_heatmap")


def render_tikz_framework() -> None:
    tex = r"""\documentclass[tikz,border=3pt]{standalone}
\usepackage{amsmath}
\usetikzlibrary{arrows.meta,positioning,fit,calc,backgrounds}
\definecolor{ink}{HTML}{1F2933}
\definecolor{muted}{HTML}{667085}
\definecolor{blue}{HTML}{2563EB}
\definecolor{teal}{HTML}{0F766E}
\definecolor{green}{HTML}{15803D}
\definecolor{amber}{HTML}{B7791F}
\definecolor{red}{HTML}{B91C1C}
\definecolor{gray}{HTML}{6B7280}
\tikzset{
  block/.style={draw=#1, rounded corners=1.8pt, line width=0.55pt, fill=white, minimum width=26mm, minimum height=8.5mm, align=center, font=\scriptsize},
  tag/.style={draw=#1, rounded corners=1.2pt, line width=0.45pt, fill=#1!7, inner xsep=2pt, inner ysep=1pt, font=\tiny},
  lane/.style={draw=#1!45, fill=#1!4, rounded corners=2pt, inner sep=3pt},
  arr/.style={-{Latex[length=1.7mm]}, line width=0.55pt, draw=ink}
}
\begin{document}
\begin{tikzpicture}[font=\sffamily, text=ink, node distance=7mm]
  \node[block=teal] (tile) {valid tile\\DOTA2 / DIOR-R};
  \node[block=blue, right=of tile] (fpn) {detector trunk\\R50--FPN};
  \node[block=blue, right=of fpn] (roi) {RoI Transformer\\rotated boxes};
  \node[block=blue, right=of roi] (scores) {detector scores\\OBB outputs};
  \node[tag=green, above=0.8mm of fpn] {implemented};
  \node[tag=green, above=0.8mm of roi] {implemented};
  \draw[arr] (tile) -- (fpn);
  \draw[arr] (fpn) -- (roi);
  \draw[arr] (roi) -- (scores);
  \begin{scope}[on background layer]
    \node[lane=blue, fit=(tile)(fpn)(roi)(scores)] (laneA) {};
  \end{scope}
  \node[anchor=west, font=\scriptsize\bfseries, text=blue] at ($(laneA.north west)+(1mm,2mm)$) {Detector geometry path};

  \node[block=amber, below=12mm of fpn] (tax) {taxonomy + aliases\\prompt templates};
  \node[block=amber, right=of tax] (clip) {RemoteCLIP text\\embeddings};
  \node[block=amber, right=of clip] (s1) {S1 prompt branch\\measured};
  \node[block=amber, right=of s1] (s2) {S2 hierarchy\\early signal};
  \node[tag=green, above=0.8mm of s1] {current};
  \node[tag=amber, above=0.8mm of s2] {unstable final};
  \draw[arr] (tax) -- (clip);
  \draw[arr] (clip) -- (s1);
  \draw[arr] (s1) -- (s2);
  \draw[arr, draw=amber] (s1.north) to[out=80,in=-100] (scores.south);
  \begin{scope}[on background layer]
    \node[lane=amber, fit=(tax)(clip)(s1)(s2)] (laneB) {};
  \end{scope}
  \node[anchor=west, font=\scriptsize\bfseries, text=amber] at ($(laneB.north west)+(1mm,2mm)$) {GeoNexus semantic prompt path};

  \node[block=gray, below=12mm of tax] (ctx) {scene/context\\adapter};
  \node[block=gray, right=of ctx] (gate) {gate / route\\selector};
  \node[block=gray, right=of gate] (pseudo) {pseudo-label\\purification};
  \node[block=red, right=of pseudo] (fair) {FAIR1M / S3--S4\\later gate};
  \node[tag=red, below=0.8mm of ctx] {paused};
  \node[tag=red, above=0.8mm of pseudo] {planned};
  \draw[arr, draw=gray] (ctx) -- (gate);
  \draw[arr, draw=gray] (gate) -- (pseudo);
  \draw[arr, draw=gray] (pseudo) -- (fair);
  \draw[arr, draw=gray, dashed] (s2.south) to[out=-80,in=100] (gate.north);
  \begin{scope}[on background layer]
    \node[lane=gray, fit=(ctx)(gate)(pseudo)(fair)] (laneC) {};
  \end{scope}
  \node[anchor=west, font=\scriptsize\bfseries, text=gray] at ($(laneC.north west)+(1mm,2mm)$) {Gated future / pseudo-label path};

  \node[anchor=west, font=\tiny, text=muted] at ($(laneC.south west)+(0,-4mm)$)
  {Current claims: DOTA2 S1 measured; DOTA2 S2 early-checkpoint candidate only; DIOR-R S1 measured; DIOR-R S2 pending; S3/S4 and pseudo-labeling paused.};
\end{tikzpicture}
\end{document}
"""
    tex_path = OUT / "fig_method_framework_tikz.tex"
    tex_path.write_text(tex, encoding="utf-8")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        tmp_tex = tmp_dir / tex_path.name
        tmp_tex.write_text(tex, encoding="utf-8")
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tmp_tex.name],
            cwd=tmp_dir,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        shutil.copyfile(tmp_dir / "fig_method_framework_tikz.pdf", OUT / "fig_method_framework_tikz.pdf")


def render_tables(e: dict) -> None:
    dota_rows = [
        ["DOTA2 S0 RoI Transformer", "complete", "0.6088", "0.6090", "matched valid-PNG detector baseline"],
        ["DOTA2 S1 RemoteCLIP", "complete", "0.6177", "0.6180", "paper-facing positive DOTA2 result"],
        ["DOTA2 S2 loss-0 best mean", "candidate", "0.620606", "-", "all-seven best-checkpoint mean; early only"],
        ["DOTA2 S2 loss-0 final mean", "unstable", "0.616655", "-", "all-seven final mean; below S1"],
    ]
    headers = ["Run", "Status", "mAP", "AP50", "Use"]
    write_csv(OUT / "table_dota2_current.csv", headers, dota_rows)
    write_table(OUT / "table_dota2_current.tex", "DOTA2 evidence after the June 13 S2 stabilization analysis.", "tab:dota2_current", headers, dota_rows, "lllrl")

    dior_rows = [
        ["DIOR-R S0 RoI Transformer", "sanitized S0", "0.6531", "0.6530", "current S0 leader"],
        ["DIOR-R S1 rep0", "complete", "0.6751", "0.675", "stronger S1 replica; source for S2"],
        ["DIOR-R S1 rep1", "complete", "0.6690", "0.669", "stability replica"],
        ["DIOR-R S2 hierarchy reps", "launched", "pending", "pending", "clean startup; no metric claim"],
    ]
    write_csv(OUT / "table_dior_r_current.csv", headers, dior_rows)
    write_table(OUT / "table_dior_r_current.tex", "DIOR-R sanitized route evidence and pending S2 status.", "tab:dior_r_current", headers, dior_rows, "lllrl")

    route_rows = [
        ["DOTA2", "primary benchmark", "S0/S1 measured; S2 best-checkpoint candidate"],
        ["DIOR-R", "required validation", "S0/S1 measured; S2 metrics pending"],
        ["DOTA v1.5", "archive only", "diagnostic history, not headline evidence"],
        ["S3/S4", "paused", "resume only after S2 and DIOR-R gates are stable"],
        ["Pseudo-label / FAIR1M", "planned", "no paper-facing metric claim yet"],
    ]
    route_headers = ["Track", "Role", "Status"]
    write_csv(OUT / "table_route_status.csv", route_headers, route_rows)
    write_table(OUT / "table_route_status.tex", "Route status used to prevent overclaiming active or paused modules.", "tab:route_status", route_headers, route_rows, "lll")

    source_rows = [
        ["DOTA2 S0", "0.6088 / 0.6090", "docs/experiments/20260603_s0_dota2_roi_trans_validpng_metrics.json"],
        ["DOTA2 S1", "0.6177 / 0.6180", "docs/experiments/20260608_dota2_s1_complete_and_s2_launch.md"],
        ["DOTA2 S2 aggregate", "best 0.620606; final 0.616655", "docs/experiments/20260611_dota2_s2_loss0_replicates_analysis.md"],
        ["DIOR-R S0", "0.6531 / 0.6530", "docs/experiments/20260613_dior_r_s0_sanitized_long_interim.md"],
        ["DIOR-R S1 replicas", "0.6751 / 0.675; 0.6690 / 0.669", "docs/experiments/20260613_dior_r_geonexus_s1_s0e52_replicas_metrics.json"],
        ["DIOR-R S2", "startup clean; metrics pending", "docs/experiments/20260613_dior_r_geonexus_s2_hierarchy_replicas_launch.md"],
    ]
    source_headers = ["Claim", "Value", "Local source"]
    write_csv(OUT / "table_artifact_sources.csv", source_headers, source_rows)
    write_table(OUT / "table_artifact_sources.tex", "Source map for the June 13 generated paper assets.", "tab:artifact_sources", source_headers, source_rows, "lll")

    asset_rows = [
        ["fig_method_framework_tikz", "TikZ/PDF", "method framework with current vs paused modules"],
        ["fig_dota2_s2_stability", "PDF/SVG", "DOTA2 S0/S1/S2 stability plot"],
        ["fig_dior_r_transfer", "PDF/SVG", "DIOR-R S0/S1 plus pending S2"],
        ["fig_route_status", "PDF/SVG", "route gate/status diagram"],
        ["fig_dota2_class_heatmap", "PDF/SVG", "appendix-style class diagnostic"],
        ["table_dota2_current", "TEX/CSV", "DOTA2 metrics table"],
        ["table_dior_r_current", "TEX/CSV", "DIOR-R metrics table"],
        ["table_route_status", "TEX/CSV", "route gate table"],
    ]
    asset_headers = ["Asset", "Format", "Use"]
    write_csv(OUT / "table_visual_asset_index.csv", asset_headers, asset_rows)
    write_table(OUT / "table_visual_asset_index.tex", "Paper asset index for the June 13 visualization refresh.", "tab:visual_asset_index", asset_headers, asset_rows, "lll")


def validate_outputs() -> None:
    required = [
        "fig_method_framework_tikz.tex",
        "fig_method_framework_tikz.pdf",
        "fig_dota2_s2_stability.pdf",
        "fig_dior_r_transfer.pdf",
        "fig_route_status.pdf",
        "fig_dota2_class_heatmap.pdf",
        "table_dota2_current.tex",
        "table_dior_r_current.tex",
        "table_route_status.tex",
        "table_visual_asset_index.tex",
    ]
    missing = [name for name in required if not (OUT / name).exists()]
    if missing:
        raise AssertionError(f"Missing required assets: {missing}")
    for path in OUT.glob("*.pdf"):
        if path.stat().st_size < 1000:
            raise AssertionError(f"{path} is unexpectedly small")
    for path in OUT.glob("*.tex"):
        text = path.read_text(encoding="utf-8")
        if path.name.startswith("table_") and "\\toprule" not in text:
            raise AssertionError(f"{path} is not a booktabs table")
    readme = OUT / "README.md"
    files = sorted(p.name for p in OUT.iterdir() if p.is_file() and p.name != "README.md")
    readme.write_text(
        "\n".join(
            [
                "# GeoNexus-RSD Paper Assets - 2026-06-13",
                "",
                "Generated by `scripts/make_paper_visuals_20260613.py`.",
                "",
                "This directory is paper-only: TikZ/LaTeX source, PDF/SVG figures, CSV tables, and booktabs LaTeX fragments. No PPT assets are emitted.",
                "",
                "Files:",
                *[f"- `{name}`" for name in files],
                "",
            ]
        ),
        encoding="utf-8",
    )


def render_all() -> None:
    ensure_out()
    evidence = load_evidence()
    render_tikz_framework()
    render_dota2_stability(evidence)
    render_dior_transfer(evidence)
    render_route_status()
    render_class_heatmap(evidence)
    render_tables(evidence)
    validate_outputs()


if __name__ == "__main__":
    if shutil.which("pdflatex") is None:
        raise SystemExit("pdflatex is required to compile fig_method_framework_tikz.tex")
    render_all()
