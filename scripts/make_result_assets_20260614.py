"""Write a single CSV audit record for experiment results and attempts.

The result package intentionally contains only:
artifacts/result_assets_20260614/all_experiment_results_20260614.csv
"""

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "docs" / "experiments"
OUT_DIR = ROOT / "artifacts" / "result_assets_20260614"
OUT_CSV = OUT_DIR / "all_experiment_results_20260614.csv"
OPENRSD = Path("/data5/2025/ldh/OpenRSD")

CSV_FIELDS = [
    "date",
    "dataset_group",
    "dataset_split",
    "protocol",
    "experiment",
    "stage",
    "detector",
    "status",
    "outcome",
    "best_epoch",
    "best_map",
    "best_ap50",
    "final_epoch",
    "final_map",
    "final_ap50",
    "n_replicas",
    "source_file",
    "work_dir",
    "notes",
]

ALLOWED_STATUS = {"completed", "failed", "invalid", "blocked", "launch_only", "diagnostic"}


@dataclass(frozen=True)
class MetricPoint:
    epoch: int
    map: float
    ap50: float | None


@dataclass(frozen=True)
class ResultRecord:
    date: str
    dataset_group: str
    dataset_split: str
    protocol: str
    experiment: str
    stage: str
    detector: str
    status: str
    outcome: str
    best_epoch: int | None = None
    best_map: float | None = None
    best_ap50: float | None = None
    final_epoch: int | None = None
    final_map: float | None = None
    final_ap50: float | None = None
    n_replicas: int | None = None
    source_file: str = ""
    work_dir: str = ""
    notes: str = ""


def rel(path: str) -> Path:
    return (ROOT / path).resolve()


def source(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except (ValueError, OSError):
        return str(path)


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
                step = row.get("step", row.get("epoch"))
                points.append(
                    MetricPoint(
                        epoch=int(step),
                        map=float(row["dota/mAP"]),
                        ap50=float(row["dota/AP50"]) if "dota/AP50" in row else None,
                    )
                )
    if not points:
        raise ValueError(f"No validation metric points in {path}")
    return points


def all_points(paths: Iterable[Path]) -> list[MetricPoint]:
    points: list[MetricPoint] = []
    for path in paths:
        points.extend(load_scalar_points(path))
    return sorted(points, key=lambda p: p.epoch)


def summarize_points(points: list[MetricPoint]) -> tuple[MetricPoint, MetricPoint]:
    return max(points, key=lambda p: p.map), points[-1]


def parse_metric_pair_from_md(path: Path, label: str) -> tuple[float, float]:
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(
        rf"{re.escape(label)}.*?dota/mAP=([0-9.]+).*?dota/AP50=([0-9.]+)",
        re.DOTALL,
    )
    match = pattern.search(text)
    if not match:
        raise ValueError(f"Could not parse {label!r} from {path}")
    return float(match.group(1)), float(match.group(2))


def rounded(x: float | None) -> float | None:
    return None if x is None else round(float(x), 4)


def assert_close(name: str, actual: float, expected: float) -> None:
    if rounded(actual) != expected:
        raise AssertionError(f"{name}: got {actual:.8f}, expected rounded {expected:.4f}")


def first_value(data: dict, keys: Iterable[str]) -> object | None:
    for key in keys:
        if key in data and data[key] not in (None, ""):
            return data[key]
    return None


def metric_value(data: dict, keys: Iterable[str]) -> float | None:
    value = first_value(data, keys)
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def int_value(data: dict, keys: Iterable[str]) -> int | None:
    value = first_value(data, keys)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def normalize_status(raw: object, text: str = "") -> tuple[str, str]:
    raw_text = str(raw or "").lower()
    hay = f"{raw_text} {text}".lower()
    if "diagnostic" in raw_text:
        return "diagnostic", "not_paper_evidence"
    if "archive" in raw_text or "superseded" in raw_text:
        return "completed", "archive_only"
    if "negative-to-neutral" in hay or "negative_to_neutral" in hay:
        return "completed", "negative_to_neutral"
    if re.search(r"\bneutral\b", hay):
        return "completed", "neutral"
    if "completed" in raw_text or "complete" in raw_text:
        return "completed", "positive"
    if "failed" in raw_text or "failure" in raw_text:
        return "failed", "not_paper_evidence"
    clean_failure_scan = (
        "no matches" in hay
        or "failure scan result: clean" in hay
        or '"result": "clean"' in hay
        or "result: clean" in hay
    )
    if clean_failure_scan and (
        "completed" in hay
        or "complete" in hay
        or "no active" in hay
        or "cleanly" in hay
    ):
        return "completed", "positive"
    if "cuda oom" in hay or "out-of-memory" in hay or "out of memory" in hay:
        return "failed", "oom"
    if re.search(r"\bnan\b|\binf\b", hay):
        return "invalid", "nan_loss"
    if "zero validation" in hay or "near-zero" in hay or "near zero" in hay:
        return "invalid", "zero_validation"
    if "blocked" in hay or "do not launch" in hay or "pause" in hay:
        return "blocked", "active_then_stopped"
    if "diagnostic" in hay or "smoke" in hay or "not paper evidence" in hay:
        return "diagnostic", "not_paper_evidence"
    if "archive" in hay or "appendix" in hay or "superseded" in hay:
        return "completed", "archive_only"
    if "completed" in hay or "complete" in hay:
        return "completed", "positive"
    if "launch" in hay or "launched" in hay or "active" in hay:
        return "launch_only", "active_then_stopped"
    if "failed" in hay or "failure" in hay or "traceback" in hay:
        return "failed", "not_paper_evidence"
    return "diagnostic", "not_paper_evidence"


def infer_stage(text: str) -> str:
    lower = text.lower()
    stage_hits: list[tuple[int, str]] = []
    for stage in ("s0", "s1", "s2", "s3", "s4"):
        match = re.search(rf"\b{stage}\b|_{stage}_|-{stage}-", lower)
        if match:
            stage_hits.append((match.start(), stage.upper()))
    if stage_hits:
        return min(stage_hits)[1]
    if "detector" in lower or "baseline" in lower:
        return "S0 detector"
    return ""


def infer_dataset_group(text: str) -> str:
    lower = text.lower()
    if "dior" in lower:
        return "DIOR-R"
    if "dota2" in lower or "dota2_1024_500" in lower:
        return "DOTA2"
    if "dota v1.5" in lower or "dota15" in lower or "dota_v15" in lower:
        return "DOTA v1.5"
    return ""


def infer_dataset_split(group: str, text: str) -> str:
    lower = text.lower()
    if "dior" in group.lower():
        return "DIOR_R_dota/test"
    if "dota2" in group.lower():
        return "DOTA2_1024_500/ss_val"
    if "v1.5" in group.lower():
        return "DOTA v1.5 reduced tiled split/val"
    if "test" in lower:
        return "test"
    if "val" in lower:
        return "val"
    return ""


def extract_date(path: Path, data: dict | None = None) -> str:
    if data:
        value = data.get("date") or data.get("snapshot_time") or data.get("launched_at")
        if value:
            match = re.search(r"20\d{2}-\d{2}-\d{2}", str(value))
            if match:
                return match.group(0)
    match = re.match(r"(\d{4})(\d{2})(\d{2})", path.name)
    if match:
        return "-".join(match.groups())
    return ""


def extract_work_dir(data: object, fallback_text: str = "") -> str:
    if isinstance(data, dict):
        direct = data.get("work_dir") or data.get("source_output_dir") or data.get("preserved_output_dir")
        if direct:
            return str(direct)
        for value in data.values():
            found = extract_work_dir(value)
            if found:
                return found
    elif isinstance(data, list):
        for item in data:
            found = extract_work_dir(item)
            if found:
                return found
    match = re.search(r"/data5/2025/ldh/OpenRSD/work_dirs/[^\s`|,)]+", fallback_text)
    return match.group(0) if match else ""


def notes_from(value: object, limit: int = 220) -> str:
    if isinstance(value, list):
        text = "; ".join(str(v) for v in value[:4])
    elif isinstance(value, dict):
        text = "; ".join(f"{k}: {v}" for k, v in list(value.items())[:4])
    elif value is None:
        text = ""
    else:
        text = str(value)
    text = " ".join(text.split())
    return text[:limit]


def record_key(record: ResultRecord) -> tuple[str, str, str, str, str]:
    return (
        record.source_file,
        record.work_dir,
        record.experiment,
        record.stage,
        str(record.final_epoch or record.best_epoch or ""),
    )


def dedupe(records: list[ResultRecord]) -> list[ResultRecord]:
    seen: set[tuple[str, str, str, str, str]] = set()
    out: list[ResultRecord] = []
    for record in records:
        key = record_key(record)
        if key in seen:
            continue
        seen.add(key)
        out.append(record)
    return out


def add(
    records: list[ResultRecord],
    *,
    date: str,
    dataset_group: str,
    dataset_split: str,
    protocol: str,
    experiment: str,
    stage: str,
    detector: str = "",
    status: str,
    outcome: str,
    best_epoch: int | None = None,
    best_map: float | None = None,
    best_ap50: float | None = None,
    final_epoch: int | None = None,
    final_map: float | None = None,
    final_ap50: float | None = None,
    n_replicas: int | None = None,
    source_file: str = "",
    work_dir: str = "",
    notes: str = "",
) -> None:
    if status not in ALLOWED_STATUS:
        raise ValueError(f"Unsupported status {status!r} for {experiment}")
    records.append(
        ResultRecord(
            date=date,
            dataset_group=dataset_group,
            dataset_split=dataset_split,
            protocol=protocol,
            experiment=experiment,
            stage=stage,
            detector=detector,
            status=status,
            outcome=outcome,
            best_epoch=best_epoch,
            best_map=best_map,
            best_ap50=best_ap50,
            final_epoch=final_epoch,
            final_map=final_map,
            final_ap50=final_ap50,
            n_replicas=n_replicas,
            source_file=source_file,
            work_dir=work_dir,
            notes=notes,
        )
    )


def add_point_record(
    records: list[ResultRecord],
    *,
    date: str,
    dataset_group: str,
    dataset_split: str,
    protocol: str,
    experiment: str,
    stage: str,
    detector: str,
    source_file: Path | str,
    work_dir: str,
    best: MetricPoint,
    final: MetricPoint,
    notes: str,
    status: str = "completed",
    outcome: str = "positive",
) -> None:
    add(
        records,
        date=date,
        dataset_group=dataset_group,
        dataset_split=dataset_split,
        protocol=protocol,
        experiment=experiment,
        stage=stage,
        detector=detector,
        status=status,
        outcome=outcome,
        best_epoch=best.epoch,
        best_map=best.map,
        best_ap50=best.ap50,
        final_epoch=final.epoch,
        final_map=final.map,
        final_ap50=final.ap50,
        source_file=source(source_file),
        work_dir=work_dir,
        notes=notes,
    )


def collect_formal_records(records: list[ResultRecord]) -> None:
    s0_dota2_path = rel("docs/experiments/20260603_s0_dota2_roi_trans_validpng_metrics.json")
    s0_dota2 = load_json(s0_dota2_path)
    s0_dota2_map = float(s0_dota2["metrics"]["dota/mAP"])
    s0_dota2_ap50 = float(s0_dota2["metrics"]["dota/AP50"])

    detector_md = rel("docs/experiments/20260605_dota2_baseline_status.md")
    followup_md = rel("docs/experiments/20260607_current_status_and_next_launch.md")
    detector_dota2 = [
        ("RoI Transformer", s0_dota2_map, s0_dota2_ap50, s0_dota2_path, s0_dota2.get("work_dir", ""), "closed-set S0"),
        ("Oriented R-CNN", *parse_metric_pair_from_md(detector_md, "Oriented R-CNN R50 bs1"), detector_md, "", "completed epoch 12"),
        ("S2ANet", *parse_metric_pair_from_md(detector_md, "S2ANet bs1"), detector_md, "", "completed epoch 12"),
        ("R3Det-KFIoU", *parse_metric_pair_from_md(followup_md, "DOTA2 R3Det-KFIoU valid-PNG bs1 completed epoch 12"), followup_md, "", "completed epoch 12"),
        ("RTMDet-M", *parse_metric_pair_from_md(detector_md, "RTMDet-M bs1"), detector_md, "", "completed epoch 12"),
        ("RTMDet-L", *parse_metric_pair_from_md(followup_md, "DOTA2 RTMDet-L valid-PNG bs1 completed epoch 12"), followup_md, "", "completed epoch 12"),
    ]
    for detector, m, ap50, src, work_dir, note in detector_dota2:
        add(
            records,
            date="2026-06-05",
            dataset_group="DOTA2 formal",
            dataset_split="DOTA2_1024_500/ss_val",
            protocol="valid-PNG formal",
            experiment=detector,
            stage="S0 detector",
            detector=detector,
            status="completed",
            outcome="positive",
            best_epoch=12,
            best_map=m,
            best_ap50=ap50,
            final_epoch=12,
            final_map=m,
            final_ap50=ap50,
            source_file=source(src),
            work_dir=work_dir,
            notes=note,
        )

    dota2_s1_specs = [
        (
            "S1 RemoteCLIP default",
            OPENRSD / "work_dirs/geonexus_dota2/roi_trans_remoteclip_s1_validpng_20260607/20260607_101146/vis_data/scalars.json",
            OPENRSD / "work_dirs/geonexus_dota2/roi_trans_remoteclip_s1_validpng_20260607",
            "default LR",
        ),
        (
            "S1 RemoteCLIP lr1e-4",
            OPENRSD / "work_dirs/geonexus_dota2/roi_trans_remoteclip_s1_validpng_lr1e4_20260607/20260607_154103/vis_data/scalars.json",
            OPENRSD / "work_dirs/geonexus_dota2/roi_trans_remoteclip_s1_validpng_lr1e4_20260607",
            "low-LR replicate",
        ),
        (
            "S1 RemoteCLIP lr5e-5",
            OPENRSD / "work_dirs/geonexus_dota2/roi_trans_remoteclip_s1_validpng_lr5e5_20260607/20260607_161931/vis_data/scalars.json",
            OPENRSD / "work_dirs/geonexus_dota2/roi_trans_remoteclip_s1_validpng_lr5e5_20260607",
            "low-LR replicate",
        ),
    ]
    dota2_s1_final = None
    for experiment, scalar_path, work_dir, note in dota2_s1_specs:
        best, final = summarize_points(load_scalar_points(scalar_path))
        if dota2_s1_final is None:
            dota2_s1_final = final.map
        add_point_record(
            records,
            date="2026-06-07",
            dataset_group="DOTA2 formal",
            dataset_split="DOTA2_1024_500/ss_val",
            protocol="valid-PNG formal",
            experiment=experiment,
            stage="S1",
            detector="RoI Transformer",
            source_file=scalar_path,
            work_dir=str(work_dir),
            best=best,
            final=final,
            notes=note,
        )

    dota2_s2_loss0_specs = [
        ("S2 loss-0", OPENRSD / "work_dirs/geonexus_dota2/roi_trans_remoteclip_s2_hierarchy_ablate_loss0_s1e12_20260610/20260610_100253/vis_data/scalars.json"),
        ("S2 loss-0 rep3407", OPENRSD / "work_dirs/geonexus_dota2/roi_trans_remoteclip_s2_hierarchy_ablate_loss0_s1e12_rep3407_20260610/20260610_191026/vis_data/scalars.json"),
        ("S2 loss-0 rep4407", OPENRSD / "work_dirs/geonexus_dota2/roi_trans_remoteclip_s2_hierarchy_ablate_loss0_s1e12_rep4407_20260610/20260610_210021/vis_data/scalars.json"),
        ("S2 loss-0 rep5407", OPENRSD / "work_dirs/geonexus_dota2/roi_trans_remoteclip_s2_hierarchy_ablate_loss0_s1e12_rep5407_20260610/20260610_210021/vis_data/scalars.json"),
        ("S2 loss-0 rep6407", OPENRSD / "work_dirs/geonexus_dota2/roi_trans_remoteclip_s2_hierarchy_ablate_loss0_s1e12_rep6407_20260611/20260611_102732/vis_data/scalars.json"),
        ("S2 loss-0 rep7407", OPENRSD / "work_dirs/geonexus_dota2/roi_trans_remoteclip_s2_hierarchy_ablate_loss0_s1e12_rep7407_20260611/20260611_102732/vis_data/scalars.json"),
        ("S2 loss-0 rep8407", OPENRSD / "work_dirs/geonexus_dota2/roi_trans_remoteclip_s2_hierarchy_ablate_loss0_s1e12_rep8407_20260611/20260611_102732/vis_data/scalars.json"),
    ]
    dota2_s2_runs: list[tuple[str, Path, MetricPoint, MetricPoint]] = []
    for experiment, scalar_path in dota2_s2_loss0_specs:
        best, final = summarize_points(load_scalar_points(scalar_path))
        dota2_s2_runs.append((experiment, scalar_path, best, final))
        add_point_record(
            records,
            date="2026-06-10",
            dataset_group="DOTA2 formal",
            dataset_split="DOTA2_1024_500/ss_val",
            protocol="valid-PNG formal",
            experiment=experiment,
            stage="S2",
            detector="RoI Transformer",
            source_file=scalar_path,
            work_dir=str(scalar_path.parents[2]),
            best=best,
            final=final,
            notes="loss-0 stability replicate",
        )
    dota2_s2_best_mean = mean(best.map for _, _, best, _ in dota2_s2_runs)
    dota2_s2_final_mean = mean(final.map for _, _, _, final in dota2_s2_runs)
    assert_close("DOTA2 S2 loss-0 best mean", dota2_s2_best_mean, 0.6206)
    assert_close("DOTA2 S2 loss-0 final mean", dota2_s2_final_mean, 0.6167)
    add(
        records,
        date="2026-06-11",
        dataset_group="DOTA2 formal",
        dataset_split="DOTA2_1024_500/ss_val",
        protocol="valid-PNG formal",
        experiment="S2 loss-0 mean",
        stage="S2 aggregate",
        detector="RoI Transformer",
        status="completed",
        outcome="positive",
        best_map=dota2_s2_best_mean,
        final_map=dota2_s2_final_mean,
        n_replicas=len(dota2_s2_runs),
        source_file="computed from DOTA2 S2 loss-0 scalar sources",
        notes="seven-replica mean",
    )

    for experiment, scalar_path in [
        ("S2 hierarchy regularizer", OPENRSD / "work_dirs/geonexus_dota2/roi_trans_remoteclip_s2_hierarchy_reg_s1e12_20260608/20260608_105856/vis_data/scalars.json"),
        ("S2 hierarchy hw1e-2", OPENRSD / "work_dirs/geonexus_dota2/roi_trans_remoteclip_s2_hierarchy_reg_s1e12_hw1e2_20260608/20260608_194400/vis_data/scalars.json"),
        ("S2 hierarchy lr1e-4", OPENRSD / "work_dirs/geonexus_dota2/roi_trans_remoteclip_s2_hierarchy_reg_s1e12_lr1e4_20260608/20260608_194400/vis_data/scalars.json"),
        ("S2 hierarchy lr5e-5", OPENRSD / "work_dirs/geonexus_dota2/roi_trans_remoteclip_s2_hierarchy_reg_s1e12_lr5e5_20260609/20260609_153415/vis_data/scalars.json"),
        ("S2 rescue lr1e-5", OPENRSD / "work_dirs/geonexus_dota2/roi_trans_remoteclip_s2_hierarchy_reg_s2e4_rescue_lr1e5_20260610/20260610_100055/vis_data/scalars.json"),
    ]:
        best, final = summarize_points(load_scalar_points(scalar_path))
        add_point_record(
            records,
            date="2026-06-09",
            dataset_group="DOTA2 formal",
            dataset_split="DOTA2_1024_500/ss_val",
            protocol="valid-PNG formal",
            experiment=experiment,
            stage="S2 diagnostic",
            detector="RoI Transformer",
            source_file=scalar_path,
            work_dir=str(scalar_path.parents[2]),
            best=best,
            final=final,
            notes="metric-bearing diagnostic run",
            status="diagnostic",
            outcome="below_baseline",
        )

    s3_dota2_path = rel("docs/experiments/20260616_dota2_s3_scene_adapter_loss0_best_complete.json")
    if s3_dota2_path.exists():
        s3_dota2 = load_json(s3_dota2_path)
        for rep in s3_dota2["replicas"]:
            final_metric = next(m for m in rep["metrics"] if m["epoch"] == rep["final_epoch"])
            best_metric = next(m for m in rep["metrics"] if m["epoch"] == rep["best_epoch"])
            add(
                records,
                date="2026-06-16",
                dataset_group="DOTA2 formal",
                dataset_split="DOTA2_1024_500/ss_val",
                protocol="valid-PNG formal",
                experiment=f"S3 scene adapter rep{rep['replica']}",
                stage="S3",
                detector="RoI Transformer",
                status="completed",
                outcome="negative_to_neutral",
                best_epoch=int(rep["best_epoch"]),
                best_map=float(rep["best_mAP"]),
                best_ap50=float(best_metric.get("dota_AP50")) if "dota_AP50" in best_metric else None,
                final_epoch=int(rep["final_epoch"]),
                final_map=float(rep["final_mAP"]),
                final_ap50=float(final_metric.get("dota_AP50")) if "dota_AP50" in final_metric else None,
                source_file=rep.get("metric_source", source(s3_dota2_path)),
                work_dir=rep.get("work_dir", ""),
                notes=f"seed {rep.get('seed')}; scene-adapter replica from loss-0 best checkpoint",
            )
        dota2_s3_best_mean = mean(float(rep["best_mAP"]) for rep in s3_dota2["replicas"])
        dota2_s3_final_mean = mean(float(rep["final_mAP"]) for rep in s3_dota2["replicas"])
        assert_close("DOTA2 S3 best mean", dota2_s3_best_mean, 0.6199)
        assert_close("DOTA2 S3 final mean", dota2_s3_final_mean, 0.6151)
        if len(s3_dota2["replicas"]) != 3:
            raise AssertionError(
                f"DOTA2 S3 archive should contain 3 replicas, got {len(s3_dota2['replicas'])}")
        add(
            records,
            date="2026-06-16",
            dataset_group="DOTA2 formal",
            dataset_split="DOTA2_1024_500/ss_val",
            protocol="valid-PNG formal",
            experiment="S3 scene adapter mean",
            stage="S3 aggregate",
            detector="RoI Transformer",
            status="completed",
            outcome="negative_to_neutral",
            best_map=dota2_s3_best_mean,
            final_map=dota2_s3_final_mean,
            n_replicas=len(s3_dota2["replicas"]),
            source_file=source(s3_dota2_path),
            notes="three-replica best/final mean; below DOTA2 S2 loss-0 best and final means",
        )

    s0_dior_path = OPENRSD / "work_dirs/dior_r_s0_roi_trans_sanitized_long_20260612_gpu1/20260612_232047/vis_data/scalars.json"
    s0_dior_best, s0_dior_final = summarize_points(load_scalar_points(s0_dior_path))
    assert_close("DIOR-R S0 final", s0_dior_final.map, 0.6544)
    add_point_record(
        records,
        date="2026-06-13",
        dataset_group="DIOR-R formal",
        dataset_split="DIOR_R_dota/test",
        protocol="sanitized DIOR-R formal",
        experiment="RoI Transformer",
        stage="S0 detector",
        detector="RoI Transformer",
        source_file=s0_dior_path,
        work_dir=str(s0_dior_path.parents[2]),
        best=s0_dior_best,
        final=s0_dior_final,
        notes="sanitized detector baseline",
    )

    for detector, scalar_paths in [
        (
            "Oriented R-CNN",
            [
                OPENRSD / "work_dirs/dior_r_s0_orcnn_sanitized_long_20260612_gpu0/20260612_181155/vis_data/scalars.json",
                OPENRSD / "work_dirs/dior_r_s0_orcnn_sanitized_long_20260612_gpu0/20260612_235635/vis_data/scalars.json",
            ],
        ),
        (
            "Rotated RetinaNet",
            [
                OPENRSD / "work_dirs/dior_r_s0_retinanet_sanitized_long_20260612_gpu2/20260612_181213/vis_data/scalars.json",
                OPENRSD / "work_dirs/dior_r_s0_retinanet_sanitized_long_20260612_gpu2/20260612_235311/vis_data/scalars.json",
            ],
        ),
    ]:
        best, final = summarize_points(all_points(scalar_paths))
        add_point_record(
            records,
            date="2026-06-13",
            dataset_group="DIOR-R formal",
            dataset_split="DIOR_R_dota/test",
            protocol="sanitized DIOR-R formal",
            experiment=detector,
            stage="S0 detector",
            detector=detector,
            source_file="; ".join(source(p) for p in scalar_paths),
            work_dir=str(scalar_paths[0].parents[2]),
            best=best,
            final=final,
            notes="sanitized detector baseline",
        )

    s1_dior_path = rel("docs/experiments/20260613_dior_r_geonexus_s1_s0e52_replicas_metrics.json")
    s1_dior = load_json(s1_dior_path)
    dior_s1_finals: list[float] = []
    for rep in s1_dior["replicas"]:
        metrics = rep["metrics"]
        best = max(metrics, key=lambda m: m["dota_mAP"])
        final = next(m for m in metrics if m["epoch"] == rep["final_epoch"])
        dior_s1_finals.append(float(final["dota_mAP"]))
        add(
            records,
            date="2026-06-13",
            dataset_group="DIOR-R formal",
            dataset_split="DIOR_R_dota/test",
            protocol="sanitized DIOR-R formal",
            experiment=f"S1 rep{rep['replica']}",
            stage="S1",
            detector="RoI Transformer",
            status="completed",
            outcome="positive",
            best_epoch=int(best["epoch"]),
            best_map=float(best["dota_mAP"]),
            best_ap50=float(best["dota_AP50"]),
            final_epoch=int(final["epoch"]),
            final_map=float(final["dota_mAP"]),
            final_ap50=float(final["dota_AP50"]),
            source_file=source(s1_dior_path),
            work_dir=rep.get("work_dir", ""),
            notes=f"seed {rep.get('seed')}",
        )
    dior_s1_final_mean = mean(dior_s1_finals)
    assert_close("DIOR-R S1 final mean", dior_s1_final_mean, 0.6720)
    add(
        records,
        date="2026-06-13",
        dataset_group="DIOR-R formal",
        dataset_split="DIOR_R_dota/test",
        protocol="sanitized DIOR-R formal",
        experiment="S1 final mean",
        stage="S1 aggregate",
        detector="RoI Transformer",
        status="completed",
        outcome="positive",
        best_map=dior_s1_final_mean,
        final_map=dior_s1_final_mean,
        n_replicas=len(dior_s1_finals),
        source_file=source(s1_dior_path),
        notes="two-replica final mean",
    )

    s2_dior_path = rel("docs/experiments/20260614_dior_r_geonexus_s2_replicas_complete.json")
    s2_dior = load_json(s2_dior_path)
    for rep in s2_dior["replicas"]:
        final_metric = next(m for m in rep["metrics"] if m["epoch"] == rep["final_epoch"])
        best_metric = next(m for m in rep["metrics"] if m["epoch"] == rep["best_epoch"])
        add(
            records,
            date="2026-06-14",
            dataset_group="DIOR-R formal",
            dataset_split="DIOR_R_dota/test",
            protocol="sanitized DIOR-R formal",
            experiment=f"S2 rep{rep['replica']}",
            stage="S2",
            detector="RoI Transformer",
            status="completed",
            outcome="positive",
            best_epoch=int(rep["best_epoch"]),
            best_map=float(rep["best_mAP"]),
            best_ap50=float(best_metric.get("dota_AP50")) if "dota_AP50" in best_metric else None,
            final_epoch=int(rep["final_epoch"]),
            final_map=float(rep["final_mAP"]),
            final_ap50=float(final_metric.get("dota_AP50")) if "dota_AP50" in final_metric else None,
            source_file=rep.get("metric_source", source(s2_dior_path)),
            work_dir=rep.get("work_dir", ""),
            notes=f"seed {rep.get('seed')}",
        )
    dior_s2_best_mean = mean(float(rep["best_mAP"]) for rep in s2_dior["replicas"])
    dior_s2_final_mean = mean(float(rep["final_mAP"]) for rep in s2_dior["replicas"])
    assert_close("DIOR-R S2 best mean", dior_s2_best_mean, 0.6887)
    assert_close("DIOR-R S2 final mean", dior_s2_final_mean, 0.6856)
    if len(s2_dior["replicas"]) != 6:
        raise AssertionError(
            f"DIOR-R S2 archive should contain 6 replicas, got {len(s2_dior['replicas'])}")
    add(
        records,
        date="2026-06-14",
        dataset_group="DIOR-R formal",
        dataset_split="DIOR_R_dota/test",
        protocol="sanitized DIOR-R formal",
        experiment="S2 mean",
        stage="S2 aggregate",
        detector="RoI Transformer",
        status="completed",
        outcome="positive",
        best_map=dior_s2_best_mean,
        final_map=dior_s2_final_mean,
        n_replicas=len(s2_dior["replicas"]),
        source_file=source(s2_dior_path),
        notes="six-replica best/final mean; best single rep4 epoch 12",
    )

    s3_dior_path = rel("docs/experiments/20260615_dior_r_geonexus_s3_scene_adapter_replicas_complete.json")
    if s3_dior_path.exists():
        s3_dior = load_json(s3_dior_path)
        for rep in s3_dior["replicas"]:
            final_metric = next(m for m in rep["metrics"] if m["epoch"] == rep["final_epoch"])
            best_metric = next(m for m in rep["metrics"] if m["epoch"] == rep["best_epoch"])
            add(
                records,
                date="2026-06-15",
                dataset_group="DIOR-R formal",
                dataset_split="DIOR_R_dota/test",
                protocol="sanitized DIOR-R formal",
                experiment=f"S3 scene adapter rep{rep['replica']}",
                stage="S3",
                detector="RoI Transformer",
                status="completed",
                outcome="positive_best_checkpoint_final_tied",
                best_epoch=int(rep["best_epoch"]),
                best_map=float(rep["best_mAP"]),
                best_ap50=float(best_metric.get("dota_AP50")) if "dota_AP50" in best_metric else None,
                final_epoch=int(rep["final_epoch"]),
                final_map=float(rep["final_mAP"]),
                final_ap50=float(final_metric.get("dota_AP50")) if "dota_AP50" in final_metric else None,
                source_file=rep.get("metric_source", source(s3_dior_path)),
                work_dir=rep.get("work_dir", ""),
                notes=f"seed {rep.get('seed')}; scene-adapter replica",
            )
        dior_s3_best_mean = mean(float(rep["best_mAP"]) for rep in s3_dior["replicas"])
        dior_s3_final_mean = mean(float(rep["final_mAP"]) for rep in s3_dior["replicas"])
        assert_close("DIOR-R S3 best mean", dior_s3_best_mean, 0.6979)
        assert_close("DIOR-R S3 final mean", dior_s3_final_mean, 0.6859)
        if len(s3_dior["replicas"]) != 3:
            raise AssertionError(
                f"DIOR-R S3 archive should contain 3 replicas, got {len(s3_dior['replicas'])}")
        add(
            records,
            date="2026-06-15",
            dataset_group="DIOR-R formal",
            dataset_split="DIOR_R_dota/test",
            protocol="sanitized DIOR-R formal",
            experiment="S3 scene adapter mean",
            stage="S3 aggregate",
            detector="RoI Transformer",
            status="completed",
            outcome="positive_best_checkpoint_final_tied",
            best_map=dior_s3_best_mean,
            final_map=dior_s3_final_mean,
            n_replicas=len(s3_dior["replicas"]),
            source_file=source(s3_dior_path),
            notes="three-replica best/final mean; best single rep0 epoch 8",
        )

    s3_stability_path = rel("docs/experiments/20260615_dior_r_s3_epoch8_lr5e5_stability_complete.json")
    if s3_stability_path.exists():
        s3_stability = load_json(s3_stability_path)
        for rep in s3_stability["replicas"]:
            final_metric = next(m for m in rep["metrics"] if m["epoch"] == rep["final_epoch"])
            best_metric = next(m for m in rep["metrics"] if m["epoch"] == rep["best_epoch"])
            add(
                records,
                date="2026-06-15",
                dataset_group="DIOR-R formal",
                dataset_split="DIOR_R_dota/test",
                protocol="sanitized DIOR-R formal",
                experiment=f"S3 epoch-8 LR5e-5 stability rep{rep['replica']}",
                stage="S3 stability",
                detector="RoI Transformer",
                status="completed",
                outcome="final_stability_improved_best_lower",
                best_epoch=int(rep["best_epoch"]),
                best_map=float(rep["best_mAP"]),
                best_ap50=float(best_metric.get("dota_AP50")) if "dota_AP50" in best_metric else None,
                final_epoch=int(rep["final_epoch"]),
                final_map=float(rep["final_mAP"]),
                final_ap50=float(final_metric.get("dota_AP50")) if "dota_AP50" in final_metric else None,
                source_file=source(s3_stability_path),
                work_dir=rep.get("work_dir", ""),
                notes="LR5e-5 continuation from original S3 epoch-8 checkpoint",
            )
        stability_best_mean = mean(float(rep["best_mAP"]) for rep in s3_stability["replicas"])
        stability_final_mean = mean(float(rep["final_mAP"]) for rep in s3_stability["replicas"])
        assert_close("DIOR-R S3 epoch-8 LR5e-5 stability best mean", stability_best_mean, 0.6922)
        assert_close("DIOR-R S3 epoch-8 LR5e-5 stability final mean", stability_final_mean, 0.6903)
        if len(s3_stability["replicas"]) != 3:
            raise AssertionError(
                f"DIOR-R S3 stability archive should contain 3 replicas, got {len(s3_stability['replicas'])}")
        add(
            records,
            date="2026-06-15",
            dataset_group="DIOR-R formal",
            dataset_split="DIOR_R_dota/test",
            protocol="sanitized DIOR-R formal",
            experiment="S3 epoch-8 LR5e-5 stability mean",
            stage="S3 stability aggregate",
            detector="RoI Transformer",
            status="completed",
            outcome="final_stability_improved_best_lower",
            best_map=stability_best_mean,
            final_map=stability_final_mean,
            n_replicas=len(s3_stability["replicas"]),
            source_file=source(s3_stability_path),
            notes="lower than original S3 best mean 0.6979; final mean improves over original S3 final 0.6859 and S2 final 0.6856",
        )

    s3_annealed_path = rel("docs/experiments/20260616_dior_r_s3_stability_e4_lr2p5e5_complete.json")
    if s3_annealed_path.exists():
        s3_annealed = load_json(s3_annealed_path)
        for rep in s3_annealed["replicas"]:
            final_metric = next(m for m in rep["metrics"] if m["epoch"] == rep["final_epoch"])
            best_metric = next(m for m in rep["metrics"] if m["epoch"] == rep["best_epoch"])
            add(
                records,
                date="2026-06-16",
                dataset_group="DIOR-R formal",
                dataset_split="DIOR_R_dota/test",
                protocol="sanitized DIOR-R formal",
                experiment=f"S3 stability e4 LR2.5e-5 rep{rep['replica']}",
                stage="S3 stability",
                detector="RoI Transformer",
                status="completed",
                outcome="neutral",
                best_epoch=int(rep["best_epoch"]),
                best_map=float(rep["best_mAP"]),
                best_ap50=float(best_metric.get("dota_AP50")) if "dota_AP50" in best_metric else None,
                final_epoch=int(rep["final_epoch"]),
                final_map=float(rep["final_mAP"]),
                final_ap50=float(final_metric.get("dota_AP50")) if "dota_AP50" in final_metric else None,
                source_file=source(s3_annealed_path),
                work_dir=rep.get("work_dir", ""),
                notes="LR2.5e-5 continuation from LR5e-5 stability epoch-4 checkpoint",
            )
        annealed_best_mean = mean(float(rep["best_mAP"]) for rep in s3_annealed["replicas"])
        annealed_final_mean = mean(float(rep["final_mAP"]) for rep in s3_annealed["replicas"])
        assert_close("DIOR-R S3 stability e4 LR2.5e-5 best mean", annealed_best_mean, 0.6908)
        assert_close("DIOR-R S3 stability e4 LR2.5e-5 final mean", annealed_final_mean, 0.6892)
        if len(s3_annealed["replicas"]) != 3:
            raise AssertionError(
                f"DIOR-R S3 annealed stability archive should contain 3 replicas, got {len(s3_annealed['replicas'])}")
        add(
            records,
            date="2026-06-16",
            dataset_group="DIOR-R formal",
            dataset_split="DIOR_R_dota/test",
            protocol="sanitized DIOR-R formal",
            experiment="S3 stability e4 LR2.5e-5 mean",
            stage="S3 stability aggregate",
            detector="RoI Transformer",
            status="completed",
            outcome="neutral",
            best_map=annealed_best_mean,
            final_map=annealed_final_mean,
            n_replicas=len(s3_annealed["replicas"]),
            source_file=source(s3_annealed_path),
            notes="neutral final-stability follow-up; final mean above S2 final 0.6856 but below useful threshold 0.6903",
        )


def add_json_metric_rows(records: list[ResultRecord]) -> None:
    for path in sorted(DOCS_DIR.glob("*.json")):
        data = load_json(path)
        if not isinstance(data, dict):
            continue
        date = extract_date(path, data)
        status, outcome = normalize_status(data.get("status") or data.get("completion_status"), notes_from(data.get("notes") or data.get("reason") or data.get("decision") or data.get("interpretation")))
        dataset_group = infer_dataset_group(" ".join(str(v) for v in [path.name, data.get("dataset"), data.get("experiment"), data.get("purpose")]))
        dataset_split = str(data.get("split") or data.get("dataset_split") or infer_dataset_split(dataset_group, path.name))
        protocol = str(data.get("protocol") or data.get("metric_implementation") or "")
        experiment = str(data.get("experiment") or data.get("archive") or path.stem)
        detector = str(data.get("detector") or ("RoI Transformer" if "roi" in experiment.lower() else ""))
        stage = infer_stage(" ".join([path.name, experiment, detector]))
        work_dir = extract_work_dir(data)
        notes = notes_from(data.get("notes") or data.get("reason") or data.get("decision") or data.get("interpretation") or data.get("purpose"))

        if "replicas" in data and isinstance(data["replicas"], list):
            for rep in data["replicas"]:
                if not isinstance(rep, dict):
                    continue
                rep_metrics = rep.get("metrics")
                best_epoch = int_value(rep, ["best_epoch"])
                final_epoch = int_value(rep, ["final_epoch"])
                best_map = metric_value(rep, ["best_mAP", "best_map"])
                final_map = metric_value(rep, ["final_mAP", "final_map"])
                best_ap50 = metric_value(rep, ["best_AP50", "best_ap50"])
                final_ap50 = metric_value(rep, ["final_AP50", "final_ap50"])
                if isinstance(rep_metrics, list) and rep_metrics:
                    best_metric = max(rep_metrics, key=lambda m: metric_value(m, ["dota_mAP", "dota/mAP", "map"]) or -1)
                    final_metric = next((m for m in rep_metrics if int_value(m, ["epoch"]) == final_epoch), rep_metrics[-1])
                    best_epoch = best_epoch or int_value(best_metric, ["epoch"])
                    final_epoch = final_epoch or int_value(final_metric, ["epoch"])
                    best_map = best_map if best_map is not None else metric_value(best_metric, ["dota_mAP", "dota/mAP", "map"])
                    final_map = final_map if final_map is not None else metric_value(final_metric, ["dota_mAP", "dota/mAP", "map"])
                    best_ap50 = best_ap50 if best_ap50 is not None else metric_value(best_metric, ["dota_AP50", "dota/AP50", "ap50"])
                    final_ap50 = final_ap50 if final_ap50 is not None else metric_value(final_metric, ["dota_AP50", "dota/AP50", "ap50"])
                add(
                    records,
                    date=date,
                    dataset_group=dataset_group,
                    dataset_split=dataset_split,
                    protocol=protocol,
                    experiment=f"{experiment} rep{rep.get('replica', '')}".strip(),
                    stage=stage,
                    detector=detector,
                    status=status,
                    outcome=outcome,
                    best_epoch=best_epoch,
                    best_map=best_map,
                    best_ap50=best_ap50,
                    final_epoch=final_epoch,
                    final_map=final_map,
                    final_ap50=final_ap50,
                    source_file=source(path),
                    work_dir=rep.get("work_dir", work_dir),
                    notes=notes or f"seed {rep.get('seed', '')}".strip(),
                )
            continue

        metrics = data.get("epoch_metrics") or data.get("metrics")
        best_epoch = int_value(data, ["best_epoch", "best_epoch_observed"])
        final_epoch = int_value(data, ["final_epoch", "epochs", "latest_checkpoint_epoch", "last_observed_log_epoch"])
        best_map = metric_value(data, ["best_map", "map", "first_epoch_map"])
        best_ap50 = metric_value(data, ["best_ap50", "ap50", "first_epoch_ap50"])
        final_map = metric_value(data, ["final_map"])
        final_ap50 = metric_value(data, ["final_ap50"])
        if isinstance(data.get("best_metrics"), dict):
            best_map = metric_value(data["best_metrics"], ["dota/mAP", "dota_mAP", "map"])
            best_ap50 = metric_value(data["best_metrics"], ["dota/AP50", "dota_AP50", "ap50"])
        if isinstance(data.get("final_metrics"), dict):
            final_map = metric_value(data["final_metrics"], ["dota/mAP", "dota_mAP", "map"])
            final_ap50 = metric_value(data["final_metrics"], ["dota/AP50", "dota_AP50", "ap50"])
        if isinstance(metrics, dict):
            best_map = best_map if best_map is not None else metric_value(metrics, ["dota/mAP", "dota_mAP", "map"])
            best_ap50 = best_ap50 if best_ap50 is not None else metric_value(metrics, ["dota/AP50", "dota_AP50", "ap50"])
            final_map = final_map if final_map is not None else best_map
            final_ap50 = final_ap50 if final_ap50 is not None else best_ap50
        if isinstance(metrics, list) and metrics:
            metric_rows = [m for m in metrics if isinstance(m, dict)]
            metric_rows_with_map = [m for m in metric_rows if metric_value(m, ["dota/mAP", "dota_mAP", "map"]) is not None]
            if metric_rows_with_map:
                best_metric = max(metric_rows_with_map, key=lambda m: metric_value(m, ["dota/mAP", "dota_mAP", "map"]) or -1)
                final_metric = metric_rows_with_map[-1]
                best_epoch = best_epoch or int_value(best_metric, ["epoch"])
                final_epoch = final_epoch or int_value(final_metric, ["epoch"])
                best_map = best_map if best_map is not None else metric_value(best_metric, ["dota/mAP", "dota_mAP", "map"])
                best_ap50 = best_ap50 if best_ap50 is not None else metric_value(best_metric, ["dota/AP50", "dota_AP50", "ap50"])
                final_map = final_map if final_map is not None else metric_value(final_metric, ["dota/mAP", "dota_mAP", "map"])
                final_ap50 = final_ap50 if final_ap50 is not None else metric_value(final_metric, ["dota/AP50", "dota_AP50", "ap50"])

        add(
            records,
            date=date,
            dataset_group=dataset_group,
            dataset_split=dataset_split,
            protocol=protocol,
            experiment=experiment,
            stage=stage,
            detector=detector,
            status=status,
            outcome=outcome,
            best_epoch=best_epoch,
            best_map=best_map,
            best_ap50=best_ap50,
            final_epoch=final_epoch,
            final_map=final_map,
            final_ap50=final_ap50,
            n_replicas=len(data["replicas"]) if isinstance(data.get("replicas"), list) else None,
            source_file=source(path),
            work_dir=work_dir,
            notes=notes,
        )


def add_markdown_attempt_rows(records: list[ResultRecord]) -> None:
    for path in sorted(DOCS_DIR.glob("*.md")):
        if path.name in {"README.md", "template.md"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        status, outcome = normalize_status("", text)
        group = infer_dataset_group(f"{path.name}\n{text[:3000]}")
        add(
            records,
            date=extract_date(path),
            dataset_group=group,
            dataset_split=infer_dataset_split(group, text),
            protocol="experiment note",
            experiment=path.stem,
            stage=infer_stage(path.name + "\n" + text[:2000]),
            detector="",
            status=status,
            outcome=outcome,
            source_file=source(path),
            work_dir=extract_work_dir({}, text),
            notes=notes_from(text.splitlines()[0] if text.splitlines() else path.stem),
        )

    # Explicit split rows for important multi-attempt notes that contain both
    # successful and failed/invalid evidence in one file.
    add(
        records,
        date="2026-05-26",
        dataset_group="DOTA v1.5",
        dataset_split="DOTA v1.5 reduced tiled split/val",
        protocol="experiment note",
        experiment="GeoNexus S1 first GPU0 launch",
        stage="S1",
        detector="RoI Transformer",
        status="failed",
        outcome="oom",
        source_file="docs/experiments/20260526_paper_evidence_dota15_summary.md",
        work_dir="/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s1",
        notes="GPU 0 launch reached training but hit CUDA OOM during RPN target assignment.",
    )
    add(
        records,
        date="2026-06-02",
        dataset_group="DOTA2",
        dataset_split="DOTA2_1024_500/ss_val",
        protocol="experiment note",
        experiment="S0 DOTA2 Oriented R-CNN R50 valid-PNG",
        stage="S0 detector",
        detector="Oriented R-CNN",
        status="failed",
        outcome="oom",
        source_file="docs/experiments/20260602_gpu_status_2025.md",
        notes="Failed after entering training at epoch 1 iter 300 with CUDA out-of-memory.",
    )
    add(
        records,
        date="2026-05-25",
        dataset_group="DOTA v1.5",
        dataset_split="DOTA v1.5 reduced tiled split/val",
        protocol="experiment note",
        experiment="RoI Transformer early lr0.005 run",
        stage="S0 detector",
        detector="RoI Transformer",
        status="invalid",
        outcome="nan_loss",
        source_file="docs/experiments/20260525_strong_detector_sweep.md",
        work_dir="/data5/2025/ldh/OpenRSD/work_dirs/strong_baseline_dota15/roi_trans_lr001_rerun",
        notes="Earlier LR 0.005 run diverged to NaN and was replaced by lower-LR runs.",
    )
    add(
        records,
        date="2026-06-07",
        dataset_group="DIOR-R",
        dataset_split="DIOR_R_dota/test",
        protocol="diagnostic",
        experiment="DIOR-R Oriented R-CNN NaN diagnostics",
        stage="S0 detector",
        detector="Oriented R-CNN",
        status="invalid",
        outcome="nan_loss",
        source_file="docs/experiments/20260607_dior_orcnn_nan_diag_and_roi_trans_launch.md",
        notes="DIOR-R detector evidence blocked by NaN/Inf diagnostics before sanitized relaunches.",
    )
    add(
        records,
        date="2026-06-04",
        dataset_group="DOTA2",
        dataset_split="DOTA2_1024_500/ss_val",
        protocol="experiment note",
        experiment="GPU pruning and detector evidence",
        stage="S0 detector",
        detector="",
        status="blocked",
        outcome="active_then_stopped",
        source_file="docs/experiments/20260604_gpu_pruning_and_next_priority.md",
        notes="Detector follow-up was blocked/paused by GPU availability and active processes.",
    )
    add(
        records,
        date="2026-06-12",
        dataset_group="DIOR-R",
        dataset_split="DIOR_R_dota/test",
        protocol="diagnostic",
        experiment="DIOR-R sanitized smoke and long launch",
        stage="S0 detector",
        detector="RoI Transformer",
        status="launch_only",
        outcome="active_then_stopped",
        source_file="docs/experiments/20260612_dior_r_s0_sanitized_smoke_and_long_launch.md",
        notes="Launch and smoke-test record before long-run completion.",
    )


def fmt(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.4f}"
    return str(value)


def record_to_row(record: ResultRecord) -> dict[str, str]:
    row = asdict(record)
    return {field: fmt(row.get(field)) for field in CSV_FIELDS}


def clean_output_dir() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in OUT_DIR.iterdir():
        if path == OUT_CSV:
            continue
        if path.is_file() or path.is_symlink():
            path.unlink()


def write_csv(records: list[ResultRecord]) -> None:
    clean_output_dir()
    records = sorted(
        dedupe(records),
        key=lambda r: (r.date, r.dataset_group, r.stage, r.experiment, r.source_file),
    )
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow(record_to_row(record))


def main() -> None:
    records: list[ResultRecord] = []
    collect_formal_records(records)
    add_json_metric_rows(records)
    add_markdown_attempt_rows(records)
    write_csv(records)
    print(f"Wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
