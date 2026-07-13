#!/usr/bin/env python3
"""Extract per-class AP50 from an MMRotate DOTAMetric runtime log (job A1).

Feeds Table V (tab:perclass) of the TGRS manuscript. Stdlib only, so it runs
in any Python on the server; no mmrotate import needed because the per-class
table is already printed into every evaluation log by DOTAMetric.

Usage (server):

    python scripts/extract_perclass_ap_20260713.py \
        --log /path/to/eval_runtime.log \
        --tag geonexus_sca_rep0_e8 \
        --out-json perclass_ap50_geonexus.json

Target logs (see docs/experiments/20260713_paper_finalization_schedule.md,
job A1, for the full provenance):
  * DIOR-R baseline: the epoch-52 RoI Transformer evaluation log under
    the paper-eval workdirs of 2026-06-17 (S0 epoch52 preds run).
  * GeoNexus best run: the scene-adapter rep0 epoch-8 evaluation log under
    the paper-eval workdirs of 2026-06-17.
  * OrientedFormer reproduction: the 2026-07-04 protocol-eval rerun log.

The script parses the LAST per-class table in the log (the final validation),
verifies 20 DIOR-R classes were found, and emits JSON, CSV, and a
ready-to-paste LaTeX row in the manuscript's class order.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

# Manuscript column order (Table V) -> DIOR-R class names used by MMRotate.
DIOR_CLASSES_IN_TABLE_ORDER = [
    ("APL", "airplane"),
    ("APO", "airport"),
    ("BF", "baseballfield"),
    ("BC", "basketballcourt"),
    ("BR", "bridge"),
    ("CH", "chimney"),
    ("DAM", "dam"),
    ("ESA", "Expressway-Service-area"),
    ("ETS", "Expressway-toll-station"),
    ("GF", "golffield"),
    ("GTF", "groundtrackfield"),
    ("HA", "harbor"),
    ("OP", "overpass"),
    ("SH", "ship"),
    ("STA", "stadium"),
    ("STO", "storagetank"),
    ("TC", "tenniscourt"),
    ("TS", "trainstation"),
    ("VE", "vehicle"),
    ("WM", "windmill"),
]

ROW_RE = re.compile(
    r"^\|\s*(?P<name>[A-Za-z][A-Za-z0-9_\- ]*?)\s*\|"
    r"(?:[^|]*\|){0,3}"          # optional gts/dets/recall columns
    r"\s*(?P<ap>[01]\.\d+)\s*\|\s*$"
)


def parse_last_table(log_path: Path) -> dict[str, float]:
    """Return {class_name: ap} from the last per-class table in the log."""
    tables: list[dict[str, float]] = []
    current: dict[str, float] = {}
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = ROW_RE.match(line.strip())
        if m:
            name = m.group("name").strip()
            if name.lower() in {"class", "map"}:
                if name.lower() == "map" and current:
                    tables.append(current)
                    current = {}
                continue
            current[name] = float(m.group("ap"))
        elif current and line.strip().startswith("+--"):
            continue
    if current:
        tables.append(current)
    if not tables:
        raise SystemExit(f"No per-class AP table found in {log_path}")
    return tables[-1]


def normalize(name: str) -> str:
    return re.sub(r"[^a-z]", "", name.lower())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", required=True, type=Path, help="MMRotate evaluation runtime log")
    ap.add_argument("--tag", required=True, help="Short method tag for outputs (e.g. baseline_e52)")
    ap.add_argument("--out-json", type=Path, default=None)
    args = ap.parse_args()

    raw = parse_last_table(args.log)
    lookup = {normalize(k): v for k, v in raw.items()}

    ordered: list[tuple[str, float]] = []
    missing: list[str] = []
    for abbrev, cls in DIOR_CLASSES_IN_TABLE_ORDER:
        key = normalize(cls)
        if key in lookup:
            ordered.append((abbrev, lookup[key]))
        else:
            missing.append(cls)
    if missing:
        print(f"WARNING: {len(missing)} classes not found in log: {missing}", file=sys.stderr)
        print(f"Classes present in log: {sorted(raw)}", file=sys.stderr)
        raise SystemExit(1)

    values_pct = [100.0 * v for _, v in ordered]
    mean_pct = sum(values_pct) / len(values_pct)

    result = {
        "tag": args.tag,
        "log": str(args.log),
        "per_class_ap50_pct": {abbrev: round(v, 2) for (abbrev, _), v in zip(ordered, values_pct)},
        "mAP_pct": round(mean_pct, 2),
    }
    print(json.dumps(result, indent=2))

    out_json = args.out_json or Path(f"perclass_ap50_{args.tag}.json")
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    out_csv = out_json.with_suffix(".csv")
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class", "ap50_pct"])
        for (abbrev, _), v in zip(ordered, values_pct):
            w.writerow([abbrev, f"{v:.2f}"])
        w.writerow(["mAP", f"{mean_pct:.2f}"])

    latex_row = " & ".join(f"{v:.2f}" for v in values_pct)
    print("\nLaTeX row (paste into tables/table_perclass.tex data row):")
    print(f"{args.tag} & {latex_row} & {mean_pct:.2f} \\\\")
    print(f"\nWrote {out_json} and {out_csv}")


if __name__ == "__main__":
    main()
