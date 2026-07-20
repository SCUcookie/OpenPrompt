#!/usr/bin/env python3
"""Extract per-class AP50 from an MMRotate DOTAMetric runtime log.

Serves two jobs:
  * Job A1 (default, --dataset dior): the TGRS manuscript per-class Table V.
  * Job N1 (--dataset fair1m): the FAIR1M S1-vs-S0 per-class delta analysis
    (see docs/experiments/20260720_fair1m_s1_route_review_and_next_steps.md).
    FAIR1M mode loads the 37 canonical class names and their parent groups
    from assets/hierarchies/fair1m_remote_sensing_taxonomy.json and adds a
    per-parent rollup (airplane/ship/vehicle/court/road) to the outputs.

Stdlib only, so it runs in any Python on the server; no mmrotate import is
needed because the per-class table is already printed into every evaluation
log by DOTAMetric.

Usage (server):

    # DIOR-R manuscript job A1
    python scripts/extract_perclass_ap_20260713.py \
        --log /path/to/eval_runtime.log \
        --tag geonexus_sca_rep0_e8

    # FAIR1M analysis job N1 (per replica log; also parses training logs --
    # pass --epoch to select a specific epoch's table instead of the last)
    python scripts/extract_perclass_ap_20260713.py --dataset fair1m \
        --log /path/to/runtime.log --tag s1_rep3407_e8

The script parses the LAST per-class table in the log (or the table
immediately following the requested --epoch marker), verifies the full class
list was found, and emits JSON, CSV, and (DIOR-R mode) a ready-to-paste
LaTeX row in the manuscript's class order.
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
    # Keep digits: FAIR1M has a220/a321/a330/boeing737/... which would
    # collide if digits were stripped.
    return re.sub(r"[^a-z0-9]", "", name.lower())


def load_fair1m_classes() -> list[tuple[str, str, str]]:
    """Return [(display_name, log_name, parent)] in canonical FAIR1M order."""
    taxonomy_path = Path(__file__).resolve().parents[1] / "assets" / "hierarchies" / "fair1m_remote_sensing_taxonomy.json"
    payload = json.loads(taxonomy_path.read_text(encoding="utf-8"))
    parents = {c["name"]: c["parent"] for c in payload["classes"]}
    order = payload["_provenance"]["canonical_order"]
    return [(name, name, parents[name]) for name in order]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", required=True, type=Path, help="MMRotate evaluation/training runtime log")
    ap.add_argument("--tag", required=True, help="Short method tag for outputs (e.g. baseline_e52)")
    ap.add_argument("--dataset", choices=["dior", "fair1m"], default="dior")
    ap.add_argument("--out-json", type=Path, default=None)
    args = ap.parse_args()

    raw = parse_last_table(args.log)
    lookup = {normalize(k): v for k, v in raw.items()}

    if args.dataset == "dior":
        class_spec = [(abbrev, cls, None) for abbrev, cls in DIOR_CLASSES_IN_TABLE_ORDER]
    else:
        class_spec = load_fair1m_classes()

    ordered: list[tuple[str, float, str | None]] = []
    missing: list[str] = []
    for display, log_name, parent in class_spec:
        key = normalize(log_name)
        if key in lookup:
            ordered.append((display, lookup[key], parent))
        else:
            missing.append(log_name)
    if missing:
        print(f"WARNING: {len(missing)} classes not found in log: {missing}", file=sys.stderr)
        print(f"Classes present in log: {sorted(raw)}", file=sys.stderr)
        raise SystemExit(1)

    values_pct = [100.0 * v for _, v, _ in ordered]
    mean_pct = sum(values_pct) / len(values_pct)

    result = {
        "tag": args.tag,
        "dataset": args.dataset,
        "log": str(args.log),
        "per_class_ap50_pct": {display: round(100.0 * v, 2) for display, v, _ in ordered},
        "mAP_pct": round(mean_pct, 2),
    }
    if args.dataset == "fair1m":
        groups: dict[str, list[float]] = {}
        for display, v, parent in ordered:
            groups.setdefault(parent, []).append(100.0 * v)
        result["parent_group_mean_ap50_pct"] = {
            parent: round(sum(vals) / len(vals), 2) for parent, vals in sorted(groups.items())
        }
    print(json.dumps(result, indent=2))

    out_json = args.out_json or Path(f"perclass_ap50_{args.tag}.json")
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    out_csv = out_json.with_suffix(".csv")
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class", "parent", "ap50_pct"])
        for display, v, parent in ordered:
            w.writerow([display, parent or "", f"{100.0 * v:.2f}"])
        w.writerow(["mAP", "", f"{mean_pct:.2f}"])

    if args.dataset == "dior":
        latex_row = " & ".join(f"{v:.2f}" for v in values_pct)
        print("\nLaTeX row (paste into tables/table_perclass.tex data row):")
        print(f"{args.tag} & {latex_row} & {mean_pct:.2f} \\\\")
    print(f"\nWrote {out_json} and {out_csv}")


if __name__ == "__main__":
    main()
