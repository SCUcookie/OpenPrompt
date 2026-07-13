#!/usr/bin/env python3
"""Convert FAIR1M XML labels (PASCAL-VOC-style, one file per image) into
DOTA-style rotated-box txt annotations so this repo's existing DOTA-format
dataloader and MMRotate DOTA-style configs can read FAIR1M directly.

MMRotate has no native FAIR1M dataset class; the documented community
practice is to convert FAIR1M's XML labels into DOTA txt offline and reuse
the standard DOTA tooling. This script does that conversion.

FAIR1M XML schema (verified against torchgeo.datasets.fair1m.parse_pascal_voc,
2026-07-09):

    <object>
      <possibleresult><name>Small Car</name></possibleresult>
      <points>
        <point>x1,y1</point>
        <point>x2,y2</point>
        <point>x3,y3</point>
        <point>x4,y4</point>
        <point>x1,y1</point>  <!-- FAIR1M repeats the first point to close
                                    the polygon; this script drops the
                                    duplicate closing point. -->
      </points>
    </object>

Output line format (DOTA v1.0 style, matches scripts/build_dota2_tiled_annfiles_from_labels.py):

    x1 y1 x2 y2 x3 y3 x4 y4 class_name difficulty

Run once per split against the raw FAIR1M labelXml directory:

    python scripts/convert_fair1m_xml_to_dota_txt.py \
        --xml-dir /path/to/FAIR1M/train/labelXml \
        --out-dir /path/to/FAIR1M/train/labelTxt \
        --report-path /path/to/FAIR1M/train/convert_report_20260709.json

This has NOT been run against real FAIR1M XML files yet (the 500GB dataset
lives on the server, not this machine) -- validate on a small sample before
trusting the full conversion. Report any schema mismatch (unexpected tag
names, point counts other than 4 or 5, unmapped class names) via the
--report-path JSON; do not silently drop objects.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from xml.etree import ElementTree as ET

# Official FAIR1M class names (as they appear in the XML <name> tag) mapped
# to this repo's kebab-case taxonomy names in
# assets/hierarchies/fair1m_remote_sensing_taxonomy.json. Verified against
# the FAIR1M paper (arXiv 2103.05569) and torchgeo.datasets.fair1m's classes
# dict, 2026-07-09.
CLASS_NAME_MAP: dict[str, str] = {
    "Passenger Ship": "passenger-ship",
    "Motorboat": "motorboat",
    "Fishing Boat": "fishing-boat",
    "Tugboat": "tugboat",
    "other-ship": "other-ship",
    "Engineering Ship": "engineering-ship",
    "Liquid Cargo Ship": "liquid-cargo-ship",
    "Dry Cargo Ship": "dry-cargo-ship",
    "Warship": "warship",
    "Small Car": "small-car",
    "Bus": "bus",
    "Cargo Truck": "cargo-truck",
    "Dump Truck": "dump-truck",
    "other-vehicle": "other-vehicle",
    "Van": "van",
    "Trailer": "trailer",
    "Tractor": "tractor",
    "Excavator": "excavator",
    "Truck Tractor": "truck-tractor",
    "Boeing737": "boeing737",
    "Boeing747": "boeing747",
    "Boeing777": "boeing777",
    "Boeing787": "boeing787",
    "ARJ21": "arj21",
    "C919": "c919",
    "A220": "a220",
    "A321": "a321",
    "A330": "a330",
    "A350": "a350",
    "other-airplane": "other-airplane",
    "Baseball Field": "baseball-field",
    "Basketball Court": "basketball-court",
    "Football Field": "football-field",
    "Tennis Court": "tennis-court",
    "Roundabout": "roundabout",
    "Intersection": "intersection",
    "Bridge": "bridge",
}


class ConversionError(Exception):
    pass


def parse_points(points_elem: ET.Element) -> list[tuple[float, float]]:
    raw = []
    for point in points_elem.findall("point"):
        text = (point.text or "").strip()
        if not text:
            raise ConversionError(f"empty <point> text")
        x_str, y_str = text.split(",")
        raw.append((float(x_str), float(y_str)))
    if len(raw) == 5 and raw[0] == raw[-1]:
        raw = raw[:4]
    if len(raw) != 4:
        raise ConversionError(f"expected 4 (or 5 closed) points, got {len(raw)}")
    if not all(math.isfinite(value) for point in raw for value in point):
        raise ConversionError("polygon contains non-finite coordinates")
    edge_lengths = [
        math.hypot(raw[(index + 1) % 4][0] - x, raw[(index + 1) % 4][1] - y)
        for index, (x, y) in enumerate(raw)
    ]
    area = abs(
        sum(
            x * raw[(index + 1) % 4][1] - raw[(index + 1) % 4][0] * y
            for index, (x, y) in enumerate(raw)
        )
    ) * 0.5
    if area <= 0 or any(length <= 0 for length in edge_lengths):
        raise ConversionError(f"degenerate polygon with area={area} and edge_lengths={edge_lengths}")
    return raw


def parse_fair1m_xml(xml_path: Path) -> tuple[list[str], list[str]]:
    """Return (dota_lines, warnings) for one FAIR1M label XML file."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    lines: list[str] = []
    warnings: list[str] = []
    for obj in root.iter("object"):
        possibleresult = obj.find("possibleresult")
        points_elem = obj.find("points")
        if possibleresult is None or points_elem is None:
            warnings.append(f"{xml_path.name}: object missing possibleresult/points, skipped")
            continue
        name_elem = possibleresult.find("name")
        raw_name = (name_elem.text or "").strip() if name_elem is not None else ""
        class_name = CLASS_NAME_MAP.get(raw_name)
        if class_name is None:
            warnings.append(f"{xml_path.name}: unmapped class name {raw_name!r}, skipped")
            continue
        try:
            points = parse_points(points_elem)
        except ConversionError as exc:
            warnings.append(f"{xml_path.name}: {exc}, skipped")
            continue
        coord_str = " ".join(f"{x:.2f} {y:.2f}" for x, y in points)
        lines.append(f"{coord_str} {class_name} 0")
    return lines, warnings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--xml-dir", required=True, type=Path, help="Directory of FAIR1M *.xml label files")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory for DOTA-style *.txt files")
    parser.add_argument("--report-path", type=Path, default=None, help="Optional JSON report of counts/warnings")
    parser.add_argument("--max-files", type=int, default=None, help="Convert only the first N sorted XML files.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    xml_files = sorted(args.xml_dir.glob("*.xml"))
    if args.max_files is not None:
        xml_files = xml_files[: args.max_files]
    if not xml_files:
        raise SystemExit(f"No .xml files found under {args.xml_dir}")

    total_objects = 0
    total_warnings: list[str] = []
    for xml_path in xml_files:
        lines, warnings = parse_fair1m_xml(xml_path)
        out_path = args.out_dir / f"{xml_path.stem}.txt"
        out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        total_objects += len(lines)
        total_warnings.extend(warnings)

    report = {
        "xml_dir": str(args.xml_dir),
        "out_dir": str(args.out_dir),
        "num_files": len(xml_files),
        "num_objects": total_objects,
        "num_warnings": len(total_warnings),
        "warnings_sample": total_warnings[:50],
    }
    print(json.dumps(report, indent=2))
    if args.report_path:
        args.report_path.parent.mkdir(parents=True, exist_ok=True)
        args.report_path.write_text(json.dumps({**report, "warnings": total_warnings}, indent=2), encoding="utf-8")

    if total_warnings:
        print(f"\n{len(total_warnings)} warnings — inspect before trusting this conversion at scale.")


if __name__ == "__main__":
    main()
