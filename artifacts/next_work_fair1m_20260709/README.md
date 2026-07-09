# Next-Work Staging: FAIR1M Supporting Files (2026-07-09)

Correction from an earlier pass in this session: I mistakenly started
downloading the FAIR1M dataset itself. **The 500GB FAIR1M dataset is already
staged on the server** — that download was killed and the partial 2.7GB
fetched so far was deleted. This directory now contains only the genuinely
*missing* small support files needed to actually use that dataset, not the
dataset itself.

## Why FAIR1M is the right next step

Per the project's own plan, not a guess:

- `docs/setup/complete_experiment_plan.md` Benchmark Matrix: FAIR1M is
  "optional fine-grained stretch... use only after DOTA2 and DIOR-R are
  stable."
- `PROJECT_INSTRUCTIONS.md` Experiment Sequence item 4: "FAIR1M: stretch
  evidence after DOTA2 and DIOR-R are stable... for fine-grained hierarchy
  claims only."

DOTA2 (S0-S3) and DIOR-R (S0-S4, S4 closed) are both stable as of
2026-07-09, so FAIR1M is the next dataset gate to open.

## What was actually missing (and is staged here / in the repo now)

**1. `assets/hierarchies/fair1m_remote_sensing_taxonomy.json`** (repo root,
not this directory) — the 37-class taxonomy (5 super-categories: ship,
vehicle, airplane, court, road), in the same schema as
`dota2_remote_sensing_taxonomy.json` / `dior_r_remote_sensing_taxonomy.json`.
Class names are verified against the official FAIR1M taxonomy (arXiv
2103.05569, cross-checked with `torchgeo.datasets.fair1m`'s classes dict).
This is what a FAIR1M S1 RemoteCLIP prompt-embedding generation pass needs
and did not exist before. Synonym/confusing-class/scene-prior/geometry/
negative-cue fields are a first-pass draft, not yet checked against real
FAIR1M imagery the way the DOTA2/DIOR-R files were — review before treating
as final.

**2. `scripts/convert_fair1m_xml_to_dota_txt.py`** (repo root, not this
directory) — a FAIR1M-XML-to-DOTA-txt label converter. This is needed
because **MMRotate has no native FAIR1M dataset class**; the documented
community practice (confirmed via MMRotate's own customize-dataset docs and
community write-ups) is to convert FAIR1M's PASCAL-VOC-style XML labels to
DOTA-style txt offline and reuse the standard DOTA tooling/configs — exactly
the pattern this repo's dataloader already expects for DOTA2/DIOR-R. Without
this, the already-staged 500GB of raw FAIR1M XML labels cannot be read by
the existing training pipeline.

- XML schema was verified against `torchgeo.datasets.fair1m.parse_pascal_voc`
  (`object/possibleresult/name` for class, `object/points/point` for the
  4-or-5-point polygon, comma-separated `x,y` text, FAIR1M repeats the first
  point to close the polygon).
- Class-name mapping (official XML strings like `"Small Car"`,
  `"Boeing737"` → this repo's kebab-case taxonomy names) is built into the
  script and cross-checked against the taxonomy file above.
- **Tested against a synthetic sample XML matching the confirmed schema**
  (3 objects: one 5-point closed polygon, one 4-point polygon, one
  deliberately-unmapped class name) — correctly converts the two valid
  objects to DOTA-txt lines and *flags* (does not silently drop) the unmapped
  one via a JSON warnings report.
- **Not yet run against real FAIR1M XML files** — the dataset lives on the
  server, not this machine. Run it there first against a small sample
  (a handful of XML files) and inspect `--report-path`'s warnings before
  converting the full 500GB label set, in case the real files have edge
  cases (different point counts, unmapped class strings, missing tags) this
  synthetic test didn't cover.

Usage on the server:

```bash
python scripts/convert_fair1m_xml_to_dota_txt.py \
  --xml-dir /path/to/FAIR1M/train/labelXml \
  --out-dir /path/to/FAIR1M/train/labelTxt \
  --report-path /path/to/FAIR1M/train/convert_report_20260709.json
```

## What's still genuinely missing (not staged, needs a decision)

- A RemoteCLIP prompt-embedding artifact for FAIR1M (like
  `remoteclip_vit_b32_dota2_prompt_embeddings.pt`) — trivial to generate
  once the taxonomy file above is on the server, since the RemoteCLIP
  checkpoint is already staged there
  (`/data5/2025/ldh/OpenRSD/checkpoints/remoteclip/RemoteCLIP-ViT-B-32.pt`
  per `docs/setup/complete_experiment_plan.md`). Not generated here because
  it needs the RemoteCLIP model running, which this laptop doesn't have
  staged.
- A bounded geometry/train-step diagnostic pass on the converted FAIR1M
  labels, mirroring what DIOR-R needed
  (`New/scripts/diagnose_dior_r_geometry_and_targets.py`-style) before any
  detector training — FAIR1M's converted labels have not been validated
  against this repo's loader at all yet.

## Unrelated to FAIR1M (from the earlier blocked-file research pass, kept for reference)

Segmentation-lane files (Pi-Seg/ConInfer/RSKT-Seg) are documented in
`PROJECT_INSTRUCTIONS.md`'s 2026-07-09 status entry and
`artifacts/blocked_files_20260709/` — that lane stays paused/secondary
behind this core DOTA2/DIOR-R/FAIR1M route by default.
