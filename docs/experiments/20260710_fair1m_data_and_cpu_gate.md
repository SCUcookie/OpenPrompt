# FAIR1M Data And CPU Gate (2026-07-10)

## Outcome

FAIR1M is present on server storage, but the S0 experiment gate is closed.
No training config, train-step run, screen, or GPU process was created.

## Repository State

- `New/main` was fast-forwarded from `943a4e97` to verified upstream
  `bb491b93` (`--ff-only` ancestry confirmed).
- The pre-existing dirty `third_party/Pi-Seg` working tree was preserved.
- `OpenRSD` was inspected read-only and remains locally dirty/untouched.
- Three unwritable reflogs owned by `nobody:nogroup` were retained as
  timestamped `.nobody_backup_20260710` files under `.git/logs`; Git recreated
  current reflogs under the repository owner.

## Dataset Inventory

Raw FAIR1M 1.0:

- Root: `/data2/2023/lcs/xyun/FAIR1M1.0` (`74G`).
- Train: `16488` images, `16488` XML files, `16488` existing DOTA txt files.
- Image and label stems reconcile exactly.
- Existing txt labels contain all 37 official classes and `393466` objects.

Pre-tiled staging:

- Root: `/data5/2025/temp/Dataset/FAIR1M_2_800_400` (`151G`).
- Train image split archive contains `134486` PNG entries.
- No train `annfiles`/`labelTxt` archive or extracted label directory was found.
- `ss_val/images.zip` and `ss_val/annfiles.zip` each contain `10970` matching
  stems; validation labels cover all 37 normalized OpenRSD class names.
- This staging is not a usable train/validation dataset until the tiled train
  annotations are restored or regenerated and validated.

No FAIR1M download was started.

## Conversion Sample

The converter now supports `--max-files` and rejects non-finite, zero-area, or
zero-edge polygons instead of silently emitting them. A non-destructive real
sample was written under `artifacts/fair1m_xml_sample_20260710/`:

- XML files: `20`
- Converted valid objects: `831`
- Warnings: `1`
- Warning: `1.xml` contains a polygon with area `0` and four zero-length edges.

The required zero-warning gate therefore failed; no full conversion was run.

## Geometry Diagnostic

Artifacts:

- `artifacts/fair1m_diagnostics_20260710_full_raw.json`
- `artifacts/fair1m_diagnostics_20260710_full_raw.md`

Results over all existing raw txt labels:

- Objects: `393466`
- Unknown classes: `0`
- Missing taxonomy classes: `0`
- Image/label stem mismatches: `0`
- Malformed/degenerate records: `173`
- MMRotate qbox-to-rbox failures: `0`
- Invalid converted rboxes: `0`
- Decoded image sample: `100`, decode failures: `0`
- Out-of-bounds qboxes in that decoded sample: `46`

The baseline config and taxonomy contain the same semantic 37-class set after
normalizing underscores to hyphens. Their orders are different. This must be
resolved before detector checkpoints or embeddings are paired with classifier
indices.

## RemoteCLIP Artifact

Path:
`artifacts/generated/remoteclip_vit_b32_fair1m_prompt_embeddings.pt`

Generated on CPU with
`/data5/2025/ldh/OpenRSD/checkpoints/remoteclip/RemoteCLIP-ViT-B-32.pt`
(`605208421` bytes) in the `geonexus_vlm` environment.

Validation:

- Keys: `backend`, `checkpoint`, `class_names`, `embeddings`, `model_name`,
  `prompts`
- Shape: `[37, 512]`
- Finite: yes
- Row norms: `0.9999998212` to `1.0000001192`
- Taxonomy order: exact
- Prompt order: exact

## Required Next Action

1. Restore or generate tiled train annotations matching the existing `134486`
   image tiles in a new non-destructive dataset root.
2. Filter the 173 degenerate raw objects before tiling and establish a written
   bounds policy; rerun the full geometry report with zero malformed records.
3. Choose one canonical 37-class order and apply it consistently to config,
   tiled labels, and RemoteCLIP artifact.
4. Only then create the S0 runtime config and run the bounded 1000-step
   `train_step` diagnostic.
5. A GPU smoke launch remains conditional on the diagnostic and three genuinely
   idle GPU polls.
