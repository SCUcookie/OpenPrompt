# Server Artifact Obstacles And Remote Handoff - 2026-07-20

## Purpose

This note records what is and is not portable through the Git remote. The
repository is `ssh://git@ssh.github.com:443/SCUcookie/OpenPrompt.git`, branch
`main`, checked at commit `33c00ae5` (`0720`). Git can carry code, configs,
small metadata, experiment notes, and generated JSON/CSV evidence. It cannot
carry the host-local FAIR1M dataset, detector checkpoints, runtime workdirs,
or the intentionally local TGRS manuscript tree.

The current working tree before this handoff contains the pre-existing dirty
submodule state `third_party/Pi-Seg` and the uncommitted July 20 A1/A1-route
documentation files. Do not reset or overwrite the submodule state.

## Large Or Host-Local Artifacts

### FAIR1M dataset

The validated FAIR1M root is external to this repository:

`/data2/2023/lcs/xyun/FAIR1M_2_800_400_sanitized_20260713`

The current `du -sh` report is `542M` on this server. It contains `train/`,
`ss_val/`, and `reports/`; the full source/archive material is not in Git.
The validation record says the root contains `208927` train image/label pairs,
`10970` ss_val pairs, `1785001` active train objects, and `199347` active
validation objects. The geometry gate found zero active malformed/zero-area
records, zero unknown classes, zero invalid MMRotate rboxes, and no decode
failures in its 1000-representative checks. The expected `6513` rejected raw
train records remain excluded and recorded in the reconstruction report.

Required recovery checks after moving or remounting the dataset:

1. Confirm exact train and ss_val image/annotation stems.
2. Re-run `scripts/diagnose_fair1m_geometry_and_targets.py` against the
   canonical taxonomy and S0 config.
3. Check representative 533/800/1024 tiles and active rboxes before training.
4. Use the absolute `data_root` in the FAIR1M configs or update it explicitly;
   do not silently substitute the old 20260710 root.

### FAIR1M checkpoints

Training checkpoints are under the external OpenRSD worktree and are not in
this repository:

`/data5/2025/ldh/OpenRSD/work_dirs/geonexus_fair1m/`

The six completed S0/TPC-S1 epoch-12 checkpoints are approximately 0.85 GB
each. The selected TPC/S1 checkpoints are:

| Artifact | Approx. bytes | Meaning |
|---|---:|---|
| `roi_trans_tpc_s1_rep3407_20260717/epoch_12.pth` | 866,192,415 | S1 rep3407 final/source for any controlled continuation |
| `roi_trans_tpc_s1_rep4407_20260717/epoch_12.pth` | 866,180,703 | S1 rep4407 final/source |
| `roi_trans_tpc_s1_rep5407_20260717/epoch_12.pth` | 866,182,175 | S1 rep5407 final/source |

The matching S0 epoch-12 checkpoints are approximately 853 MB each. Epoch-4
and epoch-8 checkpoints also exist. The exact training provenance is in each
replica's `launch_provenance.txt` and the completion records
`docs/experiments/20260717_fair1m_s0_completion.md` and
`docs/experiments/20260720_fair1m_tpc_s1_completion.md`.

Do not commit checkpoints. To move one, use a separately managed artifact
transfer and verify its SHA-256 before use. The ResNet-50 initialization used
by S0 is identified by SHA-256
`0676ba61b6795bbe1773cffd859882e5e297624d384b6993f7c9e683e722fb8a`.
SHA-256 values for the large detector checkpoints were not recomputed during
this documentation pass; do not infer integrity from file size alone.

### Runtime logs and predictions

The FAIR1M training workdirs contain multi-megabyte runtime logs and scalar
JSON files, plus evaluation `preds.pkl` outputs. These are useful for
reconciliation but are host-local and should be transferred selectively, not
committed wholesale. Preserve at minimum the launch provenance, final metric
JSON, selected evaluator JSON, failure-scan result, and the exact config.

The FAIR1M S1 paper-evaluation outputs are under:

`/data5/2025/ldh/OpenRSD/work_dirs/paper_eval_20260720/`

They include `preds.pkl`, copied configs, evaluator JSON, and evaluator logs.
The evaluator logs reached `Epoch(test) [5485/5485]` and reconciled with the
training metrics at evaluator rounding.

## Git-Excluded Repository Artifacts

The current `.gitignore` excludes `artifacts/generated/`, `checkpoints/`,
`*.log`, `*.pyc`, build/cache directories, and other generated outputs.
This is intentional for large or reproducible artifacts, but it means a fresh
clone will not contain every file referenced by the configs.

The canonical FAIR1M prompt artifact currently exists only on this server at:

`/data5/2025/ldh/New/artifacts/generated/remoteclip_vit_b32_fair1m_prompt_embeddings_canonical.pt`

It is approximately 79 KB and loads as a dictionary with `embeddings` shape
`[37,512]`, finite `float32` values, canonical `class_names`, and prompts.
It does **not** contain `relation_matrix`.

The source taxonomy is tracked and portable:

`assets/hierarchies/fair1m_remote_sensing_taxonomy.json`

It defines 37 classes and parent labels, but its provenance explicitly says
the synonyms, confusing classes, scene priors, geometry, and negative cues
were first-pass metadata. A future hierarchy artifact must be generated from
this taxonomy, reviewed, saved outside Git if large, and recorded with a
checksum and generation command.

## Current FAIR1M S2 Blocker

The implemented head
`/data5/2025/ldh/OpenRSD/geonexus_mmrotate/hierarchy_prompt_bbox_head.py`
requires all of the following:

- prompt embeddings matching `num_classes=37`;
- a `37x37` `relation_matrix` in the prompt artifact;
- an S2 config using `HierarchyPromptShared2FCBBoxHead` in both cascade stages;
- source checkpoints from the completed FAIR1M TPC/S1 replicas;
- a config/model/data gate and finite 1,000-step diagnostic before launch.

No FAIR1M S2 config or FAIR1M hierarchy relation matrix currently exists.
The DIOR-R/DOTA2 relation implementation cannot be copied as a FAIR1M
relation definition because their class sets and taxonomy semantics differ.
Therefore no S2 GPU training was launched on 2026-07-20 despite seven idle
RTX 4090s. Inventing a relation matrix would make the result irreproducible
and would not be a legitimate continuation of the archived route.

## Other Portability Boundaries

### TGRS manuscript

The TGRS source tree is intentionally local-only and remains untracked by
project policy. This workspace does not expose a writable copy of
`_local_archive_20260601_pull_backup/docs/TGRS/`; do not create a replacement
paper directory or add the local archive to Git. The route note, A1 JSON/CSV
records, and A2/A3 source paths are the portable handoff. On the host containing
the manuscript, copy the qualitative PNG, replace the A1/A3 placeholders,
compile, and run the number-consistency/undefined-reference audit.

### Pi-Seg submodule

`third_party/Pi-Seg` is a dirty submodule. Its index records commit
`6a1a25a84bf81c2cbd2a103594a4c01d376de3d6`, while the worktree reports a
modified/uncommitted state. This state predates the present handoff and was
left untouched. Recover it through the submodule's own Git history or a
separate patch/export; do not use the parent repository's reset/checkout to
clean it.

### A1/A2/A3 evidence

The portable A1 outputs are the six files beginning
`docs/experiments/20260720_dior_r_perclass_` (JSON and CSV). They contain all
20 DIOR-R classes and reproduce baseline `65.44`, GeoNexus SCA `69.92`, and
OrientedFormer parsed table mean `68.84`; the OrientedFormer runtime reports
`0.6883/0.6880`. The A2 qualitative strip and A3 efficiency JSONs remain on
the external OpenRSD worktree. Their source paths and metrics are recorded in
`docs/experiments/20260720_fair1m_route_review_and_next_analysis.md`.

## Fresh-Clone Checklist

After pushing the tracked updates and cloning elsewhere:

1. Read `PROJECT_INSTRUCTIONS.md` and this note.
2. Confirm the OpenRSD code checkout, MMRotate environment, and custom
   `geonexus_mmrotate` package are present.
3. Restore or remount the FAIR1M dataset at the configured external path.
4. Restore the ResNet-50 initialization, FAIR1M prompt artifact, and any
   selected detector checkpoint through artifact transfer, with SHA-256 logs.
5. Re-run the data/config gates before any GPU process.
6. For FAIR1M S2, first add and review the relation-matrix artifact and S2
   config; then run the 1,000-step diagnostic and record full provenance.

The Git remote is the source for code and notes only. The server artifact
inventory and this document are the source for reconstructing the experiments.
