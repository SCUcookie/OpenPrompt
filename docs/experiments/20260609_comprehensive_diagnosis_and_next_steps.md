# GeoNexus-RSD Comprehensive Diagnosis And Next Steps - 2026-06-09

This note records the current route decision after DOTA2 S1/S2 evidence,
DIOR-R detector failures, bounded DIOR-R diagnostics, and the recent
OpenRSD-adjacent literature check.

## Decision

Use a stage-gated route, not a full S1-S4 pipeline.

Current paper-path evidence is narrow: DOTA2 S1 with RoI Transformer +
RemoteCLIP improves the DOTA2 RoI Transformer S0 gate from `0.6088/0.6090` to
`0.6177/0.6180` on `DOTA2_1024_500/ss_val`. Main DOTA2 S2 completed below
S1, and active S2 variants are also below S1 so far. DIOR-R detector training
is blocked because ORCNN, RoI Transformer, and RetinaNet produced invalid
non-finite or zero-valid evidence.

Do not launch S3/S4, pseudo-labeling, FAIR1M, or more detector training until:

- DOTA2 S2 is either rescued by final metrics or archived.
- DIOR-R has a finite detector baseline.
- Every paper-facing row has a config, checkpoint, split, metric log, and
  evidence classification.

## Direct Answers To Current Questions

1. Validation remains useful as a development gate, but not as final
   SOTA-table proof by itself. `DOTA2_1024_500/ss_val` can decide whether a
   module deserves more compute. Final paper tables need an official or clearly
   reproducible protocol with exact split, tiling, checkpoint, and metric
   source.

2. CCF-B/JSTARS is the practical first target. CCF-A/TGRS/TPAMI is
   aspirational until there is cross-dataset evidence and a strong comparison
   table. Current evidence is not enough for CCF-A.

3. Recent literature raises the bar. OpenRSD is the closest ICCV 2025 anchor;
   RS-MPOD, DisDop, SOAR, and VK-Det make weak prompt-only or unstable
   pseudo-label claims hard to sell without cross-dataset evidence.

4. One module can become formal evidence if it survives the gates. The current
   candidate is DOTA2 RoI Transformer + RemoteCLIP S1. It needs final protocol
   cleanup and DIOR-R validation before becoming a persuasive paper row.

5. Do not design a SOTA claim by choosing weak comparisons after the fact.
   Stabilize the method first, then compare against defensible reproducible
   baselines under the same dataset protocol.

6. Stop and diagnose stage instability instead of finishing a full pipeline.
   S2 is below S1 so far, and DIOR-R is non-finite. Keep S1 as the current
   best module and pause deeper stages.

## DOTA2 Evidence

All DOTA2 metrics below are on `DOTA2_1024_500/ss_val`.

| Run | Status | Metric |
| --- | --- | --- |
| RoI Transformer S0 | complete | `dota/mAP=0.6088`, `dota/AP50=0.6090` |
| S1 main, GPU 1 | complete | `0.6177 / 0.6180` |
| S1 LR `1e-4`, GPU 6 | complete | `0.5997 / 0.6000` |
| S1 LR `5e-5`, GPU 0 | complete | `0.6047 / 0.6050` |
| S2 main, GPU 1 | complete | epoch 4 `0.6038 / 0.6040`; epoch 8 `0.5892 / 0.5890`; final epoch 12 `0.5924 / 0.5920` |
| S2 LR `1e-4`, GPU 0 | active at 2026-06-09 19:58 CST | epoch 4 `0.6099 / 0.6100`; epoch 8 `0.6079 / 0.6080`; epoch 12 training near `[33950/39007]`; no final metric yet |
| S2 hierarchy weight `0.01`, GPU 6 | active at 2026-06-09 19:58 CST | epoch 4 `0.6035 / 0.6040`; epoch 8 `0.6044 / 0.6040`; epoch 12 validation near `[5300/6917]`; no final metric yet |
| S2 LR `5e-5`, GPU 5 | active diagnostic at 2026-06-09 19:58 CST | epoch 2 training near `[27600/39007]`; no validation yet |

S1 main remains the strongest DOTA2 GeoNexus result. No observed S2 metric has
exceeded S1.

Main S2 checkpoint:

`/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s2_hierarchy_reg_s1e12_20260608/epoch_12.pth`

Live S2 logs:

- GPU 0 LR `1e-4`: `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s2_hierarchy_reg_s1e12_lr1e4_20260608/20260608_194400/20260608_194400.log`
- GPU 6 hierarchy weight `0.01`: `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s2_hierarchy_reg_s1e12_hw1e2_20260608/20260608_194400/20260608_194400.log`
- GPU 5 LR `5e-5`: `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s2_hierarchy_reg_s1e12_lr5e5_20260609/20260609_153415/20260609_153415.log`

## DIOR-R Evidence

Detector training is blocked for paper evidence.

Invalid detector evidence:

- ORCNN R50 completed epoch 12 but validation stayed `0.0000/0.0000`, and
  training hit `loss: nan` from epoch 2 onward.
- ORCNN low-LR diagnostic first hit NaN at epoch 1 `[650/5862]`.
- RoI Transformer S0 first hit NaN at epoch 1 `[3375/5862]`.
- Rotated RetinaNet one-stage probe first hit `loss=inf`, `loss_bbox=inf` at
  epoch 1 `[1200/5862]`.

Bounded diagnostic evidence:

- Artifact: `/data5/2025/ldh/New/artifacts/dior_r_diagnostics_20260609_bounded_cpu.md`
- JSON: `/data5/2025/ldh/New/artifacts/dior_r_diagnostics_20260609_bounded_cpu.json`
- Scope: 200 `train_val` images/labels and 200 `test` images/labels.
- Cleared checks: sampled image decode, label parsing, DIOR-R class order,
  unknown-class check, sampled qbox-to-rbox conversion, and two dataloader
  samples for RoITrans/ORCNN configs.
- Not cleared yet: first non-finite batch capture and loss-component
  traceback.

Full label geometry/statistics scan:

- Artifact: `/data5/2025/ldh/New/artifacts/dior_r_diagnostics_20260609_full_geometry.md`
- JSON: `/data5/2025/ldh/New/artifacts/dior_r_diagnostics_20260609_full_geometry.json`
- Scope: all label files in `train_val` and `test`; fallback qbox-to-rbox
  geometry; declared `800x800` image bounds. This is not a second full image
  decode audit.
- `train_val`: `11725` label files, `68072` objects, `2` bad label files
  (`04137`, `07007`) with zero-area/invalid-size `ship` boxes,
  `1210` qboxes crossing declared bounds, `0` rbox centers outside bounds,
  `2` invalid rbox sizes. Rbox width min `1.0`, height min `0.0`, aspect max
  `91.50`.
- `test`: `11738` label files, `124445` objects, `2` bad label files
  (`15504`, `16734`) with zero-area/invalid-size `ship` boxes,
  `1322` qboxes crossing declared bounds, `4` rbox centers outside bounds,
  `2` invalid rbox sizes. Rbox width min `0.0`, height min `0.0`, aspect max
  `67.19`.

The diagnostics have not yet identified the detector NaN/Inf cause, but the
full label scan found records that should be fixed or filtered before any
relaunch. The next DIOR-R step is a bounded non-finite-loss catcher after that
data cleanup, not another unchanged detector launch.

## Route Gates

### DOTA2 Gate

1. Let GPU-0 LR `1e-4` and GPU-6 hierarchy-weight `0.01` S2 finish.
2. Parse final epoch-12 `dota/mAP` and `dota/AP50`.
3. Compare against S1 `0.6177/0.6180`, not just S0.
4. If neither final S2 exceeds S1 by a meaningful margin, archive S2 and pause
   hierarchy regularizer redesign.
5. Treat GPU-5 LR `5e-5` as route-gated diagnostic evidence unless it has a
   strong epoch-4 validation.

### DIOR-R Gate

1. Fix or filter the four zero-area/invalid-size label records and review the
   four test rbox centers outside declared bounds.
2. Decide whether qboxes crossing image bounds are expected DIOR-R edge boxes
   or need clipping/filtering in the dataset pipeline.
3. Run a bounded NaN/Inf catcher until the first non-finite loss.
4. Record first failing batch, image paths, bbox stats, labels, and loss
   components.
5. Relaunch one DIOR-R S0 only after the failure cause is fixed and first-loss
   plus bounded-iteration checks are finite.

### Paper Gate

1. If DOTA2 S1 and DIOR-R S1 are both positive after DIOR-R stabilization,
   proceed with a compact JSTARS/CCF-B-style story.
2. If DIOR-R S1 is not positive or S2 remains below S1, redesign the module
   around stronger multimodal/visual-prompt or domain-prior ideas instead of
   extending S3/S4.
3. Treat CCF-A/TGRS/TPAMI as aspirational until there is cross-dataset
   evidence and at least one strong comparison table.

## Literature Context

Recent sources already checked by the previous planning pass:

- OpenRSD: https://openaccess.thecvf.com/content/ICCV2025/html/Huang_OpenRSD_Towards_Open-prompts_for_Object_Detection_in_Remote_Sensing_Images_ICCV_2025_paper.html
- RS-MPOD: https://arxiv.org/abs/2602.01954
- DisDop: https://arxiv.org/abs/2605.24639
- SOAR: https://ojs.aaai.org/index.php/AAAI/article/view/37671
- VK-Det: https://arxiv.org/abs/2511.18075

Before proposing a new route, refresh the literature tracker:

`/data5/2025/ldh/New/docs/literature/20260607_openrsd_related_recent_papers.md`

## Test Plan

- Documentation: verify `Q&A.md`, `New/PROJECT_INSTRUCTIONS.md`, and this
  note render as Markdown and contain the 2026-06-09 route gate.
- DOTA2: scan completed and live S2 logs for `Traceback`, OOM, data errors,
  true `nan`/`inf` losses, and `KeyboardInterrupt`; parse all `dota/mAP` and
  `dota/AP50` lines.
- DIOR-R: require full-scan JSON/MD outputs plus a first-nonfinite diagnostic
  record before any detector relaunch.
- Paper evidence: every table row must link config, checkpoint, dataset split,
  metric log, and whether it is paper-facing or diagnostic.
