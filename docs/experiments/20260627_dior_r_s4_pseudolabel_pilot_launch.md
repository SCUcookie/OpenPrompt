# 2026-06-27 DIOR-R S4 Pseudo-Label Pilot Launch

## Scope

Open a controlled DIOR-R S4 pseudo-label pilot after the S3 long88 archive.
This is not another S3 continuation. Pseudo-label generation treats sanitized
`DIOR_R_dota/train_val` ground truth as hidden; the ground truth is used only
for offline quality audit.

The route is gated: launch S4 training only if the pseudo-label audit improves
over confidence-only filtering at matched kept-box count, avoids catastrophic
class false-positive expansion, retains usable recall without degrading
high-confidence precision, and has a clean failure scan.

## Teachers

| Teacher | Checkpoint | Prior metric |
| --- | --- | ---: |
| `s3_original_best_rep0_e8` | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dior_r/roi_trans_remoteclip_s3_scene_adapter_s2e12_rep0_20260614/epoch_8.pth` | `0.6991876364` mAP |
| `s3_long60_rep0_e51` | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dior_r/roi_trans_remoteclip_s3_scene_adapter_stability_long60_rep0_20260624/epoch_51.pth` | `0.698892` mAP |
| `s3_long88_rep2_e88` | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dior_r/roi_trans_remoteclip_s3_scene_adapter_stability_long88_rep2_20260625/epoch_88.pth` | `0.6982299089` final mAP |

Primary confidence-only teacher: `s3_original_best_rep0_e8`.

## Artifacts

New export/audit script:

`/data5/2025/ldh/New/scripts/export_audit_dior_r_s4_pseudolabels.py`

Planned audit output directory:

`/data5/2025/ldh/New/artifacts/dior_r_s4_pseudolabel_pilot_20260627`

The script writes:

- DOTA-style filtered pseudo labels per policy under `*/labelTxt/`.
- `audit.json` with kept boxes, precision, recall, class-pair confusion,
  hierarchy consistency, scene-consistency proxy, per-class statistics, and
  gate fields.
- `audit.md` with the pseudo-label quality table and GeoReason ladder:
  R0 class/prompt grounding, R1 confusing-class relation, R2 scene-context
  consistency, and R3 final pseudo-label decision quality.

## Candidate Policies

| Policy | Definition |
| --- | --- |
| `confidence_only` | primary teacher predictions filtered by confidence |
| `hierarchy_scene` | primary teacher predictions filtered by confidence and dominant DIOR-R scene-group consistency proxy |
| `teacher_agreement_2of3` | same-class boxes retained when at least two of the three teachers agree by rotated polygon IoU |

The scene/hierarchy path is a diagnostic proxy, not a claim of direct
Vision Banana or OneReason reproduction.

## Preflight And Prediction Export Launch Trail

Preflight on `2026-06-27 11:01 CST`:

- Source checkpoints confirmed present.
- `screen -ls` showed only the persistent
  `s0_result_log_monitor_20260603` before launch.
- GPU idle gate: GPUs 0-5 were idle; GPU 6 was occupied by another user's
  process. The default mapping was used:
  `s3_original_best_rep0_e8 -> GPU 0`, `s3_long60_rep0_e51 -> GPU 1`,
  `s3_long88_rep2_e88 -> GPU 2`.
- Export config parse check passed:
  `work_dirs/geonexus_dior_r/s4_pseudolabel_pilot_20260627/roi-trans-le90_r50_fpn_remoteclip-s3-pseudolabel-export-20260627_dior_r.py`.
  The config uses `ann_file=''`, `train_val/images/`, `test_pipeline`, and
  `test_evaluator=[]` so prediction generation does not load `labelTxt`.

| Teacher | GPU | Screen | Prediction pkl | Launch log |
| --- | ---: | --- | --- | --- |
| `s3_original_best_rep0_e8` | 0 | `dior_r_s4_pl_export_s3orig_e8_20260627_gpu0` | `/data5/2025/ldh/New/artifacts/dior_r_s4_pseudolabel_pilot_20260627/predictions/s3_original_best_rep0_e8.pkl` | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dior_r/s4_pseudolabel_pilot_20260627/logs/export_s3_original_best_rep0_e8_gpu0.log` |
| `s3_long60_rep0_e51` | 1 | `dior_r_s4_pl_export_long60_e51_20260627_gpu1` | `/data5/2025/ldh/New/artifacts/dior_r_s4_pseudolabel_pilot_20260627/predictions/s3_long60_rep0_e51.pkl` | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dior_r/s4_pseudolabel_pilot_20260627/logs/export_s3_long60_rep0_e51_gpu1.log` |
| `s3_long88_rep2_e88` | 2 | `dior_r_s4_pl_export_long88_e88_20260627_gpu2` | `/data5/2025/ldh/New/artifacts/dior_r_s4_pseudolabel_pilot_20260627/predictions/s3_long88_rep2_e88.pkl` | `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dior_r/s4_pseudolabel_pilot_20260627/logs/export_s3_long88_rep2_e88_gpu2.log` |

Startup acceptance at `2026-06-27 11:02 CST`:

- Screens detached and alive.
- GPU residency confirmed on GPUs 0, 1, and 2.
- Logs confirmed intended checkpoint loads for all three teachers.
- Each export reached `Epoch(test) [450/5863]`.
- Scoped failure scan across the three export logs was clean for `Traceback`,
  CUDA OOM, `out-of-memory`, `out of memory`, `libpng`, `CRC`, `NoneType`,
  `ValueError`, `KeyboardInterrupt`, `loss: nan`, `loss: inf`,
  `grad_norm: nan`, and `grad_norm: inf`.

## Pending Audit And Gate

Audit command completed after all three prediction pickles were written:

```bash
PYTHONNOUSERSITE=1 /data1/anaconda3/envs/zwl_mmrotate/bin/python \
  /data5/2025/ldh/New/scripts/export_audit_dior_r_s4_pseudolabels.py \
  --prediction s3_original_best_rep0_e8=/data5/2025/ldh/New/artifacts/dior_r_s4_pseudolabel_pilot_20260627/predictions/s3_original_best_rep0_e8.pkl \
  --prediction s3_long60_rep0_e51=/data5/2025/ldh/New/artifacts/dior_r_s4_pseudolabel_pilot_20260627/predictions/s3_long60_rep0_e51.pkl \
  --prediction s3_long88_rep2_e88=/data5/2025/ldh/New/artifacts/dior_r_s4_pseudolabel_pilot_20260627/predictions/s3_long88_rep2_e88.pkl \
  --primary-teacher s3_original_best_rep0_e8 \
  --gt-label-dir /data5/2025/ldh/OpenRSD/data/DIOR_R_dota_sanitized_invalidsize_20260612/train_val/labelTxt \
  --output-dir /data5/2025/ldh/New/artifacts/dior_r_s4_pseudolabel_pilot_20260627
```

Audit artifacts:

- `/data5/2025/ldh/New/artifacts/dior_r_s4_pseudolabel_pilot_20260627/audit.md`
- `/data5/2025/ldh/New/artifacts/dior_r_s4_pseudolabel_pilot_20260627/audit.json`

Quality table:

| Policy | Kept boxes | Precision | Recall | Hierarchy consistency | Scene consistency | Matched confidence precision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `confidence_only` | 66260 | 0.917658 | 0.893257 | 1.000000 | 0.929374 | n/a |
| `hierarchy_scene` | 61988 | 0.928793 | 0.845806 | 1.000000 | 1.000000 | 0.955508 |
| `teacher_agreement_2of3` | 64512 | 0.944274 | 0.894917 | 1.000000 | 0.938491 | 0.934508 |

Gate:

- precision improves over confidence-only at matched kept-box count:
  `True`, via `teacher_agreement_2of3`.
- no catastrophic class false-positive expansion: `True`.
- usable recall without high-confidence precision degradation: `True`.
- failure scan clean: `True`.
- S4 launch recommended: `True`.

Accepted S4 policy: `teacher_agreement_2of3`.

Pseudo-label data root:

`/data5/2025/ldh/OpenRSD/data/DIOR_R_dota_s4_pseudo_agreement_20260627`

The pseudo-label data root is separate from the sanitized source data. It
symlinks train/test images and test labels from
`DIOR_R_dota_sanitized_invalidsize_20260612`, and symlinks
`train_val/labelTxt` to the accepted agreement pseudo labels.

## S4 Short-Pack Launch

Config print/parse checks passed for all three generated S4 configs before
launch. All three use:

- pseudo-label data root:
  `data/DIOR_R_dota_s4_pseudo_agreement_20260627/`
- source checkpoint:
  `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dior_r/roi_trans_remoteclip_s3_scene_adapter_s2e12_rep0_20260614/epoch_8.pth`
- `max_epochs=12`
- `val_interval=1`
- `lr=2.5e-5`
- `resume=False`

Launched at about `2026-06-27 11:18 CST`.

| Replica | Seed | GPU | PID | Screen | Config | Launch log | Runtime log |
| --- | ---: | ---: | ---: | --- | --- | --- | --- |
| rep23407 | 23407 | 0 | `3914126` | `dior_r_s4_pseudo_agreement_rep23407_20260627_gpu0` | `work_dirs/geonexus_dior_r/roi_trans_remoteclip_s4_pseudo_agreement_rep23407_20260627/roi-trans-le90_r50_fpn_remoteclip-s4-pseudo-agreement-rep23407-20260627_dior_r.py` | `work_dirs/geonexus_dior_r/roi_trans_remoteclip_s4_pseudo_agreement_rep23407_20260627/launch_20260627_gpu0.log` | `work_dirs/geonexus_dior_r/roi_trans_remoteclip_s4_pseudo_agreement_rep23407_20260627/20260627_111850/20260627_111850.log` |
| rep24407 | 24407 | 1 | `3914124` | `dior_r_s4_pseudo_agreement_rep24407_20260627_gpu1` | `work_dirs/geonexus_dior_r/roi_trans_remoteclip_s4_pseudo_agreement_rep24407_20260627/roi-trans-le90_r50_fpn_remoteclip-s4-pseudo-agreement-rep24407-20260627_dior_r.py` | `work_dirs/geonexus_dior_r/roi_trans_remoteclip_s4_pseudo_agreement_rep24407_20260627/launch_20260627_gpu1.log` | `work_dirs/geonexus_dior_r/roi_trans_remoteclip_s4_pseudo_agreement_rep24407_20260627/20260627_111850/20260627_111850.log` |
| rep25407 | 25407 | 2 | `3914122` | `dior_r_s4_pseudo_agreement_rep25407_20260627_gpu2` | `work_dirs/geonexus_dior_r/roi_trans_remoteclip_s4_pseudo_agreement_rep25407_20260627/roi-trans-le90_r50_fpn_remoteclip-s4-pseudo-agreement-rep25407-20260627_dior_r.py` | `work_dirs/geonexus_dior_r/roi_trans_remoteclip_s4_pseudo_agreement_rep25407_20260627/launch_20260627_gpu2.log` | `work_dirs/geonexus_dior_r/roi_trans_remoteclip_s4_pseudo_agreement_rep25407_20260627/20260627_111850/20260627_111850.log` |

Startup acceptance at `2026-06-27 11:20 CST`:

- screens detached and alive for all three replicas.
- GPU residency confirmed on GPUs 0, 1, and 2.
- `ps -p 3914126,3914124,3914122 -o pid,ppid,user,cmd --forest`
  confirmed each Python process and matching config.
- each log confirmed loading
  `roi_trans_remoteclip_s3_scene_adapter_s2e12_rep0_20260614/epoch_8.pth`.
- each log reached at least `Epoch(train) [1][450/5847]`.
- scoped failure scan across all three launch logs was clean for `Traceback`,
  CUDA OOM, `out-of-memory`, `out of memory`, `libpng`, `CRC`, `NoneType`,
  `ValueError`, `KeyboardInterrupt`, `loss: nan`, `loss: inf`,
  `grad_norm: nan`, and `grad_norm: inf`.

Expected runtime from startup log ETA is about `3.5-4` hours. Completion note
must report best and final `dota/mAP` separately and must not make a
final-only superiority claim.
