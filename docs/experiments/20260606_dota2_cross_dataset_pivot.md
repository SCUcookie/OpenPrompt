# DOTA2 Cross-Dataset Pivot

Date: 2026-06-06

## Decision

GeoNexus-RSD is pivoted from a DOTA v1.5-first paper route to a DOTA2-centered
route with DIOR-R as the required second dataset and FAIR1M as optional
fine-grained evidence.

DOTA v1.5 remains useful for debugging implementation paths, but it is now
diagnostic/archive-only. The current DOTA v1.5 GeoNexus numbers around
`0.38` mAP must not be used as headline paper evidence.

## Process Record

- Stopped DOTA v1.5 low-LR S2 refinement screen
  `geonexus_s2_hierarchy_refine_s2e4_lr5e5_20260606_gpu1`.
- Classification: stopped by research pivot, not failed evidence.
- Preserved workdir:
  `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s2_hierarchy_reg_refine_s2e4_lr5e5_20260606`.
- Preserved launch log:
  `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s2_hierarchy_reg_refine_s2e4_lr5e5_20260606/launch_20260606_gpu1.log`.
- Last observed log before stop was epoch 3 training progress, around
  `[1290/1410]`, with LR `5e-5`.
- After stopping, `screen -ls` showed no active GeoNexus DOTA v1.5 training
  screen and GPU 1 was free at `14 MiB`, `0%`.

## DOTA2 Baseline Evidence

| Detector | Status | DOTA2 split | mAP/AP50 | Source |
| --- | --- | --- | --- | --- |
| RoI Transformer | complete | `DOTA2_1024_500/ss_val` | `0.6088/0.6090` | `docs/experiments/20260603_s0_dota2_roi_trans_validpng_metrics.json` |
| Oriented R-CNN | complete | `DOTA2_1024_500/ss_val` | `0.5973/0.5970` | `docs/experiments/20260605_dota2_baseline_status.md` |
| S2ANet | complete | `DOTA2_1024_500/ss_val` | `0.5869/0.5870` | `docs/experiments/20260605_dota2_baseline_status.md` |
| RTMDet-M | complete | `DOTA2_1024_500/ss_val` | `0.3312/0.3310` | `PROJECT_INSTRUCTIONS.md` 2026-06-06 status |
| R3Det-KFIoU | active | `DOTA2_1024_500/ss_val` | pending | screen `s0_dota2_r3det_kfiou_validpng_bs1_resume_20260605_gpu5` |
| RTMDet-L | active, reassess after validation | `DOTA2_1024_500/ss_val` | pending | screen `s0_dota2_rtmdet_l_validpng_bs1_resume_20260606_gpu6` |

## Literature Anchors

- RemoteCLIP supports using remote-sensing-specific vision-language models for
  semantic alignment: `https://arxiv.org/abs/2306.11029`.
- SkyScript/SkyCLIP supports large-scale remote-sensing image-text pretraining:
  `https://arxiv.org/abs/2312.12856`.
- GeoRSCLIP is another remote-sensing vision-language foundation reference:
  `https://arxiv.org/abs/2306.11300`.
- OpenRSD motivates open-prompt remote-sensing detection across datasets:
  `https://arxiv.org/abs/2503.06146`.
- DOTA remains the oriented-detection benchmark family:
  `https://arxiv.org/abs/1711.10398`.
- DIOR justifies a second object-detection dataset for cross-dataset evidence:
  `https://arxiv.org/abs/1909.00133`.
- FAIR1M justifies fine-grained hierarchy evaluation when compute allows:
  `https://arxiv.org/abs/2103.05569`.

## Compute Priorities

1. Let DOTA2 R3Det finish.
2. Reassess DOTA2 RTMDet-L after the next validation; stop if still near
   `0.35`.
3. Archive the DOTA2 baseline table with dataset version, split, config,
   checkpoint, log, metric source, and notes about valid-PNG filtering.
4. Port only the most defensible GeoNexus module first: hierarchy-aware prompt
   scoring or hierarchy regularization on the strongest stable DOTA2 detector.
5. Do not launch DOTA2 S3/S4 until S1/S2 improves, stabilizes, or provides a
   clear complementary analysis against the strongest closed-set baseline.
6. Smoke DIOR-R loader/config with local `DIOR_R_dota/train_val` and
   `DIOR_R_dota/test`.
7. Run a DIOR-R Oriented R-CNN or RoI Transformer baseline.
8. Run the same minimal GeoNexus S1/S2 module on DIOR-R.
9. Treat FAIR1M as stretch evidence after DOTA2 and DIOR-R are stable.

## Acceptance Criteria

- `screen -ls` has no active GeoNexus DOTA v1.5 training screen.
- GPU 1 is free after stopping the low-LR refinement.
- `PROJECT_INSTRUCTIONS.md` contains the 2026-06-06 research pivot.
- `docs/setup/complete_experiment_plan.md` names DOTA2 as primary and DIOR-R as
  required cross-dataset validation.
- No formal manuscript table uses DOTA v1.5 GeoNexus `0.38` results as headline
  evidence.
