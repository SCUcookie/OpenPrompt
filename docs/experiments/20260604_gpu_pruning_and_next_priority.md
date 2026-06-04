# 2026-06-04 GPU Pruning And Next Priority

Date: 2026-06-04

Machine: `/data5/2025/ldh`

Dataset versions: DOTA v1.5 reduced tiled split; DOTA2 `DOTA2_1024_500` valid-PNG train with `ss_val` validation for S0 baselines

Metric implementation: MMRotate `DOTAMetric`

## Purpose

Record the 2026-06-04 server pruning decision and the next experiment priority.
Large checkpoints, raw logs, datasets, and training directories remain under
`/data5/2025/ldh/OpenRSD/` and are not copied into `New/`.

The pruning goal was to keep the GeoNexus paper path moving by freeing lower
priority DOTA2 baseline GPUs after checkpoint confirmation, while preserving
the active runs that are still strategically useful.

## Active Kept Runs

- GeoNexus S1 rerun remained active on GPU 5 in screen
  `geonexus_s1_rerun_retry2_20260604_gpu5`, PID `4120620`.
  Latest and best validation was epoch 25 with `dota/mAP=0.376255` and
  `dota/AP50=0.376`; checkpoint:
  `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s1_rerun_20260603/epoch_25.pth`.
- DOTA2 Oriented R-CNN R50 valid-PNG baseline remained active on GPU 6 in
  screen `s0_dota2_orcnn_r50_validpng_bs1_20260603_gpu6`, PID `2015964`.
  Latest and best validation was epoch 8 with `dota/mAP=0.585885` and
  `dota/AP50=0.586`; checkpoint:
  `/data5/2025/ldh/OpenRSD/work_dirs/s0_dota2_1024_500_orcnn_r50_validpng_bs1_20260603/epoch_8.pth`.

## Paused Runs

Lower-priority `zwl` jobs on GPUs 0, 1, 2, and 4 were stopped after confirming
saved checkpoints. GPU 3 was left untouched.

- RTMDet-L on GPU 0 was stopped after `epoch_4.pth`; observed validation
  `dota/mAP=0.350931`.
- S2ANet on GPU 1 was stopped after saved `epoch_8.pth`; best observed
  validation step 10 reached `dota/mAP=0.579767`, but no epoch-10 checkpoint
  was present.
- R3Det-KFIoU on GPU 2 was stopped after `epoch_4.pth`; observed validation
  `dota/mAP=0.504588`.
- RTMDet-M on GPU 4 was stopped after `epoch_8.pth`; best epoch 4 validation
  was `dota/mAP=0.396986`, while latest epoch 8 validation was
  `dota/mAP=0.364504`.

## Completed Context

- DOTA2 RoI Transformer valid-PNG remains the strongest completed DOTA2 S0
  evidence: final epoch 12 on `DOTA2_1024_500/ss_val` reached
  `dota/mAP=0.6088`.
- GeoNexus S2 hierarchy regularizer 144e remains completed DOTA v1.5 evidence:
  best epoch 30 reached `dota/mAP=0.381885`; final epoch 144 reached
  `dota/mAP=0.372329`.
- GeoNexus S3 scene adapter 144e remains completed DOTA v1.5 evidence: best
  epoch 65 reached `dota/mAP=0.381333`; final epoch 144 reached
  `dota/mAP=0.371248`.

## Interpretation

The immediate priority is to finish and archive the active GeoNexus S1 rerun,
then launch the next S2 hierarchy-regularizer rerun from the best S1 checkpoint.
This is higher priority than launching additional DOTA2 baselines because S1 is
the gate for the GeoNexus S2/S3 paper path, and the latest S1 validation is
still the run best.

Use `epoch_25.pth` as the current S1 initialization candidate unless a later
validation beats `dota/mAP=0.376255`. If two consecutive later validations drop
from epoch 25 without producing a new best, keep the epoch-25 checkpoint and
mark the future value of continuing S1 as limited.

## Next Action

1. Monitor GeoNexus S1 until completion or failure. Each pass must check
   `screen -ls`, `nvidia-smi`, and the active S1 log.
2. When S1 finishes normally, archive best epoch, final epoch, checkpoint path,
   and scalar JSON source.
3. Launch the next GeoNexus S2 hierarchy-regularizer rerun from the best S1
   checkpoint. Prefer GPU 1 if still free; otherwise choose the lowest-index
   allowed GPU with `memory.used <= 4000 MiB` and `util <= 10%` for three
   consecutive polls.
4. Do not use GPU 3. Avoid GPU 0 while the VLLM process is present. Keep ORCNN
   running on GPU 6 unless it fails or clearly underperforms after its next
   saved checkpoint.
5. Treat DOTA2 baseline work as secondary. Do not restart S2ANet, RTMDet, or
   R3Det until the S1-to-S2 GeoNexus rerun is secured.
