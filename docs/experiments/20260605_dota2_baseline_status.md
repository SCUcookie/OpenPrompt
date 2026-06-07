# DOTA2 Baseline Status - 2026-06-05

This note records the secondary DOTA2 valid-PNG baseline state after the GeoNexus S1/S2 recovery.

| Run | Status | Latest completed metric |
| --- | --- | --- |
| Oriented R-CNN R50 bs1 | Completed epoch 12 | `dota/mAP=0.5973`, `dota/AP50=0.5970` |
| S2ANet bs1 | Resumed and completed epoch 12 | `dota/mAP=0.5869`, `dota/AP50=0.5870` |
| R3Det-KFIoU bs1 | Resumed and completed epoch 12 | `dota/mAP=0.5633`, `dota/AP50=0.5630` |
| RTMDet-M bs1 | Resumed and completed epoch 12 | `dota/mAP=0.3312`, `dota/AP50=0.3310` |
| RTMDet-L bs1 | Active resume on GPU 6 from epoch 4 | Currently epoch 11; latest validation epoch 8 `dota/mAP=0.3521`, `dota/AP50=0.3520` |

R3Det-KFIoU source: checkpoint
`/data5/2025/ldh/OpenRSD/work_dirs/s0_dota2_1024_500_r3det_kfiou_validpng_bs1_20260603/epoch_12.pth`;
resume log
`/data5/2025/ldh/OpenRSD/work_dirs/s0_dota2_1024_500_r3det_kfiou_validpng_bs1_20260603/20260605_100954/20260605_100954.log`.

Use at most three active training GPUs total. Leave RTMDet-L running until its
epoch-12 validation completes, but treat it as low-priority unless it improves
substantially beyond the epoch-8 `0.3521/0.3520` point. Pair it with at most
one GeoNexus paper-path run.
