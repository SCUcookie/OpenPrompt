# DOTA2 Baseline Status - 2026-06-05

This note records the secondary DOTA2 valid-PNG baseline state after the GeoNexus S1/S2 recovery.

| Run | Status | Latest completed metric |
| --- | --- | --- |
| Oriented R-CNN R50 bs1 | Completed epoch 12 | `dota/mAP=0.5973`, `dota/AP50=0.5970` |
| S2ANet bs1 | Resumed and completed epoch 12 | `dota/mAP=0.5869`, `dota/AP50=0.5870` |
| R3Det-KFIoU bs1 | Active resume on GPU 5 from epoch 8 | Latest live log: epoch 10 training; best recorded validation before resume: epoch 4 `dota/mAP=0.5046`, `dota/AP50=0.5050` |
| RTMDet-M bs1 | Resumed and completed epoch 12 | `dota/mAP=0.3312`, `dota/AP50=0.3310` |
| RTMDet-L bs1 | Active resume on GPU 6 from epoch 4 | Startup passed at epoch 5 `[200/78014]`; best recorded validation before resume: epoch 4 `dota/mAP=0.3509`, `dota/AP50=0.3510` |

Use at most three active training GPUs total. Keep the active R3Det GPU-5 job, resume RTMDet-L only if GPU 6 passes the three-poll idle check, and pair it with at most one GeoNexus paper-path run.
