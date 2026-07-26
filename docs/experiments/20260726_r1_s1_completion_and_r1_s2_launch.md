# 2026-07-26 R1-S1 Completion and R1-S2 Launch

## R1-S1 Completion
The three R1-S1 Oriented R-CNN replicas completed cleanly. 
Final epoch-12 validation metrics on `DIOR_R_dota/test`:
- rep3407: `dota/mAP=0.6618`, `dota/AP50=0.6620`
- rep4407: `dota/mAP=0.6607`, `dota/AP50=0.6610`
- rep5407: `dota/mAP=0.6564`, `dota/AP50=0.6560`

This successfully ported the +TPC module (stage 1) to Oriented R-CNN on DIOR-R.

## R1-S2 Launch
The 1000-step train-step diagnostic for R1-S2 passed. Three R1-S2 replicas were launched:
- `rep3407` on physical GPU 2 in screen `geonexus_r1_s2_rep3407_gpu2_20260723`
- `rep4407` on physical GPU 3 in screen `geonexus_r1_s2_rep4407_gpu3_20260723`
- `rep5407` on physical GPU 4 in screen `geonexus_r1_s2_rep5407_gpu4_20260723`

Launch provenance files have been written to their respective `work_dirs`.
