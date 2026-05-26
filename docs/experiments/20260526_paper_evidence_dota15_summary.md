# DOTA v1.5 Paper Evidence Summary - 2026-05-26

This record separates closed-set strong-detector evidence from the GeoNexus RemoteCLIP scaffold diagnostics. Large prediction dumps, figures, and logs remain under `OpenRSD/work_dirs`; compact metrics stay in this docs directory.

## Main Comparison

| Detector | Checkpoint | Best epoch | mAP / AP50 | Prediction dump | Confusion matrix | Qualitative examples | Benchmark |
|---|---:|---:|---:|---|---|---|---|
| RoI Transformer | `/data5/2025/ldh/OpenRSD/work_dirs/strong_baseline_dota15/roi_trans_lr001_3x/epoch_34.pth` | 34 | 0.2644 / 0.264 | `/data5/2025/ldh/OpenRSD/work_dirs/paper_evidence_dota15/roi_trans_epoch34/preds.pkl` | `/data5/2025/ldh/OpenRSD/work_dirs/paper_evidence_dota15/roi_trans_epoch34/confusion/confusion_matrix.png` | `/data5/2025/ldh/OpenRSD/work_dirs/paper_evidence_dota15/roi_trans_epoch34/qualitative/` | 11.0 FPS / 90.9 ms |
| Oriented R-CNN | `/data5/2025/ldh/OpenRSD/work_dirs/strong_baseline_dota15/oriented_rcnn_3x_loadfrom/epoch_33.pth` | 33 | 0.2620 / 0.262 | `/data5/2025/ldh/OpenRSD/work_dirs/paper_evidence_dota15/oriented_rcnn_epoch33/preds.pkl` | `/data5/2025/ldh/OpenRSD/work_dirs/paper_evidence_dota15/oriented_rcnn_epoch33/confusion/confusion_matrix.png` | `/data5/2025/ldh/OpenRSD/work_dirs/paper_evidence_dota15/oriented_rcnn_epoch33/qualitative/` | 12.9 FPS / 77.5 ms |
| ReDet pretrained | `/data5/2025/ldh/OpenRSD/work_dirs/strong_baseline_dota15/redet_pretrained_rerun/epoch_12.pth` | 12 | 0.2382 / 0.238 | `/data5/2025/ldh/OpenRSD/work_dirs/paper_evidence_dota15/redet_pretrained_epoch12/preds.pkl` | `/data5/2025/ldh/OpenRSD/work_dirs/paper_evidence_dota15/redet_pretrained_epoch12/confusion/confusion_matrix.png` | `/data5/2025/ldh/OpenRSD/work_dirs/paper_evidence_dota15/redet_pretrained_epoch12/qualitative/` | 9.3 FPS / 107.5 ms |

Raw MMRotate test output JSONs:

| Detector | Raw JSON | mAP | AP50 | mean iter time |
|---|---|---:|---:|---:|
| RoI Transformer | `/data5/2025/ldh/OpenRSD/work_dirs/paper_evidence_dota15/roi_trans_epoch34/20260526_104338/20260526_104338.json` | 0.2644376755 | 0.264 | 0.2256 s |
| Oriented R-CNN | `/data5/2025/ldh/OpenRSD/work_dirs/paper_evidence_dota15/oriented_rcnn_epoch33/20260526_104337/20260526_104337.json` | 0.2619543076 | 0.262 | 0.1426 s |
| ReDet pretrained | `/data5/2025/ldh/OpenRSD/work_dirs/paper_evidence_dota15/redet_pretrained_epoch12/20260526_104338/20260526_104338.json` | 0.2382197678 | 0.238 | 0.1423 s |

## Per-Epoch AP50 Curves

Values are copied from the tracked S0 metric JSONs:

| Epoch | RoI Transformer | Oriented R-CNN | ReDet pretrained |
|---:|---:|---:|---:|
| 1 | 0.243 | 0.237 | 0.101 |
| 2 | 0.249 | 0.240 | 0.145 |
| 3 | 0.244 | 0.236 | 0.176 |
| 4 | 0.245 | 0.248 | 0.191 |
| 5 | 0.241 | 0.238 | 0.201 |
| 6 | 0.244 | 0.241 | 0.213 |
| 7 | 0.253 | 0.259 | 0.225 |
| 8 | 0.248 | 0.256 | 0.223 |
| 9 | 0.249 | 0.257 | 0.230 |
| 10 | 0.243 | 0.235 | 0.230 |
| 11 | 0.250 | 0.258 | 0.236 |
| 12 | 0.249 | 0.255 | 0.238 |
| 13 | 0.255 | 0.253 | - |
| 14 | 0.250 | 0.242 | - |
| 15 | 0.252 | 0.259 | - |
| 16 | 0.247 | 0.249 | - |
| 17 | 0.250 | 0.249 | - |
| 18 | 0.244 | 0.258 | - |
| 19 | 0.254 | 0.249 | - |
| 20 | 0.256 | 0.248 | - |
| 21 | 0.254 | 0.260 | - |
| 22 | 0.263 | 0.259 | - |
| 23 | 0.253 | 0.255 | - |
| 24 | 0.249 | 0.252 | - |
| 25 | 0.260 | 0.262 | - |
| 26 | 0.262 | 0.261 | - |
| 27 | 0.262 | 0.258 | - |
| 28 | 0.260 | 0.258 | - |
| 29 | 0.257 | 0.258 | - |
| 30 | 0.260 | 0.260 | - |
| 31 | 0.258 | 0.257 | - |
| 32 | 0.261 | 0.258 | - |
| 33 | 0.261 | 0.262 | - |
| 34 | 0.264 | 0.262 | - |
| 35 | 0.261 | 0.259 | - |
| 36 | 0.261 | 0.261 | - |

## Qualitative Artifacts

Each detector has 5 `good` and 5 `bad` rendered examples. Per-image ranking used mean detection confidence as a fallback because MMDet's per-image HBB mAP helper cannot compare 5-parameter rotated boxes directly.

## Efficiency Notes

Controlled benchmark runs now use sanitized standalone configs under `/data5/2025/ldh/OpenRSD/work_dirs/paper_evidence_dota15/benchmark_configs/`. The local benchmark path removes legacy `pretrained` fields and uses `test_step` for MMEngine-style detectors. Logs:

- RoI Transformer: `/data5/2025/ldh/OpenRSD/work_dirs/paper_evidence_dota15/roi_trans_epoch34/benchmark.log`
- Oriented R-CNN: `/data5/2025/ldh/OpenRSD/work_dirs/paper_evidence_dota15/oriented_rcnn_epoch33/benchmark.log`
- ReDet pretrained: `/data5/2025/ldh/OpenRSD/work_dirs/paper_evidence_dota15/redet_pretrained_epoch12/benchmark.log`

## GeoNexus S1 Status

Paper-facing S1 is implemented in the MMRotate RoI Transformer path, not the scaffold trainer:

- Prompt head: `/data5/2025/ldh/OpenRSD/geonexus_mmrotate/prompt_bbox_head.py`
- Source config: `/data5/2025/ldh/OpenRSD/mmrotate_configs/geonexus_dota15/roi-trans-le90_r50_fpn_remoteclip-s1_dota15.py`
- Sanitized runtime config: `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s1/roi_trans_remoteclip_s1_sanitized.py`
- RemoteCLIP prompt cache: `/data5/2025/ldh/New/artifacts/generated/remoteclip_vit_b32_dota15_prompt_embeddings.pt`

One-batch GPU loss smoke test passed on GPU 0 with finite losses: `s0.loss_cls=2.8070`, `s0.loss_bbox=0.0270`, `s1.loss_cls=2.7331`, `s1.loss_bbox=0.0725`. No S1 mAP claim is recorded yet.

Launch/update on 2026-05-26:

- Required code fixes before launch: `tools/bootstrap_run.py` now re-adds the repo root after preloading installed OpenMMLab packages, and `geonexus_mmrotate/prompt_bbox_head.py` registers `PromptShared2FCBBoxHead` into `mmrotate.registry.MODELS`.
- GPU 0 launch reached training but hit CUDA OOM during RPN target assignment with another process/reserved memory on the device. The failed log is `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s1/screen_train.log`.
- Relaunched on GPU 1 at `2026-05-26T20:47:21+08:00`; log `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s1/screen_train_gpu1.log` shows finite losses through epoch 1 and wrote `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s1/epoch_1.pth`.
- Epoch-1 validation started but crashed before DOTAMetric output because spawned workers imported repo-local `mmengine` instead of installed `mmengine`. `tools/bootstrap_run.py` was adjusted again to append, not prepend, the repo root so installed OpenMMLab remains first in `sys.path`.
- Relaunched/resumed from epoch 1 on GPU 1 at `2026-05-26T20:59:27+08:00`; log `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s1/screen_train_gpu1_resume.log` shows training resumed at epoch 2 with finite losses.
- The first resume then hit another CUDA OOM on a dense training image during RPN assignment. The sanitized runtime config now sets `gpu_assign_thr=256` on the RPN and both cascade-stage assigners to move dense assignment to CPU.
- Relaunched/resumed again from epoch 1 at `2026-05-26T21:03:18+08:00`; log `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s1/screen_train_gpu1_cpuassign.log` shows epoch 2+ training continuing with finite losses and materially lower GPU memory use.
- Epoch-2 validation completed with DOTAMetric `dota/mAP=0.2510`, `dota/AP50=0.2510`; this is an early S1 checkpoint and is not yet promoted to the main comparison.
- Parallel S1 variants were launched from S0 RoI Transformer epoch 34 with `resume=False`: frozen backbone on GPU 4, learnable prompt offsets on GPU 0, and prompt logit scale 5.0 on GPU 3. Their configs and live logs are summarized in `/data5/2025/ldh/New/docs/experiments/20260526_dota15_geonexus_s1_archive.md`.
- The run loads S0 RoI Transformer epoch 34 and only misses the new prompt classifier parameters, which is expected for S1.
- S1 remains status/early-validation evidence only. Add S1 to the main comparison only after a completed run or selected best checkpoint has a DOTAMetric mAP/AP50 record.

## GeoNexus Scaffold Separation

The `dota_v15_geonexus_remoteclip` scaffold run is diagnostic-only. It was preserved at `/data5/2025/ldh/New/outputs/preserved_20260526_dota_v15_geonexus_remoteclip/`; metrics and tracebacks are recorded in `/data5/2025/ldh/New/docs/experiments/20260526_dota15_geonexus_remoteclip_scaffold_metrics.json`. Current source-output metrics reached epoch 10, while the log later reached epoch 11 before the screen/session died without an epoch 11 checkpoint. `positive_cls_acc` stayed unstable and never recovered the epoch-1 value, so it should not be used as final paper evidence.

## Recommendation

Finish the live S1 MMRotate RoI Transformer RemoteCLIP run and evaluate its best checkpoint with `DOTAMetric` before starting another scaffold experiment. If S1 does not beat the S0 RoI Transformer 0.2644 AP50 baseline, the next controlled experiment should freeze the S0 detector trunk and fine-tune only the prompt projection/bias for a short schedule, then unfreeze the RoI heads. That isolates whether the performance gap is caused by destabilizing the strong detector or by the RemoteCLIP prompt classifier itself.
