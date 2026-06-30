# Project Instructions

This file is the persistent project memory. Keep it updated when the research
scope, repository structure, server workflow, or experiment protocol changes.
Paper-first rule: if the research direction, claim, experiment sequence, or
submission target changes, update the canonical manuscript and this file before
changing code, configs, or secondary docs.

## Token-Saving Startup Protocol

For a fresh agent session, do not load the full project history by default.
Read only `AGENTS.md`, these top operational sections of
`PROJECT_INSTRUCTIONS.md`, and the latest dated experiment note relevant to the
current task. Load older route history only when route evidence, provenance, or
paper claims require it.

Use tight, path-scoped commands. Prefer `rg`/`find`/`grep` filters over broad
repo scans, and avoid searching datasets, checkpoints, logs, binary artifacts,
or large generated workdirs unless the task explicitly targets them.

For experiment status checks, report only the screen names, GPU residency,
latest log marker, scoped failure-scan result, and next action. Do not paste or
summarize the full historical record unless explicitly asked.

When context grows large, compact aggressively into the current goal, active
runs, exact paths, exact commands, and unresolved blockers.

## Experiment Provenance Rule

Every time a GPU run is started, record the full operation trail: task intent,
timestamp, screen name, GPU IDs, working directory, config path, source
checkpoint, exact launch command, log paths, PID/process check, startup marker,
and any GPU remapping decision.

When asked to record a finished experiment result, align the final record with
the original launch record. The completion note must reference the first-run
record and reconcile screen name, GPU assignment, workdir, config, checkpoint
inputs, final/best metrics, failure scan, and any deviations from launch.

## Command/GPU Failure Playbook

Normal sandbox execution may not see host GPU devices. In that mode,
`/dev/nvidia*` can be invisible, `nvidia-smi` can fail, and PyTorch may report
`cuda_available=False` even when GPUs are usable from the host. The fix is to
use approved or escalated host access for real GPU checks and training launches.

Use this GPU status check:

```bash
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
```

Use this process ownership check, replacing the PID list as needed:

```bash
ps -p PID1,PID2,PID3 -o pid,ppid,user,cmd --forest
```

Avoid occupied GPUs. If requested GPU IDs are busy, remap jobs to idle GPUs
instead of forcing the original IDs.

For reruns, do not copy whole completed workdirs with checkpoints and logs.
Create clean rerun workdirs and copy only the required config files or minimal
metadata needed for provenance.

Keep searches narrow: scope `rg`, `find`, and `grep` to docs, configs, and code
paths, and exclude artifacts, datasets, checkpoints, generated logs, and binary
outputs unless those files are the explicit target.

Literature-first advisor rule: before changing the research route, adding a new
module, making a paper-positioning claim, or proposing a submission argument,
search recent primary sources for OpenRSD-citing or adjacent remote-sensing
open-vocabulary detection work. Prioritize arXiv/e-print pages, OpenReview, CVF
open access, AAAI/OJS, IEEE/JSTARS/IJCV official pages, and official project
pages. Blogs and repositories are discovery aids only. Record source date,
venue/status, relation to OpenRSD/GeoNexus, math principle, reusable idea, and
route impact in `docs/literature/20260607_openrsd_related_recent_papers.md`
or its successor before changing experiments.

## Research Direction

Project name: GeoNexus-RSD.

Primary goal: hierarchy- and context-aware vision-language prompting for
DOTA2-centered oriented remote sensing object detection, with DIOR-R as the
required cross-dataset validation and FAIR1M as stretch fine-grained evidence.

Practical first target: IEEE JSTARS. Consider TGRS or ISPRS P&RS only if final
results are strong across at least two datasets. Consider GRSL, IGARSS, or a
workshop if results are modest or incomplete.

Main paper claim:

Hierarchy- and context-aware vision-language prompting improves fine-grained
oriented object detection and semi-supervised pseudo-label quality in remote
sensing imagery.

2026-06-06 research pivot:

- DOTA v1.5 GeoNexus runs are diagnostic/archive-only evidence. They remain
  useful for debugging hierarchy/context code paths, but they are no longer the
  formal benchmark route for the paper and must not be used as headline table
  evidence.
- The formal benchmark order is now DOTA2 first, DIOR-R second, and FAIR1M
  only after DOTA2 plus DIOR-R are stable.
- Stop extending the DOTA v1.5 S2/S3/S4 chain. The lower-LR DOTA v1.5 S2
  refinement in screen
  `geonexus_s2_hierarchy_refine_s2e4_lr5e5_20260606_gpu1` was stopped by
  research pivot on 2026-06-06 at about epoch 3, not classified as a failed
  experiment. Preserve its workdir, launch log, and any partial checkpoints:
  `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s2_hierarchy_reg_refine_s2e4_lr5e5_20260606`.
- Active priority order: finish DOTA2 baselines and archive their exact
  config/checkpoint/metric sources; run only the most defensible DOTA2
  GeoNexus S1/S2 module on the strongest stable detector; establish a DIOR-R
  baseline on `DIOR_R_dota/train_val` and `DIOR_R_dota/test`; then repeat the
  same minimal GeoNexus module on DIOR-R. FAIR1M is stretch evidence for
  fine-grained hierarchy claims, not the first cross-dataset proof.
- 2026-06-08 route gate update: DOTA2 GeoNexus S1 on GPU 1 completed cleanly
  at epoch 12 and exceeded DOTA2 RoI Transformer S0 `dota/mAP=0.6088`,
  `dota/AP50=0.6090`. Launch DOTA2 S2 from the GPU-1 S1 epoch-12 checkpoint
  while the GPU-0/GPU-6 S1 replicates continue as comparison evidence. Keep
  S3/S4, pseudo-label purification, and routing paused until DOTA2 S2 and
  DIOR-R numeric stability are resolved.
- 2026-06-09 route gate update: use a stage-gated route, not a full S1-S4
  pipeline. DOTA2 S1 with RoI Transformer + RemoteCLIP is the current best
  paper-path module on `DOTA2_1024_500/ss_val`, improving S0
  `0.6088/0.6090` to `0.6177/0.6180`. Main DOTA2 S2 completed below S1 at
  `0.5924/0.5920`; live S2 variants have not beaten S1 so far. As of
  `2026-06-09 19:58 CST`, GPU-6 hierarchy-weight `0.01` S2 is in epoch-12
  validation, GPU-0 LR `1e-4` S2 is still in epoch-12 training, and GPU-5 LR
  `5e-5` S2 is still epoch-2 diagnostic training. Do not launch S3/S4,
  pseudo-labeling, FAIR1M, or additional detector training until DOTA2 S2 is
  either rescued by final metrics or archived, and DIOR-R has a finite detector
  baseline.
- 2026-06-10 route gate update: the S2 hierarchy loss-0 ablation briefly
  exceeded S1 at epoch 2 (`0.6204/0.6200`) but finished epoch 4 at
  `0.6179/0.6180`. A controlled loss-0 replication with seed `3407` was
  launched on GPU 1 in screen
  `geonexus_dota2_s2_loss0_rep3407_20260610_gpu1`. Workdir:
  `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s2_hierarchy_ablate_loss0_s1e12_rep3407_20260610`.
  Startup reached `Epoch(train) [1][200/39007]` with no failure signature.
  Keep S3/S4, pseudo-labeling, FAIR1M, and DIOR-R detector relaunch paused
  until this replication's best and final checkpoints are compared separately
  against S1 `0.6177/0.6180`.
- 2026-06-10 concurrency update: two additional controlled DOTA2 S2 loss-0
  replications were launched to maintain three concurrent GeoNexus GPU
  experiments while keeping the same route gate. Seed `4407` is running on GPU
  0 in screen `geonexus_dota2_s2_loss0_rep4407_20260610_gpu0`; seed `5407` is
  running on GPU 2 in screen
  `geonexus_dota2_s2_loss0_rep5407_20260610_gpu2`. Both configs were copied
  from `rep3407` and changed only for `randomness.seed` plus `work_dir`.
  Runtime logs:
  `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s2_hierarchy_ablate_loss0_s1e12_rep4407_20260610/20260610_210021/20260610_210021.log`
  and
  `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s2_hierarchy_ablate_loss0_s1e12_rep5407_20260610/20260610_210021/20260610_210021.log`.
  At `2026-06-10 21:05 CST`, both jobs were GPU-resident but still waiting in
  startup I/O before the `Epoch(train) [1][200/39007]` acceptance point; do
  not mark them startup-accepted until that line appears and the scoped failure
  signature scan remains clean.
- 2026-06-12 DOTA2 S2 loss-0 stabilization update: the three 3-epoch
  stabilization runs completed. rep6407 finished with epoch metrics
  `0.620483/0.6200`, `0.618960/0.6190`, and final `0.618167/0.6180`;
  rep7407 finished with `0.620785/0.6210`, `0.614483/0.6140`, and final
  `0.618315/0.6180`; rep8407 finished with `0.616526/0.6170`,
  `0.619625/0.6200`, and final `0.612147/0.6120`. Original four loss-0 runs
  have best mean `0.620837` (`+0.003137` over S1) and final mean `0.616990`
  (`-0.000710` below S1). The new three runs have best mean `0.620298`
  (`+0.002598`) and final mean `0.616209` (`-0.001491`). All seven runs have
  best mean `0.620606` (`+0.002906`) and final mean `0.616655`
  (`-0.001045`). Classify this as repeatable early-checkpoint S2 evidence but
  unstable final-checkpoint S2 evidence; do not cite the final epoch as S2
  evidence. Keep S3/S4, pseudo-labeling, and FAIR1M paused.
- 2026-06-07 DIOR-R gate: after ORCNN/RoITrans NaN and RetinaNet `loss=inf`,
  DIOR-R detector training is blocked. The next DIOR-R work is diagnosis of
  data records, rotated-box conversion, class mapping, and loss targets, not
  another unchanged detector launch.
- 2026-06-09 DIOR-R geometry update: full label geometry scan artifact
  `New/artifacts/dior_r_diagnostics_20260609_full_geometry.md` covers
  `11725` train_val label files / `68072` objects and `11738` test label files
  / `124445` objects using fallback rbox geometry and declared `800x800`
  bounds. It found 2 bad label files per split with zero-area/invalid-size ship
  boxes, `1210` train_val and `1322` test qboxes crossing the declared bounds,
  and 4 test rbox centers outside bounds. Next DIOR-R step is fixing or
  filtering these records and then running a bounded first-non-finite-loss
  catcher before any detector relaunch.
- 2026-06-12 DIOR-R sanitized-label update: sanitized label directories were
  created at
  `/data5/2025/ldh/OpenRSD/data/DIOR_R_dota/train_val/labelTxt_sanitized_invalidsize_20260612`
  and
  `/data5/2025/ldh/OpenRSD/data/DIOR_R_dota/test/labelTxt_sanitized_invalidsize_20260612`,
  with scanner-compatible root
  `/data5/2025/ldh/OpenRSD/data/DIOR_R_dota_sanitized_invalidsize_20260612`.
  The current raw labels no longer contain the zero-area/invalid-size records
  reported by the 2026-06-09 artifact, so these sanitized dirs mirror current
  raw labels and raw `labelTxt` dirs were not modified. Fresh artifacts
  `New/artifacts/dior_r_diagnostics_20260612_sanitized_invalidsize_geometry.json`
  and `.md` report `11725` train_val files / `68070` objects and `11738` test
  files / `124443` objects, with `num_bad_label_files=0` and
  `invalid_rbox_size=0` for both splits. Next DIOR-R action is bounded
  `train-step` non-finite diagnostics on sanitized labels only.
- 2026-06-12 DIOR-R train-step update: `OpenRSD/tools/diagnose_first_nonfinite_loss.py`
  supports `--mode train-step` and records real `model.train_step` progress,
  non-finite losses, exceptions, and batch context. Sanitized-label diagnostic
  configs live under
  `/data5/2025/ldh/OpenRSD/work_dirs/dior_r_trainstep_diag_20260612/`. ORCNN
  completed `1000/1000`, RoI Transformer completed `4000/4000`, and Rotated
  RetinaNet completed `1500/1500` train-step batches with
  `status=finite_within_limit`; JSON artifacts are
  `.../orcnn/trainstep_diag_20260612.json`,
  `.../roi_trans/trainstep_diag_20260612.json`, and
  `.../retinanet/trainstep_diag_20260612.json`. Completion scans found no
  `Traceback`, CUDA OOM, `out of memory`, `libpng`, `CRC`, `NoneType`,
  `ValueError`, true `nan`, or true `inf`.
- 2026-06-12 DIOR-R S0 smoke update: because all three train-step diagnostics
  completed finite, the one allowed sanitized DIOR-R RoI Transformer S0 smoke
  was launched on GPU 3 in screen
  `dior_r_roi_trans_s0_sanitized_smoke_20260612_gpu3`, PID `3363864`.
  Workdir:
  `/data5/2025/ldh/OpenRSD/work_dirs/dior_r_s0_roi_trans_sanitized_smoke_20260612`;
  config:
  `/data5/2025/ldh/OpenRSD/work_dirs/dior_r_s0_roi_trans_sanitized_smoke_20260612/dior_r_roi_trans_sanitized_s0_smoke_1e_20260612.py`;
  launch log:
  `/data5/2025/ldh/OpenRSD/work_dirs/dior_r_s0_roi_trans_sanitized_smoke_20260612/launch_20260612_gpu3.log`;
  runtime log:
  `/data5/2025/ldh/OpenRSD/work_dirs/dior_r_s0_roi_trans_sanitized_smoke_20260612/20260612_153506/20260612_153506.log`.
  Startup acceptance passed at `Epoch(train) [1][  50/5862]`; keep this as
  diagnostic evidence only and do not launch further DIOR-R detector runs until
  its 1-epoch result is archived.
- 2026-06-13 DIOR-R S0 sanitized-long status: the extended sanitized DIOR-R S0
  runs are finite so far. RoI Transformer is the current leader with epoch 48
  `dota/mAP=0.6531`, `dota/AP50=0.6530`; ORCNN is secondary with best epoch 28
  `0.6341/0.6340`; Rotated RetinaNet trails with best epoch 80
  `0.5686/0.5690` and epoch 84 dipping to `0.5667/0.5670`. ORCNN is still
  active toward epoch 36, RoI Transformer toward epoch 52, and RetinaNet toward
  epoch 96. Do not launch DIOR-R GeoNexus S1 until ORCNN and RoI Transformer
  finish, best and final metrics are archived separately, and completion scans
  remain clean. Detailed interim record:
  `New/docs/experiments/20260613_dior_r_s0_sanitized_long_interim.md`.
- 2026-06-14 conference-result status: DIOR-R S0/S1/S2 are complete on the
  sanitized `DIOR_R_dota/test` path. RoI Transformer S0 final epoch 52 is
  `dota/mAP=0.6544`, DIOR-R S1 final mean over two replicas is `0.6720`, and
  DIOR-R S2 over six replicas has best mean `0.6887` and final mean
  `0.6856`; the best single S2 replica is rep4 epoch 12 at `0.6914`. DOTA2
  remains a modest S1/S2 story on `DOTA2_1024_500/ss_val`: S0 RoITrans
  `0.6088`, S1 `0.6177`, S2 loss-0 best mean `0.6206`, and S2 loss-0 final
  mean `0.6167`. Use DOTA2 S2 as repeatable early-checkpoint evidence but
  final-unstable; use DIOR-R as the stronger S0 -> S1 -> S2 progression. Keep
  DOTA v1.5 in appendix/archive-only slides. Keep S3/S4, pseudo-labeling, and
  FAIR1M paused unless a new route decision is made. Conference slide assets
  are generated by `scripts/make_ppt_assets_20260614.py` into
  `artifacts/ppt_assets_20260614/`.
- 2026-06-15 archive status: DIOR-R S3 scene-adapter replicas completed
  cleanly from S2 rep4 epoch 12. S3 best mean over three replicas is `0.6979`,
  final mean is `0.6859`, and the best single checkpoint is rep0 epoch 8
  `0.6992`. Classify S3 as strong best-checkpoint scene-adapter evidence, but
  only roughly tied with S2 at final checkpoints. No active training screens
  remain except `s0_result_log_monitor_20260603`, and GPUs 0-6 reported idle
  after completion. The next work package is paper/result analysis and archive
  hygiene, not launching a new route. Keep pseudo-labeling, FAIR1M, S4, and
  submission-positioning claims paused until a separate route decision.
- 2026-06-23 archive and route status: the missing DOTA2 S3 long-12 override
  completion and continue13 completion records were archived. Long-12 finished
  with best mean `0.6203` and final epoch-12 mean `0.6130`; continue13
  finished with primary epoch-12 mean `0.6137` and final epoch-13 mean
  `0.6128`. Both are clean negative-to-neutral DOTA2 S3 evidence and remain
  below the useful DOTA2 S1/S2 story. Today's only new training route is a
  DIOR-R S3 long-stability confirmation continuation from the 2026-06-16
  annealed stability epoch-4 checkpoints. Keep DOTA2 follow-up training, S4,
  pseudo-labeling, and FAIR1M paused.
- 2026-06-15 follow-up archive: DIOR-R S3 epoch-8 LR5e-5 stability completed
  cleanly. Stability best mean is `0.692193` and final mean is `0.690303`.
  This is lower than original S3 best mean `0.6979`, but improves final
  stability over original S3 final `0.6859` and S2 final `0.6856`. Keep best
  and final metrics separate in claims.
- 2026-06-15 DOTA2 S3 launch status: three S3 scene-adapter replicas were
  launched from DOTA2 S2 loss-0 best early checkpoints on GPUs 0, 1, and 2 in
  screens `geonexus_dota2_s3_scene_rep3407_20260615_gpu0`,
  `geonexus_dota2_s3_scene_rep4407_20260615_gpu1`, and
  `geonexus_dota2_s3_scene_rep5407_20260615_gpu2`. Config preflight passed
  for 18 classes/prompts, scene-adapter heads, source checkpoints, and seeds
  `93407/94407/95407`. Startup acceptance is pending while the jobs finish
  DOTA2 annotation initialization; require `Epoch(train) [1][  200/39007]`
  before treating startup as accepted.
- 2026-06-16 DOTA2 S3 completion status: the three S3 scene-adapter replicas
  completed cleanly. Best mean is `0.6199` (`0.6199271768`) and final mean is
  `0.6151` (`0.6150780916`). This is below DOTA2 S2 loss-0 best mean
  `0.620606` and below S2 loss-0 final mean `0.616655`, so classify DOTA2 S3
  as exploratory/negative-to-neutral evidence. No active training screens
  remain except `s0_result_log_monitor_20260603`; do not launch S4,
  pseudo-labeling, FAIR1M, or route-changing experiments from this result.
- 2026-06-19 DOTA2 S3 rerun completion status: the 2026-06-18 controlled
  loss-0-best scene-adapter rerun completed through epoch 4 for all three
  replicas with a clean scoped failure scan. Rounded per-replica epoch mAPs
  were rep3407 `0.6189/0.6213/0.6141/0.6130`, rep4407
  `0.6156/0.6160/0.6160/0.6155`, and rep5407
  `0.6207/0.6207/0.6133/0.6165`; best mean is `0.6193` and final mean is
  `0.6150`. This remains exploratory/negative-to-neutral DOTA2 S3 evidence
  below the useful DOTA2 S2 best/final story. Keep the DOTA2 follow-up
  training route paused unless explicitly overridden.
- 2026-06-17 archive and analysis status: archive hygiene for the 2026-06-14
  through 2026-06-16 result package is recorded in
  `docs/experiments/20260617_archive_and_analysis_launch.md`. Three
  analysis-only DIOR-R checkpoint evaluation jobs were launched/planned for
  GPUs 0, 1, and 2 to produce metrics, logs, and prediction pickles for
  qualitative and claim-boundary analysis. This does not change the route
  gate: S4, pseudo-labeling, FAIR1M, routing, DOTA2 follow-ups, and new
  training remain paused unless a separate route decision is made.
- 2026-06-20 paper-eval archive status: the DOTA2 paper-evaluation 3-GPU note
  `docs/experiments/20260620_dota2_paper_eval_3gpu_complete.md` is complete
  and preserved. Metrics are S0 epoch 12 `0.608833/0.6090`, S1 epoch 12
  `0.617687/0.6180`, and S2 loss-0 rep3407 epoch 1 `0.621121/0.6210`. This
  was analysis-only work and did not reopen S4, pseudo-labeling, FAIR1M, or a
  route change.
- 2026-06-21 DOTA2 S3 long12 completion status: the June 20
  3-replica long12 scene-adapter pack on GPUs 1/2/3 completed cleanly and is
  archived in
  `docs/experiments/20260620_dota2_s3_scene_adapter_long12_override_launch.md`.
  All screens exited and GPUs are free. Final epoch-12 metrics are rep3407
  `0.6122/0.6120` (exact best `0.6216441989/0.622`, final
  `0.6122340560/0.612`), rep4407 `0.6150/0.6150` (exact best
  `0.6176995635/0.618`, final `0.6150146723/0.615`), and rep5407
  `0.6118/0.6120` (exact best `0.6214531064/0.621`, final
  `0.6118243933/0.612`). Final mean mAP is about `0.6130`. Classify this pack
  as negative-to-neutral DOTA2 S3 evidence.
- 2026-06-21 DOTA2 S3 confirmation launch: the next launched work is the
  2026-06-21 3-GPU confirmation pack from DOTA2 S2 loss-0 reps
  6407/7407/8407 on GPUs 1/2/3, launched from clean workdirs with 6 epochs.
  Record its operation trail in
  `docs/experiments/20260621_dota2_s3_confirm6_rep6407_7407_8407_launch.md`.
- 2026-06-22 DOTA2 S3 confirm6/continue13 status: the 2026-06-21 confirm6
  pack completed cleanly and is archived in
  `docs/experiments/20260622_dota2_s3_confirm6_complete.md`. Final rounded
  mAPs are rep6407 `0.6162`, rep7407 `0.6151`, and rep8407 `0.6165`; best
  mean rounded mAP is `0.6193`, final mean rounded mAP is `0.6159`. A
  conservative continuation from the three epoch-6 checkpoints to epoch 13 was
  launched on GPUs 1/2/3 and recorded in
  `docs/experiments/20260622_dota2_s3_continue13_launch.md`. Epoch 12 is the
  primary comparable result; epoch 13 is a stability/occupancy tail. This does
  not reopen S4, pseudo-labeling, FAIR1M, or route-changing claims.
- 2026-06-16 DIOR-R S3 final-stability follow-up: three annealed stability
  continuations from the LR5e-5 stability `epoch_4.pth` checkpoints completed
  cleanly on GPUs 0, 1, and 2. LR was `2.5e-5` for 4 epochs with validation
  every epoch. Best mean is `0.6908363303`; final mean is `0.6892216007`.
  Final replica mAPs are rep0 `0.6923807859`, rep1 `0.6871313453`, and rep2
  `0.6881526709`. By the 2026-06-16 decision rules this is neutral: above
  DIOR-R S2 final `0.6856`, but below the useful final-stability threshold
  `0.6903`. Archive without automatic extension. Keep S4, pseudo-labeling,
  FAIR1M, routing, DOTA2 follow-ups, and other route-changing experiments
  paused unless a separate route decision is made.

Core modules:

1. Hierarchical prompt bank.
2. Scene/context prompt adapter.
3. VLM-assisted pseudo-label purification.

Secondary only:

- Routing is optional after the three core modules are stable.
- Compression is a later-paper topic.
- Segmentation is not the primary task for this paper.

Do not claim open-vocabulary detection unless the final system uses real
vision-language embeddings and evaluates a real open-vocabulary or vocabulary
robustness setting.

## Current Code Reality

This repository is a research scaffold, not a competitive detector yet.

Current limitations:

- The local backbone is lightweight.
- The current text embedder is deterministic hash-based unless replaced.
- Official DOTA validation is integrated for the reduced tiled baseline, but the current validation mAP is still extremely low.
- The reduced DOTA v1.0 validation result is `map50=3.326794065590851e-06` on 4055 images; treat it as a pipeline sanity check only, not a paper result.

Current server evidence:

- The matched DOTA v1.5 baseline training and validation evaluation have completed.
- The v1.5 validation result is `map50=1.0926445202230628e-05` on 4055 images; it is still only a sanity-check baseline.
- The baseline comparison should stay tied to the reduced tiled setup and the same dataset/version split used for the recorded metrics.

Current diagnosis:

- Quick baseline diagnostics show the issue is not thresholding; decoded scores stay above the tested thresholds.
- Predictions collapse toward `small-vehicle`, `harbor`, `plane`, and `ship`, and a spot-checked validation tile shows center-biased boxes with very low same-class IoU.
- `QueryGenerator` computes `query_centers`, but the current box heads do not consume them, so the scaffold currently regresses boxes without an explicit spatial anchor.
- The anchor-repair quick test completed and wrote `outputs/dota_v15_anchor_repair/epoch_001.pt`; final training metrics were `loss=0.07363908355801901`, `loss_cls=0.001671954903589549`, `loss_box=0.035983564312892485`, `positive_cls_acc=0.5529336195676059`, and `positive_box_l1=0.10294117139314753`.
- S0 strong-detector baselines are complete for the controlled DOTA v1.5 split.
- The best current S0 detector is RoI Transformer 3x, epoch 34, with MMRotate DOTAMetric `dota/mAP=0.2644` and `dota/AP50=0.2640`; checkpoint `/data5/2025/ldh/OpenRSD/work_dirs/strong_baseline_dota15/roi_trans_lr001_3x/epoch_34.pth`; metric summary `docs/experiments/20260526_roi_transformer_3x_dota15_metrics.json`.
- GeoNexus S2 hierarchy regularizer 12e completed on the same DOTA v1.5 reduced tiled split. Final epoch 12: `dota/mAP=0.3644`, `dota/AP50=0.3640`; best observed epoch 11: `dota/mAP=0.3652`, `dota/AP50=0.3650`; checkpoint `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s2_hierarchy_reg_frozen_12e/epoch_12.pth`; metric summary `docs/experiments/20260601_s2_hierarchy_regularizer_12e_metrics.json`.
- GeoNexus S2 hierarchy regularizer 72e completed on the same DOTA v1.5 reduced tiled split. Final epoch 72: `dota/mAP=0.3738`, `dota/AP50=0.3740`; best observed epoch 56: `dota/mAP=0.3757`, `dota/AP50=0.3760`; checkpoint `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s2_hierarchy_reg_frozen_72e/epoch_72.pth`; metric summary `docs/experiments/20260601_s2_hierarchy_regularizer_72e_metrics.json`.
- GeoNexus S2 hierarchy regularizer 144e completed on the same DOTA v1.5 reduced tiled split. Final epoch 144: `dota/mAP=0.3723`, `dota/AP50=0.3720`; best observed epoch 30: `dota/mAP=0.3819`, `dota/AP50=0.3820`; checkpoint `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s2_hierarchy_reg_frozen_144e/epoch_144.pth`; metric summary `docs/experiments/20260602_s2_hierarchy_regularizer_144e_metrics.json`. Treat best and final numbers separately: the 144e best is the strongest observed S2 validation point, while the 144e final is slightly below the 72e final.
- S3 scene-adapter 72e first queue launch failed before training because the inherited base config nested `roi_head.bbox_head` incorrectly and the child config also dropped full RCNN assigner definitions. The owned child config `/data5/2025/ldh/OpenRSD/mmrotate_configs/geonexus_dota15/roi-trans-le90_r50_fpn_remoteclip-s3-72e_dota15.py` was corrected to inherit from S1 directly, define scene-adapter heads with a proper `bbox_head` list, and keep full assigner/sampler configs. The failed base file `/data5/2025/ldh/OpenRSD/mmrotate_configs/geonexus_dota15/roi-trans-le90_r50_fpn_remoteclip-s3_dota15.py` is owned by `nobody:nogroup`; avoid relying on it until permissions are fixed.
- GeoNexus S3 scene-adapter 72e completed on the same DOTA v1.5 reduced tiled split. Final epoch 72: `dota/mAP=0.3759`, `dota/AP50=0.3760`; best observed epoch 51: `dota/mAP=0.3800`, `dota/AP50=0.3800`; checkpoint `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s3_scene_adapter_72e/epoch_72.pth`; metric summary `docs/experiments/20260602_s3_scene_adapter_72e_metrics.json`. Treat this as completed S3 evidence; do not make stronger context-adapter claims until S3 144e and the active follow-up runs finish.
- GeoNexus S3 scene-adapter 144e completed on the same DOTA v1.5 reduced tiled split. Final epoch 144: `dota/mAP=0.3712`, `dota/AP50=0.3710`; best observed epochs 65 and 73 tied at rounded log `dota/mAP=0.3813`, `dota/AP50=0.3810`; checkpoint `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s3_scene_adapter_144e/epoch_144.pth`; metric summary `docs/experiments/20260603_s3_scene_adapter_144e_metrics.json`. Treat best and final separately: the best S3 144e validation is slightly below the S2 144e best, and the final S3 144e is below the S3 72e final.
- Current GPU status at `2026-06-03 17:08 +0800`: our active valid-PNG DOTA2 baseline jobs occupy GPUs 0, 1, 2, 4, and 6; other users occupy GPUs 3 and 5. The DOTA2 RoI Transformer valid-PNG recovery completed and released GPU 0, then RTMDet-L was launched there.
- The manual S0 DOTA2 RoI Transformer rebuild `s0_dota2_roi_trans_rebuild_20260601` was launched on GPU 0 and marked `launched_manually=true` in the queue metadata, then failed during epoch 1 with `libpng error: IDAT: CRC error` and `AttributeError: 'NoneType' object has no attribute 'shape'` from image loading. Preserve `/data5/2025/ldh/OpenRSD/work_dirs/s0_dota2_1024_500_roi_trans_rebuild_20260601/queue_launch_20260601.log`; failure note `docs/experiments/20260602_s0_dota2_roi_trans_rebuild_failure.md`.
- S0 DOTA2 RoI Transformer valid-PNG recovery completed on `2026-06-03`: a full Pillow decode scan of `/data5/2025/ldh/OpenRSD/data/DOTA2_1024_500/train/images` found `47` corrupt PNGs out of `170878`; corrupt list `/data5/2025/ldh/OpenRSD/work_dirs/s0_dota2_1024_500_roi_trans_rebuild_20260602_validpng_restart/corrupt_train_pngs_20260602.txt`; scan summary `/data5/2025/ldh/OpenRSD/work_dirs/s0_dota2_1024_500_roi_trans_rebuild_20260602_validpng_restart/corrupt_train_pngs_scan_summary_20260602.txt`. Filtered annotation symlink dir `/data5/2025/ldh/OpenRSD/data/DOTA2_1024_500/train/annfiles_validpng_20260602` links `170831` valid annotations and excludes the `47` corrupt-image annotations. Restart config `/data5/2025/ldh/OpenRSD/work_dirs/s0_dota2_1024_500_roi_trans_rebuild_20260602_validpng_restart/G02_Baselines_Data1_DOTA2_M2_RoITrans_validpng_20260602.py`; launch log `/data5/2025/ldh/OpenRSD/work_dirs/s0_dota2_1024_500_roi_trans_rebuild_20260602_validpng_restart/launch_20260602.log`; checkpoint `/data5/2025/ldh/OpenRSD/work_dirs/s0_dota2_1024_500_roi_trans_rebuild_20260602_validpng_restart/epoch_12.pth`. Final epoch 12 on `DOTA2_1024_500/ss_val`: `dota/mAP=0.6088`, `dota/AP50=0.6090` at `2026-06-03 14:31:57 +0800`; metric summary `docs/experiments/20260603_s0_dota2_roi_trans_validpng_metrics.json`; record `docs/experiments/20260602_s0_dota2_roi_trans_rebuild_validpng_restart.md`. Treat this as completed S0 DOTA2 `ss_val` evidence only, not GeoNexus S1/S2/S3/S4 evidence.
- S0 DOTA2 Oriented R-CNN R50 valid-PNG baseline launched on GPU 1 at `2026-06-02 14:53 +0800`: workdir `/data5/2025/ldh/OpenRSD/work_dirs/s0_dota2_1024_500_orcnn_r50_validpng_20260602`; runtime config `/data5/2025/ldh/OpenRSD/work_dirs/s0_dota2_1024_500_orcnn_r50_validpng_20260602/G02_Baselines_Data1_DOTA2_M5_ORCNN_R50_validpng_20260602.py`; launch log `/data5/2025/ldh/OpenRSD/work_dirs/s0_dota2_1024_500_orcnn_r50_validpng_20260602/launch_20260602.log`; screen `s0_dota2_orcnn_r50_validpng_20260602_gpu1`; PID `1598732`. It passed filtered annotation preparation and entered training, but failed at `Epoch(train) [1][300/39007]` with CUDA out-of-memory while computing anchor-target IoU. Preserve the log and do not cite as completed ORCNN evidence. The next ORCNN retry should reduce memory pressure before relaunching.
- OpenRSD DOTA2 epoch-12 checkpoint evaluation on `DOTA2_1024_500/ss_val` completed on `2026-06-02`: `dota/mAP=0.4202`, `dota/AP50=0.4200`. Checkpoint `/data5/2025/ldh/OpenRSD/work_dirs/opensrd_step2_dota2_nozero_full_20260531/epoch_12.pth`; config `/data5/2025/ldh/OpenRSD/work_dirs/opensrd_formal_dota2_ssval_eval/a10_formal_dota2_eval_no_star.py`; predictions `/data5/2025/ldh/OpenRSD/work_dirs/opensrd_dota2_epoch12_ssval_eval_20260602/preds.pkl`; metric summary `docs/experiments/20260602_opensrd_dota2_epoch12_ssval_metrics.json`. This is below the prior official DOTA2 `ss_val` evaluator result `dota/mAP=0.6510`, `dota/AP50=0.6510` by `-0.2308` mAP and `-0.2310` AP50. Keep claims narrow: this is DOTA2 `ss_val` evidence for the completed OpenRSD DOTA2 epoch-12 checkpoint, not a GeoNexus S2/S3 result.
- Oriented R-CNN 3x is the close secondary baseline, with best epoch 33/34 `dota/mAP=0.2620` and `dota/AP50=0.2620`; primary checkpoint path for the summary is `/data5/2025/ldh/OpenRSD/work_dirs/strong_baseline_dota15/oriented_rcnn_3x_loadfrom/epoch_33.pth`; metric summary `docs/experiments/20260526_oriented_rcnn_3x_dota15_metrics.json`.
- ReDet pretrained completed 12 epochs with best/final `dota/mAP=0.2382` and `dota/AP50=0.2380`; checkpoint `/data5/2025/ldh/OpenRSD/work_dirs/strong_baseline_dota15/redet_pretrained_rerun/epoch_12.pth`; metric summary `docs/experiments/20260526_redet_pretrained_dota15_metrics.json`.
- The earlier corrected Oriented R-CNN 12-epoch baseline remains an archived reference with `map=0.2561` and `AP50=0.2560`; checkpoint `/data5/2025/ldh/OpenRSD/work_dirs/strong_baseline_dota15/oriented_rcnn/epoch_12.pth`; metric summary `docs/experiments/20260525_oriented_rcnn_dota15_epoch12_metrics.json`.
- The key strong-baseline fix was validation/test pipeline ordering: resize the image before `LoadAnnotations`, then convert qbox to rbox and pack explicit meta keys. Loading annotations before resize produced near-zero AP by evaluating against mis-scaled GT boxes.
- The earlier RoI Transformer 1x low-LR rerun and ReDet scratch rerun are superseded by the completed 3x RoI Transformer and pretrained ReDet records. Keep their logs only as troubleshooting history.
- Mid-run detector curves, class-wise snapshots, and figure/table TODOs are recorded in `docs/experiments/20260525_strong_detector_midrun_records.md`.
- The complete paper-indicator experiment matrix and current download/staging list are recorded in `docs/setup/complete_experiment_plan.md`.
- Use `docs/experiments/20260524_dota_v15_anchor_repair_quick_test.md` and `docs/setup/strong_baseline_checklist.md` as the active planning anchors.
- S1 real VLM embedding support passed the RemoteCLIP smoke test (`classes=16`,
  `embedding_shape=[16, 512]`), using the checkpoint symlink at
  `/data5/2025/ldh/OpenRSD/checkpoints/remoteclip/RemoteCLIP-ViT-B-32.pt`.
  Earlier S1 launches on 2026-06-03 failed with CUDA OOM before checkpointing,
  but the patched retry 2 completed 36 epochs on 2026-06-04.
- 2026-06-04 GPU pruning is archived in
  `docs/experiments/20260604_gpu_pruning_and_next_priority.md`: lower-priority
  `zwl` jobs on GPUs 0/1/2/4 were stopped after checkpoint confirmation, GPU 3
  was left untouched, GeoNexus S1 retry 2 stayed active on GPU 5 with current
  best epoch 25 `dota/mAP=0.376255`, and DOTA2 ORCNN stayed active on GPU 6
  with current best epoch 8 `dota/mAP=0.585885`. The next priority is to finish
  and archive S1, then launch the next S2 hierarchy-regularizer rerun from the
  best S1 checkpoint before restarting secondary DOTA2 baselines.
- 2026-06-05 recovery update: GeoNexus S1 retry 2 completed 36 epochs. Best
  epoch 32: `dota/mAP=0.3800`, `dota/AP50=0.3800`; final epoch 36:
  `dota/mAP=0.3793`, `dota/AP50=0.3790`; metric summary
  `docs/experiments/20260605_geonexus_s1_retry2_metrics.json`.
- 2026-06-05 recovery update: GeoNexus S2 hierarchy-regularizer rerun from S1
  epoch 32 completed 12 epochs. Best epoch 4: `dota/mAP=0.3858`,
  `dota/AP50=0.3860`; final epoch 12: `dota/mAP=0.3784`,
  `dota/AP50=0.3780`; metric summary
  `docs/experiments/20260605_geonexus_s2_rerun_s1e32_metrics.json`. Use
  `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s2_hierarchy_reg_rerun_s1e32_20260604/epoch_4.pth`
  as the S3 initialization checkpoint.
- 2026-06-05 recovery update: GeoNexus S3 scene-adapter rerun from S2 epoch 4
  completed 12 epochs and released GPU 1. Best epoch 2: `dota/mAP=0.3827`,
  `dota/AP50=0.3830`; final epoch 12: `dota/mAP=0.3756`,
  `dota/AP50=0.3760`; metric summary
  `docs/experiments/20260605_geonexus_s3_rerun_s2e4_metrics.json`.
  Compared with S1 retry2 best epoch 32 `0.3800/0.3800` and S2 rerun best
  epoch 4 `0.3858/0.3860`, hierarchy is currently positive but the scene
  adapter is not yet stable enough to support the paper claim. Do not launch S4
  yet. The next diagnostic priority is a controlled S3 repair from S2 epoch 4
  using an identity-initialized scene adapter that honors `scene_adapter_dim`
  and a reduced residual scale.
- 2026-06-05 launch update: the controlled S3 repair run was launched on GPU 1
  in screen `geonexus_s3_identity_rerun_s2e4_20260605_gpu1`. Work dir:
  `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s3_scene_adapter_identity_rerun_s2e4_20260605`;
  config:
  `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s3_scene_adapter_identity_rerun_s2e4_20260605/roi-trans-le90_r50_fpn_remoteclip-s3-identity-rerun-s2e4-20260605_dota15.py`;
  launch log:
  `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s3_scene_adapter_identity_rerun_s2e4_20260605/launch_20260605_gpu1.log`.
  Startup acceptance passed at `Epoch(train) [1][200/1410]` with no
  `Traceback`, CUDA OOM, `libpng`, `CRC`, or `NoneType` signature in the log.
- 2026-06-05 recovery update: the controlled S3 identity repair from S2 epoch
  4 completed 12 epochs. Best epoch 9: `dota/mAP=0.3806`,
  `dota/AP50=0.3810`; final epoch 12: `dota/mAP=0.3792`,
  `dota/AP50=0.3790`; metric summary
  `docs/experiments/20260605_geonexus_s3_identity_rerun_s2e4_metrics.json`.
  This did not recover the prior S3 rerun best `0.3827/0.3830` or the S2
  rerun best `0.3858/0.3860`. The final S3 diagnostic is an adapter-off rerun
  from the same S2 epoch-4 checkpoint to isolate whether degradation comes from
  scene modulation or the S3 head/config transition.
- 2026-06-05 launch update: the final S3 adapter-off diagnostic was launched
  on GPU 3 in screen `geonexus_s3_adapter_off_rerun_s2e4_20260605_gpu3`.
  Work dir:
  `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s3_adapter_off_rerun_s2e4_20260605`;
  config:
  `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s3_adapter_off_rerun_s2e4_20260605/roi-trans-le90_r50_fpn_remoteclip-s3-adapter-off-rerun-s2e4-20260605_dota15.py`;
  launch log:
  `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s3_adapter_off_rerun_s2e4_20260605/launch_20260605_gpu3.log`.
  Preflight passed: GPU 3 stayed at `14 MiB` and `0%` over three polls, the
  config parsed with both cascade bbox heads reporting `use_scene_adapter=False`,
  and startup reached `Epoch(train) [1][200/1410]` with no `Traceback`, CUDA
  OOM, `libpng`, `CRC`, or `NoneType` signature in the launch log.
- 2026-06-06 recovery update: the final S3 adapter-off diagnostic from S2
  epoch 4 completed 12 epochs. Best epoch 3: `dota/mAP=0.3772`,
  `dota/AP50=0.3770`; final epoch 12: `dota/mAP=0.3758`,
  `dota/AP50=0.3760`; metric summary
  `docs/experiments/20260606_geonexus_s3_adapter_off_rerun_s2e4_metrics.json`.
  This stayed below the stop threshold `0.3827` and below the S2 rerun best
  `0.3858/0.3860`. Stop S3 diagnostics and do not launch S4 from S3. The next
  GeoNexus paper-path run is an S2 hierarchy-stabilization refinement from S2
  epoch 4 with LR `1e-4`, not S4.
- 2026-06-05 DOTA2 secondary status: ORCNN completed epoch 12 at
  `dota/mAP=0.5973`, `dota/AP50=0.5970`; S2ANet completed epoch 12 at
  `dota/mAP=0.5869`, `dota/AP50=0.5870`; R3Det, RTMDet-M, and RTMDet-L were
  interrupted by `KeyboardInterrupt`. Do not auto-restart those secondary runs
  before the next GeoNexus paper-path run. Status note:
  `docs/experiments/20260605_dota2_baseline_status.md`.
- 2026-06-05 GPU state before S3 launch planning: screens for our training runs
  are gone; GPUs 0, 1, 3, 5, and 6 are effectively free, while user `lyc`
  owns compute jobs on GPUs 2 and 4. Prefer GPU 1 for S3 if three consecutive
  `nvidia-smi` polls keep it at `memory.used <= 4000 MiB` and `util <= 10%`;
  otherwise use GPU 5 or GPU 6. Do not use GPU 3 for the next S3 launch.
- 2026-06-06 DOTA2 secondary update: RTMDet-M resumed and completed epoch 12
  with `dota/mAP=0.3312`, `dota/AP50=0.3310`. R3Det-KFIoU is active on GPU 5
  in screen `s0_dota2_r3det_kfiou_validpng_bs1_resume_20260605_gpu5`, currently
  epoch 10 with about nine hours remaining as of `2026-06-06 09:45 +0800`.
  RTMDet-L remains the only unfinished secondary baseline resume candidate;
  resume from
  `/data5/2025/ldh/OpenRSD/work_dirs/s0_dota2_1024_500_rtmdet_l_validpng_bs1_20260603/epoch_4.pth`
  only if GPU 6 passes the three-poll idle check.
- 2026-06-06 launch update: with R3Det still active on GPU 5, GPU 1 and GPU 6
  passed three idle polls (`14 MiB`, `0%` each). RTMDet-L was resumed on GPU 6
  in screen `s0_dota2_rtmdet_l_validpng_bs1_resume_20260606_gpu6`; launch log
  `/data5/2025/ldh/OpenRSD/work_dirs/s0_dota2_1024_500_rtmdet_l_validpng_bs1_20260603/launch_resume_20260606_gpu6.log`;
  startup passed at epoch 5 `[200/78014]` with no `Traceback`, CUDA OOM,
  `libpng`, `CRC`, `NoneType`, or immediate `KeyboardInterrupt`. GeoNexus S2
  hierarchy refinement from S2 epoch 4 was launched on GPU 1 in screen
  `geonexus_s2_hierarchy_refine_s2e4_lr1e4_20260606_gpu1`; work dir
  `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s2_hierarchy_reg_refine_s2e4_lr1e4_20260606`;
  config
  `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s2_hierarchy_reg_refine_s2e4_lr1e4_20260606/roi-trans-le90_r50_fpn_remoteclip-s2-hierarchy-reg-refine-s2e4-lr1e4-20260606_dota15.py`;
  launch log
  `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s2_hierarchy_reg_refine_s2e4_lr1e4_20260606/launch_20260606_gpu1.log`;
  startup passed at epoch 1 `[200/1410]` with no listed failure signatures.
- 2026-06-06 recovery update: GeoNexus S2 hierarchy refinement from S2 epoch 4
  with LR `1e-4` completed 12 epochs and released GPU 1. Best epoch 1:
  `dota/mAP=0.3804`, `dota/AP50=0.3800`; final epoch 12:
  `dota/mAP=0.3765`, `dota/AP50=0.3760`; metric summary
  `docs/experiments/20260606_geonexus_s2_refine_s2e4_lr1e4_metrics.json`.
  This did not improve the S2 rerun best epoch 4 `0.3858/0.3860`. The later
  lower-LR S2 refinement is now archive-only under the 2026-06-06 DOTA2
  cross-dataset pivot.
- 2026-06-06 launch update: the lower-LR S2 refinement from S2 epoch 4 was
  launched on GPU 1 with LR `5e-5` in screen
  `geonexus_s2_hierarchy_refine_s2e4_lr5e5_20260606_gpu1`. Work dir:
  `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s2_hierarchy_reg_refine_s2e4_lr5e5_20260606`;
  config:
  `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s2_hierarchy_reg_refine_s2e4_lr5e5_20260606/roi-trans-le90_r50_fpn_remoteclip-s2-hierarchy-reg-refine-s2e4-lr5e5-20260606_dota15.py`;
  launch log:
  `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota15/roi_trans_remoteclip_s2_hierarchy_reg_refine_s2e4_lr5e5_20260606/launch_20260606_gpu1.log`.
  Startup passed at epoch 1 `[200/1410]` with no `Traceback`, CUDA OOM,
  `libpng`, `CRC`, `NoneType`, `KeyboardInterrupt`, or early exit signature.
  It was stopped by research pivot on 2026-06-06 at about epoch 3; preserve the
  partial log/checkpoint artifacts and do not relaunch as paper-path evidence.
- 2026-06-07 DOTA2 secondary update: R3Det-KFIoU valid-PNG bs1 completed epoch
  12 with `dota/mAP=0.5633`, `dota/AP50=0.5630`. Checkpoint:
  `/data5/2025/ldh/OpenRSD/work_dirs/s0_dota2_1024_500_r3det_kfiou_validpng_bs1_20260603/epoch_12.pth`;
  metric log:
  `/data5/2025/ldh/OpenRSD/work_dirs/s0_dota2_1024_500_r3det_kfiou_validpng_bs1_20260603/20260605_100954/20260605_100954.log`.
  RTMDet-L remains active on GPU 6, currently epoch 11; latest validation is
  epoch 8 `dota/mAP=0.3521`, `dota/AP50=0.3520`.
- 2026-06-07 DIOR-R update: the Oriented R-CNN R50 baseline completed epoch 12
  but is invalid evidence. Epoch 4/8/12 validation stayed `0.0000/0.0000`, and
  training hit `loss: nan` from epoch 2 onward. Preserve the checkpoint/logs,
  but do not cite it as DIOR-R baseline evidence until the NaN cause is fixed.
- 2026-06-07 DIOR-R diagnostic update: the ORCNN R50 low-LR diagnostic
  `/data5/2025/ldh/OpenRSD/work_dirs/s0_dior_r_orcnn_r50_nan_diag_lr2p5e4_2e_20260607`
  also failed as evidence. It first hit NaN at `2026-06-07 10:28:33 +0800`,
  `Epoch(train) [1][650/5862]`, `lr=2.5000e-04`, with `grad_norm: nan` and
  `loss: nan`; final epoch-2 validation at `2026-06-07 11:07:21 +0800` was
  `dota/mAP=0.0000`, `dota/AP50=0.0000`, with all classes at `dets=0`.
  Checkpoints `epoch_1.pth` and `epoch_2.pth` plus launch log
  `launch_20260607_gpu5.log` are preserved. Low LR alone did not fix DIOR-R
  ORCNN; do not cite it as DIOR-R baseline evidence. Use DIOR-R RoI
  Transformer S0 next.
- 2026-06-07 DOTA2 GeoNexus launch update: created DOTA2-specific taxonomy
  `/data5/2025/ldh/New/assets/hierarchies/dota2_remote_sensing_taxonomy.json`
  in the exact 18-class DOTA2 config order, including `airport` and `helipad`.
  Generated RemoteCLIP artifact
  `/data5/2025/ldh/New/artifacts/generated/remoteclip_vit_b32_dota2_prompt_embeddings.pt`;
  validation confirmed `class_names` length 18, `embeddings` shape `[18, 512]`,
  finite values, and L2-normalized vectors.
- 2026-06-07 launch update: DOTA2 GeoNexus RoI Transformer + RemoteCLIP S1 was
  launched on GPU 1 in screen
  `geonexus_dota2_roi_trans_s1_validpng_20260607_gpu1`. Work dir:
  `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s1_validpng_20260607`;
  config:
  `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s1_validpng_20260607/roi-trans-le90_r50_fpn_remoteclip-s1-validpng-20260607_dota2.py`;
  launch log:
  `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s1_validpng_20260607/launch_20260607_gpu1.log`.
  The run initializes from the completed DOTA2 RoI Transformer valid-PNG
  checkpoint
  `/data5/2025/ldh/OpenRSD/work_dirs/s0_dota2_1024_500_roi_trans_rebuild_20260602_validpng_restart/epoch_12.pth`,
  trains on `DOTA2_1024_500/train/annfiles_validpng_20260602`, and validates
  on `DOTA2_1024_500/ss_val/annfiles`. Monitor until startup reaches
  `Epoch(train) [1][200/... ]` with no `Traceback`, CUDA OOM, `libpng`, `CRC`,
  `NoneType`, `ValueError`, or prompt class-count mismatch. Compare first
  validation against DOTA2 RoI Transformer S0 `0.6088/0.6090`; do not launch
  S2/S3/S4, FAIR1M, or more secondary baselines before S1 validates cleanly.
- 2026-06-07 DOTA2 RTMDet-L completion: resumed RTMDet-L valid-PNG bs1
  finished epoch 12 at `2026-06-07 15:04:34 +0800` with
  `dota/mAP=0.2779`, `dota/AP50=0.2780`; checkpoint
  `/data5/2025/ldh/OpenRSD/work_dirs/s0_dota2_1024_500_rtmdet_l_validpng_bs1_20260603/epoch_12.pth`.
  This degraded from epoch 8 `0.3521/0.3520`, so do not prioritize RTMDet-L
  further on DOTA2.
- 2026-06-07 fill-GPU launch plan: keep active DOTA2 GeoNexus S1 on GPU 1
  unchanged. Launch a lower-LR S1 replicate on GPU 6 from the same DOTA2 RoI
  Transformer S0 epoch-12 checkpoint, workdir
  `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s1_validpng_lr1e4_20260607`,
  LR `1e-4`, screen
  `geonexus_dota2_roi_trans_s1_validpng_lr1e4_20260607_gpu6`. Launch DIOR-R
  Rotated RetinaNet one-stage NaN probe on GPU 5, workdir
  `/data5/2025/ldh/OpenRSD/work_dirs/s0_dior_r_rotated_retinanet_r50_nan_probe_2e_20260607`,
  LR `1e-4`, `max_epochs=2`, `val_interval=1`, screen
  `s0_dior_r_rotated_retinanet_nan_probe_2e_20260607_gpu5`. If the DIOR-R
  probe hits NaN, stop it and record the first NaN before any more DIOR-R
  training.
- 2026-06-07 literature update: the living tracker
  `docs/literature/20260607_openrsd_related_recent_papers.md` records recent
  OpenRSD-adjacent and RS-OVD sources including OpenRSD, RS-MPOD, DisDop,
  SOAR, VK-Det, OTA-Det, InstructSAM, OS-W2S, CastDet, LAE, CoseDet, SCORE,
  and FLAME. Future agents must read and refresh this tracker before proposing
  route changes, paper claims, S3/S4, pseudo-labeling, or new prompt modules.

Paper-level claims require:

- A credible oriented detector baseline, preferably from MMRotate or an
  equivalent strong implementation.
- Real text/image embeddings such as CLIP, SkyCLIP, or RemoteCLIP.
- Verified tiling, class mapping, rotated IoU/NMS, and mAP.
- Complete ablations with real numbers.

## Experiment Sequence

Run experiments in this order:

1. DOTA2 S0: complete and archive strong closed-set baselines. Completed
   baselines: RoI Transformer `0.6088/0.6090`, Oriented R-CNN
   `0.5973/0.5970`, S2ANet `0.5869/0.5870`, R3Det-KFIoU `0.5633/0.5630`,
   RTMDet-M `0.3312/0.3310`, and RTMDet-L final `0.2779/0.2780` on
   `DOTA2_1024_500/ss_val`. RTMDet-L is completed and deprioritized.
2. DOTA2 GeoNexus S1/S2: port only the strongest defensible module first:
   hierarchy-aware prompt scoring or hierarchy regularization on the strongest
   stable DOTA2 detector. GPU-1 S1 completed cleanly at epoch 12 with
   `0.6177/0.6180`, so launch DOTA2 S2 from that checkpoint while the GPU-0
   and GPU-6 S1 replicates continue as comparison evidence. Do not run S3/S4
   until DOTA2 S1/S2 and the DIOR-R path are stable.
3. DIOR-R S0/S1/S2: complete on sanitized `DIOR_R_dota/train_val` to
   `DIOR_R_dota/test`. Use RoI Transformer S0 final `0.6544`, S1 final mean
   `0.6720`, S2 best mean `0.6884`, and S2 final mean `0.6853` as the current
   cross-dataset progression evidence.
4. FAIR1M: stretch evidence after DOTA2 and DIOR-R are stable. Use it for
   fine-grained hierarchy claims only.
5. S3/S4 and optional routing: run only after S1/S2 provide credible evidence
   on the DOTA2 and DIOR-R path.

Do not add S5 to the main story unless S2-S4 already show stable gains.

If the scaffold baseline is still near zero after diagnosis, pause S1-S5 and
fix the detector-localization path first or pivot to the stronger detector path.

Required final analyses:

- Main comparison table.
- Core ablation table.
- Prompt robustness table.
- Pseudo-label quality table.
- Efficiency table.
- Qualitative detections.
- Accepted/rejected pseudo-label examples.
- Confusion matrix or fine-grained class-pair analysis.

No final submission may contain pending/planned result tables.

## Paper-First Workflow

Canonical manuscript source:

- `docs/geonexus_short_paper.tex`

Supporting drafts and presentation notes may exist, but they must not override
the canonical manuscript. Keep method wording aligned with the real claim:
hierarchical prompts, scene/context adaptation, and VLM-assisted pseudo-label
purification. When code exposes routing or compression hooks, document them as
optional ablations or future work unless measured results justify making them
central.

Before any paper-facing claim is added:

- identify which experiment record supports it
- identify whether the support is paper-facing evidence, archive/debug
  evidence, or future inspiration
- link the config and command used to produce it
- record the exact dataset version and split, especially distinguishing
  DOTA v1.0, DOTA v1.5, DOTA2, DIOR-R, and FAIR1M
- record whether embeddings are hash fallback or real VLM embeddings
- record whether metrics are from scaffold evaluation or accepted DOTA-style
  evaluation
- read and, if needed, refresh the recent-paper tracker before making
  literature or route claims

## Local And Server Workflow

Use GitHub as the shared code and result-metadata transport between:

- local machine: code editing, documentation, lightweight tests
- experiment server: training, evaluation, logs, heavy artifacts

Recommended loop:

1. Local: edit code/docs/configs.
2. Local: run unit tests and smoke tests.
3. Local: commit and push to GitHub.
4. Server: pull GitHub.
5. Server: link datasets/checkpoints outside Git.
6. Server: run experiments.
7. Server: save logs and small structured summaries in Git-tracked locations.
8. Server: keep large outputs/checkpoints outside Git.
9. Server: commit and push code/config/log-summary changes.
10. Local: pull and continue analysis or code improvement.

Never put datasets, model checkpoints, raw large logs, or generated training
directories in Git. Commit configs, scripts, documentation, small metrics JSON,
environment notes, and experiment summaries.

## Repository Boundary

Tracked in Git:

- `assets/hierarchies/`
- `assets/prompts/`
- `configs/`
- `docs/`
- `scripts/`
- `src/`
- `tests/`
- root metadata and instruction files
- canonical paper source only, not duplicate generated PDFs

Ignored or external:

- `DOTA/`
- `DOTAv2/`
- `images/`
- `labels/`
- `outputs/`
- `checkpoints/`
- `artifacts/generated/`
- `wandb/`
- generated PDFs and LaTeX auxiliary files

## Future-Agent Prompt

When starting a new coding session, give the agent this instruction:

Read `PROJECT_INSTRUCTIONS.md`, then inspect the current Git status. Preserve
unrelated user changes. Continue the GeoNexus-RSD DOTA2-first JSTARS path:
read `docs/literature/20260607_openrsd_related_recent_papers.md` before
proposing new modules or route changes; search recent primary sources when the
user asks for planning, route changes, or paper claims; distinguish
paper-facing evidence, archive/debug evidence, and future inspiration; do not
make unsupported performance claims; keep routing/compression secondary;
maintain the local/server GitHub workflow; treat DOTA v1.5 as archive-only
diagnostic evidence; make DOTA2 the primary benchmark; make DIOR-R the required
cross-dataset validation; and update the canonical manuscript before code/docs
when the research direction changes.

For active experiment monitoring, every pass must check `screen -ls`,
`nvidia-smi`, and the active run log before reporting status. If a run is gone
or the log shows a failure, first read the traceback and classify the reason.
For `CUDA out of memory`, wait for an allowed physical GPU with
`memory.used <= 4000 MiB` and `util <= 10%` for three consecutive polls before
restarting there. For `libpng`, `CRC`, `NoneType`, or other data-read errors,
identify the bad file/sample first and do not relaunch unchanged unless that
input is fixed or excluded. For import/config errors, fix the environment or
config before relaunch. For an unknown traceback, record the traceback and
allow one clean-GPU relaunch; if the same traceback repeats, stop and fix it.
Cap automatic retries at three per experiment. Each retry must use a new log
name containing the retry index and GPU, and the handoff note must record the
failure reason plus restart command. If `last_checkpoint` exists, resume from
it; otherwise restart from epoch 0. Do not relaunch DOTA v1.5 GeoNexus
refinements unless the user explicitly asks for archive/debug work.

## Active Server Runs

- DOTA2 RTMDet-L baseline completed on GPU 6. Final epoch 12:
  `dota/mAP=0.2779`, `dota/AP50=0.2780`; epoch 8 was stronger at
  `0.3521/0.3520`, so RTMDet-L should not be prioritized further.
- DOTA2 GeoNexus RoI Transformer + RemoteCLIP S1 on GPU 1 completed cleanly.
  Workdir:
  `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s1_validpng_20260607`;
  checkpoint:
  `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s1_validpng_20260607/epoch_12.pth`;
  final metric: `dota/mAP=0.6177`, `dota/AP50=0.6180`.
- DOTA2 GeoNexus S2 hierarchy regularizer is active on GPU 1 in screen
  `geonexus_dota2_roi_trans_s2_hierarchy_reg_s1e12_20260608_gpu1`. Workdir:
  `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s2_hierarchy_reg_s1e12_20260608`;
  config:
  `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s2_hierarchy_reg_s1e12_20260608/roi-trans-le90_r50_fpn_remoteclip-s2-hierarchy-reg-s1e12-20260608_dota2.py`;
  launch log:
  `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s2_hierarchy_reg_s1e12_20260608/launch_20260608_gpu1.log`.
  It initializes from GPU-1 S1 `epoch_12.pth`; startup acceptance passed at
  `Epoch(train) [1][200/39007]` with finite `s0.loss_hierarchy` and
  `s1.loss_hierarchy`.
- DOTA2 GeoNexus RoI Transformer + RemoteCLIP S1 low-LR replicate is active
  on GPU 6 in screen
  `geonexus_dota2_roi_trans_s1_validpng_lr1e4_20260607_gpu6`. Workdir:
  `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s1_validpng_lr1e4_20260607`;
  config:
  `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s1_validpng_lr1e4_20260607/roi-trans-le90_r50_fpn_remoteclip-s1-validpng-lr1e4-20260607_dota2.py`;
  launch log:
  `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s1_validpng_lr1e4_20260607/launch_20260607_gpu6.log`.
  It is a hedge for the active GPU-1 S1 run, changes only optimizer LR to
  `1e-4` plus workdir/log identity, and should be compared against DOTA2 RoI
  Transformer S0 `dota/mAP=0.6088`, `dota/AP50=0.6090` after first validation.
- DOTA2 GeoNexus RoI Transformer + RemoteCLIP S1 lower-LR replicate is active
  on GPU 0 in screen
  `geonexus_dota2_roi_trans_s1_validpng_lr5e5_20260607_gpu0`. Workdir:
  `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s1_validpng_lr5e5_20260607`;
  launch log:
  `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s1_validpng_lr5e5_20260607/launch_20260607_gpu0.log`.
  Live check at `2026-06-08 09:28 +0800`: screen still exists and
  `nvidia-smi` shows PID `799811` using GPU 0. Latest log status was
  `Epoch(train) [9][32800/39007]` with ETA about `5:57:02`. Validation so far:
  epoch 4 `dota/mAP=0.5929`, `dota/AP50=0.5930`; epoch 8
  `dota/mAP=0.6046`, `dota/AP50=0.6050`, still below DOTA2 RoI Transformer S0
  `0.6088/0.6090`.
- 2026-06-07 DIOR-R one-stage probe result: Rotated RetinaNet was launched on
  GPU 5 in screen `s0_dior_r_rotated_retinanet_nan_probe_2e_20260607_gpu5`
  after three idle polls, reached startup acceptance at
  `2026-06-07 15:41:44 +0800`, `Epoch(train) [1][200/5862]`, with
  `grad_norm=1.0693` and `loss=2.1723`, then hit first non-finite loss at
  `2026-06-07 15:43:59 +0800`, `Epoch(train) [1][1200/5862]`,
  `lr=1.0000e-04`, `grad_norm=1.3637`, `loss=inf`, `loss_cls=1.2532`,
  `loss_bbox=inf`. The screen was stopped and GPU 5 returned idle. Preserve
  workdir
  `/data5/2025/ldh/OpenRSD/work_dirs/s0_dior_r_rotated_retinanet_r50_nan_probe_2e_20260607`
  and launch log
  `/data5/2025/ldh/OpenRSD/work_dirs/s0_dior_r_rotated_retinanet_r50_nan_probe_2e_20260607/launch_20260607_gpu5.log`.
  Do not launch more DIOR-R detector training until DIOR-R data, rotated-box
  conversion, and loss targets are diagnosed.
- 2026-06-08 DOTA2 GeoNexus S1 completion: the GPU-1 S1 run completed cleanly
  at `2026-06-08 09:33:42 +0800`. Workdir:
  `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s1_validpng_20260607`;
  config:
  `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s1_validpng_20260607/roi-trans-le90_r50_fpn_remoteclip-s1-validpng-20260607_dota2.py`;
  log:
  `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s1_validpng_20260607/20260607_101146/20260607_101146.log`;
  checkpoint:
  `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s1_validpng_20260607/epoch_12.pth`.
  Final epoch 12 on `DOTA2_1024_500/ss_val`: `dota/mAP=0.6177`,
  `dota/AP50=0.6180`, above DOTA2 RoI Transformer S0 `0.6088/0.6090`, GPU-6
  S1 epoch 8 `0.6017/0.6020`, and GPU-0 S1 epoch 8 `0.6046/0.6050`. Use this
  checkpoint as the DOTA2 S2 initialization point; do not wait for the active
  GPU-0/GPU-6 replicates before launching S2.
- 2026-06-08 DOTA2 S2 preparation: generated hierarchy prompt artifact
  `/data5/2025/ldh/New/artifacts/generated/remoteclip_vit_b32_dota2_s2_hierarchy_prompt_embeddings.pt`
  with RemoteCLIP ViT-B-32. Validation confirmed 18 classes, `embeddings`
  shape `[18, 512]`, finite embeddings, normalized embedding rows, and
  `relation_matrix` shape `[18, 18]`. Runtime config:
  `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s2_hierarchy_reg_s1e12_20260608/roi-trans-le90_r50_fpn_remoteclip-s2-hierarchy-reg-s1e12-20260608_dota2.py`.
- 2026-06-08 DOTA2 S2 launch update: DOTA2 hierarchy regularizer S2 launched
  on GPU 1 in screen
  `geonexus_dota2_roi_trans_s2_hierarchy_reg_s1e12_20260608_gpu1`. Workdir:
  `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s2_hierarchy_reg_s1e12_20260608`;
  launch log:
  `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s2_hierarchy_reg_s1e12_20260608/launch_20260608_gpu1.log`.
  It initializes from
  `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s1_validpng_20260607/epoch_12.pth`
  and startup acceptance passed at `2026-06-08 11:00:05 +0800` with
  `Epoch(train) [1][200/39007]`, finite losses and hierarchy losses, detached
  screen present, and PID `716070` active on GPU 1. No `Traceback`, CUDA OOM,
  `libpng`, `CRC`, `NoneType`, `ValueError`, `nan`, or `inf` signature was
  found before acceptance. Keep S3/S4 paused until DOTA2 S2 and the DIOR-R path
  are stable. DIOR-R detector training remains blocked pending data, rotated
  box conversion, class mapping, and loss-target diagnosis.
- Result monitor `s0_result_log_monitor_20260603` remains active.
- No active GeoNexus DOTA v1.5 training screen should exist after the
  2026-06-06 pivot. GPU 1 was freed after stopping
  `geonexus_s2_hierarchy_refine_s2e4_lr5e5_20260606_gpu1`.
- No DIOR-R detector training screen is active after the failed RoI Transformer
  attempt; GPU 5 is idle.

## Recent Stopped Runs

- DIOR-R RoI Transformer S0 was launched on GPU 5 in screen
  `s0_dior_r_roi_trans_r50_20260607_gpu5`, but was stopped after a NaN.
  Workdir:
  `/data5/2025/ldh/OpenRSD/work_dirs/s0_dior_r_roi_trans_r50_20260607`;
  config:
  `/data5/2025/ldh/OpenRSD/work_dirs/s0_dior_r_roi_trans_r50_20260607/G02_Baselines_Data2_DIOR_R_M2_RoITrans_20260607.py`;
  launch log:
  `/data5/2025/ldh/OpenRSD/work_dirs/s0_dior_r_roi_trans_r50_20260607/launch_20260607_gpu5.log`.
  Preflight passed with three GPU-5 idle polls at `14 MiB`, `0%`; startup
  passed at `2026-06-07 11:21:15 +0800` by reaching
  `Epoch(train) [1][200/5862]` with finite early loss/grad values. Acceptance
  failed at `2026-06-07 11:30:25 +0800`, `Epoch(train) [1][3375/5862]`,
  `lr=2.5000e-03`, with `grad_norm: nan`, `loss: nan`, and NaN RPN/cascade
  losses. No checkpoint was written; GPU 5 returned idle. Because both ORCNN
  and RoI Transformer hit NaN on DIOR-R, treat this as a DIOR-R
  detector/data/box-coder path issue and diagnose inputs/box conversion/loss
  targets before launching another DIOR-R detector unchanged.

## 2026-06-09 DOTA2 S2 Status And DIOR-R Policy

- As of `2026-06-09 09:37 CST`, the target of three concurrent DOTA2 S2 jobs
  is already met. Do not launch another detector training run while GPUs 0, 1,
  and 6 remain occupied.
- Active screens:
  `geonexus_dota2_roi_trans_s2_hierarchy_reg_s1e12_lr1e4_20260608_gpu0`
  (PID `2711973` on GPU 0),
  `geonexus_dota2_roi_trans_s2_hierarchy_reg_s1e12_20260608_gpu1`
  (PID `716070` on GPU 1), and
  `geonexus_dota2_roi_trans_s2_hierarchy_reg_s1e12_hw1e2_20260608_gpu6`
  (PID `2711971` on GPU 6).
- Completed DOTA2 S1 metrics on `DOTA2_1024_500/ss_val`: main GPU-1 S1 final
  `dota/mAP=0.6177`, `dota/AP50=0.6180`; GPU-6 LR `1e-4` final
  `0.5997/0.6000`; GPU-0 LR `5e-5` final `0.6047/0.6050`.
- Active DOTA2 S2 interim evidence on `DOTA2_1024_500/ss_val`: main GPU-1 S2
  epoch 4 `0.6038/0.6040`, epoch 8 `0.5892/0.5890`, currently epoch 12;
  GPU-0 low-LR S2 epoch 4 `0.6099/0.6100`, currently epoch 8; GPU-6 reduced
  hierarchy-weight S2 epoch 4 `0.6035/0.6040`, currently epoch 8. The low-LR
  S2 variant is the best S2 result so far, but it is still below main S1
  `0.6177/0.6180`.
- Scoped active-log checks found no training failure signature in the live
  tails; the broad substring `inf` only matched static config text such as
  `metainfo`. Continue to watch for `Traceback`, CUDA OOM, `libpng`, `CRC`,
  `NoneType`, `ValueError`, non-finite losses, and `KeyboardInterrupt`.
- Keep S3/S4, pseudo-labeling, FAIR1M, and DIOR-R detector training paused.
  When one GPU frees, use it for DIOR-R diagnostics only. Do not cite the
  invalid DIOR-R detector runs as DIOR-R baseline evidence.
- 2026-06-09 completion update: main DOTA2 S2 on GPU 1 finished at
  `2026-06-09 11:10:06 CST +0800`. Final epoch 12 on
  `DOTA2_1024_500/ss_val`: `dota/mAP=0.5924431681632996`,
  `dota/AP50=0.5920`, checkpoint
  `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s2_hierarchy_reg_s1e12_20260608/epoch_12.pth`.
  This is below main S1 `0.6177/0.6180`; do not launch S3/S4 from this result.
  GPU 0 low-LR S2 and GPU 6 reduced hierarchy-weight S2 remain active. The
  completed GPU-1 S2 log reached `Epoch(val) [12][6917/6917]`, and the
  completed-log failure scan found no `Traceback`, CUDA OOM, `libpng`, `CRC`,
  `NoneType`, `ValueError`, true non-finite loss/metric, or
  `KeyboardInterrupt` signature.
- 2026-06-09 DIOR-R diagnostic launch: after GPU 1 freed, launched diagnostic
  screen `dior_r_geometry_targets_diag_20260609_gpu1` using
  `New/scripts/diagnose_dior_r_geometry_and_targets.py --check-dataloader --check-first-loss`.
  This is diagnostic-only, not detector training. As of `2026-06-09 11:17 CST`,
  the screen is active with Python PID `2024594`, GPU 1 is Xorg-only, and
  `New/artifacts/dior_r_diagnostics_20260609_gpu1.log` exists but JSON/Markdown
  outputs are pending until the full scan and optional checks complete.
- 2026-06-09 evening live update: as of `2026-06-09 19:23 CST`, GPU-0 low-LR
  S2 and GPU-6 reduced hierarchy-weight S2 are still running in epoch 12.
  Their best observed validation metrics are GPU-0 epoch 4 `0.6099/0.6100`
  and GPU-6 epoch 8 `0.6044/0.6040`, both below S1 `0.6177/0.6180`. GPU-5
  LR `5e-5` S2 was already launched and is only diagnostic unless it produces
  a strong epoch-4 validation. The bounded CPU DIOR-R diagnostic completed on
  200 `train_val` and 200 `test` samples, confirming sampled image decode,
  label parsing, class order, and dataloader basics, but it is not a full data
  scan and did not produce a first non-finite batch record. Next DIOR-R work is
  a full geometry/statistics scan plus a bounded non-finite-loss catcher.
- 2026-06-09 full DIOR-R label-geometry follow-up: artifact
  `New/artifacts/dior_r_diagnostics_20260609_full_geometry.md` completed the
  full label scan with fallback rbox geometry and declared `800x800` bounds.
  It found four zero-area/invalid-size `ship` records across train/test and
  four test rbox centers outside declared bounds. Fix/filter those records and
  then run the bounded non-finite-loss catcher before any detector relaunch.
- 2026-06-10 DOTA2 S2 loss-0 replication launch: screen
  `geonexus_dota2_s2_loss0_rep3407_20260610_gpu1`, PID `1559651`, GPU 1.
  Config:
  `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s2_hierarchy_ablate_loss0_s1e12_rep3407_20260610/roi-trans-le90_r50_fpn_remoteclip-s2-hierarchy-ablate-loss0-s1e12-rep3407-20260610_dota2.py`.
  Log:
  `/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s2_hierarchy_ablate_loss0_s1e12_rep3407_20260610/20260610_191026/20260610_191026.log`.
  Startup acceptance passed at `Epoch(train) [1][200/39007]`; metrics are
  pending until epoch validation completes.

## 2026-06-20 Paper-Eval Status

- The 2026-06-17 DIOR-R paper-eval pack completed cleanly under
  `/data5/2025/ldh/OpenRSD/work_dirs/paper_eval_20260617/`: S0 epoch52
  `dota/mAP=0.654401421546936`, `dota/AP50=0.654`; S2 rep4 epoch12
  `0.6914003491401672/0.691`; S3 rep0 epoch8 `0.6991876363754272/0.699`.
  Each run produced `preds.pkl`, a runtime log, and a JSON metric file; scoped
  failure scans were clean.
- Today work is analysis-only DOTA2 paper evaluation on existing checkpoints.
  Keep S4, pseudo-labeling, FAIR1M, routing changes, and all new training
  paused unless explicitly overridden.

## 2026-06-24 DIOR-R S3 Route Status

- DIOR-R S3 stability long32 completed cleanly on `2026-06-23 CST`: rep0
  final `0.6960/0.6960`, rep1 final `0.6897/0.6900`, rep2 final
  `0.6941/0.6940`; final mean rounded mAP `0.6933`, best mean rounded mAP
  `0.6965`. Archive:
  `New/docs/experiments/20260624_dior_r_s3_stability_long32_complete.md`.
- Approved follow-up is DIOR-R S3 long60 continuation from each long32
  `epoch_32.pth`. Keep S4, pseudo-labeling, FAIR1M, DOTA2 follow-up training,
  and route changes paused unless separately approved.

## 2026-06-25 DIOR-R S3 Route Status

- DIOR-R S3 stability long60 completed cleanly on `2026-06-25 CST`: rep0
  final `0.694532/0.6950` with best epoch 51 `0.698892`, rep1 final
  `0.688295/0.6880` with best epoch 33 `0.692448`, and rep2 final
  `0.696215/0.6960` with best epoch 58 `0.698467`. Final mean exact mAP is
  `0.693014`; best mean exact mAP is `0.696602`. Archive:
  `New/docs/experiments/20260625_dior_r_s3_stability_long60_complete.md`.
- Approved follow-up is DIOR-R S3 long88 continuation from each long60
  `epoch_60.pth`, keeping all three replicas for controlled aggregate
  stability. Keep S4, pseudo-labeling, FAIR1M, DOTA2 follow-up training, and
  route changes paused unless separately approved.

## 2026-06-26 DIOR-R S3 Route Status

- DIOR-R S3 stability long88 completed cleanly on `2026-06-26 CST`: rep0
  final epoch 88 `0.6874858141/0.6870` with best epoch 66
  `0.6985080242/0.6990`, rep1 final epoch 88 `0.6918782592/0.6920` with best
  epoch 86 `0.6920918226/0.6920`, and rep2 final epoch 88
  `0.6982299089/0.6980` with best epoch 86 `0.6985765696/0.6990`. Final mean
  exact mAP is `0.6925313274`; best mean exact mAP is `0.6963921388`.
  Archive:
  `New/docs/experiments/20260626_dior_r_s3_stability_long88_complete.md`.
- Archive verification found only `s0_result_log_monitor_20260603` remaining
  in `screen`; original long88 PIDs `1652371`, `1652558`, and `1652680` were
  absent. GPUs 0-5 were idle, and GPU 6 was occupied by another user/process.
  The scoped failure scan over long88 launch/runtime logs was clean.
- Treat long88 as useful stability evidence but not an improvement over long60
  best mean `0.696602`. Pause further DIOR-R S3 continuation. Keep S4,
  pseudo-labeling, FAIR1M, DOTA2 follow-up training, route-changing
  experiments, and new long-continuation launches paused unless separately
  approved.

## 2026-06-28 DIOR-R S4 Route Status

- DIOR-R S4 pseudo-label short-pack from `2026-06-27` completed cleanly but
  degraded after epoch 1: best mean `dota/mAP=0.696903` with all three best
  checkpoints at epoch 1, final epoch-12 mean `dota/mAP=0.691337`. Treat S4
  as a stabilization test only, not as a new superiority claim. Archive and
  launch record:
  `New/docs/experiments/20260628_dior_r_s4_pseudolabel_low_lr_from_e1_launch.md`.
- Controlled S4 low-LR stabilization runs were launched on `2026-06-28
  10:15 CST` from each replica's S4 epoch-1 checkpoint with `lr=1e-5`,
  `max_epochs=8`, `val_interval=1`, `resume=False`, and the same pseudo-label
  data root `data/DIOR_R_dota_s4_pseudo_agreement_20260627/`. GPU mapping:
  rep23407 -> GPU 0, rep24407 -> GPU 2, rep25407 -> GPU 3; GPU 1 was avoided
  because PID `616621` was resident there. Accepted screens:
  `dior_r_s4_e1_lr1e5_rep23407_20260628_gpu0`,
  `dior_r_s4_e1_lr1e5_rep24407_20260628_gpu2`, and
  `dior_r_s4_e1_lr1e5_rep25407_20260628_gpu3`. PIDs: `743669`, `743673`,
  `743672`. Startup reached `Epoch(train) [1][ 450/5847]` in all three
  bootstrap logs with a clean scoped failure scan.
- Completion must report best and final `dota/mAP` plus `dota/AP50` for each
  replica and aggregate best/final means. Strong S4 evidence requires best mean
  above original DIOR-R S3 best mean `0.6979` or final mean clearly above S3
  long60 final mean `0.693014`; stabilization evidence requires final mean
  above S4 short-pack final mean `0.691337` with clean logs.

## 2026-06-29 DIOR-R S4 Route Status

- DIOR-R S4 low-LR continuation from `2026-06-28` completed through epoch 8
  for all three replicas. Accepted bootstrap/runtime logs were clean; the
  preserved direct-launch `tools/train.py` logs contain expected
  `geonexus_mmrotate` import `Traceback` entries before bootstrap relaunch.
  Archive:
  `New/docs/experiments/20260629_dior_r_s4_low_lr_complete.md`.
- Metrics: rep23407 best epoch 2 `0.6935/0.6930`, final epoch 8
  `0.6892/0.6890`; rep24407 best epoch 6 `0.6966/0.6970`, final epoch 8
  `0.6963/0.6960`; rep25407 best epoch 2 `0.6967/0.6970`, final epoch 8
  `0.6923/0.6920`. Aggregate best mean mAP is `0.6956`; aggregate final mean
  mAP is `0.6926`.
- Treat this as weak stabilization only, not paper-facing S4 superiority. The
  final mean improves over the S4 short-pack final `0.691337` by about
  `+0.0013`, but remains below S3 long60 final `0.693014`. The best mean
  remains below S4 short-pack best `0.696903` and original S3 best threshold
  `0.6979`.
- Pause further S4 training unless separately approved. Use `2026-06-29` for
  paper-facing evaluation artifacts on the best low-LR checkpoint from each
  replica.
- DIOR-R S4 paper-eval best-checkpoint audit was launched on `2026-06-29
  09:07 CST` with `tools/bootstrap_run.py tools/test.py --out preds.pkl`.
  Mapping: rep23407 epoch 2 -> GPU 0, rep24407 epoch 6 -> GPU 2, rep25407
  epoch 2 -> GPU 3. Startup acceptance passed at `Epoch(test) [ 350/5869]`
  or later with clean scoped failure scan. Launch record:
  `New/docs/experiments/20260629_dior_r_s4_paper_eval_best_launch.md`.
- The paper-eval audit completed at `2026-06-29 09:17 CST`. Each workdir
  contains `preds.pkl`, a runtime `.log`, and JSON metric file. Metrics match
  the training-log best-checkpoint values: rep23407 epoch 2 `0.6935/0.6930`,
  rep24407 epoch 6 `0.6966/0.6970`, rep25407 epoch 2 `0.6967/0.6970`.
  Final screen state returned to only `s0_result_log_monitor_20260603`, and
  GPUs 0, 2, and 3 returned to idle.
