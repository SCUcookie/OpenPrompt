# 2026-07-22 R1 Launch: Next Actions After The Port-And-Gate Status

## Basis

`20260722_r1_orcnn_port_and_gate_status.md` reports R1-P0 complete and
CPU-gated: `PromptRotatedShared2FCBBoxHead` and
`HierarchyPromptRotatedShared2FCBBoxHead` implemented, R1-S1/R1-S2 replica
configs staged for seeds 3407/4407/5407, artifacts verified (`[20,512]`
prompts, `[20,20]` relation matrix), source checkpoint epoch 28 verified by
SHA-256, checkpoint compatibility clean, forward/loss finite. Two blockers
stopped short of launch: the 1,000-step diagnostic was aborted at 50 steps
because two foreground GPU-0 jobs lacked detached-screen provenance, and
the shared EMA hook raises `AttributeError: ema_model` in the standalone
diagnostic (documented fix: `custom_hooks=[]` for the diagnostic only).

This note is the executable continuation. Steps N1–N3 are the server's
next session, in order; N4 runs in parallel on CPU; the paper track is
unaffected and listed at the end.

## N0 — GPU hygiene check (minutes)

Account for the two foreground jobs that interrupted the last diagnostic:
identify owner/PID, confirm they are finished or legitimately owned by
another user, and record the outcome in the launch note. Do not reuse
GPUs 0–1 while they carry pre-existing work; GPUs 2–6 were idle at the
last poll.

## N1 — Clean 1,000-step diagnostic rerun (one GPU, ~1 h)

Run from `/data5/2025/ldh/OpenRSD`, in a detached screen this time (the
diagnostic itself gets provenance, same as a training job):

```bash
screen -dmS r1_s1_diag1000_20260722_gpu<N> bash -lc '
CUDA_VISIBLE_DEVICES=<N> /data1/anaconda3/envs/zwl_mmrotate/bin/python \
  tools/bootstrap_run.py tools/diagnose_first_nonfinite_loss.py \
  <R1-S1 rep3407 config, with custom_hooks=[] override for this run only> \
  --work-dir work_dirs/geonexus_dior_r/r1_s1_precision_diag1000_20260722 \
  --out work_dirs/geonexus_dior_r/r1_s1_precision_diag1000_20260722/result.json \
  --max-batches 1000 --mode train-step --progress-interval 10 \
  2>&1 | tee work_dirs/geonexus_dior_r/r1_s1_precision_diag1000_20260722/launch_20260722.log'
```

Use a config copy with `custom_hooks=[]` for the diagnostic only — the
full training configs keep their original hooks. Acceptance:
`checked_batches == 1000`, `finite_within_limit`, clean scoped scan.

## N2 — R1-S1 launch (3 GPUs, ~8 h/run)

Only after N1 accepts. Three-poll GPU selection (dynamic; each selected
GPU below 1 GB and 0% in all three polls), then the three staged replica
configs (seeds 3407/4407/5407), each in a named detached screen
`r1_s1_rep<seed>_20260722_gpu<N>` with `launch_provenance.txt`
(timestamp, GPU, PID, exact command, config path, source checkpoint
epoch-28 path + SHA-256 `9f19f6dd…201b93`, seed) and a teed log. 12
epochs, validation/checkpoint interval 4, initialized from the Oriented
R-CNN epoch-28 best checkpoint. Startup acceptance: finite losses at
iterations 200 and 1,000 plus a clean scoped scan.

## N3 — Completion and the pre-registered R1 criteria

Archive per the completion-record conventions (per-replica best AND final,
mean/std for both, per-class table, provenance, failure scan, decision
boundary). The R1 success criteria are pre-registered in
`20260721_next_training_route_plan.md` and are not renegotiated after
seeing results: monotone stage ordering, total best-mean gain over the
63.41 baseline ≥ +2.0 for the full-stack generality claim; any stage
regression stops the campaign at that stage with no variant chasing.
After R1-S1 completes and reconciles: launch R1-S2 from each replica's
strongest S1 checkpoint using the already-staged S2 configs (same gate
chain, the 1,000-step diagnostic repeated for the S2 head).

## N4 — R1-S3 port gap (CPU, parallel with N2's training)

The port status covers the S1 prompt head and S2 hierarchy head only —
**the scene-context adapter head for Oriented R-CNN does not exist yet.**
Implement `SceneContextPromptRotatedShared2FCBBoxHead` (tile-descriptor
pooling + adapter MLP + adapted-prototype cosine scoring, mirroring the
cascade SCA head), stage R1-S3 replica configs, and run the CPU gate now
so R1-S3 can launch the day R1-S2 completes instead of losing a porting
day mid-campaign.

## Server skills installation (one-time, minutes)

Install the three server-side skills recorded in
`docs/setup/server_skills_20260722.md` into `/home/zwl/.codex/skills/`
(`gpu-launch-gate`, `experiment-completion-record`,
`mmrotate-failure-playbook`). They encode this project's launch chain,
completion-record format, and the verified failure fixes — including the
exact EMA-hook repair that N1 needs — so future server sessions enforce
them without re-reading project history. Local counterparts
(`tgrs-manuscript`, `server-update-review`) are already installed under
`New/.claude/skills/` and travel with the repo.

## Paper track (local, parallel, unaffected)

The manuscript is complete (8 pages, zero placeholders). Remaining:
advisor review → author-list finalization → submission package. Optional
polish: harbor/bridge-focused qualitative re-render. R1/R2 results are the
revision arsenal and do not block submission.

## Order of execution

1. N0 hygiene → N1 diagnostic → N2 launch (server, sequential).
2. N4 SCA-head port (server or local CPU, parallel with N2).
3. Server skills install (any time).
4. R1-S1 completion record → R1-S2 launch → (R1-S3 once N4 lands).
5. Paper: advisor review proceeds independently.
