# 2026-07-22 R1 Oriented R-CNN Port And Gate Status

## Scope

Implemented and staged R1-P0 for the DIOR-R cross-detector generality route.
R1-S1 remains gated and was not launched.

## Implementation

- Added `PromptRotatedShared2FCBBoxHead`, inheriting the rotated MMRotate bbox
  head so Oriented R-CNN retains its five-parameter rotated-box loss path.
- Added `HierarchyPromptRotatedShared2FCBBoxHead` for R1-S2 relation loss.
- Added R1-S1 and R1-S2 configs, including replica configs for seeds `3407`,
  `4407`, and `5407`.
- Added `runtime_imports.py` to register the repository-local
  `OrientedRPNHead` alongside the installed OpenMMLab packages used by the
  project bootstrap wrapper.
- Installed the curated `jupyter-notebook` skill at
  `/home/zwl/.codex/skills/jupyter-notebook`.

## Passed gates

The CPU-safe gate report is
`OpenRSD/work_dirs/geonexus_dior_r/r1_p0_cpu_gate_20260722.json`.

- Prompt artifact: `[20,512]`; SHA-256
  `a7cde254fcb87a741fde89a0d448e72452687c122644b75a5247615b296389ad`.
- Hierarchy artifact: embeddings `[20,512]`, relation matrix `[20,20]`.
- Sanitized data: `11725` train image-label pairs and `11738` validation
  image-label pairs.
- Source checkpoint: epoch 28, SHA-256
  `9f19f6dd7a9e2818544d345a597901ec5bb9d247c1b1880f28bceb0426201b93`.
- Schedule: 12 epochs, validation and checkpoint interval 4.
- Rotated-head forward/loss test: finite classification and regression loss.
- Checkpoint compatibility: six expected missing prompt parameters, zero
  unexpected keys.
- Host model construction through `run_bootstrap.sh`: passed.

## Blocking result

The required exact 1,000-step train-step diagnostic is incomplete. It reached
50 finite steps on the host, then was stopped because two foreground GPU-0
jobs had been unintentionally left running without detached-screen provenance.
The shared diagnostic's EMA hook also raises `AttributeError: ema_model` during
checkpoint loading; use `custom_hooks=[]` for this diagnostic only.

No R1-S1 replica was launched, no launch provenance record was created, and no
scientific result is claimed. Final GPU poll: GPUs 2-6 idle, GPUs 0-1 occupied
by pre-existing work. The next authorized action is to rerun the 1,000-step
diagnostic cleanly, then perform three consecutive GPU polls and launch only
with detached-screen provenance.
