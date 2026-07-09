# OpenPrompt Session Start

Use this file as the single source of truth for future sessions.

If you open a new session, the only instruction you need is:

```text
Read /data5/2025/ldh/OpenPrompt/SESSION_START.md and start working from it.
```

## Startup Protocol

Prefer `New/` as the project memory root for current agent startup state.
Start with `New/AGENTS.md`, then read only the top operational sections of
`New/PROJECT_INSTRUCTIONS.md` plus the latest dated experiment note relevant to
the task. Do not load or summarize the full history unless the task requires
route evidence or provenance.

Use concise, status-first workflows. For experiment status, report screen names,
GPU residency, latest log marker, scoped failure-scan result, and next action.

Every GPU launch must leave a first-run record with intent, timestamp, screen,
GPU IDs, workdir, config, source checkpoint, exact launch command, log paths,
PID/process check, startup marker, and GPU remapping decisions. When recording
a finished experiment, align the final result back to that first-run record and
reconcile metrics, checkpoints, failure scan, and any deviations.

GPU checks from the normal sandbox can be misleading because host `/dev/nvidia*`
may be hidden. Use approved or escalated host access for real GPU process checks
and launches. The standard checks are:

```bash
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
ps -p PID1,PID2,PID3 -o pid,ppid,user,cmd --forest
```

## Project Identity

- Main research repo: `/data5/2025/ldh/OpenPrompt`
- Official-style reference repo: `/data5/2025/ldh/OpenRSD`
- Main paper direction: `GeoNexus-RSD`
- Working scope: open-prompt rotated remote sensing detection built on top of OpenRSD ideas, but developed mainly inside `OpenPrompt`

## Hard Constraints

- The full original OpenRSD asset package is too large to reproduce completely.
- The practical plan must work with a reduced local setup.
- The deadline is short:
  finish the main experiments and begin paper writing within 1 month.
- Do not spend the month on broad re-planning.
- Do not drift into a large new framework or agent-heavy story.
- Before planning route changes, paper claims, or new modules, read and update
  the living literature tracker:
  `docs/literature/20260607_openrsd_related_recent_papers.md`.

## Current Status

- `OpenRSD` is locally available and usable as a reference implementation.
- `OpenPrompt` is the main repo for research and paper-facing experiments.
- On this server, `zwl_oneformer_ViT_P` is the working training environment;
  `dlp` is not suitable because it lacks `torch`, and cuDNN must stay disabled
  in the repo runtime hook to avoid CUDA convolution segfaults on the RTX 4090.
- The previous full-image baseline was trainable but produced nearly zero useful validation detection quality.
- The main suspected causes were:
  - too few query slots
  - destructive resizing for dense DOTA scenes
- A better reproduction baseline already exists:
  [`configs/experiments/dota_v1_baseline_repro.yaml`](./configs/experiments/dota_v1_baseline_repro.yaml)
  and [`configs/experiments/dota_v15_baseline_repro.yaml`](./configs/experiments/dota_v15_baseline_repro.yaml)
- Those baselines use:
  - `1024x1024` tiles
  - `16x16` query grid
  - a more realistic setup for dense DOTA tiles
- The reduced tiled DOTA v1.0 validation has now completed with `map50=3.326794065590851e-06` on 4055 images; the pipeline is verified end-to-end, but the detector is still a weak sanity-check baseline.
- The matched DOTA v1.5 baseline training and validation evaluation have completed; the result is `map50=1.0926445202230628e-05` on 4055 images, which is still only a sanity-check baseline.
- Quick diagnostics show the weak v1.5 result is not a thresholding artifact: decoded scores stay above `0.05`, `0.01`, and `0.001`, predictions collapse toward `small-vehicle`/`harbor`/`plane`/`ship`, and a sample tile shows center-biased boxes with very low same-class IoU.
- `QueryGenerator` currently produces `query_centers`, but the box heads ignore them. The next step is to inspect and repair the scaffold localization path before starting S1-S5 prompt ablations.
- Anchor-repair quick test completed and produced `outputs/dota_v15_anchor_repair/epoch_001.pt`; the final training metrics were `loss=0.07363908355801901`, `loss_cls=0.001671954903589549`, `loss_box=0.035983564312892485`, `positive_cls_acc=0.5529336195676059`, and `positive_box_l1=0.10294117139314753`.
- Strong-baseline preparation should proceed in parallel using `docs/setup/strong_baseline_checklist.md`.
- The detector sweep order is Oriented R-CNN -> RoI Transformer -> ReDet; with 7 visible RTX 4090s, the first wave can be started in parallel as separate jobs once the detector environment is ready, with ReDet using distributed training.
- The anchor-repair run is archived in `docs/experiments/20260524_dota_v15_anchor_repair_quick_test.md`.
- 2026-06-07 route update: the paper path is DOTA2 first, DIOR-R second, and
  FAIR1M only after both are stable. DOTA v1.5 GeoNexus evidence is
  archive/debug evidence, not headline paper evidence.
- 2026-06-07 DOTA2 gate: wait for both active DOTA2 GeoNexus S1 candidates to
  reach first validation and compare them against RoI Transformer S0
  `dota/mAP=0.6088`, `dota/AP50=0.6090`. Launch S2 only from the better clean
  S1 checkpoint. Keep S3/S4, pseudo-labeling, and routing paused until DOTA2
  S1/S2 and DIOR-R numeric stability are resolved.
- 2026-06-07 DIOR-R gate: detector training is blocked after ORCNN/RoITrans
  NaN and RetinaNet `loss=inf`. The next DIOR-R task is diagnosing data,
  rotated-box conversion, class mapping, and loss targets, not launching
  another unchanged detector.
- 2026-06-07 RTMDet-L status: completed and deprioritized after epoch-12
  `dota/mAP=0.2779`, `dota/AP50=0.2780`.
- 2026-07-09 status: DIOR-R S3 scene/context adapter (best mean `0.6979`,
  final mean `0.6859`) is the strongest local result and the primary
  GeoNexus-RSD DIOR-R claim. DIOR-R S4 pseudo-label purification is formally
  closed after three attempts topped out at best mean `0.697272`, below the
  S3 gate. The TGRS manuscript
  (`_local_archive_20260601_pull_backup/docs/TGRS/geonexus_tgrs.tex`) was
  refreshed to this evidence state, including a new DIOR-R comparator
  section/table/figure (OrientedFormer Swin-T confirmed by our own rerun at
  `0.6883`; Strip R-CNN-S and AOPG are unverified public-paper rows; this is
  comparator context, not a SOTA claim). This was a no-GPU local session; see
  `PROJECT_INSTRUCTIONS.md`'s `2026-07-09 local-only status` entry for the
  full list of what was and was not done. The segmentation lane
  (`BRIEF_LIST.md`) stays paused/secondary behind the core DOTA2/DIOR-R route
  by default. Strip R-CNN-S, LSKNet, and PKINet/PKINet-v2 are now confirmed
  (not just "blocked pending research") to have no usable DIOR-R checkpoint
  anywhere; stop re-attempting those specific downloads without new evidence.
  A background research workflow hung 4+ hours on a `WebFetch` to a OneDrive
  folder redirect link before being killed manually — avoid unbounded
  `WebFetch` calls on `1drv.ms` folder URLs in future sessions.

## Main Decision

Use a hybrid workflow:

- `OpenRSD` is only the local oracle for reference behavior, config ideas, and sanity checks.
- `OpenPrompt` is the only repo that should be actively shaped for ablations, results, and writing.

Do not switch back to heavy OpenRSD-first reproduction unless a hard blocker appears.

## One-Month Objective

The objective is not to finish a huge final paper.

The objective is to reach this state within 1 month:

1. a credible reduced baseline exists
2. 2 to 3 focused ablations are completed
3. at least one method story is defensible
4. figures, tables, and notes are sufficient to start paper writing

## Recommended Scope For This Month

Keep the paper narrow.

The preferred paper story is:

- failure mode:
  small and confusing rotated objects under prompt ambiguity
- method:
  hierarchy + scene context + consistency
- minimal modules for the first paper:
  - hierarchy-aware prompt bank
  - scene-context prompt adapter
  - hierarchy-consistent pseudo-label filtering

If time becomes tight, skip routing in the first paper version.

## Strict Priority Order

### Priority 1: Make the baseline credible

Use the version that matches the staged server asset:

- [`configs/experiments/dota_v1_baseline_repro.yaml`](./configs/experiments/dota_v1_baseline_repro.yaml)
- [`configs/experiments/dota_v15_baseline_repro.yaml`](./configs/experiments/dota_v15_baseline_repro.yaml)

Before adding novelty, confirm:

- dataset loading is correct
- tiled training works
- validation metrics are nonzero and interpretable
- failure cases are recorded

If the tiled baseline still fails badly, spend time fixing baseline capacity first.
Do not add new modules on top of a broken baseline.

### Priority 2: Run the fastest defensible ablations

Recommended order:

1. baseline only
2. `+ hierarchy bank`
3. `+ scene adapter`
4. `+ hierarchy-consistent pseudo-label score`

Only add routing if the first three stages already show signal.

Current 2026-06-07 override: for the DOTA2-first paper route, do not start
S2 until both S1 candidates validate; do not start S3/S4 or pseudo-labeling
until DOTA2 S1/S2 and DIOR-R diagnosis are stable.

### Priority 3: Start paper-facing evidence collection early

Do not wait until the end to prepare evidence.

As soon as experiments run, start collecting:

- AP50 tables
- confusion examples
- small-object examples
- prompt failure examples
- qualitative visualizations

## Best Research Choices Under The Current Constraint

### Choice A: Baseline-Fix Study

Question:
Can a tiled and higher-query OpenRSD-like baseline become credible on the reduced local DOTA setup?

Why this is good:

- lowest risk
- necessary before any real claim
- immediately useful for the paper

Expected output:

- one baseline table
- one failure analysis section
- one justification for the chosen training setup

### Choice B: Hierarchy-Only Study

Question:
Does a hierarchy-aware prompt bank improve confusing fine-grained classes?

Why this is good:

- low engineering risk
- easy to explain
- cheap to ablate

Relevant code already exists in:

- [`src/openprompt_rs/models/prompt_bank.py`](./src/openprompt_rs/models/prompt_bank.py)
- [`src/openprompt_rs/models/losses.py`](./src/openprompt_rs/models/losses.py)

### Choice C: Scene-Context Study

Question:
Does scene-conditioned prompt adaptation help small and ambiguous categories?

Why this is good:

- still lightweight
- useful for a clear method story
- fits the remote sensing setting well

Relevant code already exists in:

- [`src/openprompt_rs/models/detector.py`](./src/openprompt_rs/models/detector.py)
- [`src/openprompt_rs/models/innovations.py`](./src/openprompt_rs/models/innovations.py)

### Choice D: Pseudo-Label Consistency Study

Question:
Can hierarchy-aware pseudo-label filtering improve results when labeled data is limited?

Why this is good:

- matches the storage constraint better than requiring huge new labeled assets
- gives a stronger cross-dataset or semi-supervised story
- already partially implemented

Relevant code already exists in:

- [`src/openprompt_rs/models/pseudo_label.py`](./src/openprompt_rs/models/pseudo_label.py)

### Choice E: Routing Study

Question:
Do hard or tiny proposals benefit from stronger prompt fusion than easy proposals?

Why this is weaker as a first-month target:

- more novel but higher risk
- easier to fail without enough time
- should only be attempted after the earlier studies show value

Relevant code already exists in:

- [`src/openprompt_rs/models/routing.py`](./src/openprompt_rs/models/routing.py)

## Recommended One-Month Plan

### Week 1

- verify the tiled baseline setup
- run short train/eval loops
- confirm dataset counts and metric outputs
- record one exact OpenRSD reference run for sanity comparison

### Week 2

- run the main reduced baseline
- inspect errors
- run `hierarchy-only` ablation

### Week 3

- run `scene-adapter-only` ablation
- run `hierarchy + scene` or `pseudo-label` ablation, depending on earlier signal
- start drafting figures and experiment tables

### Week 4

- finish the smallest strong experiment set
- rerun only critical experiments if needed
- freeze scope
- start paper writing with available results

## Minimum Acceptable Experiment Package

If time is very tight, the minimum acceptable package is:

1. reduced tiled baseline
2. hierarchy-only ablation
3. scene-only or pseudo-label-only ablation
4. one combined variant if possible
5. qualitative failure and improvement examples

That is enough to start writing a focused paper draft.

## Go / No-Go Rules

### Go

Keep the current thesis if at least one of these happens:

- average AP improvement is clearly positive
- small-object performance improves meaningfully
- confusing-class errors decrease visibly
- prompt robustness improves in a way that can be shown clearly

### No-Go

Narrow the paper or pivot if:

- baseline remains broken after tiled/query fixes
- hierarchy and scene both give negligible signal
- the added modules increase cost without a clear benefit

## What To Avoid

- Do not restart from zero.
- Do not expand the project into too many datasets at once.
- Do not claim official OpenRSD reproduction unless verified.
- Do not spend much time on large-scale engineering cleanup unless it directly helps experiments.
- Do not add a new module just because it sounds novel.

## Default Working Rules For Future Sessions

When continuing this project:

1. Read this file first.
2. Audit the current repo state before making claims.
3. Read the living literature tracker before proposing new modules or route
   changes.
4. Search recent primary sources when the user asks for planning, route
   changes, or paper claims.
5. Distinguish paper-facing evidence, archive/debug evidence, and future
   inspiration.
6. Prefer execution over brainstorming.
7. Ask only the minimal missing questions that block valid work.
8. Update Markdown records inside `OpenPrompt` so the repo stays self-contained.
9. Treat baseline reproduction as higher priority than novelty.
10. Keep the paper scope small enough to start writing within 1 month.

## Default Experiment Sequence

Use this order unless there is a clear reason not to:

1. DOTA2 S1 first-validation comparison against RoI Transformer S0
   `0.6088/0.6090`
2. DOTA2 S2 from the better clean S1 checkpoint
3. DIOR-R data/box/loss-target diagnosis before any detector relaunch
4. DIOR-R S1/S2 only after numeric stability is fixed
5. S3/S4 and pseudo-label consistency only after the DOTA2 S1/S2 and DIOR-R
   gates pass
6. routing only if earlier modules already show value

## Default Writing Story

If results are modest, write the paper around:

- a real failure mode
- a focused and lightweight method
- strong qualitative analysis
- a clean ablation story

Do not force a large SOTA claim if the evidence does not support it.

## Ready-To-Use Next-Session Prompt

Copy this into a future session if needed:

```text
You are continuing the OpenPrompt remote sensing research project.

Read /data5/2025/ldh/OpenPrompt/SESSION_START.md first and treat it as the single source of truth for this session.

Then:
1. Audit the current repo state and recent experiment artifacts.
2. Read /data5/2025/ldh/OpenPrompt/docs/literature/20260607_openrsd_related_recent_papers.md before proposing route changes or new modules.
3. Search recent primary sources when planning, changing route, or making paper claims.
4. Continue from the highest-priority unfinished item.
5. Avoid broad re-planning unless the current path is clearly blocked.
6. Keep the scope aligned with a 1-month deadline for finishing experiments and starting paper writing.
7. Update Markdown notes in /data5/2025/ldh/OpenPrompt when you learn something important.

Default priority order (updated 2026-07-09; superseded the original bullets
below once DOTA2 S1/S2 and DIOR-R S0-S4 all completed):
- If no GPU/server access this session: resolve blocked-file research
  (Strip R-CNN DIOR-R checkpoint, PKINet/PKINet-v2/LSKNet status, literature
  tracker refresh), reconcile manuscript/doc drift, and stage a ready-to-run
  GPU queue for the next server session. Do not open new training routes from
  local reasoning alone.
- If GPU/server access is available: first confirm whether the blocked-file
  research from the last local session found a usable Strip R-CNN-S DIOR-R
  checkpoint or new PKINet/PKINet-v2/LSKNet comparator target; run whichever
  of those became unblocked before anything else.
- Do not relaunch DIOR-R S4 pseudo-label training with the same recipe; it is
  closed (best mean `0.697272` < S3 gate `0.6979`) unless a new design change
  is proposed and separately approved.
- FAIR1M, routing, and the segmentation lane (`BRIEF_LIST.md`) stay paused
  behind the core DOTA2/DIOR-R route unless explicitly reprioritized.

Original priority order (superseded, kept for history):
- DOTA2 S1 first-validation comparison against RoI Transformer S0 0.6088/0.6090
- DOTA2 S2 from the better clean S1 checkpoint
- DIOR-R data/box/loss-target diagnosis before detector relaunch
- S3/S4 and pseudo-labeling only after DOTA2 S1/S2 and DIOR-R gates pass
- routing only if earlier modules already show value

Use OpenRSD only as a reference implementation, not as the main development repo.
```

## When To Update This File

Update this file only if one of these changes:

1. the main paper direction changes
2. the baseline target changes
3. the deadline changes
4. the hardware or storage constraints change
5. the preferred experiment order changes
