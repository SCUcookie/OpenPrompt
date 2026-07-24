# Server-Side Skills To Install (2026-07-22)

The server agent already uses a Codex-style skills directory
(`/home/zwl/.codex/skills/`; the `jupyter-notebook` skill was installed
there on 2026-07-22). The three skills below encode the server's own hard
rules as reusable prompts so every server session enforces them without
re-reading the full project history. Install each by creating
`/home/zwl/.codex/skills/<name>/SKILL.md` with the exact content in its
block.

Division of labor context: **local** = planning, manuscript, analysis,
route decisions (skills installed under `New/.claude/skills/`:
`tgrs-manuscript`, `server-update-review`); **server** = GPU execution
only, governed by the three skills below.

---

## 1. `/home/zwl/.codex/skills/gpu-launch-gate/SKILL.md`

```markdown
---
name: gpu-launch-gate
description: Mandatory pre-launch protocol for any GPU training or long evaluation in the GeoNexus-RSD project (OpenRSD/MMRotate). Use this skill before launching ANY process that will occupy a GPU — training runs, train-step diagnostics, long eval sweeps — even for "quick" or "small" jobs, and whenever the user says launch, train, run, resume, or continue an experiment.
---

# GPU Launch Gate

Launching without this chain has already cost the project: foreground jobs
left on GPU 0 without provenance forced a diagnostic to be aborted and
redone (2026-07-22), and unverified data once burned a full training run
on corrupt tiles. Walk every step; none are optional.

## The chain (in order, no skipping)

1. **Config/model/data gate (CPU-safe).** Load the exact config, construct
   the model, verify checkpoint compatibility (expected-missing keys only,
   zero unexpected), verify artifact shapes (prompt embeddings, relation
   matrix) and dataset pair counts against the recorded numbers. Record
   SHA-256 of every input checkpoint and artifact. Write the gate report
   JSON into the target workdir.
2. **Real 1,000-step train-step diagnostic** with
   `tools/diagnose_first_nonfinite_loss.py --mode train-step
   --max-batches 1000`. Accept only `checked_batches == 1000` and
   `finite_within_limit`. Known repair: the shared EMA hook raises
   `AttributeError: ema_model` when loading checkpoints in this standalone
   diagnostic — set `custom_hooks=[]` for the diagnostic ONLY; full launch
   configs keep their original hooks.
3. **Three consecutive GPU polls** (`nvidia-smi
   --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader`).
   A GPU qualifies only below 1 GB memory and at 0% utilization in all
   three polls. Select dynamically; never reserve by habit; never touch
   GPUs occupied by other users' processes.
   **Sandbox device-isolation gotcha (verified 2026-07-23):** if
   `nvidia-smi` reports it "cannot communicate with the NVIDIA driver" or
   `/dev/nvidia*` nodes are absent, this is sandbox isolation, NOT a driver
   outage — the ordinary sandbox hides the devices. Use elevated/escalated
   host access, under which all seven RTX 4090s are visible, then re-poll.
   Do not declare a GPU-access failure until host access has been tried.
4. **Detached launch with provenance.** Every job runs in a named detached
   screen (`<experiment>_<seed>_<date>_gpu<N>`), writes
   `launch_provenance.txt` (timestamp, GPU, PID, exact command, config
   path, source checkpoint + SHA-256, seed) into its workdir, and tees its
   log. Foreground GPU jobs are forbidden — if you find one, record it and
   stop it before proceeding.
5. **Startup acceptance.** Finite losses at iterations 200 and 1,000 and a
   clean scoped scan (Traceback, CUDA OOM, libpng/CRC/decode, NoneType,
   ValueError, nan, inf, KeyboardInterrupt) before the run is declared
   accepted. Cap automatic retries at three, each with a new dated log
   name; identical repeated tracebacks mean stop and diagnose, not retry.

## Standing closures — do not launch these at all

FAIR1M S2/GeoNexus, DIOR-R S4 pseudo-labeling, segmentation lane, and
DOTA-v2.0 follow-up training are closed by route decision. A closed route
reopens only through an explicit dated decision note in
docs/experiments/, never through a launch.
```

---

## 2. `/home/zwl/.codex/skills/experiment-completion-record/SKILL.md`

```markdown
---
name: experiment-completion-record
description: How to archive a finished GeoNexus-RSD training or evaluation campaign so local planning can trust it. Use this skill whenever a run finishes, whenever the user asks to record/archive/reconcile results, and before reporting any new metric — even for partial results or failed runs.
---

# Experiment Completion Record

Local route decisions are made entirely from these records; an
unreconciled number can silently steer the paper. The record must let a
reader reproduce the run and audit the metric without shell access.

## Required content (one dated markdown in docs/experiments/)

1. **Reconciliation with the launch record**: same screens, GPUs, seeds,
   configs, workdirs; name any deviation explicitly.
2. **Metrics table**: per-replica best AND final checkpoint values
   (mAP/AP50), with epochs; aggregate mean/std for both. Never blend best
   and final into one number — checkpoint selection is a finding here,
   not a nuisance.
3. **Per-class table** for the final epoch when the campaign is
   route-relevant (fine-grained movements have reversed mean-level
   conclusions in this project).
4. **Provenance**: checkpoint paths that exist on disk, runtime log paths,
   source-checkpoint SHA-256, exact eval command for any analysis pass.
5. **Scoped failure scan result** over all logs (the standard signature
   list), stating "clean" or listing hits. Pre-launch tool-mistake logs
   (wrong path, wrong env) are recorded as non-experiment failures.
6. **Decision boundary**: one closing paragraph stating what this record
   does and does not authorize next ("X remains closed", "next authorized
   action is Y"). Partial results get an explicit partial record that
   forbids promotion until the missing replica lands.

Also append a dated summary entry to PROJECT_INSTRUCTIONS.md and commit
JSON metric files alongside the note whenever the numbers will feed the
manuscript generator (evidence must be machine-readable, not only prose).
```

---

## 3. `/home/zwl/.codex/skills/mmrotate-failure-playbook/SKILL.md`

```markdown
---
name: mmrotate-failure-playbook
description: Known failure modes and verified fixes for this project's MMRotate/OpenRSD stack. Consult this skill whenever a training or eval job crashes, hangs, produces NaN/Inf, hits registry or import errors, or a diagnostic aborts — before attempting any fix, so known repairs are applied instead of rediscovered.
---

# MMRotate Failure Playbook (GeoNexus-RSD stack)

Verified failures and their working repairs. Apply the known fix first;
only debug fresh when the signature is genuinely new, and then add the new
signature here.

- **`AttributeError: ema_model` in the standalone train-step diagnostic**
  when loading a checkpoint: the shared EMA hook expects an initialized
  training-run EMA object. Fix: `custom_hooks=[]` in the DIAGNOSTIC config
  only. Full training configs keep the original hook.
- **Unregistered module / KeyError in registry** when running scripts
  outside `tools/train.py`: package imports alone are insufficient in this
  installed mix. Fix: call `register_all_modules` for mmdet AND mmrotate
  with `init_default_scope=False`, set
  `DefaultScope.get_instance("mmrotate", scope_name="mmrotate")`, then
  explicitly import the repo-local modules
  (`geonexus_mmrotate.prompt_bbox_head`, `runtime_imports` for
  `OrientedRPNHead`, the hbbox coder module).
- **`libpng`/`CRC`/`NoneType` during data loading**: corrupt image tile.
  Identify the exact file, exclude or deterministically replace it, record
  the SHA-256 of any replacement; never relaunch unchanged.
- **`loss: nan` / `loss: inf` early in training**: stop, do not lower LR
  blindly. Run the geometry/target diagnostic on the dataset first —
  every NaN in this project traced to data/box-conversion issues, not
  optimizer settings.
- **CUDA OOM**: pick a different qualifying GPU via the three-poll rule or
  reduce batch size in a NEW dated config; never force the original GPU.
- **Google-Drive/OneDrive fetches hanging**: never block a session on an
  unbounded fetch of `1drv.ms`/Drive folder links — set explicit timeouts;
  these have hung sessions for hours before.
- **Evaluator vs training-log mismatch**: per-class table means can differ
  from the full-precision evaluator mAP by ±0.01 (rounding accumulation).
  Report which one you cite; do not "correct" one to the other.
```

---

## Install command (server, one paste)

```bash
mkdir -p /home/zwl/.codex/skills/{gpu-launch-gate,experiment-completion-record,mmrotate-failure-playbook}
# then create each SKILL.md with the exact contents of the blocks above
```

Keep these synchronized: when a new failure signature or gate step is
added on either side, update both this file (committed, portable) and the
installed copies on the server.

---

# Third-Party Skills Evaluation Log (2026-07-22)

GitHub was searched for community Claude-Code skills useful to this
project. Every candidate's SKILL.md was **read in full locally (shallow
clone) before any installation** — third-party skills are prompt text that
future sessions will obey, so unreviewed installation is an injection
risk. Community signals (stars, releases, maintainer identity) and
licenses were checked for each.

## Installed into `New/.claude/skills/` (content-reviewed, MIT-licensed, attributed)

| Skill | Source | Review verdict | Project addendum |
|---|---|---|---|
| `paper-verification` | `fcakyon/phd-skills` (338 stars, v1.3.0 2026-05, MIT) | Clean: local-only operations, no network calls, tightly scoped numerical/terminology/code-paper/citation audits | Added GeoNexus context: evidence-JSON tracing rule, generator-only tables, the 68.83/68.84 verified split, component-name policy |
| `reviewer-defense` | same repo | Clean: weakness analysis, question generation, ablation selection, rebuttal structure; no external deps | Added TGRS venue framing + our real vulnerability list with current mitigation states (R1/R2 campaigns, seed-replica framing, dagger convention) |

## Adapted rather than installed

| Source skill | Why not verbatim | What was taken |
|---|---|---|
| `flonat/flonat-research` `skills/proofread` (120 stars, v0.2.1, MIT) | Hard dependencies on that repo's `_shared` infrastructure (sibling `devils-advocate` skill, quality-scoring/review-state schemas, council-mode protocol) — a verbatim copy would reference files that do not exist here | The novel check categories (notation consistency, equation completeness, causal-language audit, citation-voice balance, preprint staleness) folded into `tgrs-manuscript` as a pre-submission checklist, with attribution |
| same repo, `skills/pre-submission-report` | Same dependency problem | Covered by the combination of `tgrs-manuscript` checklist + `paper-verification` + `reviewer-defense` |

## Evaluated and rejected

| Candidate | Reason |
|---|---|
| `ndpvt-web/latex-document-skill` (27 templates/27 scripts) | Generic template library; our LaTeX pipeline is already fixed (IEEEtran + generator + compile chain); bulk would dilute triggering |
| `ndpvt-web/arxiv-claude-skills` (10k auto-generated skills) | Machine-generated at scale, unreviewable surface area — fails the review-before-install bar |
| `imbad0202/academic-research-skills`, `lingzhi227/agent-research-skills` | Heavyweight multi-agent frameworks (13+ agents); overlap with existing `deep-research` skill and this project's established workflow |
| `ultimatile/arxiv-skills` | Useful concept (arXiv→markdown reference docs) but overlaps the existing literature-tracker workflow + `deep-research`; revisit only if literature volume grows |

## Full local skills inventory after this pass

- `tgrs-manuscript` (custom) — manuscript hard rules + pre-submission
  proofread checklist
- `server-update-review` (custom) — server-commit review loop
- `paper-verification` (third-party, reviewed) — claims-vs-evidence audit
- `reviewer-defense` (third-party, reviewed) — pre-review vulnerability
  analysis and rebuttal preparation

Server-side prompts (this file, top half): `gpu-launch-gate`,
`experiment-completion-record`, `mmrotate-failure-playbook` — plus the
server's own `jupyter-notebook`. The phd-skills repo also contains
`launch`/`debug`/`compare` skills that overlap our server prompts; ours
are project-specific and stay authoritative, but the repo
(`fcakyon/phd-skills`) is worth a look server-side if the team wants the
generic `reproduce` workflow for future comparator reproductions.
