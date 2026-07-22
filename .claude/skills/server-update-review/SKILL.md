---
name: server-update-review
description: Workflow for reviewing new server commits pulled into this repo and turning them into judgments, route decisions, and next-step plans. Use this skill whenever the user says they pulled/pushed new updates from the server, asks to "read the new update", "judge it", "review the server's changes", "update our next route/plan", or after any git pull that brings dated experiment notes — even if they don't use the word "server". Also use it before proposing or scheduling any GPU training.
---

# Server Update Review (GeoNexus-RSD)

This project splits into a local half (planning, papers, analysis) and a
server half (GPU runs, pushed back as dated evidence notes). The review
loop below has caught real problems before — stale checkpoints presented
as new results, README venue claims that were copy-paste bugs, a mean-level
metric hiding a per-class story — so walk it in order rather than skimming
the newest file.

## Review sequence

1. **Enumerate what actually arrived.**
   `git log --oneline -8` then `git show --stat <hash>` for each new commit
   since the last reviewed one. Authors matter: `lyc` commits are
   server-side runs; `SCUcookie` commits are usually the user relaying
   local work. Never assume the newest note is the only change.
2. **Read completion records against their launch records.** Every
   completed run must reconcile with its launch note (screen names, GPUs,
   seeds, configs, checkpoint paths). A result without launch provenance is
   flagged, not trusted.
3. **Check the numbers before the narrative.** Recompute means/deltas from
   the tables in the note; compare against the documented baselines in
   `PROJECT_INSTRUCTIONS.md`. Ask: is a gain larger than seed noise? Is the
   best/final checkpoint split reported? Does a mean hide a per-class
   story worth extracting?
4. **Check route-gate compliance.** The project runs on pre-registered
   gates and explicit closures (currently closed: FAIR1M S2, DIOR-R S4
   pseudo-labeling, segmentation lane, DOTA-v2.0 follow-up training).
   Anything that reopens a closed route without an explicit decision note
   is an error to surface, not adopt. New training proposals must carry the
   full gate chain: config/model/data gate → real 1000-step train-step
   diagnostic → three consecutive idle-GPU polls → detached screen with
   `launch_provenance.txt` → scoped failure scan.
5. **Update the standing documents** so the next session inherits the
   state, not the archaeology:
   - `PROJECT_INSTRUCTIONS.md` — dated status entry at the top block;
   - the active schedule/status table (currently in
     `docs/experiments/20260713_paper_finalization_schedule.md`);
   - a dated analysis/route note in `docs/experiments/` when a decision was
     made or evidence judged (thresholds quoted next to measured values so
     decisions stay auditable).
6. **Report to the user** with the verdict first (one sentence), then the
   evidence, then flags. Praise process only when the provenance actually
   checks out.

## Judgment heuristics that have paid off here

- Seed-noise scale on these benchmarks is ~0.1–0.3 mAP points at n=3;
  treat sub-0.5 mean deltas as noise unless per-seed signs agree.
- Best checkpoints peak early (epoch 4–8) and final epochs erode ~1 point
  on DOTA-v2.0 and FAIR1M; always ask which checkpoint a number comes from.
- A class at exactly 0.000 AP across all seeds is a data question
  (instances present?) before it is a model question.
- Repo README badges are not venue evidence; only proceedings pages,
  OpenReview decisions, or journal pages count.

## Writing the next plan

Plans the server will execute go into a dated
`docs/experiments/<date>_*.md` with copy-paste-ready commands, exact
artifact paths, pre-registered decision branches, and acceptance criteria —
the server session should never need this conversation's context. Keep the
paper's critical path explicitly separated from training campaigns so
neither blocks the other, and end with an ordered execution checklist.
