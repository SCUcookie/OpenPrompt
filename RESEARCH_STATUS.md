# Research Status

Date: 2026-04-06

## Scope

You currently have two complementary local repos:

- `OpenRSD`: the official-style reproduction workspace using the
  `CKpoint_pkl` asset bundle.
- `OpenPrompt`: a cleaner research repo rebuilt around the OpenRSD paper, with
  extra hooks for your own follow-up work.

## What Is Already In Place

### OpenRSD

- The local asset bundle is present at `OpenRSD/OpenRSD_Ckpoint_pkl/`.
- `data` and `results` are already linked into the repo.
- `run_bootstrap.sh` and `tools/bootstrap_run.py` let you run OpenRSD scripts
  with the intended Python environment while preferring installed OpenMMLab
  packages.
- `tools/link_local_assets.py` exists for re-linking local `data/` and
  `results/` if you move the bundle.
- `SimpleRun/step2_trans_to_weights_only.py` converts the full checkpoint into
  a lighter weights-only checkpoint.
- `SimpleRun/local_ckpts/epoch_24_weights_only.pth` shows that checkpoint
  extraction has already been done locally.
- `SimpleRun/step1_inference.py` and `SimpleRun/test_rotate.py` were adjusted
  to use environment variables for config, checkpoint, data root, classes, and
  thresholds instead of hardcoded single-machine paths.

### OpenPrompt

- The DOTA loader now supports both original DOTA labels and your converted
  `class_id + normalized polygon` labels.
- The active taxonomy has been expanded from the earlier 8-class subset to the
  16 classes present in the converted labels.
- Validation split support has been added.
- A lightweight rotated `mAP@50` evaluator has been added.
- Tiled DOTA sampling has been added so you can avoid crushing large DOTA
  images into a single `512x512` input.
- New experiment configs exist for:
  - baseline
  - stronger reproduction-oriented baseline
  - tiled baseline
  - GeoNexus-style follow-up work
- New tests were added for the DOTA dataset loader and detection metrics.

## What The Current Results Mean

### OpenRSD side

You have moved beyond just reading the paper:

- the `CKpoint_pkl` bundle is wired into a runnable local repo
- checkpoint conversion is in place
- there is a simple local inference path

That means the official codebase is already usable as a reference teacher and
sanity anchor for later research.

### OpenPrompt side

The repo is not just notes anymore. It already contains meaningful
reproduction-oriented engineering:

- converted-label parsing
- validation support
- tiled sampling
- detection-style evaluation

From the existing notes and metrics:

- the earlier full-image baseline was trainable but produced effectively zero
  validation detection quality
- the most likely causes were too few query slots and destructive full-image
  resizing for dense DOTA scenes

So the important transition is this:

- phase 1 was "make OpenPrompt runnable on your local DOTA conversion"
- phase 2 is now "test whether the tiled reproduction baseline is a credible
  baseline before adding new research modules"

## Cleanup Performed In OpenPrompt

To make `OpenPrompt` pushable:

- local-only assets are now ignored more explicitly
- the tracked DOTA configs now use repo-relative paths instead of a
  machine-specific absolute path
- local setup instructions were added in `LOCAL_SETUP.md`
- a local-asset linker was added in `scripts/link_local_assets.py`
- the duplicate root PDF and the nested `OpenRSD_official` copy are no longer
  needed inside `OpenPrompt`
- duplicate `images/` data can be removed because `DOTAv2/images/` already
  contains the same files

The intended repo boundary is now:

- keep source, configs, docs, and tests in Git
- keep datasets, outputs, checkpoints, and scratch logs local only

## Recommended Next Decisions

### Decision A: Baseline-first in OpenPrompt

Choose this if your near-term goal is a paper-ready research repo.

Do next:

1. Treat `configs/experiments/dota_v2_baseline_repro.yaml` as the main baseline.
2. Re-run training and validation from scratch with the tiled setup.
3. If validation `mAP@50` is still near zero, fix baseline capacity before
   adding any new innovation module.
4. Only after the tiled baseline is credible, add GeoNexus-style modules one by
   one.

Why choose it:

- cleaner codebase
- easier ablations
- easier GitHub presentation
- better foundation for your own paper

### Decision B: Reproduce OpenRSD first, then port ideas

Choose this if you want the strongest possible claim that your later work is
grounded in the official implementation.

Do next:

1. Freeze `OpenPrompt` after this cleanup.
2. Use `OpenRSD` to document one repeatable inference/evaluation path with the
   current `CKpoint_pkl` bundle.
3. Record exactly which config, checkpoint, and dataset subset correspond to a
   stable reproduced result.
4. Port only the verified pieces you need into `OpenPrompt`.

Why choose it:

- tighter connection to the official codebase
- lower risk of drifting too far from the original method
- stronger story if reviewers ask what was reproduced exactly

### Decision C: Hybrid workflow

This is the most practical option if you want both speed and credibility.

Do next:

1. Use `OpenRSD` as the local oracle for checkpoint behavior and data layout.
2. Use `OpenPrompt` as the main research repo for ablations and new ideas.
3. Keep a short mapping note for every concept you port:
   OpenRSD file/config -> OpenPrompt module/config.

Why choose it:

- fastest iteration
- keeps the official repo intact
- avoids turning `OpenPrompt` back into a messy vendor dump

## My Recommendation

Take Decision C first, then converge to Decision A.

That means:

1. keep `OpenRSD` only as your local reference implementation
2. make `OpenPrompt` the only repo you actively shape for GitHub and paper work
3. spend the next serious experiment budget on the tiled reproduction baseline
4. do not add more novelty until the tiled baseline either works or clearly
   fails for a diagnosed architectural reason

## Immediate Next Actions

1. Commit the cleaned `OpenPrompt` source/config/docs/tests without any local
   assets.
2. Verify the tiled DOTA dataset counts and a short train/eval loop.
3. Write down one exact OpenRSD reference run:
   config, checkpoint, classes, thresholds, and output file.
4. Decide whether your next week is a baseline-fix week or a novelty-design
   week. Baseline-fix is the safer choice.
