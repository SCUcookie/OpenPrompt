# Next Steps

This plan is fitted to the GeoNexus-RSD paper direction: DOTA-style oriented
detection with hierarchy/context prompting and VLM-assisted pseudo-label
purification.

## Immediate Local Checks

Run these after code changes:

```bash
PYTHONPATH=src python -m pytest
PYTHONPATH=src python scripts/smoke_test.py --config configs/experiments/geonexus_synthetic.yaml
```

If the package is installed with `pip install -e .`, `PYTHONPATH=src` is not
needed.

## Baseline-First Experiment Order

1. Diagnose the completed DOTA v1.5 scaffold baseline from
   `docs/experiments/20260523_dota_v15_baseline_server_followup.md`.
2. Because DOTA v1.5 is also near zero, diagnose the baseline scaffold before any
   hierarchy/context/pseudo-label novelty work.
3. Establish a credible strong oriented detector baseline on DOTA v1.0 or
   DOTA v1.5.
4. Replace hash text embeddings with a documented real VLM encoder such as
   CLIP, SkyCLIP, or RemoteCLIP before making vision-language claims.
5. Run flat class-name prompt classification.
6. Add hierarchical prompt bank.
7. Add scene/context prompt adapter.
8. Add VLM-assisted pseudo-label purification.
9. Add optional routing only if the core modules already help.

## Current Diagnosis

The first diagnosis pass is already in hand:

- threshold sweeps do not explain the failure; decoded scores are above the tested thresholds
- the predictions collapse toward a few classes, especially `small-vehicle` and `harbor`
- the inspected validation tile shows boxes clustered near the center instead of aligned with GT objects
- `QueryGenerator` emits `query_centers`, but the current box heads do not use them

So the next step is not S1-S5 prompt work. The next step is to inspect the
localization scaffold in `src/openprompt_rs/models/backbone.py`,
`src/openprompt_rs/models/heads.py`, and `src/openprompt_rs/models/detector.py`,
then decide whether to add an explicit spatial anchor or swap to a stronger
baseline path.

## Current Local Scaffold Commands

Build a prompt-bank artifact for inspection:

```bash
PYTHONPATH=src python scripts/build_prompt_bank.py \
  --taxonomy assets/hierarchies/remote_sensing_taxonomy.json \
  --templates assets/prompts/prompt_templates.json \
  --output artifacts/generated/prompt_bank_remote_sensing.pt \
  --embedding-dim 256
```

Run the local DOTA v1.0/v1.5 reproduction-style scaffold:

```bash
PYTHONPATH=src python scripts/train.py \
  --config configs/experiments/dota_v1_baseline_repro.yaml
# Swap to configs/experiments/dota_v15_baseline_repro.yaml if the staged asset is DOTA v1.5.
```

Evaluate a checkpoint:

```bash
PYTHONPATH=src python scripts/evaluate.py \
  --config configs/experiments/dota_v1_baseline_repro.yaml \
  --checkpoint outputs/dota_v1_baseline_repro/epoch_001.pt \
  --metric-set both
# Swap to the matching v1.5 config and checkpoint names if needed.
```

Export hierarchy-consistent pseudo labels:

```bash
PYTHONPATH=src python scripts/self_train.py \
  --config configs/experiments/dota_v1_geonexus.yaml \
  --checkpoint outputs/dota_v1_geonexus/epoch_001.pt \
  --output outputs/dota_v1_geonexus/pseudo_labels.pt
# Swap to the matching v1.5 config and checkpoint names if needed.
```

## What Must Improve Before Paper Claims

- Replace hash text embeddings with CLIP, SkyCLIP, RemoteCLIP, or an equivalent
  documented VLM encoder.
- Use a credible oriented detector baseline, preferably through MMRotate or a
  similarly strong implementation.
- Verify DOTA tiling, class mapping, rotated IoU/NMS, and mAP.
- Record complete experiments in `docs/experiments/`.
- Report prompt robustness and pseudo-label quality, not only final mAP.

The current state does not justify prompt ablations yet; restore a credible
localizer first.

## Current Baseline Gate

The DOTA v1.0 reduced tiled scaffold run completed on 2026-05-23 with
`map50=3.326794065590851e-06`, `mean_precision=0.00015695091957847277`, and
`mean_recall=0.00037193994697493814` on 4055 validation images. This passed the
nonzero sanity gate but is far too weak for paper claims.

The DOTA v1.5 matched scaffold run completed on 2026-05-24 with
`map50=1.0926445202230628e-05`, `mean_precision=0.0006667361585641629`, and
`mean_recall=0.0011823561703749874` on 4055 validation images. This is also
near zero, so pause S1-S5 prompt experiments and diagnose:

1. train/val image count, tile count, label count, and class distribution
2. decoded prediction score distribution before thresholding
3. threshold sweeps at `0.05`, `0.01`, and `0.001`
4. predicted box coordinates after tiling and normalization reversal
5. DOTA v1.0/v1.5 class mapping and ignored/difficulty label handling
6. prediction visualizations on a small validation-tile subset

The first three checks are now covered by `scripts/diagnose_baseline.py`; run it
with the matched config and checkpoint before changing the model.

Preliminary quick diagnostics on the staged v1.5 checkpoint show that the issue
is not thresholding: raw scores stay above the tested thresholds, predicted
classes collapse toward `small-vehicle` / `harbor`, and the inspected sample is
center-biased with very low same-class IoU. Treat the next step as scaffold
inspection of decoding / assignment / baseline capacity before any S1-S5 prompt
work.

One concrete code hypothesis to test next: `QueryGenerator` computes
`query_centers`, but the current box heads do not consume them, so the detector
is regressing boxes without an explicit spatial anchor. If that hypothesis
holds, the next fix should touch the box head or a stronger baseline rather
than prompt modules.

## Acceptance Criteria For JSTARS

- Complete DOTA v1.0/v1.5 results.
- At least one strong oriented detector baseline.
- Clear improvement over flat prompts.
- Clear improvement over confidence-only pseudo-label self-training.
- Ablations for hierarchy, context, and purification.
- Prompt robustness analysis.
- Pseudo-label precision/recall on a held-out labeled subset.
- Efficiency analysis.
- Qualitative detection and rejected pseudo-label examples.

## Stop Conditions

Do not proceed to paper writing if:

- baseline mAP is near zero
- official or accepted mAP evaluation is missing
- prompt/VLM embeddings are still hash-only
- tables contain pending or planned values
- routing becomes the main story before core prompt/VLM modules are validated
