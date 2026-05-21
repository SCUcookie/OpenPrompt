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

1. Verify dataset loading and tiling.
2. Run synthetic smoke tests.
3. Run the local lightweight baseline only as a code sanity check.
4. Establish a credible strong oriented detector baseline on DOTA v2.
5. Run flat class-name prompt classification.
6. Add hierarchical prompt bank.
7. Add scene/context prompt adapter.
8. Add VLM-assisted pseudo-label purification.
9. Add optional routing only if the core modules already help.

## Current Local Scaffold Commands

Build a prompt-bank artifact for inspection:

```bash
PYTHONPATH=src python scripts/build_prompt_bank.py \
  --taxonomy assets/hierarchies/remote_sensing_taxonomy.json \
  --templates assets/prompts/prompt_templates.json \
  --output artifacts/generated/prompt_bank_remote_sensing.pt \
  --embedding-dim 256
```

Run the local DOTA reproduction-style scaffold:

```bash
PYTHONPATH=src python scripts/train.py \
  --config configs/experiments/dota_v2_baseline_repro.yaml
```

Evaluate a checkpoint:

```bash
PYTHONPATH=src python scripts/evaluate.py \
  --config configs/experiments/dota_v2_baseline_repro.yaml \
  --checkpoint outputs/dota_v2_baseline_repro/epoch_001.pt \
  --metric-set both
```

Export hierarchy-consistent pseudo labels:

```bash
PYTHONPATH=src python scripts/self_train.py \
  --config configs/experiments/dota_v2_geonexus.yaml \
  --checkpoint outputs/dota_v2_geonexus/epoch_001.pt \
  --output outputs/dota_v2_geonexus/pseudo_labels.pt
```

## What Must Improve Before Paper Claims

- Replace hash text embeddings with CLIP, SkyCLIP, RemoteCLIP, or an equivalent
  documented VLM encoder.
- Use a credible oriented detector baseline, preferably through MMRotate or a
  similarly strong implementation.
- Verify DOTA tiling, class mapping, rotated IoU/NMS, and mAP.
- Record complete experiments in `docs/experiments/`.
- Report prompt robustness and pseudo-label quality, not only final mAP.

## Acceptance Criteria For JSTARS

- Complete DOTA v2 results.
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
