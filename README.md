# openprompt

`openprompt` is a rebuilt GitHub-style research repository for **baseline-first open-prompt rotated remote sensing detection** under the working paper direction:

**GeoNexus-RSD: Hierarchy- and Context-Aware Open-Prompt Rotated Detection for Remote Sensing**

Repository identity:

- GitHub owner: `SCUcookie`
- Intended repo URL: `https://github.com/SCUcookie/openprompt`
- Local path: `/opt/pangu/ldh/openprompt`

## What this rebuild is

The original directory only contained:

- the `OpenRSD` ICCV 2025 paper PDF
- planning notes for the next paper
- no runnable OpenRSD source code

This rebuild turns that handoff folder into a usable research repo with:

- a clean Python package under `src/openprompt_rs`
- config-driven train and evaluation entrypoints
- a pseudo-label export entrypoint for self-training
- a lightweight `OpenRSD`-like baseline scaffold
- the planned `GeoNexus-RSD` innovation modules
- a modular innovation registry for later ablations
- sample taxonomy assets and prompt-bank generation
- preserved handoff notes under `docs/research_handoff`

## What is implemented

- `OpenRSD`-style dual-head prompt detector scaffold:
  - alignment head
  - fusion head
  - class-embedding refinement
- `GeoNexus-RSD` extensions:
  - hierarchy-aware prompt bank
  - scene-context prompt adapter
  - scale/rotation-aware routing
  - hierarchy-consistent pseudo-label filtering
  - scene-conditioned temperature scaling hook
  - confusing-class margin-loss hook
- dataset support:
  - synthetic smoke-test dataset
  - DOTA-style rotated annotation loader template
- experiment assets:
  - YAML configs
  - prompt taxonomy JSON
  - scripts for prompt-bank build, training, evaluation, and smoke testing
- documentation for next-step experiments and publication-facing innovation ideas

## What is not claimed

This repo does **not** claim to be the official `OpenRSD` implementation.

The following remain external or incomplete because they were not present locally:

- exact original `OpenRSD` training code
- official paper hyperparameters
- dataset copies and paths
- official DOTA/FAIR1M evaluation wrappers
- verified reproduction numbers

The code is therefore a **research-faithful rebuild and extension scaffold**, not a false claim of official reproduction.

## Quick start

```bash
cd /opt/pangu/ldh/openprompt
python3 -m pip install -e .
python3 scripts/smoke_test.py --config configs/experiments/geonexus_synthetic.yaml
python3 scripts/build_prompt_bank.py \
  --taxonomy assets/hierarchies/remote_sensing_taxonomy.json \
  --output artifacts/generated/prompt_bank_remote_sensing.pt
python3 scripts/self_train.py \
  --config configs/experiments/geonexus_synthetic.yaml \
  --checkpoint outputs/geonexus_synthetic/last.pt \
  --output outputs/geonexus_synthetic/pseudo_labels.pt
```

## Main structure

```text
openprompt/
├── assets/                  # taxonomy and prompt assets
├── configs/                 # model, dataset, experiment configs
├── docs/                    # method notes, setup guides, preserved handoff docs
├── scripts/                 # train/eval/prompt-bank entrypoints
├── src/openprompt_rs/       # source package
└── tests/                   # smoke and unit tests
```

## Recommended execution order

1. Run the synthetic smoke test to verify the environment.
2. Fill in real dataset paths in `configs/datasets/*.yaml`.
3. Replace the hash text embedder with CLIP, SkyCLIP, or another stronger text encoder if available.
4. Run the pure baseline configuration first.
5. Enable `GeoNexus-RSD` modules one by one for ablations.
6. Try the math-ready config after the structure modules are stable.
7. Add official rotated mAP evaluation before making any paper-level claims.

## Innovation toggles

You can now switch most research ideas by config instead of rewriting the detector. Example:

```yaml
model:
  innovations:
    scene_adapter:
      enabled: true
    router:
      enabled: true
      hidden_dim: 128
    scene_temperature:
      enabled: true
      hidden_dim: 128
      min_tau: 0.70
      max_tau: 1.40

criterion:
  hierarchy_weight: 0.10
  margin_weight: 0.10
  margin_value: 0.20
```

This lets you test structure innovations and mathematical regularizers as separate ablations.

## Innovation tracks already prepared

Low-risk directions:

- hierarchy-aware prompt smoothing
- scene-conditioned prompt gating
- hierarchy-consistent pseudo-label scoring
- scene-conditioned temperature scaling

Medium-risk directions:

- query routing by uncertainty and geometry
- confusing-class margin regularization
- scene-aware hard-negative suppression

Reviewer-friendly upgrades:

- fine-grained confusion analysis
- mixed-prompt robustness evaluation
- small-object transfer experiments
- efficiency versus accuracy trade-off table

Detailed next steps are in [docs/setup/next_steps.md](/opt/pangu/ldh/openprompt/docs/setup/next_steps.md) and [docs/method/innovation_playbook.md](/opt/pangu/ldh/openprompt/docs/method/innovation_playbook.md).

## Preserved research handoff

The original planning package is preserved in [docs/research_handoff/README.md](/opt/pangu/ldh/openprompt/docs/research_handoff/README.md).

## Baseline citation

Primary source used for this rebuild:

- Huang et al., `OpenRSD: Towards Open-prompts for Object Detection in Remote Sensing Images`, ICCV 2025.
