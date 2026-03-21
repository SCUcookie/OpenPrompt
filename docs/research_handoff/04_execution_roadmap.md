# Execution Roadmap

## Phase 0: Reproduction and sanity check

Goal:

- confirm that you can reproduce a meaningful OpenRSD-like baseline before inventing anything new

Tasks:

1. Rebuild the baseline on a reduced dataset set first:
   - DOTA-v2.0
   - DIOR-R
   - FAIR1M-2.0
   - HRSC2016
2. Reproduce:
   - baseline AP50
   - cross-dataset trends
   - prompt type behavior
3. Verify:
   - text prompt > image prompt on average
   - self-training helps
   - SODA-A style small-object gain exists

Exit condition:

- your reproduced baseline is within an acceptable gap from the paper trend

## Phase 1: Fast ablations on the semantic side

Goal:

- test whether hierarchy and context are worth pursuing before doing full-scale training

### Experiment 1: Hierarchy bank only

Change:

- replace flat class prompt bank with hierarchy-aware prompt bank

Keep:

- detector unchanged
- no scene-context module yet

What to measure:

- average OBB AP50
- small-object subset
- confusion pairs
- mixed-granularity prompts

Decision rule:

- if this gives no signal, do not continue with a hierarchy-heavy paper

### Experiment 2: Scene-context adapter only

Change:

- add scene-conditioned prompt reweighting

Keep:

- no hierarchy consistency in pseudo labels yet

What to measure:

- gains on buildings, ships, small vehicles
- cross-dataset performance

Decision rule:

- if gains appear mainly on ambiguous categories, this module is worth keeping

## Phase 2: Strong prototype

Goal:

- combine the useful semantic modules into a paper-worthy system

Tasks:

1. Merge hierarchy bank + scene-context adapter
2. Add hierarchy-consistent pseudo-label filtering
3. Run on the main training set
4. Evaluate on:
   - in-domain OBB AP50
   - cross-dataset AP50
   - prompt-granularity robustness
   - runtime

## Phase 3: Full paper validation

Goal:

- build the full evidence package for a submission

Required experiments:

1. Comparison with OpenRSD
2. Comparison with RTMDet-R-L
3. Comparison with another OV detector if practical
4. Ablation by module
5. Ablation by prompt granularity
6. Ablation by small/medium/large object size
7. Cross-dataset evaluation
8. Qualitative visualizations

## Hardware plan

Your hard limit is **4 x 4090**, so the roadmap should assume:

- mixed precision training
- frozen LLM and frozen VLM encoders whenever possible
- no training of a diffusion backbone from scratch
- full experiments only after fast ablations show value

## Recommended compute strategy

### Stage A: cheap validation

- 1 to 2 GPUs
- reduced dataset mix
- short schedules
- goal: see directionality, not final SOTA

### Stage B: serious training

- 4 GPUs
- full mixed dataset training
- only after Stage A works

### Stage C: final ablations

- reuse checkpoints whenever possible
- avoid retraining everything from zero

## Suggested 10-week schedule

### Weeks 1-2

- reproduce reduced OpenRSD baseline
- prepare prompt-bank pipeline
- prepare cross-dataset evaluation scripts

### Weeks 3-4

- run hierarchy-only and context-only ablations
- decide whether the paper thesis survives

### Weeks 5-6

- implement full GeoNexus-RSD prototype
- run main training on 4 GPUs

### Weeks 7-8

- run ablations
- run cross-dataset evaluation
- collect visualizations

### Weeks 9-10

- write the paper
- clean the contribution story
- prepare fallback results if one module is weak

## Risks and mitigation

### Risk 1: Gains are too small

Mitigation:

- focus more strongly on small-object and mixed-prompt benchmarks
- strengthen hierarchy-consistent pseudo-labeling

### Risk 2: Scene context hurts generalization

Mitigation:

- use context only as prompt gating, not as hard class suppression

### Risk 3: Training becomes too heavy

Mitigation:

- freeze external encoders
- keep only lightweight adapters trainable

### Risk 4: Paper feels incremental

Mitigation:

- add a new evaluation setting for prompt granularity and small-object cross-domain transfer
- make the story about a real failure mode, not just +0.x AP

## Clear go/no-go checkpoints

### Go

- +1 AP average or strong gains on small-object/cross-dataset tasks
- stable improvements across at least 2 seeds or 2 settings

### No-go

- tiny gains only on one dataset
- no semantic robustness improvement
- too much extra compute for negligible accuracy
