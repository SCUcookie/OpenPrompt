# GeoNexus-RSD Blueprint

## Proposed title

**GeoNexus-RSD: Hierarchy- and Context-Aware Open-Prompt Rotated Detection for Remote Sensing**

## One-sentence idea

Use **offline LLM-built semantic hierarchy** and **lightweight scene-context prompt adaptation** to improve **small, ambiguous, rotated object detection** under cross-dataset shift, while keeping the detector practical under 4 x 4090.

## Main hypothesis

OpenRSD still loses performance when:

- prompt granularity is mismatched
- categories are visually confusing
- the scene gives strong contextual clues but the prompt bank does not use them
- pseudo labels are filtered only by confidence and shallow similarity

If we inject:

1. **semantic hierarchy**
2. **scene context**
3. **hierarchy-consistent pseudo-labeling**

then the detector should gain the most on:

- small objects
- fine-grained classes
- cross-dataset transfer
- mixed-prompt settings

## Method overview

### Module A: Offline semantic hierarchy bank

Use an LLM offline to build a structured prompt graph for each category.

For each class, store:

- parent class
- child classes if any
- synonyms
- visually confusing neighbors
- scene priors
- geometry attributes
- negative prompt cues

Examples:

- ship:
  - parent: vessel
  - confusing classes: vehicle, storage tank, harbor structure
  - scene priors: harbor, coastline, river, dock
  - geometry cues: elongated, high aspect ratio, arbitrary orientation

- small vehicle:
  - parent: vehicle
  - confusing classes: container, roof structure, dense parked car regions
  - scene priors: roads, parking lots, industrial areas
  - geometry cues: tiny footprint, dense cluster, weak texture

This is not just prompt expansion.
This is a **hierarchy-controlled semantic prior**.

### Module B: Scene-context prompt adapter

Extract a lightweight global or regional scene representation from the image.

Use it to reweight the prompt bank before detection.

The goal is:

- suppress scene-inconsistent classes
- strengthen scene-consistent classes
- improve fine-grained separation when local crops are weak

Implementation options:

1. global image token from a frozen VLM encoder
2. low-resolution scene classifier branch
3. regional context token around proposals or feature map cells

Recommended first version:

- use a frozen encoder
- add a small MLP gate over prompt embeddings
- do not fine-tune the encoder in v1

### Module C: Scale- and rotation-aware prompt routing

Not all proposals need the same prompt reasoning depth.

Idea:

- easy proposals use the alignment head
- hard or tiny proposals trigger stronger context-conditioned prompt fusion

Routing signals can include:

- FPN level
- proposal size proxy
- aspect ratio/orientation uncertainty
- scene ambiguity score

This keeps compute controlled.

### Module D: Hierarchy-consistent self-training

Replace OpenRSD's relatively simple pseudo-label logic with a hierarchy-aware version.

Keep a pseudo label if:

- confidence is high
- semantic similarity is high
- predicted label matches scene prior
- parent-child relation is consistent

Reject or downweight a pseudo label if:

- it conflicts with the scene context
- it violates parent-child hierarchy
- it matches a known hard negative class

This is where LLM-generated semantic priors are useful in a concrete way.

## Why this is publishable

### Contribution 1

A lightweight **hierarchy-aware prompt representation** for open-prompt rotated remote sensing detection.

### Contribution 2

A **scene-context adaptation mechanism** tailored to small and ambiguous aerial objects.

### Contribution 3

A **hierarchy-consistent self-training scheme** for cross-dataset generalization.

### Contribution 4

A more realistic evaluation protocol for:

- mixed prompts
- prompt granularity shift
- small-object cross-dataset transfer

## Minimal viable paper version

If you want the fastest path to a submission, keep only:

1. Module A
2. Module B
3. Basic pseudo-label consistency rules from Module D

Skip Module C in the first strong prototype if time is tight.

## Full paper version

Use all four modules if the first ablations show clean gains.

## What to compare against

### Main baselines

- OpenRSD
- RTMDet-R-L
- CastDet
- YOLO-World or YOLOE for efficiency/OV comparison if possible

### Internal ablations

- OpenRSD baseline
- + hierarchy bank only
- + scene-context adapter
- + hierarchy-aware pseudo-labeling
- + full model

## Datasets to prioritize

### Main training/evaluation

- DOTA-v2.0
- DIOR-R
- FAIR1M-2.0
- HRSC2016
- SpaceNet

### Cross-dataset evaluation

- VEDAI
- CORS
- DOSR
- SODA-A

### Why this set

- It includes rotated detection
- It includes fine-grained ships and aircraft
- It includes building categories
- It includes small-object transfer stress

## Primary success criteria

You should treat the project as promising only if it satisfies at least two of these:

1. Beats OpenRSD on average OBB AP50 by **>= 1.0**
2. Beats OpenRSD on SODA-A by **>= 2.0**
3. Improves HRSC2016 or FAIR1M-2.0 fine-grained performance without hurting DIOR-R/DOTA
4. Improves mixed-prompt robustness clearly
5. Keeps runtime and memory in a practical range

## Failure signals

Stop or pivot if:

1. hierarchy prompts only help by less than 0.3 AP everywhere
2. scene context helps large objects but hurts small ones
3. pseudo-label refinement is unstable across seeds
4. the method becomes too heavy and loses the "practical" story

## Backup pivot

If GeoNexus-RSD underperforms, pivot to:

**LLM-assisted sparse or partial annotation detection on top of OpenRSD**

That is still publishable and more contained.
