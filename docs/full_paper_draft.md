# GeoNexus-RSD: Hierarchy- and Context-Aware Prompting for Oriented Remote Sensing Detection

This draft is a paper scaffold, not a submission-ready manuscript. Replace all
protocol language with measured results before submission.

## 1. Claim

GeoNexus-RSD should make one clean claim:

Hierarchy- and context-aware vision-language prompting improves fine-grained
oriented object detection and semi-supervised pseudo-label quality in remote
sensing imagery.

The paper should focus on DOTA-style rotated detection. Routing and compression
should not be central contributions in this manuscript.

## 2. Motivation

Remote sensing detection differs from natural-image detection because objects
are often tiny, rotated, densely packed, and context dependent. Strong oriented
detectors model box geometry, but fixed class classifiers still struggle with
fine-grained ambiguity such as small vehicle versus large vehicle, ship versus
harbor structures, and storage tank versus roundabout. Annotation is also
expensive, so semi-supervised learning is attractive but sensitive to noisy
pseudo labels.

Vision-language models can introduce semantic priors, but flat class-name
prompts are weak for aerial imagery. A useful prompt representation should
encode taxonomy, aliases, scene priors, geometry priors, and known confusing
classes.

## 3. Method

### Hierarchical Prompt Bank

For each DOTA class, build prompts from:

- class name and aliases
- parent category
- scene priors
- geometry priors
- confusing classes
- negative scene cues

The current repository implements this through
`assets/hierarchies/remote_sensing_taxonomy.json` and `PromptBank`.

### Context Prompt Adapter

Use tile-level and region-level visual context to adapt class prompt embeddings.
The goal is to help categories where local appearance is insufficient and scene
context is informative.

### Pseudo-Label Purification

Use pseudo labels conservatively. A candidate pseudo box should be weighted or
filtered using detector confidence, hierarchy consistency, VLM crop-text
agreement, and geometry plausibility. Report pseudo-label precision/recall on a
held-out labeled subset before claiming semi-supervised gains.

### Optional Routing

Routing can be included only as a later ablation after hierarchy, context, and
purification are stable. It should not appear in the title or main contribution
unless it becomes empirically necessary and clearly beneficial.

## 4. Experiment Sequence

Use this order:

1. S0: strong closed-set oriented detector baseline.
2. S1: flat class-name prompt classifier.
3. S2: hierarchical prompt bank.
4. S3: hierarchy plus scene/context adapter.
5. S4: hierarchy plus context plus pseudo-label purification.
6. S5: optional routing.

Acceptance for moving beyond S0: baseline mAP must be in a plausible range
relative to published DOTA-style detectors. A near-zero baseline means the
dataset, tiling, model capacity, evaluation, or training setup must be fixed
before adding novelty.

## 5. Required Final Tables

The final manuscript should contain real numbers for:

- main comparison against strong oriented detectors and prompt baselines
- ablation of hierarchy, context, purification, and optional routing
- prompt robustness: class names, aliases, parent prompts, mixed prompts
- pseudo-label quality: teacher only, hierarchy filter, VLM filter, full filter
- efficiency: parameters, memory, latency per tile

Do not submit tables with `pending`, `planned`, or expected values.

## 6. Required Figures

Include:

- method diagram
- qualitative detections
- accepted and rejected pseudo-label examples
- confusion matrix or class-pair failure visualization

## 7. Limitations

The final paper should explicitly acknowledge that VLMs can be weak on tiny
aerial objects, prompt quality depends on taxonomy design, and fixed DOTA-class
experiments do not by themselves prove open-vocabulary detection.

## 8. Venue Guidance

JSTARS is the practical first journal target if DOTA v2 experiments and
ablation evidence are complete. TGRS or ISPRS P&RS should be considered only if
results are strong across at least two datasets with convincing methodological
depth. If results are modest or incomplete, use GRSL, IGARSS, or a workshop
target instead.
