# GeoNexus-RSD Abstract Draft

GeoNexus-RSD studies hierarchy- and context-aware vision-language prompting for
DOTA-style oriented object detection in remote sensing imagery. The central
claim is intentionally narrow: taxonomy-aware prompts, scene-context adaptation,
and conservative VLM-assisted pseudo-label purification may improve
fine-grained rotated detection and semi-supervised pseudo-label quality over
flat class-name prompts and confidence-only self-training.

The method has three core modules. First, a hierarchical prompt bank represents
each class with leaf names, parent categories, aliases, scene priors, geometry
priors, confusing classes, and negative cues. Second, a context adapter modulates
class prompts using tile-level and region-level visual features. Third, a
pseudo-label purification stage combines detector confidence, hierarchy
consistency, VLM crop-text agreement, and geometry plausibility before using
unlabeled examples as soft supervision.

The immediate publication target is a complete JSTARS-style study on DOTA v2.
The final paper must report real rotated mAP, class-wise AP, prompt robustness,
pseudo-label precision/recall, efficiency, and qualitative failure analysis. It
should not include pending result tables or claim open-vocabulary performance
unless the final implementation uses real CLIP/SkyCLIP/RemoteCLIP-style
embeddings and evaluates beyond fixed DOTA class names.

Routing and compression are not the main contribution for this paper. They can
remain optional ablations or follow-up work after the baseline, hierarchy,
context, and purification modules are empirically stable.
