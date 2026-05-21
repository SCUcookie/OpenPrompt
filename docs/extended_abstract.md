# GeoNexus-RSD Extended Abstract Draft

Remote sensing object detection must handle dense layouts, tiny targets,
arbitrary orientations, and fine-grained category ambiguity. Strong oriented
detectors address the geometric side of this problem, but most still use fixed
class classifiers and rely heavily on dense box annotation. GeoNexus-RSD focuses
on a complementary question: whether structured language priors and context can
make oriented detection and semi-supervised pseudo labels more reliable for
DOTA-style aerial imagery.

The proposed study is built around three modules. A hierarchical prompt bank
constructs class representations from class names, parent categories, aliases,
scene priors, geometry priors, confusing classes, and negative cues. A
scene-context adapter adjusts prompt embeddings using tile-level and region-level
visual features, targeting cases where local appearance alone is ambiguous. A
VLM-assisted pseudo-label purification module filters candidate pseudo boxes
using detector confidence, hierarchy consistency, crop-text agreement, and
geometry plausibility before using them as weighted soft supervision.

The experimental path is baseline-first. Stage S0 establishes a credible
closed-set oriented detector baseline with verified DOTA tiling, class mapping,
rotated IoU/NMS, and mAP evaluation. Stage S1 adds flat class-name prompt
classification. Stage S2 replaces flat prompts with the hierarchical prompt
bank. Stage S3 adds context adaptation. Stage S4 adds pseudo-label purification
and reports pseudo-label precision/recall on a held-out labeled subset before
reporting final mAP gains. Optional routing is evaluated only if S2-S4 already
show stable improvements.

For a JSTARS-ready submission, the final paper should include complete DOTA v2
results with at least one strong oriented detector baseline, ablations for each
core module, prompt robustness analysis, pseudo-label quality analysis,
efficiency reporting, qualitative accepted/rejected pseudo-label examples, and
limitations. A TGRS/ISPRS submission should only be considered if the method
also shows strong multi-dataset evidence on FAIR1M, DIOR-R, or another suitable
rotated detection benchmark.

The current local OpenPrompt scaffold is useful for implementation and ablation
plumbing, but it is not sufficient for paper-level claims by itself because it
uses a tiny backbone and deterministic hash text embeddings. The final system
should use a strong oriented detection framework and real vision-language
embeddings such as CLIP, SkyCLIP, or RemoteCLIP.
