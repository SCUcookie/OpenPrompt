# Basic Knowledge For This Project

This file is a clean reference for the concepts used in GeoNexus-RSD.

## Oriented Remote Sensing Detection

Remote sensing object detection predicts object categories and oriented boxes in
large aerial or satellite images. An oriented box is commonly represented as:

```text
(x, y, w, h, theta)
```

DOTA-style datasets often store boxes as four polygon points:

```text
x1 y1 x2 y2 x3 y3 x4 y4 class difficulty
```

Large images are tiled before training and evaluation. After inference, tile
predictions must be mapped back to source-image coordinates and merged with
rotated NMS.

## Strong Baselines

Paper-level experiments should use strong oriented detectors such as:

- Oriented R-CNN
- RoI Transformer
- ReDet
- Oriented RepPoints
- modern MMRotate-style baselines

The lightweight local detector is useful for code plumbing and ablation
scaffolding, but it is not enough for a serious journal claim by itself.

## Prompt Bank

A prompt bank is a structured set of text descriptions for each class. In this
project, class prompts should include:

- class name
- aliases
- parent category
- scene priors
- geometry priors
- confusing classes
- negative cues

The prompt bank converts taxonomy metadata into class embeddings. The current
code supports this flow through `PromptBank`, but the default embedder is a hash
fallback for smoke tests.

## VLM Embeddings

For paper-level work, prompt text and image crops should be embedded with a real
vision-language model:

- CLIP
- SkyCLIP
- RemoteCLIP
- another documented remote-sensing VLM

The VLM is needed for two reasons:

1. Better semantic class embeddings than hash vectors.
2. Crop-text agreement scores for pseudo-label purification.

## Context Adapter

A context adapter modifies class prompt embeddings using image context. This is
important because small aerial objects may be visually ambiguous without scene
cues. For example, compact bright regions may be vehicles, rooftop structures,
containers, or small facilities depending on surrounding roads, ports, or
industrial areas.

## Pseudo-Label Purification

Pseudo labels should not be accepted by detector confidence alone. A safer score
combines:

- detector confidence
- hierarchy consistency
- VLM crop-text agreement
- geometry plausibility

The purified label should be used as soft supervision, weighted by confidence.

## Paper Rule

Do not claim performance before real results exist. Planned or pending result
tables are allowed in internal notes only, not in a final submission.
