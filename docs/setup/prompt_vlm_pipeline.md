# Prompt And VLM Pipeline

The paper is not only an oriented detection paper. The prompt/VLM path must be
explicit in the implementation and in the experiment logs.

## Prompt Assets

Tracked prompt assets:

- `assets/hierarchies/remote_sensing_taxonomy.json`
- `assets/prompts/prompt_templates.json`

Each class should include:

- class name
- parent category
- aliases
- scene priors
- geometry priors
- confusing classes
- negative cues

The current `PromptBank` builds prompt strings from these fields. This is useful
for testing, but the current embedding backend is still only a deterministic
hash fallback.

## Required Upgrade

Before paper-level semantic claims, replace the hash embedder with one of:

- CLIP
- SkyCLIP
- RemoteCLIP
- another documented remote-sensing vision-language encoder

Implementation target:

- keep hash embeddings for offline smoke tests
- add a real embedder selected by config
- cache text embeddings to a generated artifact
- record encoder name, checkpoint, preprocessing, embedding dimension, and
  prompt templates in the experiment summary

## Experiment Stages

Use this semantic sequence:

1. Class-name prompts only.
2. Class names plus aliases.
3. Hierarchical prompts with parent categories.
4. Full taxonomy prompts with scene, geometry, confusing, and negative cues.
5. Context-adapted prompt embeddings.
6. VLM-assisted pseudo-label purification.

Each stage should keep the detector backbone fixed unless the experiment is
explicitly a detector-baseline comparison.

## VLM-Assisted Pseudo-Label Purification

For each candidate pseudo box:

1. Crop the predicted region from the source tile.
2. Encode the crop with the VLM image encoder.
3. Encode class prompts with the VLM text encoder.
4. Compute crop-text agreement for the predicted class.
5. Combine:
   - detector confidence
   - hierarchy consistency
   - VLM agreement
   - geometry plausibility
6. Accept only conservative pseudo labels and train with soft weights.

Report pseudo-label precision and recall on a held-out labeled subset before
claiming final mAP improvement from self-training.

## Required Tables

Prompt robustness:

- exact class names
- aliases
- parent prompts
- mixed/full prompts

Pseudo-label quality:

- teacher only
- hierarchy filter
- VLM filter
- full purification

Failure analysis:

- small vehicle versus large vehicle
- ship versus harbor structures
- storage tank versus roundabout
- bridge versus road segment
- sports-field subclasses

