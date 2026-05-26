# Complete Experiment Plan For Paper Indicators

Date: 2026-05-26

Goal: move from small sanity experiments to paper-facing GeoNexus-RSD results.
The paper tables should close around DOTA-style oriented detection mAP,
class-wise AP, prompt robustness, pseudo-label quality, and efficiency.

## Current S0 Status

Use DOTA v1.5 as the controlled first benchmark unless a later note documents a
dataset switch.

Credible closed-set baselines now exist:

- RoI Transformer 3x: completed 36 epochs, best epoch 34
  `dota/mAP=0.2644`, `AP50=0.2640`.
- Oriented R-CNN 3x: completed 36 epochs, best epoch 33/34
  `dota/mAP=0.2620`, `AP50=0.2620`.
- ReDet pretrained: completed 12 epochs, best/final epoch 12
  `dota/mAP=0.2382`, `AP50=0.2380`.
- Oriented R-CNN 12e remains an archived reference at `dota/mAP=0.2561`,
  `AP50=0.2560`.

Paper-facing baseline choice:

- Primary baseline: RoI Transformer 3x epoch 34.
- Secondary baseline: Oriented R-CNN 3x epoch 33/34; use it if implementation
  simplicity/stability matters more than the small mAP lead.
- Comparison baseline: ReDet pretrained epoch 12.

## Required Paper Tables

### Main Comparison

Rows:

- Oriented R-CNN closed-set baseline.
- RoI Transformer closed-set baseline.
- ReDet, only if pretrained or clearly marked scratch.
- GeoNexus-RSD S1-S4 on the selected detector.

Columns:

- detector, prompt/VLM setting, dataset split, mAP/AP50, per-class AP summary,
  FPS or images/s, GPU memory, checkpoint.

### Core Ablation

Keep the detector, data split, schedule, and evaluator fixed.

- S0: closed-set detector.
- S1: flat class-name prompt classifier.
- S2: hierarchical prompt bank.
- S3: hierarchy plus scene/context adapter.
- S4: hierarchy plus context plus VLM-assisted pseudo-label purification.
- S5: optional routing only if S2-S4 are stable.

### Prompt Robustness

Evaluate with frozen detector weights where possible:

- exact class names only.
- aliases only or class names plus aliases.
- parent-category prompts.
- full mixed prompts with hierarchy, scene, geometry, confusing, and negative
  cues.

Report overall mAP/AP50 plus fine-grained pairs:

- small-vehicle vs large-vehicle.
- ship vs harbor.
- storage-tank vs roundabout.
- bridge vs road-like background.
- sports-field subclasses.

### Pseudo-Label Quality

Use a labeled holdout subset as if it were unlabeled, then compare pseudo labels
against ground truth before retraining:

- teacher detector only.
- detector plus hierarchy consistency.
- detector plus VLM crop-text agreement.
- full purification score.

Report pseudo-label precision, recall, F1, accepted-label count, and class-wise
quality. Then run the actual S4 retraining and report final detection mAP.

### Efficiency

Measure on the same GPU type:

- training time per epoch.
- validation time.
- inference FPS or images/s.
- peak GPU memory.
- prompt embedding cache build time.
- VLM crop-filtering throughput.

## Run Order

1. S0 is archived in compact JSON/Markdown summaries; keep checkpoints and raw
   logs outside Git.
2. Port the strong-baseline MMRotate config wrappers and `tools/test.py`
   compatibility patch into a tracked transport location, or document them as
   server-local required files.
3. Use RoI Transformer 3x epoch 34 as the fixed detector for S1 unless a later
   note chooses Oriented R-CNN 3x for implementation stability.
4. Install or configure real VLM dependencies. The active
   `/data1/anaconda3/envs/zwl_mmrotate/bin/python` currently has `torch` but is
   missing both `open_clip` and `clip`.
5. Run `scripts/smoke_vlm_embeddings.py` on all 16 DOTA v1.5 class prompts
   with a real backend such as RemoteCLIP before launching S1.
6. Run S1-S3 on DOTA v1.5 with the same split and evaluator.
7. Generate pseudo labels with the selected teacher, evaluate purification on a
   heldout labeled subset, then run S4 retraining.
8. Produce qualitative examples and confusion analysis only after S1-S4 numbers
   are complete.

## Assets To Download Or Stage

Required:

- DOTA v1.5 train/val images and annotations, already expected under
  `/data5/2025/ldh/OpenPrompt/DOTA/`.
- A real VLM checkpoint. Prefer RemoteCLIP first because it is remote-sensing
  specific and provides Hugging Face checkpoint download instructions in the
  official repository.
- OpenAI CLIP or OpenCLIP weights as a natural-image baseline for the prompt
  robustness table.

Recommended:

- SkyCLIP ViT-L/14 weights for a second remote-sensing VLM comparison.
- ReDet ReResNet-50 ImageNet pretrained checkpoint, so the ReDet run is not
  compared as scratch against ImageNet-pretrained R50 baselines.
- MMRotate pretrained detector checkpoints for DOTA v1.0 as reference sanity
  checks, not as direct DOTA v1.5 results.

Optional if compute and storage allow:

- DOTA v1.0 for cross-version reproducibility.
- DOTA v2.0 only after DOTA v1.5 S1-S4 is complete.
- Extra unlabeled DOTA-style tiles for pseudo-label experiments, provided the
  source and preprocessing are documented.

Public source anchors checked on 2026-05-25:

- DOTA official dataset page: `https://captain-whu.github.io/DOTA/dataset`
- RemoteCLIP official repository: `https://github.com/ChenDelong1999/RemoteCLIP`
- SkyCLIP / SkyScript release: `https://github.com/wangzhecheng/SkyScript`
- SkyCLIP Hugging Face mirror: `https://huggingface.co/BiliSakura/SkyCLIP-ViT-L-14`
- MMRotate model zoo: `https://mmrotate.readthedocs.io/en/stable/model_zoo.html`
