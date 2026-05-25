# Complete Experiment Plan For Paper Indicators

Date: 2026-05-25

Goal: move from small sanity experiments to paper-facing GeoNexus-RSD results.
The paper tables should close around DOTA-style oriented detection mAP,
class-wise AP, prompt robustness, pseudo-label quality, and efficiency.

## Current S0 Status

Use DOTA v1.5 as the controlled first benchmark unless a later note documents a
dataset switch.

Credible closed-set baselines now exist:

- Oriented R-CNN: completed 12 epochs, `dota/mAP=0.2561`, `AP50=0.2560`.
- RoI Transformer: stable low-LR rerun reached epoch 11, best observed epoch 10
  `dota/mAP=0.2485`, `AP50=0.2480`; epoch 12 had not written a checkpoint at
  the latest check.
- ReDet scratch: completed 12 epochs, `dota/mAP=0.1221`, `AP50=0.1220`.

Paper-facing baseline choice:

- Primary baseline: Oriented R-CNN epoch 12.
- Secondary baseline: RoI Transformer epoch 10 if epoch 12 remains unavailable
  or worse.
- Diagnostic only: ReDet scratch, unless ReResNet ImageNet pretraining is
  restored and rerun.

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

1. Finish or diagnose the RoI Transformer epoch-12 job. If it remains stalled,
   keep epoch 10 as the best RoI Transformer result and record the stall.
2. Export small JSON summaries for RoI Transformer and ReDet, matching the
   Oriented R-CNN metrics JSON style.
3. Port the strong-baseline MMRotate config wrappers and `tools/test.py`
   compatibility patch into a tracked transport location, or document them as
   server-local required files.
4. Choose the primary detector for prompt experiments. Use Oriented R-CNN first
   because it is the best completed stable result.
5. Implement real VLM embedding support and text-embedding caching; keep hash
   embeddings only for smoke tests.
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

