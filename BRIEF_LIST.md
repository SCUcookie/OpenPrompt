# Brief List

## Current Research Direction

Main direction:

**Gumbel-routed, compression-aware open-vocabulary remote sensing perception.**

Recommended primary task:

- open-vocabulary remote sensing semantic segmentation

Fallback task:

- open-vocabulary remote sensing detection if segmentation reproduction is
  blocked by data or environment issues

Use OpenPrompt/OpenRSD as prior work and a lightweight local scaffold, not as
the main reproduction burden.

## Baseline Priority

Segmentation baselines:

- RSKT-Seg / Pi-Seg
- ConInfer

Detection fallback baselines:

- LAE-DINO
- RF-DETR
- LMW-YOLO

Historical/local baseline:

- OpenPrompt/OpenRSD-style prompt detection scaffold

## Implemented Local Hook

The local detector router supports:

- `soft`: softmax blend over alignment and fusion branches
- `gumbel`: Gumbel-Softmax branch routing
- `random`: random branch selection for ablation

Config location:

```yaml
model:
  innovations:
    router:
      enabled: true
      hidden_dim: 128
      mode: gumbel
      temperature: 0.7
      hard: true
```

Relevant files:

- `src/openprompt_rs/models/routing.py`
- `src/openprompt_rs/models/detector.py`
- `src/openprompt_rs/models/innovations.py`
- `configs/models/geonexus_rsd.yaml`
- `tests/test_innovations.py`

## Four-Week Plan

Week 1:

- reproduce one small RSKT-Seg/Pi-Seg run
- reproduce one ConInfer run
- record dataset size, memory, runtime, mIoU/AP, and code friction

Week 2:

- port the router to the selected baseline
- ablate no router, soft router, Gumbel router, and random router

Week 3:

- add compression experiments
- start with LoRA/adapters
- add quantization or pruning only after the baseline is stable

Week 4:

- add offline LLM prompt expansion
- generate aliases, fine-grained descriptions, and confusing-class prompts
- finalize accuracy-latency-memory tables

## Directory Map

- `src/openprompt_rs/`: local OpenPrompt research scaffold
- `configs/`: dataset, model, and experiment YAML files
- `tests/`: smoke and unit tests
- `scripts/`: train/eval/prompt-bank utilities
- `assets/`: taxonomy and prompt templates
- `docs/method/`: older method notes kept for reference
- `DOTAv2/`, `labels/`, `outputs/`: local data/results paths, ignored by Git

## Cleanup Rule

Keep in Git:

- source code
- configs
- scripts
- tests
- concise docs

Keep local or delete when stale:

- datasets
- checkpoints
- logs
- long superseded handoff notes
- generated experiment outputs
