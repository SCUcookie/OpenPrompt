# GitHub And Experiment Server Workflow

Use GitHub as the synchronization layer between the local coding machine and the
experiment server.

Paper-first rule:

- update `docs/geonexus_short_paper.tex` and `PROJECT_INSTRUCTIONS.md` before
  changing code when the claim or experiment sequence changes
- keep code, configs, and experiment notes aligned with the manuscript

## Local Machine

Responsibilities:

- edit code
- edit configs
- write docs
- run unit tests and smoke tests
- commit and push clean changes

Recommended commands:

```bash
git status
python -m pytest
git add <changed files>
git commit -m "short factual message"
git push
```

If the package is not installed, use:

```bash
PYTHONPATH=src python -m pytest
```

## Experiment Server

Responsibilities:

- pull code
- link datasets and checkpoints outside Git
- run training and evaluation
- preserve logs and small summaries
- push reproducibility metadata back to GitHub

Runtime note for this host:

- use the `zwl_oneformer_ViT_P` conda environment for OpenPrompt training
- avoid `dlp` for training; it currently lacks `torch`
- the repo disables cuDNN in `seed_everything()` because CUDA convolutions segfault on the host RTX 4090 unless cuDNN is off

Recommended commands:

```bash
git pull
python scripts/link_local_assets.py --dota-root /path/to/DOTA --outputs-dir /path/to/outputs
PYTHONPATH=src python scripts/smoke_test.py --config configs/experiments/geonexus_synthetic.yaml
PYTHONPATH=src python scripts/train.py --config configs/experiments/dota_v1_baseline_repro.yaml
# Swap to configs/experiments/dota_v15_baseline_repro.yaml if the staged asset is DOTA v1.5.
```

For long runs, prefer a detached `screen` session so SSH disconnects do not
stop training:

```bash
screen -dmS openprompt_dota_v1_baseline bash -lc '
  source /data1/anaconda3/etc/profile.d/conda.sh &&
  conda activate zwl_oneformer_ViT_P &&
  cd /data5/2025/ldh/OpenPrompt &&
  PYTHONPATH=src python scripts/train.py --config configs/experiments/dota_v1_baseline_repro.yaml |& tee -a outputs/openprompt_dota_v1_baseline/train.log
'
screen -r openprompt_dota_v1_baseline
```

For long runs, save the command, Git commit, config path, machine, GPU count,
dataset path, and result summary in `docs/experiments/`.

When the server produces a paper-facing metric, add a short record in
`docs/experiments/` that states whether the result came from the scaffold,
synthetic smoke tests, or accepted DOTA-style evaluation, and record whether
the run used DOTA v1.0 or DOTA v1.5.

## What To Commit From Server

Commit:

- changed configs
- bug fixes
- small metrics summaries
- experiment notes
- environment notes
- scripts needed to reproduce the run

Do not commit:

- datasets
- checkpoints
- raw output directories
- large logs
- generated prompt-bank tensors
- `wandb/` directories

## Pulling Back Locally

After the server pushes:

```bash
git pull
git status
```

Then inspect the experiment note and decide the next code change locally.
