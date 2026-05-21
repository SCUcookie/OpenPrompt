# GitHub And Experiment Server Workflow

Use GitHub as the synchronization layer between the local coding machine and the
experiment server.

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

Recommended commands:

```bash
git pull
python scripts/link_local_assets.py --dotav2-root /path/to/DOTAv2 --outputs-dir /path/to/outputs
PYTHONPATH=src python scripts/smoke_test.py --config configs/experiments/geonexus_synthetic.yaml
PYTHONPATH=src python scripts/train.py --config configs/experiments/dota_v2_baseline_repro.yaml
```

For long runs, save the command, Git commit, config path, machine, GPU count,
dataset path, and result summary in `docs/experiments/`.

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

