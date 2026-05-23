# Experiment: dota_v1_baseline_server_progress

Date: 2026-05-22

Git commit: 663439d

Machine: nuosen (server)

GPU: 1 x NVIDIA GeForce RTX 4090

Dataset: /data2/2023/lcs/xyun/DOTA (train/val; DOTA v1.0 labels for the first baseline run)

Config: `configs/experiments/dota_v1_baseline_repro.yaml`

Command:

```bash
source /data1/anaconda3/etc/profile.d/conda.sh && \
conda activate zwl_oneformer_ViT_P && \
cd /data5/2025/ldh/OpenPrompt && \
PYTHONPATH=src python scripts/train.py --config configs/experiments/dota_v1_baseline_repro.yaml
```

External checkpoint path: `outputs/dota_v1_baseline_repro/epoch_012.pt`

Detached screen session: `openprompt_dota_v1_baseline` (completed)

External log path: `outputs/openprompt_dota_v1_baseline/train.log`

Resume command:

```bash
screen -r openprompt_dota_v1_baseline
```

## Purpose

Start the first server-side baseline reproduction run on DOTA v1.0 and verify that the training pipeline is stable on the available GPU environment.

## Result

- Training completed successfully in `zwl_oneformer_ViT_P` inside a detached `screen` session.
- The run progressed past the first batches after disabling cuDNN in the shared runtime hook.
- All 12 epochs finished and checkpoints were written through `outputs/dota_v1_baseline_repro/epoch_012.pt`.
- Final epoch metrics: `loss=0.18908667655497577`, `loss_cls=0.0010149941903454095`, `loss_box=0.09403584113557859`, `positive_cls_acc=0.3241933747263396`, `positive_box_l1=0.18381485000583453`.
- The detached session successfully protected the run from SSH disconnects.

## Notes

- `dlp` is not suitable for training this repo on this server because it does not have `torch` installed.
- The RTX 4090 on this host segfaults on the cuDNN convolution path, so the repo now disables cuDNN in `seed_everything()`.
- Long runs should be started with `scripts/run_train_in_screen.sh` or an equivalent detached `screen` command.
- If the SSH session drops, reconnect with `screen -r openprompt_dota_v1_baseline`.
- Earlier dataset configuration issues were fixed before this run.

## Next Action

Run evaluation on the completed DOTA v1.0 checkpoint, record the metrics, and then launch the matched DOTA v1.5 baseline if the v1.0 baseline is acceptable.
