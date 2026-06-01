# 2026-05-31 Resume OpenRSD DOTA2 Launch

## Context

- Repository state: `New/` tracked files were clean at handoff. `OpenRSD/` is dirty from existing local work, so this resume only edits new experiment/archive files.
- Correct next experiment: OpenRSD DOTA2 tiled-data online federated-label training, not original-image DOTA S0 baselines.
- Do not interrupt other-user jobs on GPU 0 or GPU 5.

## GPU Status

Checked with `nvidia-smi` before launch:

| GPU | Memory | Util | Status |
| --- | --- | --- | --- |
| 0 | 3745 / 24564 MiB | 49% | occupied by other-user training |
| 1 | 14 / 24564 MiB | 0% | usable |
| 2 | 14 / 24564 MiB | 0% | selected for smoke |
| 3 | 14 / 24564 MiB | 0% | usable |
| 4 | 14 / 24564 MiB | 0% | usable |
| 5 | 21671 / 24564 MiB | 45% | occupied by other-user training |
| 6 | 14 / 24564 MiB | 0% | usable |

## Completed Diagnostics

- ORCNN seed9281 best `mAP/AP50 = 0.2918/0.2920`.
- RoITrans seed3407 final `mAP/AP50 = 0.2533/0.2530`.

## Dataset Readiness

Verified paths:

- `OpenRSD/data/DOTA2_1024_500/train/images`: 170878 files.
- `OpenRSD/data/DOTA2_1024_500/ss_val/images`: 13833 files.
- `OpenRSD/data/DOTA2_1024_500/ss_val/annfiles`: 13833 files.
- `OpenRSD/data/Formatted_FederatedLabels/Data1_DOTA2`: 78044 `.pkl` files.
- `OpenRSD/data/Formatted_FederatedLabels/Data1_DOTA2_nozero`: 78024 symlinks.
- `OpenRSD/data/Neg_supports_v2.pkl`: exists.
- `OpenRSD/data/normalized_class_dict.pkl`: exists.

Missing classic training label directories:

- `OpenRSD/data/DOTA2_1024_500/train/annfiles`
- `OpenRSD/data/DOTA2_1024_500/train/Step6_Format_labels`

Supplement uploaded by user:

- Location: `/data5/2025/temp/Dataset/DOTA2_supplement`.
- OBB source used: `labelTxt-v2.0/DOTA-v2.0_train.zip`.
- HBB source present but not used: `labelTxt-v2.0/DOTA-v2.0_train_hbb.zip`.
- `meta.zip` present.

Generated classic tiled train labels from the uploaded OBB labels and existing train tile names:

- Script: `New/scripts/build_dota2_tiled_annfiles_from_labels.py`.
- Output: `OpenRSD/data/DOTA2_1024_500/train/annfiles_generated_20260531`.
- Count: `170878` txt files, matching train image count.
- Non-empty/empty: `78044` / `92834`.
- Total objects: `2317821`.
- Symlinks created:
  - `OpenRSD/data/DOTA2_1024_500/train/annfiles -> annfiles_generated_20260531`
  - `OpenRSD/data/DOTA2_1024_500/train/Step6_Format_labels -> annfiles_generated_20260531`

Conclusion for S0 strong-detector work: original DOTA2 train images do not need to be downloaded again for label generation; the existing tiled train images plus uploaded `DOTA-v2.0_train.zip` were sufficient. `ss_val/annfiles` is usable as validation annotations, but it could not replace the missing `train/annfiles` required by classic S0 training.

## Config

Smoke config:

- `OpenRSD/work_dirs/opensrd_step2_dota2_nozero_smoke/opensrd_step2_dota2_nozero_smoke.py`
- Base: `M_configs/Step2_A10_Large_Pretrain_Stage3/A10_flex_rtm_v3_1_formal.py`
- DOTA2-only overrides:
  - `model.support_feat_dict` uses `_delete_=True` and keeps only `Data1_DOTA2`.
  - Training dataset is only `DOTADatasetOnline` on `data/DOTA2_1024_500/train/images`.
  - Labels use `/data5/2025/ldh/OpenRSD/data/Formatted_FederatedLabels/Data1_DOTA2_nozero`.
  - `max_epochs = 1`, `max_iter_per_epoch = 200`, `batch_size = 1`.
  - Validation/test loops and evaluators are disabled for smoke.

Previous observed failure in `work_dirs/opensrd_step2_dota2_nozero_smoke/launch.log`: the config merged `support_feat_dict` with the base dict and still loaded missing `./data/STAR_800_200/val/Step5_3_Prepare_Visual_Text_DINOv2_support.pkl`. The resumed config fixes that by deleting the base support dictionary before setting DOTA2 support.

## Launch Command

```bash
cd /data5/2025/ldh/OpenRSD
CUDA_VISIBLE_DEVICES=2 MPLCONFIGDIR=/tmp/matplotlib_opensrd_dota2_smoke \
  /data1/anaconda3/envs/zwl_mmrotate/bin/python tools/bootstrap_run.py tools/train.py \
  work_dirs/opensrd_step2_dota2_nozero_smoke/opensrd_step2_dota2_nozero_smoke.py \
  --work-dir work_dirs/opensrd_step2_dota2_nozero_smoke
```

## First Observed Status

- First resumed launch passed model/support loading but failed before iteration 1 because `DOTADatasetOnline._join_prefix()` prepended `data_root` to the relative label directory, making dataset length 0.
- Config was patched again to use the absolute nozero label path.
- MMEngine resolved the effective `train_dataloader.dataset.datasets[0].ann_file` from the final dumped block in the config, so both the top-level `Data1_DOTA2` definition and the nested train dataloader copy must use the absolute nozero path.

## Smoke Result

- Command: the smoke launch command above, on GPU 2.
- Effective dataset: `Data1_DOTA2` only, sampler size `78024`, `max_iter_per_epoch=200`.
- Training reached `Epoch(train) [1][200/200]`.
- Checkpoint saved: `OpenRSD/work_dirs/opensrd_step2_dota2_nozero_smoke/epoch_1.pth`.
- Latest smoke log: `OpenRSD/work_dirs/opensrd_step2_dota2_nozero_smoke/20260531_094052/20260531_094052.log`.

## Full Run

- Config: `OpenRSD/work_dirs/opensrd_step2_dota2_nozero_full_20260531/opensrd_step2_dota2_nozero_full_20260531.py`.
- Command:

```bash
cd /data5/2025/ldh/OpenRSD
CUDA_VISIBLE_DEVICES=2 MPLCONFIGDIR=/tmp/matplotlib_opensrd_dota2_full \
  /data1/anaconda3/envs/zwl_mmrotate/bin/python tools/bootstrap_run.py tools/train.py \
  work_dirs/opensrd_step2_dota2_nozero_full_20260531/opensrd_step2_dota2_nozero_full_20260531.py \
  --work-dir work_dirs/opensrd_step2_dota2_nozero_full_20260531
```

- Detached session: `screen -r opensrd_dota2_full_20260531`.
- Main PID at launch verification: `2232790`.
- Launch log: `OpenRSD/work_dirs/opensrd_step2_dota2_nozero_full_20260531/launch.log`.
- Verified status after detached relaunch: `Epoch(train) [1][250/12000]`, GPU 2 active at about `4647 MiB`, ETA about `7:23:02`.
- Validation is disabled for this run. This is training-only evidence because the online training labels are ready, while validation compatibility against `ss_val/annfiles` has not been proven in this config.

## Full Run Crash and Resume

- Original full run saved `epoch_1.pth`, then crashed in epoch 2 after `Epoch(train) [2][3500/12000]`.
- Crash signature:
  - `libpng error: IDAT: CRC error`
  - `AttributeError: 'NoneType' object has no attribute 'shape'` in `mmcv.transforms.loading.LoadImageFromFile`.
- Root cause found by PIL full decode scan:
  - `OpenRSD/data/DOTA2_1024_500/train/images/P1497__1024__1572___1572.png`
  - `OpenRSD/data/DOTA2_1024_500/train/images/P2612__682__4188___698.png`
  - `OpenRSD/data/DOTA2_1024_500/train/images/P2756__1024__2096___3144.png`
  - `OpenRSD/data/DOTA2_1024_500/train/images/P3536__2048__4192___1048.png`
  - `OpenRSD/data/DOTA2_1024_500/train/images/P3536__2048__4192___3144.png`
  - `OpenRSD/data/DOTA2_1024_500/train/images/P3536__2048__4192___5240.png`
  - `OpenRSD/data/DOTA2_1024_500/train/images/P4076__2048__2049___0.png`
  - `OpenRSD/data/DOTA2_1024_500/train/images/P7584__1024__1572___1048.png`
  - `OpenRSD/data/DOTA2_1024_500/train/images/P7584__1024__1572___524.png`
  - `OpenRSD/data/DOTA2_1024_500/train/images/P8461__682__1396___4188.png`
- `identify` over all `78024` `Data1_DOTA2_nozero` referenced images reported no errors, so PIL decode was needed to catch this PNG corruption.
- Added a narrow guard in `OpenRSD/M_AD/datasets/dota_online_v1.py`: when `LoadImageFromFile` returns `None` and causes the known `None.shape` `AttributeError`, log `Skipping unreadable image: ...` and let the dataset refetch another sample.
- Created future-restart label set excluding only the corrupt sample:
  - `OpenRSD/data/Formatted_FederatedLabels/Data1_DOTA2_nozero_validpng_20260531`
  - Count after excluding the ten known corrupt referenced tiles: `78014` `.pkl` symlinks.
- Patched future full-run config references from `Data1_DOTA2_nozero` to `Data1_DOTA2_nozero_validpng_20260531`.
- Resumed active run command:

```bash
cd /data5/2025/ldh/OpenRSD
CUDA_VISIBLE_DEVICES=2 MPLCONFIGDIR=/tmp/matplotlib_opensrd_dota2_full \
  /data1/anaconda3/envs/zwl_mmrotate/bin/python tools/bootstrap_run.py tools/train.py \
  work_dirs/opensrd_step2_dota2_nozero_full_20260531/opensrd_step2_dota2_nozero_full_20260531.py \
  --work-dir work_dirs/opensrd_step2_dota2_nozero_full_20260531 \
  --resume auto
```

- Resume log: `OpenRSD/work_dirs/opensrd_step2_dota2_nozero_full_20260531/launch_resume_20260531_1101.log`.
- Detached session: `2396455.opensrd_dota2_full_20260531`.
- Resume checkpoint: `epoch_1.pth`, reported `resumed epoch: 1, iter: 12000`.
- Status at `2026-05-31 11:22:49 CST`: active on GPU 2, observed past `Epoch(train) [2][1950/12000]`, ETA about `6:24:34`.

## Final Training Gate Result

- The resumed full run completed all 12 epochs on `2026-05-31`.
- Final checkpoint: `OpenRSD/work_dirs/opensrd_step2_dota2_nozero_full_20260531/epoch_12.pth`.
- Last checkpoint pointer: `OpenRSD/work_dirs/opensrd_step2_dota2_nozero_full_20260531/last_checkpoint`.
- Final resume log: `OpenRSD/work_dirs/opensrd_step2_dota2_nozero_full_20260531/launch_resume_20260531_1101.log`.
- Final observed state: `Epoch(train) [12][12000/12000]`, finite losses, checkpoint saved at 12 epochs.
- The unreadable-image guard was exercised during epoch 12 for `P3536__2048__4192___3144.png` and training continued.

This is a training-only gate. Validation/test loops were disabled, so DOTA2
performance is not claimable yet. Treat this run as evidence that the DOTA2
online-label training path is now runnable through 12 epochs. Before reporting
DOTA2 mAP/AP50, adapt and run validation against
`OpenRSD/data/DOTA2_1024_500/ss_val/annfiles` with a compatible evaluator.

Next paper-facing experiment gate: S2 hierarchy regularizer on the existing
DOTA v1.5 reduced tiled split. Compare it against S1 frozen and the S2
hierarchy-offset epoch-1 evidence before deciding whether to proceed to S3.
