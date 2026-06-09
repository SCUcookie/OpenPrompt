# DOTA2 S2 Live Status And DIOR-R Diagnostic Policy - 2026-06-09

This note records the DOTA2 three-GPU S2 state, the completed main GPU-1 S2
result, and the decision to use the freed GPU for DIOR-R diagnostics only.

## Live Server Snapshot

Checked at `2026-06-09 09:37 CST`.

| GPU | Screen | Process PID | Status |
| --- | --- | --- | --- |
| 0 | `2711972.geonexus_dota2_roi_trans_s2_hierarchy_reg_s1e12_lr1e4_20260608_gpu0` | `2711973` | active low-LR S2 |
| 1 | `715887.geonexus_dota2_roi_trans_s2_hierarchy_reg_s1e12_20260608_gpu1` | `716070` | active main S2 |
| 6 | `2711969.geonexus_dota2_roi_trans_s2_hierarchy_reg_s1e12_hw1e2_20260608_gpu6` | `2711971` | active reduced hierarchy-weight S2 |

`nvidia-smi` showed GPU memory in use by the three Python jobs: GPU 0 about
`15400 MiB`, GPU 1 about `19116 MiB`, and GPU 6 about `12618 MiB`. GPUs 2-5
had only Xorg memory.

Completion check at `2026-06-09 11:10:06 CST +0800`: the GPU-1 main S2
process `716070` was gone, the screen
`geonexus_dota2_roi_trans_s2_hierarchy_reg_s1e12_20260608_gpu1` was no longer
listed by `screen -ls`, and GPU 1 had returned to Xorg-only memory. GPU 0 and
GPU 6 S2 variants remained active.

## Completed DOTA2 S1 Evidence

All metrics below are DOTA2 `DOTA2_1024_500/ss_val` results.

| Run | Final metric |
| --- | --- |
| Main S1, GPU 1 | `dota/mAP=0.6177`, `dota/AP50=0.6180` |
| S1 LR `1e-4`, GPU 6 | `0.5997 / 0.6000` |
| S1 LR `5e-5`, GPU 0 | `0.6047 / 0.6050` |

Main S1 remains the strongest DOTA2 GeoNexus result in this set.

## Active DOTA2 S2 Evidence

All metrics below are DOTA2 `DOTA2_1024_500/ss_val` results.

| Run | Interim metric | Current log position |
| --- | --- | --- |
| Main S2, GPU 1 | epoch 4 `0.6038 / 0.6040`; epoch 8 `0.5892 / 0.5890`; final epoch 12 `dota/mAP=0.5924431681632996`, `dota/AP50=0.5920` | completed at `2026-06-09 11:10:06 CST +0800`; final log `Epoch(val) [12][6917/6917]` |
| Low-LR S2, GPU 0 | epoch 4 `0.6099 / 0.6100` | epoch 8, latest checked around `[8][300/39007]` |
| Reduced hierarchy-weight S2, GPU 6 | epoch 4 `0.6035 / 0.6040` | epoch 8, latest checked around `[8][6850/39007]` |

The completed main GPU-1 S2 result is below the main S1 final metric
`0.6177 / 0.6180` by `-0.0252568318367004` mAP and `-0.0260` AP50. Do not
launch S3/S4 from this main S2 result. GPU 0 and GPU 6 S2 variants remain
active.

Main GPU-1 S2 checkpoint:

`/data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s2_hierarchy_reg_s1e12_20260608/epoch_12.pth`

## Failure Scan

Scoped checks covered the three S2 launch/training logs:

- `roi_trans_remoteclip_s2_hierarchy_reg_s1e12_20260608`
- `roi_trans_remoteclip_s2_hierarchy_reg_s1e12_lr1e4_20260608`
- `roi_trans_remoteclip_s2_hierarchy_reg_s1e12_hw1e2_20260608`

No active tail showed `Traceback`, CUDA OOM, `libpng`, `CRC`, `NoneType`,
`ValueError`, non-finite training losses, or `KeyboardInterrupt`. A literal
substring scan for `inf` matches static config text such as `metainfo`, so
future scans should distinguish that from true `nan`/`inf` tensor or loss
values.

After the main GPU-1 S2 completed, a scoped scan of its completed launch and
training logs found no `Traceback`, CUDA OOM, `libpng`, `CRC`, `NoneType`,
`ValueError`, `KeyboardInterrupt`, or true non-finite loss/metric signature.

## Decision

Do not launch a new detector training job while the three DOTA2 S2 jobs occupy
GPUs 0, 1, and 6. Keep S3/S4, pseudo-labeling, FAIR1M, and DIOR-R detector
training paused.

GPU 1 became free after the main S2 completion. Use it for DIOR-R diagnostics
only. The invalid DIOR-R detector runs should not be cited as DIOR-R baseline
evidence; they are evidence that the DIOR-R data, rotated-box conversion, class
mapping, or loss-target path needs diagnosis before detector training resumes.

Prepared diagnostic utility:

`/data5/2025/ldh/New/scripts/diagnose_dior_r_geometry_and_targets.py`

## Diagnostic Launch

After GPU 1 freed, launched detached screen
`dior_r_geometry_targets_diag_20260609_gpu1` at `2026-06-09 11:11 CST`:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 \
python3 New/scripts/diagnose_dior_r_geometry_and_targets.py \
  --check-dataloader \
  --check-first-loss \
  --output-json New/artifacts/dior_r_diagnostics_20260609_gpu1.json \
  --output-md New/artifacts/dior_r_diagnostics_20260609_gpu1.md \
  2>&1 | tee New/artifacts/dior_r_diagnostics_20260609_gpu1.log
```

Status at `2026-06-09 11:17 CST`: the screen is active, Python PID `2024594`
is running the diagnostic script, GPU 1 remains Xorg-only, and the log file
exists at `New/artifacts/dior_r_diagnostics_20260609_gpu1.log`. The JSON and
Markdown outputs are pending until the full scan plus dataloader/first-loss
checks finish. No detector training command was launched.
