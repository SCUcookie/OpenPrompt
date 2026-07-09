# 2026-07-04 OrientedFormer Swin-T DIOR-R Protocol Eval Rerun

## Intent

Evaluation-only reproducibility rerun of the already validated OrientedFormer Swin-T DIOR-R protocol eval. No new training, no FAIR1M work, and no Strip R-CNN DIOR-R continuation.

## Inputs

- Config: `/data5/2025/ldh/orientedformer_protocol_eval_20260702/orientedformer_swin_t_dior_r_geonexus_test_20260702.py`
- Checkpoint: `/data5/2025/ldh/orientedformer_protocol_eval_20260702/checkpoints/orientedformer_hf/orientedformer_le90_swin-tiny_q300_layer2_head64_point32_1x_dior/epoch_12.pth`
- PYTHONPATH shim: `/data5/2025/ldh/orientedformer_protocol_eval_20260702/pyshim`
- Prior baseline note: `/data5/2025/ldh/New/docs/experiments/20260702_orientedformer_dior_r_swin_t_protocol_eval_launch.md`
- Fresh rerun workdir: `/data5/2025/ldh/orientedformer_protocol_eval_20260702/orientedformer_swin_t_dior_r_geonexus_eval_20260704_rerun`

## Preflight

- `screen -ls` before launch showed only `3470174.s0_result_log_monitor_20260603`.
- Config and checkpoint paths existed.
- `tools/misc/print_config.py` passed with:

```bash
PYTHONPATH=/data5/2025/ldh/orientedformer_protocol_eval_20260702/pyshim \
PYTHONNOUSERSITE=1 \
MPLCONFIGDIR=/tmp/matplotlib_orientedformer_swin_t_dior_r_20260704 \
/data1/anaconda3/envs/zwl_mmrotate/bin/python \
/data5/2025/ldh/OrientedFormer/tools/misc/print_config.py \
/data5/2025/ldh/orientedformer_protocol_eval_20260702/orientedformer_swin_t_dior_r_geonexus_test_20260702.py
```

- GPU snapshot before launch:
  - `0, 14 MiB, 0 %`
  - `1, 14 MiB, 0 %`
  - `2, 14 MiB, 0 %`
  - `3, 14 MiB, 0 %`
  - `4, 14 MiB, 0 %`
  - `5, 14 MiB, 0 %`
  - `6, 14 MiB, 0 %`
- Launch GPU: `0`
- GPU remap: none

## Launch Command

```bash
screen -dmS orientedformer_swin_t_dior_r_eval_20260704_gpu0 bash -lc 'cd /data5/2025/ldh/OrientedFormer && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/data5/2025/ldh/orientedformer_protocol_eval_20260702/pyshim PYTHONNOUSERSITE=1 MPLCONFIGDIR=/tmp/matplotlib_orientedformer_swin_t_dior_r_20260704 /data1/anaconda3/envs/zwl_mmrotate/bin/python tools/test.py /data5/2025/ldh/orientedformer_protocol_eval_20260702/orientedformer_swin_t_dior_r_geonexus_test_20260702.py /data5/2025/ldh/orientedformer_protocol_eval_20260702/checkpoints/orientedformer_hf/orientedformer_le90_swin-tiny_q300_layer2_head64_point32_1x_dior/epoch_12.pth --work-dir /data5/2025/ldh/orientedformer_protocol_eval_20260702/orientedformer_swin_t_dior_r_geonexus_eval_20260704_rerun --out /data5/2025/ldh/orientedformer_protocol_eval_20260702/orientedformer_swin_t_dior_r_geonexus_eval_20260704_rerun/preds.pkl > /data5/2025/ldh/orientedformer_protocol_eval_20260702/orientedformer_swin_t_dior_r_geonexus_eval_20260704_rerun/test_stdout.log 2>&1'
```

## Runtime Acceptance

- Screen name at launch: `orientedformer_swin_t_dior_r_eval_20260704_gpu0`
- Active GPU process was observed on GPU 0 via `nvidia-smi --query-compute-apps`, with the eval Python worker peaking at `3062 MiB`.
- Checkpoint load confirmed:
  - `Load checkpoint from /data5/2025/ldh/orientedformer_protocol_eval_20260702/checkpoints/orientedformer_hf/orientedformer_le90_swin-tiny_q300_layer2_head64_point32_1x_dior/epoch_12.pth`
- Test-loop progress confirmed from log:
  - `Epoch(test) [   50/11738]`
  - `Epoch(test) [11738/11738]    dota/mAP: 0.6883  dota/AP50: 0.6880 ...`

## Completion

- Start: `2026-07-04 10:52:35`
- End: `2026-07-04 11:08:44`
- Screen state after completion: exited; `screen -ls` again showed only `3470174.s0_result_log_monitor_20260603`
- Metrics JSON: `/data5/2025/ldh/orientedformer_protocol_eval_20260702/orientedformer_swin_t_dior_r_geonexus_eval_20260704_rerun/20260704_105235/20260704_105235.json`
- Runtime log: `/data5/2025/ldh/orientedformer_protocol_eval_20260702/orientedformer_swin_t_dior_r_geonexus_eval_20260704_rerun/20260704_105235/20260704_105235.log`
- Stdout log: `/data5/2025/ldh/orientedformer_protocol_eval_20260702/orientedformer_swin_t_dior_r_geonexus_eval_20260704_rerun/test_stdout.log`
- Prediction dump: `/data5/2025/ldh/orientedformer_protocol_eval_20260702/orientedformer_swin_t_dior_r_geonexus_eval_20260704_rerun/preds.pkl`

Final metrics:

```json
{"dota/mAP": 0.6882880926132202, "dota/AP50": 0.688, "data_time": 0.018639041021057484, "time": 0.07046237812880703}
```

## Baseline Comparison

- 2026-07-02 baseline: `{"dota/mAP": 0.6882880926132202, "dota/AP50": 0.688}`
- 2026-07-04 rerun: `{"dota/mAP": 0.6882880926132202, "dota/AP50": 0.688}`
- `dota/AP50` rounded acceptance: pass (`0.688`)
- `dota/mAP` exact-match tolerance check: pass (`delta=0.0`)
- Classification: reproducibility confirmed, no drift

## Failure Scan

- Scoped scan pattern: `Traceback|Error|Exception|CUDA out|out of memory|No such file|AssertionError|RuntimeError|failed|Failed`
- Matches were limited to the config text `allow_failed_imports=False`.
- No real traceback, exception, CUDA OOM, missing-file, assertion, or runtime-error signature appeared.
- Non-fatal warnings seen during evaluation:
  - `torch.meshgrid` upcoming indexing argument requirement
  - PyTorch `__floordiv__` deprecation warnings inside OrientedFormer project code

## Outcome

This rerun reproduced the prior July 2 protocol result exactly with the same config, checkpoint, and DIOR-R XML bridge assets. No further route was opened.
