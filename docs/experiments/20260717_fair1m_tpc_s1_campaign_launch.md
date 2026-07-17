# FAIR1M TPC/S1 Campaign Launch (2026-07-17)

## Gate result

The S1 campaign passed the configuration, prompt artifact, complete-batch,
checkpoint, and real 1,000-batch train-step gates. Three clean 12-epoch
replicas were launched from the matching FAIR1M S0 epoch-12 checkpoints.

Evidence:

- Config gate: `OpenRSD/work_dirs/geonexus_fair1m/fair1m_tpc_s1_config_gate_20260717/config_gate_rep3407.json`
- Checkpoint gate log: `OpenRSD/work_dirs/geonexus_fair1m/roi_trans_tpc_s1_checkpoint_gate_20260717/checkpoint_gate_noema.log`
- 1,000-step diagnostic: `OpenRSD/work_dirs/geonexus_fair1m/roi_trans_tpc_s1_precision_diag1000_20260717/result.json`
- Prompt artifact: `New/artifacts/generated/remoteclip_vit_b32_fair1m_prompt_embeddings_canonical.pt`, shape `[37,512]`
- Source checkpoints: S0 `epoch_12.pth` in each matching replica workdir.

The checkpoint load reported only the expected newly initialized S1 prompt
keys (`prompt_offsets`, `prompt_bias`, `bg_logit`, `logit_scale`, prompt
embedding buffer, and projection) for both cascade stages. No unexpected
critical parameters were reported. The standalone diagnostic disabled EMA
because the repository EMA hook expects an initialized training-run EMA
object; full launch configs retain the original S0 EMA hook.

## Launch provenance

The three-poll GPU gate at `2026-07-17T16:25:16+08:00` showed GPUs 0–5 at
14 MiB / 0%; GPU 6 was occupied at 23,183 MiB and was excluded. No remapping
was required.

| Seed | Screen | GPU | Workdir | Config | Source checkpoint |
|---:|---|---:|---|---|---|
| 3407 | `fair1m_tpc_s1_rep3407_20260717_gpu0` | 0 | `OpenRSD/work_dirs/geonexus_fair1m/roi_trans_tpc_s1_rep3407_20260717` | `..._rep3407_20260717.py` | `.../roi_trans_s0_rep3407_20260715/epoch_12.pth` |
| 4407 | `fair1m_tpc_s1_rep4407_20260717_gpu1` | 1 | `OpenRSD/work_dirs/geonexus_fair1m/roi_trans_tpc_s1_rep4407_20260717` | `..._rep4407_20260717.py` | `.../roi_trans_s0_rep4407_20260715/epoch_12.pth` |
| 5407 | `fair1m_tpc_s1_rep5407_20260717_gpu2` | 2 | `OpenRSD/work_dirs/geonexus_fair1m/roi_trans_tpc_s1_rep5407_20260717` | `..._rep5407_20260717.py` | `.../roi_trans_s0_rep5407_20260715/epoch_12.pth` |

Each workdir retains `launch_provenance.txt` and `launch_20260717_gpu*.log`
with the exact command, PID/startup marker, and scoped failure scan. Initial
startup acceptance was reached cleanly at iteration 200 for all three
replicas: `3407` loss `1.0167`, `4407` loss `0.9555`, and `5407` loss `0.9152`.
The latest observed markers are iteration 500 for `3407`, and iteration 350
for `4407` and `5407`; all losses remain finite. Replica acceptance requires
finite losses at iterations 200 and 1,000 and no
traceback, CUDA OOM, decode/CRC, invalid-box, NaN/Inf, or interruption
signature. Do not launch FAIR1M S2/GeoNexus or unrelated paused work.
