# FAIR1M S0 Campaign Launch (2026-07-15)

## Gate result

The repaired precision-v2 root passed the final data/runtime gate and the
three-replica S0 campaign is active. The prior 2026-07-10 root remains
preserved. The expected `6513` rejected raw train records remain excluded and
recorded by the reconstruction provenance.

Evidence:

- Full audit: `artifacts/fair1m_geometry_gate_20260715_precision_v2_mmrotate.json`
- Full audit markdown: `artifacts/fair1m_geometry_gate_20260715_precision_v2_mmrotate.md`
- Config/batch gate: `OpenRSD/work_dirs/geonexus_fair1m/fair1m_config_gate_20260715/config_gate_rep3407.json`
- 1000-step diagnostic: `OpenRSD/work_dirs/geonexus_fair1m/roi_trans_s0_precision_diag1000_20260715/console.log`
- Checkpoint SHA-256: `0676ba61b6795bbe1773cffd859882e5e297624d384b6993f7c9e683e722fb8a`
- RemoteCLIP embeddings: `artifacts/generated/remoteclip_vit_b32_fair1m_prompt_embeddings_canonical.pt`, `[37,512]`

The audit found 208,927 train pairs / 10,970 validation pairs, 1,785,001 /
199,347 active objects, exact stems, zero malformed or zero-area active
records, zero unknown classes, zero invalid MMRotate rboxes, and no decode
failures across 1,000 representatives per split. It observed 155 out-of-bounds
objects in the decoded train representatives; this remains a follow-up review
item. The 1000-batch real train-step diagnostic ended
`finite_within_limit`.

## Launch provenance

The final three host polls at 10:31:25, 10:31:35, and 10:31:45 +08:00 showed
all seven RTX 4090s at 14 MiB / 0%. Dynamic selection assigned physical and
logical GPUs 0, 1, and 2:

| Seed | Screen | GPU | Main PID | Workdir | Launch log |
|---|---|---:|---:|---|---|
| 3407 | `fair1m_s0_rep3407_20260715_gpu0` | 0 | 1022098 | `OpenRSD/work_dirs/geonexus_fair1m/roi_trans_s0_rep3407_20260715` | `launch_20260715_gpu0.log` |
| 4407 | `fair1m_s0_rep4407_20260715_gpu1` | 1 | 1022112 | `OpenRSD/work_dirs/geonexus_fair1m/roi_trans_s0_rep4407_20260715` | `launch_20260715_gpu1.log` |
| 5407 | `fair1m_s0_rep5407_20260715_gpu2` | 2 | 1022131 | `OpenRSD/work_dirs/geonexus_fair1m/roi_trans_s0_rep5407_20260715` | `launch_20260715_gpu2.log` |

Each workdir contains `launch_provenance.txt` with the exact command and
config. The configs are the existing 3407/4407/5407 replicas, run for 12
epochs with validation/checkpoints at epochs 4, 8, and 12, and no resume.

## Startup acceptance

At the first check, all three reached `Epoch(train) [1][200/66467]` with finite
nonzero losses and no traceback, CUDA OOM, decode/CRC, invalid-box, NaN/Inf, or
keyboard-interruption signatures. The subsequent check reached
`[1][1000/66467]` for all three, still with finite losses. GPU residency was
approximately 10.1, 10.1, and 10.4 GiB on GPUs 0, 1, and 2.

Do not launch FAIR1M TPC/GeoNexus automatically. Review epoch-4/8/12 metrics,
best versus final checkpoints, replica mean/std, and complete failure scans
before the next route decision. DOTA2 follow-up, DIOR-R S4, pseudo-labeling,
and segmentation experiments remain paused.
