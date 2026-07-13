# FAIR1M Sanitized Staging Handoff (2026-07-10)

## Current Gate

Dataset reconstruction and canonical class ordering are staged. No GPU job was
launched. The 1,000-step diagnostic and one-epoch smoke remain pending, so full
FAIR1M training is closed.

## Completed

- Raw source preserved: `/data2/2023/lcs/xyun/FAIR1M1.0`.
- Source archives preserved: `/data5/2025/temp/Dataset/FAIR1M_2_800_400`.
- New root: `/data2/2023/lcs/xyun/FAIR1M_2_800_400_sanitized_20260710`.
- Complete train archive count corrected to `208927`; `134486` belongs to
  `Step5_3_Prepare_Visual_Text_DINOv2_support.pkl` and is not the image count.
- Train image/annotation pairs: `208927 / 208927`.
- ss_val image/sanitized-annotation pairs: `10970 / 10970`.
- Train active objects: `1785001`.
- Train difficulty counts: `0 = 1730100`, `2 = 54901`.
- Rejected raw train records: `6513`, fully recorded in JSON.
- ss_val active objects: `199347`.
- Rejected ss_val archive records: `2020`, fully recorded in JSON.
- All 37 canonical hyphenated classes are present in both active splits.
- Reconstruction unit tests pass for valid, malformed, unknown-class,
  out-of-bounds, fully contained, truncated, and below-threshold polygons.

Reports:

- `/data2/2023/lcs/xyun/FAIR1M_2_800_400_sanitized_20260710/reports/train_annotation_reconstruction.json`
- `/data2/2023/lcs/xyun/FAIR1M_2_800_400_sanitized_20260710/reports/ss_val_annotation_sanitization.json`

## Archive Integrity Exception

Both ss_val archives pass `7z t` with `10970` files each. The 14-volume train
archive contains `208927` files but fails CRC for exactly:

`images/14777__533__0___801.png`

7-Zip extracted a corrupt file with a valid PNG header; OpenCV/libpng rejected
it with `bad adaptive filter value`. Only the staged copy was replaced using
the original splitter geometry:

`cv2.imread(raw_14777)[801:1334, 0:533]`

The replacement decodes as `(533, 533, 3)` and has SHA-256:

`66fdbadb2bf7ac5c6a0de8e49bd55fec604e862fb294a1dcca1b03fbb4e96f96`

Do not describe the source train archive itself as CRC-clean. The staged set is
repaired, while the source failure remains provenance evidence.

## Code And Canonical Artifacts

New utilities in `New/`:

- `scripts/reconstruct_fair1m_tiled_annotations.py`
- `scripts/sanitize_fair1m_dota_annotations.py`
- `scripts/canonicalize_fair1m_taxonomy_and_embeddings.py`
- `tests/test_reconstruct_fair1m_tiled_annotations.py`

Canonical taxonomy:

- `assets/hierarchies/fair1m_remote_sensing_taxonomy.json`

Superseded artifact, preserved:

- `artifacts/generated/remoteclip_vit_b32_fair1m_prompt_embeddings.pt`

New canonical artifact:

- `artifacts/generated/remoteclip_vit_b32_fair1m_prompt_embeddings_canonical.pt`
- Shape `[37, 512]`, all finite.
- `class_names`, prompts, and embedding rows use the required detector order.

Staged OpenRSD config:

- `/data5/2025/ldh/OpenRSD/M_configs/G02_Baselines/Data3_FAIR1M/G02_Baselines_Data3_FAIR1M_M2_RoITrans_S0_Sanitized_20260710.py`
- Batch size `2`, `diff_thr=1`, one epoch, validation/checkpoint at epoch 1.
- Uses sanitized train annotations and `ss_val/annfiles_sanitized`.
- Points to `checkpoints/pretrained/resnet50-0676ba61.pth`.

## Not Completed

- Official torchvision ResNet-50 download/checksum. The approved curl was
  interrupted and no usable weight file was found afterward.
- Full scan proving zero malformed active records, zero invalid MMRotate rboxes,
  and zero unknown active classes.
- Representative decoding at every train tile size and ss_val size.
- MMEngine config load, dataloader construction, and one complete batch.
- Three host GPU idle polls.
- Detached 1,000-batch `train-step` diagnostic.
- One-epoch S0 smoke, checkpoint, validation metric, and provenance record.

No screen or GPU process was created.

## Next Commands

Run from `/data5/2025/ldh/OpenRSD`.

1. Download and checksum official initialization:

```bash
mkdir -p checkpoints/pretrained
curl -fL --retry 3 -o checkpoints/pretrained/resnet50-0676ba61.pth \
  https://download.pytorch.org/models/resnet50-0676ba61.pth
sha256sum checkpoints/pretrained/resnet50-0676ba61.pth
```

2. Run the full active-split geometry/rbox scan, exact stem comparison,
   representative decode checks for `533`, `800`, and `1024`, config load,
   dataloader construction, and one full batch. Fix any issue before GPUs.

3. Poll host GPUs three times. Each selected GPU must remain below 1 GB memory
   and at zero utilization. Select dynamically; do not reserve old candidates.

4. Launch exactly 1,000 train steps in a detached screen:

```bash
CUDA_VISIBLE_DEVICES=<gpu> /data1/anaconda3/envs/zwl_mmrotate/bin/python \
  tools/bootstrap_run.py tools/diagnose_first_nonfinite_loss.py \
  M_configs/G02_Baselines/Data3_FAIR1M/G02_Baselines_Data3_FAIR1M_M2_RoITrans_S0_Sanitized_20260710.py \
  --work-dir work_dirs/geonexus_fair1m/roi_trans_s0_sanitized_diag1000_20260710 \
  --out work_dirs/geonexus_fair1m/roi_trans_s0_sanitized_diag1000_20260710/result.json \
  --max-batches 1000 --mode train-step --progress-interval 10
```

Proceed only when `checked_batches == 1000`, status is
`finite_within_limit`, and logs contain no traceback, OOM, decode, invalid-box,
NaN, or Inf signatures.

5. Launch a separate one-epoch smoke from the staged config. Accept startup
   only after iteration 200 and a clean failure scan. Stop after epoch-1
   validation. A finite, nonzero metric and epoch-1 checkpoint are required;
   do not automatically launch 12 epochs.

6. Record timestamp, screen, GPU and remapping, PID, config, workdir, weight
   checksum, exact command, logs, startup marker, diagnostic result, checkpoint,
   validation metric, and failure scan in a new dated experiment note.
