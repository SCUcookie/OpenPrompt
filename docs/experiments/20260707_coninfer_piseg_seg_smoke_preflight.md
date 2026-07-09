# 2026-07-07 Segmentation Baseline Smoke Preflight

## Intent

Start the `New/BRIEF_LIST.md` segmentation lane for open-vocabulary remote
sensing semantic segmentation. Detection fallback work is intentionally not
started in this run.

## Timestamp

- Local time: 2026-07-07T10:13:26+08:00
- Workspace: `/data5/2025/ldh`

## Source Verification

### ConInfer

- Official repo: https://github.com/Dog-Yang/ConInfer
- Local path: `/data5/2025/ldh/New/third_party/ConInfer`
- Commit: `ebaddccacce05fe3cc9a79eb401d3a62bff58c6a`
- Paper link from README: https://arxiv.org/abs/2603.29271
- Claimed task: training-free open-vocabulary remote-sensing segmentation
- Documented command: `bash ./dist_test.sh ./config/cfg_DATASET.py`
- Actual config directory in repo: `configs_ConInfer/`
- Dataset preparation reference: `dataset_prepare.md`, derived from SegEarth-OV
- Smallest plausible smoke config inspected:
  `configs_ConInfer/cfg_whu_aerial.py`

### RSKT-Seg

- Official repo: https://github.com/LiBingyu01/RSKT-Seg
- Local path: `/data5/2025/ldh/New/third_party/RSKT-Seg`
- Commit: `7b84091598e1edc3236dfbf45cc27e7e3436ffcb`
- Paper link from README: https://arxiv.org/pdf/2509.12040.pdf
- Claimed task: open-vocabulary remote sensing image segmentation
- README notes that Pi-Seg moved to a separate repo on 2026-06-30.
- Dataset source documented by README: OVSISBench via Baidu Netdisk/OneDrive.
- Pretrained weights documented by README: Baidu Netdisk/OneDrive plus CLIP,
  DINO, RemoteCLIP prerequisites.

### Pi-Seg

- Official repo: https://github.com/LiBingyu01/Pi-Seg
- Local path: `/data5/2025/ldh/New/third_party/Pi-Seg`
- Commit: `6a1a25a84bf81c2cbd2a103594a4c01d376de3d6`
- Paper link from README: https://arxiv.org/html/2604.15652v1
- Claimed task: Pi-Seg on CAT-Seg for OVRSISBenchV2
- Dataset sources documented by README:
  - https://huggingface.co/datasets/kkk2026/OVRSIS95K
  - https://huggingface.co/datasets/kkk2026/OVRSISBenchtest
  - https://huggingface.co/datasets/kkk2026/OVRSISBenchV2o3
- Weight sources documented by README:
  - https://huggingface.co/kkk2026/Pi-Seg_for_OVRSISBenchV1
  - https://huggingface.co/kkk2026/Pi-Seg_for_OVRSISBenchV2
- Documented single-dataset eval command:

```bash
python train_net.py \
    --config-file configs/vitl_336_OVRSIS95K.yaml \
    --eval-only \
    MODEL.WEIGHTS output/piseg_vitl/model_final.pth \
    DATASETS.TEST '("OpenEarthMap_sem_seg",)'
```

## Machine State

`screen -ls`:

```text
There is a screen on:
	3470174.s0_result_log_monitor_20260603	(06/03/26 19:55:38)	(Detached)
1 Socket in /run/screen/S-zwl.
```

`nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader`:

```text
0, 14 MiB, 0 %
1, 14 MiB, 0 %
2, 14 MiB, 0 %
3, 14 MiB, 0 %
4, 14 MiB, 0 %
5, 14 MiB, 0 %
6, 4765 MiB, 38 %
```

GPU 0 was idle and would have been selected for a launch if preflight had
passed.

## Environment Checks

### `dlp`

Command:

```bash
source ~/.bashrc >/dev/null 2>&1
conda activate dlp >/dev/null 2>&1
python - <<'PY'
mods=["torch","mmengine","mmseg","mmcv","transformers","timm","cv2","PIL"]
for m in mods:
    try:
        mod=__import__(m)
        print(m, getattr(mod, "__version__", "ok"))
    except Exception as e:
        print(m, "FAIL", type(e).__name__, str(e)[:160])
PY
```

Result:

```text
torch 2.4.1+cu121
mmengine FAIL ModuleNotFoundError No module named 'mmengine'
mmseg FAIL ModuleNotFoundError No module named 'mmseg'
mmcv FAIL ModuleNotFoundError No module named 'mmcv'
transformers FAIL ModuleNotFoundError No module named 'transformers'
timm FAIL ModuleNotFoundError No module named 'timm'
cv2 4.10.0
PIL 10.4.0
```

Conclusion: not suitable for ConInfer or Pi-Seg preflight.

### `lcs_mmseg`

Command:

```bash
source ~/.bashrc >/dev/null 2>&1
conda activate lcs_mmseg >/dev/null 2>&1
python - <<'PY'
mods=["torch","mmengine","mmseg","mmcv","transformers","timm"]
for m in mods:
    try:
        mod=__import__(m)
        print(m, getattr(mod, "__version__", "ok"))
    except Exception as e:
        print(m, "FAIL", type(e).__name__, str(e)[:120])
PY
```

Result:

```text
torch 2.0.0+cu118
mmengine 0.10.3
mmseg 1.2.2
mmcv 2.0.1
transformers 4.40.0
timm 0.9.16
```

This environment is close enough for OpenMMLab config/import preflight, but it
is not the exact ConInfer requirement set and lacks at least `fairscale`.

## ConInfer Preflight

Command:

```bash
source ~/.bashrc >/dev/null 2>&1
conda activate lcs_mmseg >/dev/null 2>&1
mkdir -p ../../runs/coninfer_smoke_20260707/{hf,torch,tmp}
HF_HOME=$PWD/../../runs/coninfer_smoke_20260707/hf \
TRANSFORMERS_CACHE=$PWD/../../runs/coninfer_smoke_20260707/hf \
TORCH_HOME=$PWD/../../runs/coninfer_smoke_20260707/torch \
TMPDIR=$PWD/../../runs/coninfer_smoke_20260707/tmp \
python - <<'PY'
from mmengine.config import Config
cfg=Config.fromfile("configs_ConInfer/cfg_whu_aerial.py")
print("config-ok", cfg.dataset_type, cfg.test_dataloader.dataset.data_prefix)
try:
    import eval as eval_module
    print("eval-import-ok")
except Exception as e:
    print("eval-import-fail", type(e).__name__, str(e))
try:
    import ConInfer_segmentor
    print("coninfer-import-ok")
except Exception as e:
    print("coninfer-import-fail", type(e).__name__, str(e))
PY
```

Result:

```text
config-ok WHUDataset {'img_path': 'data/WHU_Aerial/val/image', 'seg_map_path': 'data/WHU_Aerial/val/label_cvt'}
eval-import-fail ModuleNotFoundError No module named 'segearth_segmentor'
coninfer-import-fail ModuleNotFoundError No module named 'fairscale'
```

Dataset requirement from inspected config:

```text
data/WHU_Aerial/val/image
data/WHU_Aerial/val/label_cvt
```

Local dataset scan found WHU/SpaceNet-like OpenRSD data, but not the ConInfer
prepared layout:

```text
/data5/2025/ldh/OpenRSD/OpenRSD_Ckpoint_pkl/data/WHU_Mix
/data5/2025/ldh/OpenRSD/OpenRSD_Ckpoint_pkl/data/Spacenet_Merge
/data5/2025/ldh/OpenRSD/M_Tools/Data11_WHU_Mix
/data5/2025/ldh/OpenRSD/M_Tools/Data5_SpaceNet_New
/data5/2025/ldh/OpenRSD/M_Tools/Data5_SpaceNet
/data5/2025/ldh/OpenRSD/M_configs/G02_Baselines/Data11_WHU_Mix
/data5/2025/ldh/OpenRSD/M_configs/G02_Baselines/Data5_SpaceNet
```

Blockers:

- Official ConInfer `eval.py` imports `segearth_segmentor`, but no matching file
  exists in the cloned repo.
- The closest existing environment is missing `fairscale`.
- No supported dataset is available in the documented prepared layout.

## Pi-Seg Preflight

Command:

```bash
source ~/.bashrc >/dev/null 2>&1
conda activate lcs_mmseg >/dev/null 2>&1
python - <<'PY'
mods=["torch","detectron2","timm","einops","cv2","PIL"]
for m in mods:
    try:
        mod=__import__(m)
        print(m, getattr(mod, "__version__", "ok"))
    except Exception as e:
        print(m, "FAIL", type(e).__name__, str(e)[:160])
try:
    import train_net
    print("train_net-import-ok")
except Exception as e:
    print("train_net-import-fail", type(e).__name__, str(e)[:240])
PY
```

Result:

```text
torch 2.0.0+cu118
detectron2 FAIL ModuleNotFoundError No module named 'detectron2'
timm 0.9.16
einops 0.7.0
cv2 4.6.0
PIL 10.2.0
train_net-import-fail ModuleNotFoundError No module named 'detectron2'
```

Local dataset scan:

```bash
find /data5/2025/ldh -maxdepth 5 -type d \( -iname '*OVRSIS*' -o -iname '*OVSIS*' -o -iname '*OVRSIS95K*' -o -iname '*OVRSISBench*' \)
```

Result: no matching local dataset directories.

Blockers:

- Existing tested environment lacks Detectron2.
- Required OVRSISBenchV2/OVRSIS95K data is not present locally.
- Required Pi-Seg weights are not present in the documented `output/piseg_*`
  paths.

## Launch Decision

No GPU job was launched.

Reason: all source-verified segmentation baselines failed non-training preflight
before the dataset/dataloader or smoke inference stage. Launching on GPU 0 would
only produce missing-module or missing-data failures.

## Next Concrete Unblockers

1. For ConInfer, resolve the official repo mismatch around
   `segearth_segmentor.py` and install the missing Python dependency set,
   starting with `fairscale`.
2. Prepare one documented ConInfer dataset layout, preferably
   `data/WHU_Aerial/val/image` and `data/WHU_Aerial/val/label_cvt` for the
   inspected WHU smoke config.
3. For Pi-Seg, create or locate a compatible Detectron2 environment and download
   the Hugging Face OVRSISBenchV2 test data plus Pi-Seg checkpoint.
4. Re-run the same non-training checks before starting a screen session on GPU 0
   or the lowest idle GPU.
