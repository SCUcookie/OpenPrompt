# DIOR-R Geometry And Target Diagnostics

- Data root: `/data5/2025/ldh/OpenRSD/data/DIOR_R_dota`
- Config class-order checks: `[{'config': PosixPath('/data5/2025/ldh/OpenRSD/M_configs/G02_Baselines/Data2_DIOR_R/G02_Baselines_Data2_DIOR_R_M2_RoITrans.py'), 'exists': True, 'class_names': ['airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge', 'chimney', 'dam', 'Expressway-Service-area', 'Expressway-toll-station', 'golffield', 'groundtrackfield', 'harbor', 'overpass', 'ship', 'stadium', 'storagetank', 'tenniscourt', 'trainstation', 'vehicle', 'windmill'], 'matches_dior_r_order': True}, {'config': PosixPath('/data5/2025/ldh/OpenRSD/M_configs/G02_Baselines/Data2_DIOR_R/G02_Baselines_Data2_DIOR_R_M5_ORCNN_R50.py'), 'exists': True, 'class_names': ['airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge', 'chimney', 'dam', 'Expressway-Service-area', 'Expressway-toll-station', 'golffield', 'groundtrackfield', 'harbor', 'overpass', 'ship', 'stadium', 'storagetank', 'tenniscourt', 'trainstation', 'vehicle', 'windmill'], 'matches_dior_r_order': True}]`

## Split Summary

### train_val

- Images checked: `200`
- Label files checked: `200`
- Objects: `1019`
- Bad image decodes: `0`
- Bad label files: `0`
- Unknown classes: `{}`
- First bad conversion: `None`

### test

- Images checked: `200`
- Label files checked: `200`
- Objects: `1782`
- Bad image decodes: `0`
- Bad label files: `0`
- Unknown classes: `{}`
- First bad conversion: `None`

## Dataloader Checks

[
  {
    "status": "ok",
    "config": "/data5/2025/ldh/OpenRSD/M_configs/G02_Baselines/Data2_DIOR_R/G02_Baselines_Data2_DIOR_R_M2_RoITrans.py",
    "samples": [
      {
        "index": 0,
        "img_path": "data/DIOR_R_dota/train_val/images/06865.png",
        "bbox_shape": [
          1,
          5
        ],
        "bbox_finite": true,
        "num_labels": 1
      },
      {
        "index": 1,
        "img_path": "data/DIOR_R_dota/train_val/images/07632.png",
        "bbox_shape": [
          5,
          5
        ],
        "bbox_finite": true,
        "num_labels": 5
      }
    ]
  },
  {
    "status": "ok",
    "config": "/data5/2025/ldh/OpenRSD/M_configs/G02_Baselines/Data2_DIOR_R/G02_Baselines_Data2_DIOR_R_M5_ORCNN_R50.py",
    "samples": [
      {
        "index": 0,
        "img_path": "data/DIOR_R_dota/train_val/images/06865.png",
        "bbox_shape": [
          1,
          5
        ],
        "bbox_finite": true,
        "num_labels": 1
      },
      {
        "index": 1,
        "img_path": "data/DIOR_R_dota/train_val/images/07632.png",
        "bbox_shape": [
          5,
          5
        ],
        "bbox_finite": true,
        "num_labels": 5
      }
    ]
  }
]
