# FAIR1M Geometry And Dataloader Gate

- Taxonomy: `/data5/2025/ldh/New/assets/hierarchies/fair1m_remote_sensing_taxonomy.json`
- Config: `/data5/2025/ldh/OpenRSD/M_configs/G02_Baselines/Data3_FAIR1M/G02_Baselines_Data3_FAIR1M_M2_RoITrans_S0_Sanitized_20260713.py`
- Config class check: `{'path': '/data5/2025/ldh/OpenRSD/M_configs/G02_Baselines/Data3_FAIR1M/G02_Baselines_Data3_FAIR1M_M2_RoITrans_S0_Sanitized_20260713.py', 'raw_class_names': None, 'normalized_class_names': None, 'same_class_set': False, 'exact_taxonomy_order': False}`

## train

```json
{
  "name": "train",
  "image_dir": "/data2/2023/lcs/xyun/FAIR1M_2_800_400_sanitized_20260713/train/images",
  "label_dir": "/data2/2023/lcs/xyun/FAIR1M_2_800_400_sanitized_20260713/train/annfiles",
  "num_images": 208927,
  "num_label_files": 208927,
  "num_label_files_checked": 208927,
  "num_objects": 1785001,
  "missing_image_stems": [],
  "missing_label_stems": [],
  "num_missing_image_stems": 0,
  "num_missing_label_stems": 0,
  "images_decoded": 300,
  "decode_errors": [],
  "class_counts": {
    "a220": 28283,
    "a321": 11876,
    "a330": 7467,
    "a350": 4821,
    "arj21": 827,
    "baseball-field": 4233,
    "basketball-court": 5531,
    "boeing737": 18454,
    "boeing747": 6864,
    "boeing777": 7578,
    "boeing787": 7186,
    "bridge": 4450,
    "bus": 4597,
    "c919": 648,
    "cargo-truck": 43304,
    "dry-cargo-ship": 45077,
    "dump-truck": 113798,
    "engineering-ship": 7532,
    "excavator": 3969,
    "fishing-boat": 30421,
    "football-field": 2970,
    "intersection": 30906,
    "liquid-cargo-ship": 13012,
    "motorboat": 35836,
    "other-airplane": 46644,
    "other-ship": 11185,
    "other-vehicle": 14170,
    "passenger-ship": 3031,
    "roundabout": 2475,
    "small-car": 640616,
    "tennis-court": 12601,
    "tractor": 1240,
    "trailer": 2673,
    "truck-tractor": 3824,
    "tugboat": 8307,
    "van": 595618,
    "warship": 2977
  },
  "missing_taxonomy_classes": [],
  "unknown_class_counts": {},
  "malformed_records": [],
  "num_malformed_records": 0,
  "qbox_area": {
    "count": 1785001,
    "min": 1.5516116036451422e-07,
    "mean": 809.3695725667902,
    "max": 186444.0
  },
  "edge_length": {
    "count": 7140004,
    "min": 0.00032739120330146037,
    "mean": 21.74529275480377,
    "max": 1291.223063610622
  },
  "qboxes_out_of_bounds_for_decoded_images": 93,
  "invalid_mmrotate_rboxes": 0,
  "mmrotate_conversion_errors": []
}
```

## ss_val

```json
{
  "name": "ss_val",
  "image_dir": "/data2/2023/lcs/xyun/FAIR1M_2_800_400_sanitized_20260713/ss_val/images",
  "label_dir": "/data2/2023/lcs/xyun/FAIR1M_2_800_400_sanitized_20260713/ss_val/annfiles_sanitized",
  "num_images": 10970,
  "num_label_files": 10970,
  "num_label_files_checked": 10970,
  "num_objects": 199347,
  "missing_image_stems": [],
  "missing_label_stems": [],
  "num_missing_image_stems": 0,
  "num_missing_label_stems": 0,
  "images_decoded": 300,
  "decode_errors": [],
  "class_counts": {
    "a220": 2299,
    "a321": 1167,
    "a330": 592,
    "a350": 372,
    "arj21": 135,
    "baseball-field": 403,
    "basketball-court": 423,
    "boeing737": 2045,
    "boeing747": 833,
    "boeing777": 332,
    "boeing787": 711,
    "bridge": 675,
    "bus": 659,
    "c919": 25,
    "cargo-truck": 7037,
    "dry-cargo-ship": 7687,
    "dump-truck": 8155,
    "engineering-ship": 2350,
    "excavator": 397,
    "fishing-boat": 3773,
    "football-field": 432,
    "intersection": 3793,
    "liquid-cargo-ship": 923,
    "motorboat": 7681,
    "other-airplane": 4635,
    "other-ship": 2060,
    "other-vehicle": 3383,
    "passenger-ship": 1301,
    "roundabout": 267,
    "small-car": 63398,
    "tennis-court": 1381,
    "tractor": 37,
    "trailer": 600,
    "truck-tractor": 427,
    "tugboat": 427,
    "van": 67988,
    "warship": 544
  },
  "missing_taxonomy_classes": [],
  "unknown_class_counts": {},
  "malformed_records": [],
  "num_malformed_records": 0,
  "qbox_area": {
    "count": 199347,
    "min": 21.0,
    "mean": 777.1868422900774,
    "max": 169757.9
  },
  "edge_length": {
    "count": 797388,
    "min": 1.0,
    "mean": 21.57940552941092,
    "max": 1003.1575150493566
  },
  "qboxes_out_of_bounds_for_decoded_images": 0,
  "invalid_mmrotate_rboxes": 0,
  "mmrotate_conversion_errors": []
}
```
