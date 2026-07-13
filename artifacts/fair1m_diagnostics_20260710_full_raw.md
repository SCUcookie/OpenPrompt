# FAIR1M Geometry And Dataloader Gate

- Taxonomy: `/data5/2025/ldh/New/assets/hierarchies/fair1m_remote_sensing_taxonomy.json`
- Config: `/data5/2025/ldh/OpenRSD/M_configs/G02_Baselines/Data3_FAIR1M/G02_Baselines_Data3_FAIR1M_M2_RoITrans.py`
- Config class check: `{'path': '/data5/2025/ldh/OpenRSD/M_configs/G02_Baselines/Data3_FAIR1M/G02_Baselines_Data3_FAIR1M_M2_RoITrans.py', 'raw_class_names': ['a220', 'a321', 'a330', 'a350', 'arj21', 'baseball_field', 'basketball_court', 'boeing737', 'boeing747', 'boeing777', 'boeing787', 'bridge', 'bus', 'c919', 'cargo_truck', 'dry_cargo_ship', 'dump_truck', 'engineering_ship', 'excavator', 'fishing_boat', 'football_field', 'intersection', 'liquid_cargo_ship', 'motorboat', 'other-airplane', 'other-ship', 'other-vehicle', 'passenger_ship', 'roundabout', 'small_car', 'tennis_court', 'tractor', 'trailer', 'truck_tractor', 'tugboat', 'van', 'warship'], 'normalized_class_names': ['a220', 'a321', 'a330', 'a350', 'arj21', 'baseball-field', 'basketball-court', 'boeing737', 'boeing747', 'boeing777', 'boeing787', 'bridge', 'bus', 'c919', 'cargo-truck', 'dry-cargo-ship', 'dump-truck', 'engineering-ship', 'excavator', 'fishing-boat', 'football-field', 'intersection', 'liquid-cargo-ship', 'motorboat', 'other-airplane', 'other-ship', 'other-vehicle', 'passenger-ship', 'roundabout', 'small-car', 'tennis-court', 'tractor', 'trailer', 'truck-tractor', 'tugboat', 'van', 'warship'], 'same_class_set': True, 'exact_taxonomy_order': False}`

## raw_train

```json
{
  "name": "raw_train",
  "image_dir": "/data2/2023/lcs/xyun/FAIR1M1.0/train/images",
  "label_dir": "/data2/2023/lcs/xyun/FAIR1M1.0/train/labelTxt",
  "num_images": 16488,
  "num_label_files": 16488,
  "num_label_files_checked": 16488,
  "num_objects": 393466,
  "missing_image_stems": [],
  "missing_label_stems": [],
  "num_missing_image_stems": 0,
  "num_missing_label_stems": 0,
  "images_decoded": 100,
  "decode_errors": [],
  "class_counts": {
    "a220": 6057,
    "a321": 2505,
    "a330": 1599,
    "a350": 1064,
    "arj21": 166,
    "baseball-field": 1062,
    "basketball-court": 1271,
    "boeing737": 3949,
    "boeing747": 1673,
    "boeing777": 1532,
    "boeing787": 1669,
    "bridge": 1008,
    "bus": 1022,
    "c919": 135,
    "cargo-truck": 9257,
    "dry-cargo-ship": 9435,
    "dump-truck": 25794,
    "engineering-ship": 1425,
    "excavator": 891,
    "fishing-boat": 5174,
    "football-field": 853,
    "intersection": 6368,
    "liquid-cargo-ship": 2898,
    "motorboat": 7706,
    "other-airplane": 9975,
    "other-ship": 2197,
    "other-vehicle": 3065,
    "passenger-ship": 575,
    "roundabout": 563,
    "small-car": 143390,
    "tennis-court": 2924,
    "tractor": 262,
    "trailer": 589,
    "truck-tractor": 923,
    "tugboat": 1453,
    "van": 132438,
    "warship": 599
  },
  "missing_taxonomy_classes": [],
  "unknown_class_counts": {},
  "malformed_records": [
    {
      "path": "/data2/2023/lcs/xyun/FAIR1M1.0/train/labelTxt/1.txt",
      "line": 19,
      "class_name": "cargo-truck",
      "area": 0.0,
      "edge_lengths": [
        0.0,
        0.0,
        0.0,
        0.0
      ],
      "known_class": true
    },
    {
      "path": "/data2/2023/lcs/xyun/FAIR1M1.0/train/labelTxt/1005.txt",
      "line": 57,
      "class_name": "small-car",
      "area": 0.0,
      "edge_lengths": [
        0.0,
        0.0,
        0.0,
        0.0
      ],
      "known_class": true
    },
    {
      "path": "/data2/2023/lcs/xyun/FAIR1M1.0/train/labelTxt/105.txt",
      "line": 3,
      "class_name": "small-car",
      "area": 0.0,
      "edge_lengths": [
        0.0,
        0.0,
        0.0,
        0.0
      ],
      "known_class": true
    },
    {
      "path": "/data2/2023/lcs/xyun/FAIR1M1.0/train/labelTxt/1052.txt",
      "line": 70,
      "class_name": "van",
      "area": 0.0,
      "edge_lengths": [
        0.0,
        2.0,
        0.0,
        2.0
      ],
      "known_class": true
    },
    {
      "path": "/data2/2023/lcs/xyun/FAIR1M1.0/train/labelTxt/1082.txt",
      "line": 21,
      "class_name": "van",
      "area": 0.0,
      "edge_lengths": [
        0.0,
        11.0,
        0.0,
        11.0
      ],
      "known_class": true
    },
    {
      "path": "/data2/2023/lcs/xyun/FAIR1M1.0/train/labelTxt/1104.txt",
      "line": 225,
      "class_name": "small-car",
      "area": 0.0,
      "edge_lengths": [
        0.0,
        0.0,
        0.0,
        0.0
      ],
      "known_class": true
    },
    {
      "path": "/data2/2023/lcs/xyun/FAIR1M1.0/train/labelTxt/1126.txt",
      "line": 102,
      "class_name": "van",
      "area": 0.0,
      "edge_lengths": [
        0.0,
        17.0,
        0.0,
        17.0
      ],
      "known_class": true
    },
    {
      "path": "/data2/2023/lcs/xyun/FAIR1M1.0/train/labelTxt/1126.txt",
      "line": 110,
      "class_name": "van",
      "area": 0.0,
      "edge_lengths": [
        0.0,
        16.0,
        0.0,
        16.0
      ],
      "known_class": true
    },
    {
      "path": "/data2/2023/lcs/xyun/FAIR1M1.0/train/labelTxt/1135.txt",
      "line": 93,
      "class_name": "small-car",
      "area": 0.0,
      "edge_lengths": [
        0.0,
        0.0,
        0.0,
        0.0
      ],
      "known_class": true
    },
    {
      "path": "/data2/2023/lcs/xyun/FAIR1M1.0/train/labelTxt/1148.txt",
      "line": 16,
      "class_name": "van",
      "area": 0.0,
      "edge_lengths": [
        0.0,
        0.0,
        0.0,
        0.0
      ],
      "known_class": true
    },
    {
      "path": "/data2/2023/lcs/xyun/FAIR1M1.0/train/labelTxt/115.txt",
      "line": 24,
      "class_name": "small-car",
      "area": 0.0,
      "edge_lengths": [
        0.0,
        0.0,
        0.0,
        0.0
      ],
      "known_class": true
    },
    {
      "path": "/data2/2023/lcs/xyun/FAIR1M1.0/train/labelTxt/1150.txt",
      "line": 5,
      "class_name": "intersection",
      "area": 0.0,
      "edge_lengths": [
        0.0,
        17.0,
        0.0,
        17.0
      ],
      "known_class": true
    },
    {
      "path": "/data2/2023/lcs/xyun/FAIR1M1.0/train/labelTxt/1150.txt",
      "line": 7,
      "class_name": "van",
      "area": 0.0,
      "edge_lengths": [
        0.0,
        6.0,
        0.0,
        6.0
      ],
      "known_class": true
    },
    {
      "path": "/data2/2023/lcs/xyun/FAIR1M1.0/train/labelTxt/1161.txt",
      "line": 142,
      "class_name": "dump-truck",
      "area": 0.0,
      "edge_lengths": [
        0.0,
        7.0,
        0.0,
        7.0
      ],
      "known_class": true
    },
    {
      "path": "/data2/2023/lcs/xyun/FAIR1M1.0/train/labelTxt/1169.txt",
      "line": 35,
      "class_name": "cargo-truck",
      "area": 0.0,
      "edge_lengths": [
        0.0,
        0.0,
        0.0,
        0.0
      ],
      "known_class": true
    },
    {
      "path": "/data2/2023/lcs/xyun/FAIR1M1.0/train/labelTxt/1170.txt",
      "line": 17,
      "class_name": "small-car",
      "area": 0.0,
      "edge_lengths": [
        0.0,
        0.0,
        0.0,
        0.0
      ],
      "known_class": true
    },
    {
      "path": "/data2/2023/lcs/xyun/FAIR1M1.0/train/labelTxt/1179.txt",
      "line": 133,
      "class_name": "van",
      "area": 0.0,
      "edge_lengths": [
        0.0,
        0.0,
        0.0,
        0.0
      ],
      "known_class": true
    },
    {
      "path": "/data2/2023/lcs/xyun/FAIR1M1.0/train/labelTxt/1182.txt",
      "line": 205,
      "class_name": "small-car",
      "area": 0.0,
      "edge_lengths": [
        0.0,
        0.0,
        0.0,
        0.0
      ],
      "known_class": true
    },
    {
      "path": "/data2/2023/lcs/xyun/FAIR1M1.0/train/labelTxt/1191.txt",
      "line": 84,
      "class_name": "small-car",
      "area": 0.0,
      "edge_lengths": [
        0.0,
        1.0,
        0.0,
        1.0
      ],
      "known_class": true
    },
    {
      "path": "/data2/2023/lcs/xyun/FAIR1M1.0/train/labelTxt/1204.txt",
      "line": 24,
      "class_name": "small-car",
      "area": 0.0,
      "edge_lengths": [
        0.0,
        0.0,
        0.0,
        0.0
      ],
      "known_class": true
    },
    {
      "path": "/data2/2023/lcs/xyun/FAIR1M1.0/train/labelTxt/1211.txt",
      "line": 13,
      "class_name": "van",
      "area": 0.0,
      "edge_lengths": [
        0.0,
        2.0,
        0.0,
        2.0
      ],
      "known_class": true
    },
    {
      "path": "/data2/2023/lcs/xyun/FAIR1M1.0/train/labelTxt/1216.txt",
      "line": 66,
      "class_name": "van",
      "area": 0.0,
      "edge_lengths": [
        0.0,
        0.0,
        0.0,
        0.0
      ],
      "known_class": true
    },
    {
      "path": "/data2/2023/lcs/xyun/FAIR1M1.0/train/labelTxt/1222.txt",
      "line": 87,
      "class_name": "van",
      "area": 0.0,
      "edge_lengths": [
        0.0,
        0.0,
        0.0,
        0.0
      ],
      "known_class": true
    },
    {
      "path": "/data2/2023/lcs/xyun/FAIR1M1.0/train/labelTxt/1229.txt",
      "line": 100,
      "class_name": "van",
      "area": 0.0,
      "edge_lengths": [
        0.0,
        2.0,
        0.0,
        2.0
      ],
      "known_class": true
    },
    {
      "path": "/data2/2023/lcs/xyun/FAIR1M1.0/train/labelTxt/1243.txt",
      "line": 37,
      "class_name": "van",
      "area": 0.0,
      "edge_lengths": [
        0.0,
        0.0,
        0.0,
        0.0
      ],
      "known_class": true
    },
    {
      "path": "/data2/2023/lcs/xyun/FAIR1M1.0/train/labelTxt/1246.txt",
      "line": 2,
      "class_name": "van",
      "area": 0.0,
      "edge_lengths": [
        0.0,
        0.0,
        0.0,
        0.0
      ],
      "known_class": true
    },
    {
      "path": "/data2/2023/lcs/xyun/FAIR1M1.0/train/labelTxt/127.txt",
      "line": 25,
      "class_name": "van",
      "area": 0.0,
      "edge_lengths": [
        0.0,
        14.0,
        0.0,
        14.0
      ],
      "known_class": true
    },
    {
      "path": "/data2/2023/lcs/xyun/FAIR1M1.0/train/labelTxt/1281.txt",
      "line": 9,
      "class_name": "small-car",
      "area": 0.0,
      "edge_lengths": [
        0.0,
        0.0,
        0.0,
        0.0
      ],
      "known_class": true
    },
    {
      "path": "/data2/2023/lcs/xyun/FAIR1M1.0/train/labelTxt/1297.txt",
      "line": 5,
      "class_name": "van",
      "area": 0.0,
      "edge_lengths": [
        0.0,
        6.0,
        0.0,
        6.0
      ],
      "known_class": true
    },
    {
      "path": "/data2/2023/lcs/xyun/FAIR1M1.0/train/labelTxt/1297.txt",
      "line": 6,
      "class_name": "van",
      "area": 0.0,
      "edge_lengths": [
        0.0,
        6.0,
        0.0,
        6.0
      ],
      "known_class": true
    },
    {
      "path": "/data2/2023/lcs/xyun/FAIR1M1.0/train/labelTxt/1300.txt",
      "line": 137,
      "class_name": "small-car",
      "area": 0.0,
      "edge_lengths": [
        0.0,
        0.0,
        0.0,
        0.0
      ],
      "known_class": true
    },
    {
      "path": "/data2/2023/lcs/xyun/FAIR1M1.0/train/labelTxt/1300.txt",
      "line": 138,
      "class_name": "van",
      "area": 0.0,
      "edge_lengths": [
        0.0,
        0.0,
        0.0,
        0.0
      ],
      "known_class": true
    },
    {
      "path": "/data2/2023/lcs/xyun/FAIR1M1.0/train/labelTxt/132.txt",
      "line": 1,
      "class_name": "van",
      "area": 0.0,
      "edge_lengths": [
        0.0,
        9.0,
        0.0,
        9.0
      ],
      "known_class": true
    },
    {
      "path": "/data2/2023/lcs/xyun/FAIR1M1.0/train/labelTxt/1324.txt",
      "line": 1,
      "class_name": "small-car",
      "area": 0.0,
      "edge_lengths": [
        0.0,
        0.0,
        0.0,
        0.0
      ],
      "known_class": true
    },
    {
      "path": "/data2/2023/lcs/xyun/FAIR1M1.0/train/labelTxt/1327.txt",
      "line": 120,
      "class_name": "small-car",
      "area": 0.0,
      "edge_lengths": [
        0.0,
        0.0,
        0.0,
        0.0
      ],
      "known_class": true
    },
    {
      "path": "/data2/2023/lcs/xyun/FAIR1M1.0/train/labelTxt/1327.txt",
      "line": 123,
      "class_name": "van",
      "area": 0.0,
      "edge_lengths": [
        0.0,
        2.0,
        0.0,
        2.0
      ],
      "known_class": true
    },
    {
      "path": "/data2/2023/lcs/xyun/FAIR1M1.0/train/labelTxt/1332.txt",
      "line": 75,
      "class_name": "dump-truck",
      "area": 0.0,
      "edge_lengths": [
        0.0,
        11.0,
        0.0,
        11.0
      ],
      "known_class": true
    },
    {
      "path": "/data2/2023/lcs/xyun/FAIR1M1.0/train/labelTxt/13444.txt",
      "line": 24,
      "class_name": "fishing-boat",
      "area": 0.5,
      "edge_lengths": [
        0.0,
        1.0,
        1.4142135623730951,
        1.0
      ],
      "known_class": true
    },
    {
      "path": "/data2/2023/lcs/xyun/FAIR1M1.0/train/labelTxt/1377.txt",
      "line": 79,
      "class_name": "cargo-truck",
      "area": 0.0,
      "edge_lengths": [
        0.0,
        2.0,
        0.0,
        2.0
      ],
      "known_class": true
    },
    {
      "path": "/data2/2023/lcs/xyun/FAIR1M1.0/train/labelTxt/1377.txt",
      "line": 100,
      "class_name": "small-car",
      "area": 0.0,
      "edge_lengths": [
        0.0,
        0.0,
        0.0,
        0.0
      ],
      "known_class": true
    },
    {
      "path": "/data2/2023/lcs/xyun/FAIR1M1.0/train/labelTxt/1377.txt",
      "line": 101,
      "class_name": "van",
      "area": 0.0,
      "edge_lengths": [
        0.0,
        0.0,
        0.0,
        0.0
      ],
      "known_class": true
    },
    {
      "path": "/data2/2023/lcs/xyun/FAIR1M1.0/train/labelTxt/1388.txt",
      "line": 47,
      "class_name": "small-car",
      "area": 0.0,
      "edge_lengths": [
        0.0,
        0.0,
        0.0,
        0.0
      ],
      "known_class": true
    },
    {
      "path": "/data2/2023/lcs/xyun/FAIR1M1.0/train/labelTxt/1410.txt",
      "line": 4,
      "class_name": "van",
      "area": 0.0,
      "edge_lengths": [
        0.0,
        0.0,
        0.0,
        0.0
      ],
      "known_class": true
    },
    {
      "path": "/data2/2023/lcs/xyun/FAIR1M1.0/train/labelTxt/1432.txt",
      "line": 1,
      "class_name": "dump-truck",
      "area": 0.0,
      "edge_lengths": [
        0.0,
        0.0,
        0.0,
        0.0
      ],
      "known_class": true
    },
    {
      "path": "/data2/2023/lcs/xyun/FAIR1M1.0/train/labelTxt/1436.txt",
      "line": 296,
      "class_name": "small-car",
      "area": 0.0,
      "edge_lengths": [
        0.0,
        0.0,
        0.0,
        0.0
      ],
      "known_class": true
    },
    {
      "path": "/data2/2023/lcs/xyun/FAIR1M1.0/train/labelTxt/1436.txt",
      "line": 317,
      "class_name": "van",
      "area": 0.0,
      "edge_lengths": [
        0.0,
        0.0,
        0.0,
        0.0
      ],
      "known_class": true
    },
    {
      "path": "/data2/2023/lcs/xyun/FAIR1M1.0/train/labelTxt/144.txt",
      "line": 44,
      "class_name": "van",
      "area": 0.0,
      "edge_lengths": [
        0.0,
        0.0,
        0.0,
        0.0
      ],
      "known_class": true
    },
    {
      "path": "/data2/2023/lcs/xyun/FAIR1M1.0/train/labelTxt/1450.txt",
      "line": 50,
      "class_name": "van",
      "area": 0.0,
      "edge_lengths": [
        0.0,
        0.0,
        0.0,
        0.0
      ],
      "known_class": true
    },
    {
      "path": "/data2/2023/lcs/xyun/FAIR1M1.0/train/labelTxt/1454.txt",
      "line": 92,
      "class_name": "small-car",
      "area": 0.0,
      "edge_lengths": [
        0.0,
        0.0,
        0.0,
        0.0
      ],
      "known_class": true
    },
    {
      "path": "/data2/2023/lcs/xyun/FAIR1M1.0/train/labelTxt/1463.txt",
      "line": 36,
      "class_name": "small-car",
      "area": 0.0,
      "edge_lengths": [
        0.0,
        0.0,
        0.0,
        0.0
      ],
      "known_class": true
    },
    {
      "path": "/data2/2023/lcs/xyun/FAIR1M1.0/train/labelTxt/147.txt",
      "line": 119,
      "class_name": "small-car",
      "area": 0.0,
      "edge_lengths": [
        0.0,
        7.0,
        0.0,
        7.0
      ],
      "known_class": true
    },
    {
      "path": "/data2/2023/lcs/xyun/FAIR1M1.0/train/labelTxt/1490.txt",
      "line": 48,
      "class_name": "small-car",
      "area": 0.0,
      "edge_lengths": [
        0.0,
        0.0,
        0.0,
        0.0
      ],
      "known_class": true
    },
    {
      "path": "/data2/2023/lcs/xyun/FAIR1M1.0/train/labelTxt/1491.txt",
      "line": 12,
      "class_name": "bus",
      "area": 0.0,
      "edge_lengths": [
        0.0,
        0.0,
        0.0,
        0.0
      ],
      "known_class": true
    },
    {
      "path": "/data2/2023/lcs/xyun/FAIR1M1.0/train/labelTxt/1506.txt",
      "line": 1,
      "class_name": "other-vehicle",
      "area": 0.0,
      "edge_lengths": [
        0.0,
        0.0,
        0.0,
        0.0
      ],
      "known_class": true
    },
    {
      "path": "/data2/2023/lcs/xyun/FAIR1M1.0/train/labelTxt/153.txt",
      "line": 54,
      "class_name": "van",
      "area": 0.0,
      "edge_lengths": [
        0.0,
        9.0,
        0.0,
        9.0
      ],
      "known_class": true
    },
    {
      "path": "/data2/2023/lcs/xyun/FAIR1M1.0/train/labelTxt/1530.txt",
      "line": 77,
      "class_name": "van",
      "area": 0.0,
      "edge_lengths": [
        0.0,
        0.0,
        0.0,
        0.0
      ],
      "known_class": true
    },
    {
      "path": "/data2/2023/lcs/xyun/FAIR1M1.0/train/labelTxt/1533.txt",
      "line": 68,
      "class_name": "small-car",
      "area": 0.0,
      "edge_lengths": [
        0.0,
        0.0,
        0.0,
        0.0
      ],
      "known_class": true
    },
    {
      "path": "/data2/2023/lcs/xyun/FAIR1M1.0/train/labelTxt/154.txt",
      "line": 1,
      "class_name": "small-car",
      "area": 0.0,
      "edge_lengths": [
        0.0,
        7.0,
        0.0,
        7.0
      ],
      "known_class": true
    },
    {
      "path": "/data2/2023/lcs/xyun/FAIR1M1.0/train/labelTxt/1571.txt",
      "line": 77,
      "class_name": "small-car",
      "area": 0.0,
      "edge_lengths": [
        0.0,
        0.0,
        0.0,
        0.0
      ],
      "known_class": true
    },
    {
      "path": "/data2/2023/lcs/xyun/FAIR1M1.0/train/labelTxt/1588.txt",
      "line": 1,
      "class_name": "small-car",
      "area": 0.0,
      "edge_lengths": [
        0.0,
        21.0,
        0.0,
        21.0
      ],
      "known_class": true
    },
    {
      "path": "/data2/2023/lcs/xyun/FAIR1M1.0/train/labelTxt/1588.txt",
      "line": 52,
      "class_name": "small-car",
      "area": 0.0,
      "edge_lengths": [
        0.0,
        0.0,
        0.0,
        0.0
      ],
      "known_class": true
    },
    {
      "path": "/data2/2023/lcs/xyun/FAIR1M1.0/train/labelTxt/161.txt",
      "line": 201,
      "class_name": "small-car",
      "area": 0.0,
      "edge_lengths": [
        0.0,
        0.0,
        0.0,
        0.0
      ],
      "known_class": true
    },
    {
      "path": "/data2/2023/lcs/xyun/FAIR1M1.0/train/labelTxt/1616.txt",
      "line": 55,
      "class_name": "van",
      "area": 0.0,
      "edge_lengths": [
        0.0,
        0.0,
        0.0,
        0.0
      ],
      "known_class": true
    },
    {
      "path": "/data2/2023/lcs/xyun/FAIR1M1.0/train/labelTxt/1622.txt",
      "line": 55,
      "class_name": "small-car",
      "area": 0.0,
      "edge_lengths": [
        0.0,
        0.0,
        0.0,
        0.0
      ],
      "known_class": true
    },
    {
      "path": "/data2/2023/lcs/xyun/FAIR1M1.0/train/labelTxt/1643.txt",
      "line": 1,
      "class_name": "van",
      "area": 0.0,
      "edge_lengths": [
        0.0,
        10.0,
        0.0,
        10.0
      ],
      "known_class": true
    },
    {
      "path": "/data2/2023/lcs/xyun/FAIR1M1.0/train/labelTxt/1645.txt",
      "line": 20,
      "class_name": "van",
      "area": 0.0,
      "edge_lengths": [
        0.0,
        0.0,
        0.0,
        0.0
      ],
      "known_class": true
    },
    {
      "path": "/data2/2023/lcs/xyun/FAIR1M1.0/train/labelTxt/1646.txt",
      "line": 94,
      "class_name": "van",
      "area": 0.0,
      "edge_lengths": [
        0.0,
        0.0,
        0.0,
        0.0
      ],
      "known_class": true
    },
    {
      "path": "/data2/2023/lcs/xyun/FAIR1M1.0/train/labelTxt/1655.txt",
      "line": 53,
      "class_name": "van",
      "area": 0.0,
      "edge_lengths": [
        0.0,
        0.0,
        0.0,
        0.0
      ],
      "known_class": true
    },
    {
      "path": "/data2/2023/lcs/xyun/FAIR1M1.0/train/labelTxt/1656.txt",
      "line": 8,
      "class_name": "van",
      "area": 0.0,
      "edge_lengths": [
        0.0,
        0.0,
        0.0,
        0.0
      ],
      "known_class": true
    },
    {
      "path": "/data2/2023/lcs/xyun/FAIR1M1.0/train/labelTxt/1686.txt",
      "line": 57,
      "class_name": "small-car",
      "area": 0.0,
      "edge_lengths": [
        2.0,
        0.0,
        2.0,
        0.0
      ],
      "known_class": true
    },
    {
      "path": "/data2/2023/lcs/xyun/FAIR1M1.0/train/labelTxt/1695.txt",
      "line": 86,
      "class_name": "small-car",
      "area": 0.0,
      "edge_lengths": [
        0.0,
        0.0,
        0.0,
        0.0
      ],
      "known_class": true
    },
    {
      "path": "/data2/2023/lcs/xyun/FAIR1M1.0/train/labelTxt/1702.txt",
      "line": 91,
      "class_name": "van",
      "area": 0.0,
      "edge_lengths": [
        0.0,
        0.0,
        0.0,
        0.0
      ],
      "known_class": true
    },
    {
      "path": "/data2/2023/lcs/xyun/FAIR1M1.0/train/labelTxt/171.txt",
      "line": 78,
      "class_name": "other-vehicle",
      "area": 0.0,
      "edge_lengths": [
        0.0,
        5.0,
        0.0,
        5.0
      ],
      "known_class": true
    },
    {
      "path": "/data2/2023/lcs/xyun/FAIR1M1.0/train/labelTxt/1720.txt",
      "line": 60,
      "class_name": "small-car",
      "area": 0.0,
      "edge_lengths": [
        0.0,
        3.0,
        0.0,
        3.0
      ],
      "known_class": true
    },
    {
      "path": "/data2/2023/lcs/xyun/FAIR1M1.0/train/labelTxt/1725.txt",
      "line": 1,
      "class_name": "small-car",
      "area": 0.0,
      "edge_lengths": [
        0.0,
        0.0,
        0.0,
        0.0
      ],
      "known_class": true
    },
    {
      "path": "/data2/2023/lcs/xyun/FAIR1M1.0/train/labelTxt/1727.txt",
      "line": 25,
      "class_name": "van",
      "area": 0.0,
      "edge_lengths": [
        0.0,
        0.0,
        0.0,
        0.0
      ],
      "known_class": true
    },
    {
      "path": "/data2/2023/lcs/xyun/FAIR1M1.0/train/labelTxt/182.txt",
      "line": 14,
      "class_name": "van",
      "area": 0.0,
      "edge_lengths": [
        0.0,
        7.0,
        0.0,
        7.0
      ],
      "known_class": true
    },
    {
      "path": "/data2/2023/lcs/xyun/FAIR1M1.0/train/labelTxt/182.txt",
      "line": 17,
      "class_name": "small-car",
      "area": 0.0,
      "edge_lengths": [
        0.0,
        6.0,
        0.0,
        6.0
      ],
      "known_class": true
    },
    {
      "path": "/data2/2023/lcs/xyun/FAIR1M1.0/train/labelTxt/186.txt",
      "line": 11,
      "class_name": "van",
      "area": 0.0,
      "edge_lengths": [
        0.0,
        0.0,
        0.0,
        0.0
      ],
      "known_class": true
    },
    {
      "path": "/data2/2023/lcs/xyun/FAIR1M1.0/train/labelTxt/188.txt",
      "line": 27,
      "class_name": "van",
      "area": 0.0,
      "edge_lengths": [
        0.0,
        18.0,
        0.0,
        18.0
      ],
      "known_class": true
    },
    {
      "path": "/data2/2023/lcs/xyun/FAIR1M1.0/train/labelTxt/188.txt",
      "line": 32,
      "class_name": "small-car",
      "area": 0.0,
      "edge_lengths": [
        0.0,
        8.0,
        0.0,
        8.0
      ],
      "known_class": true
    },
    {
      "path": "/data2/2023/lcs/xyun/FAIR1M1.0/train/labelTxt/197.txt",
      "line": 18,
      "class_name": "van",
      "area": 0.0,
      "edge_lengths": [
        0.0,
        0.0,
        0.0,
        0.0
      ],
      "known_class": true
    },
    {
      "path": "/data2/2023/lcs/xyun/FAIR1M1.0/train/labelTxt/199.txt",
      "line": 9,
      "class_name": "small-car",
      "area": 0.0,
      "edge_lengths": [
        0.0,
        15.0,
        0.0,
        15.0
      ],
      "known_class": true
    },
    {
      "path": "/data2/2023/lcs/xyun/FAIR1M1.0/train/labelTxt/199.txt",
      "line": 194,
      "class_name": "small-car",
      "area": 0.0,
      "edge_lengths": [
        0.0,
        8.0,
        0.0,
        8.0
      ],
      "known_class": true
    },
    {
      "path": "/data2/2023/lcs/xyun/FAIR1M1.0/train/labelTxt/217.txt",
      "line": 5,
      "class_name": "van",
      "area": 0.0,
      "edge_lengths": [
        0.0,
        16.0,
        0.0,
        16.0
      ],
      "known_class": true
    },
    {
      "path": "/data2/2023/lcs/xyun/FAIR1M1.0/train/labelTxt/217.txt",
      "line": 6,
      "class_name": "small-car",
      "area": 0.0,
      "edge_lengths": [
        0.0,
        17.0,
        0.0,
        17.0
      ],
      "known_class": true
    },
    {
      "path": "/data2/2023/lcs/xyun/FAIR1M1.0/train/labelTxt/217.txt",
      "line": 27,
      "class_name": "small-car",
      "area": 0.0,
      "edge_lengths": [
        0.0,
        17.0,
        0.0,
        17.0
      ],
      "known_class": true
    },
    {
      "path": "/data2/2023/lcs/xyun/FAIR1M1.0/train/labelTxt/217.txt",
      "line": 48,
      "class_name": "small-car",
      "area": 0.0,
      "edge_lengths": [
        0.0,
        18.0,
        0.0,
        18.0
      ],
      "known_class": true
    },
    {
      "path": "/data2/2023/lcs/xyun/FAIR1M1.0/train/labelTxt/226.txt",
      "line": 43,
      "class_name": "small-car",
      "area": 0.0,
      "edge_lengths": [
        0.0,
        0.0,
        0.0,
        0.0
      ],
      "known_class": true
    },
    {
      "path": "/data2/2023/lcs/xyun/FAIR1M1.0/train/labelTxt/244.txt",
      "line": 52,
      "class_name": "small-car",
      "area": 0.0,
      "edge_lengths": [
        0.0,
        0.0,
        0.0,
        0.0
      ],
      "known_class": true
    },
    {
      "path": "/data2/2023/lcs/xyun/FAIR1M1.0/train/labelTxt/249.txt",
      "line": 3,
      "class_name": "small-car",
      "area": 0.0,
      "edge_lengths": [
        0.0,
        0.0,
        0.0,
        0.0
      ],
      "known_class": true
    },
    {
      "path": "/data2/2023/lcs/xyun/FAIR1M1.0/train/labelTxt/262.txt",
      "line": 129,
      "class_name": "small-car",
      "area": 0.0,
      "edge_lengths": [
        0.0,
        20.0,
        0.0,
        20.0
      ],
      "known_class": true
    },
    {
      "path": "/data2/2023/lcs/xyun/FAIR1M1.0/train/labelTxt/264.txt",
      "line": 30,
      "class_name": "small-car",
      "area": 0.0,
      "edge_lengths": [
        1.0,
        1.0,
        1.0,
        1.0
      ],
      "known_class": true
    },
    {
      "path": "/data2/2023/lcs/xyun/FAIR1M1.0/train/labelTxt/27.txt",
      "line": 2,
      "class_name": "bus",
      "area": 3.5,
      "edge_lengths": [
        7.0,
        1.0,
        7.0710678118654755,
        0.0
      ],
      "known_class": true
    },
    {
      "path": "/data2/2023/lcs/xyun/FAIR1M1.0/train/labelTxt/285.txt",
      "line": 1,
      "class_name": "van",
      "area": 0.0,
      "edge_lengths": [
        0.0,
        16.0,
        0.0,
        16.0
      ],
      "known_class": true
    },
    {
      "path": "/data2/2023/lcs/xyun/FAIR1M1.0/train/labelTxt/290.txt",
      "line": 65,
      "class_name": "van",
      "area": 0.0,
      "edge_lengths": [
        0.0,
        4.0,
        0.0,
        4.0
      ],
      "known_class": true
    },
    {
      "path": "/data2/2023/lcs/xyun/FAIR1M1.0/train/labelTxt/296.txt",
      "line": 97,
      "class_name": "van",
      "area": 0.0,
      "edge_lengths": [
        0.0,
        9.0,
        0.0,
        9.0
      ],
      "known_class": true
    },
    {
      "path": "/data2/2023/lcs/xyun/FAIR1M1.0/train/labelTxt/297.txt",
      "line": 34,
      "class_name": "small-car",
      "area": 0.0,
      "edge_lengths": [
        0.0,
        0.0,
        0.0,
        0.0
      ],
      "known_class": true
    },
    {
      "path": "/data2/2023/lcs/xyun/FAIR1M1.0/train/labelTxt/298.txt",
      "line": 31,
      "class_name": "van",
      "area": 0.0,
      "edge_lengths": [
        0.0,
        11.0,
        0.0,
        11.0
      ],
      "known_class": true
    },
    {
      "path": "/data2/2023/lcs/xyun/FAIR1M1.0/train/labelTxt/302.txt",
      "line": 38,
      "class_name": "small-car",
      "area": 0.0,
      "edge_lengths": [
        0.0,
        21.0,
        0.0,
        21.0
      ],
      "known_class": true
    }
  ],
  "num_malformed_records": 173,
  "qbox_area": {
    "count": 393466,
    "min": 0.0,
    "mean": 844.9152380681483,
    "max": 186444.0
  },
  "edge_length": {
    "count": 1573864,
    "min": 0.0,
    "mean": 21.673839841649013,
    "max": 1291.223063610622
  },
  "qboxes_out_of_bounds_for_decoded_images": 46,
  "invalid_mmrotate_rboxes": 0,
  "mmrotate_conversion_errors": []
}
```
