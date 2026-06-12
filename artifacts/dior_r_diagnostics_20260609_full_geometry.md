# DIOR-R Geometry And Target Diagnostics

- Data root: `/data5/2025/ldh/OpenRSD/data/DIOR_R_dota`
- Config class-order checks: `[{'config': PosixPath('/data5/2025/ldh/OpenRSD/M_configs/G02_Baselines/Data2_DIOR_R/G02_Baselines_Data2_DIOR_R_M2_RoITrans.py'), 'exists': True, 'class_names': ['airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge', 'chimney', 'dam', 'Expressway-Service-area', 'Expressway-toll-station', 'golffield', 'groundtrackfield', 'harbor', 'overpass', 'ship', 'stadium', 'storagetank', 'tenniscourt', 'trainstation', 'vehicle', 'windmill'], 'matches_dior_r_order': True}, {'config': PosixPath('/data5/2025/ldh/OpenRSD/M_configs/G02_Baselines/Data2_DIOR_R/G02_Baselines_Data2_DIOR_R_M5_ORCNN_R50.py'), 'exists': True, 'class_names': ['airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge', 'chimney', 'dam', 'Expressway-Service-area', 'Expressway-toll-station', 'golffield', 'groundtrackfield', 'harbor', 'overpass', 'ship', 'stadium', 'storagetank', 'tenniscourt', 'trainstation', 'vehicle', 'windmill'], 'matches_dior_r_order': True}]`

## Split Summary

### train_val

- Conversion backend: `fallback`
- Image check mode: `dimensions`
- Assumed image size: `(800, 800)`
- Images checked: `0`
- Images assumed: `11725`
- Label files checked: `11725`
- Objects: `68072`
- Bad image decodes: `0`
- Bad label files: `2`
- Unknown classes: `{}`
- First bad conversion: `None`
- Missing image info for labels: `0`
- Bounds checks: `{'qbox_out_of_bounds': 1210, 'qbox_out_of_bounds_examples': [{'path': '/data5/2025/ldh/OpenRSD/data/DIOR_R_dota/train_val/labelTxt/00008.txt', 'class_name': 'vehicle'}, {'path': '/data5/2025/ldh/OpenRSD/data/DIOR_R_dota/train_val/labelTxt/00029.txt', 'class_name': 'airplane'}, {'path': '/data5/2025/ldh/OpenRSD/data/DIOR_R_dota/train_val/labelTxt/00029.txt', 'class_name': 'airplane'}, {'path': '/data5/2025/ldh/OpenRSD/data/DIOR_R_dota/train_val/labelTxt/00044.txt', 'class_name': 'vehicle'}, {'path': '/data5/2025/ldh/OpenRSD/data/DIOR_R_dota/train_val/labelTxt/00087.txt', 'class_name': 'groundtrackfield'}, {'path': '/data5/2025/ldh/OpenRSD/data/DIOR_R_dota/train_val/labelTxt/00090.txt', 'class_name': 'harbor'}, {'path': '/data5/2025/ldh/OpenRSD/data/DIOR_R_dota/train_val/labelTxt/00090.txt', 'class_name': 'harbor'}, {'path': '/data5/2025/ldh/OpenRSD/data/DIOR_R_dota/train_val/labelTxt/00107.txt', 'class_name': 'groundtrackfield'}, {'path': '/data5/2025/ldh/OpenRSD/data/DIOR_R_dota/train_val/labelTxt/00107.txt', 'class_name': 'stadium'}, {'path': '/data5/2025/ldh/OpenRSD/data/DIOR_R_dota/train_val/labelTxt/00126.txt', 'class_name': 'Expressway-Service-area'}], 'rbox_center_out_of_bounds': 0, 'rbox_center_out_of_bounds_examples': [], 'invalid_rbox_size': 2}`

RBox statistics:

{
  "qbox_area": {
    "count": 68072,
    "min": 0.0,
    "p01": 42.0,
    "p05": 100.49880000000121,
    "p25": 363.1693999999952,
    "p50": 1110.8631999999961,
    "p75": 10570.0,
    "p95": 81617.91438601195,
    "p99": 228046.0613831818,
    "max": 513989.0,
    "mean": 15596.85002871948
  },
  "rbox_width": {
    "count": 68072,
    "min": 1.0,
    "p01": 9.486832980505138,
    "p05": 15.620492776172174,
    "p25": 31.384732785222543,
    "p50": 55.077210237150304,
    "p75": 139.0,
    "p95": 495.98336642838024,
    "p99": 695.300863056997,
    "max": 1027.3276000787528,
    "mean": 117.85301787860548
  },
  "rbox_height": {
    "count": 68072,
    "min": 0.0,
    "p01": 4.0,
    "p05": 6.0,
    "p25": 11.401724378356143,
    "p50": 20.00004177608134,
    "p75": 69.46219477807682,
    "p95": 221.0,
    "p99": 370.0,
    "max": 707.0,
    "mean": 54.8580550276537
  },
  "rbox_area": {
    "count": 68070,
    "min": 6.325200012647165,
    "p01": 42.00000000000001,
    "p05": 100.7245600070049,
    "p25": 363.277300006826,
    "p50": 1111.0171000030414,
    "p75": 10570.0,
    "p95": 81619.56631582817,
    "p99": 228047.03672743263,
    "max": 513989.0,
    "mean": 15597.308937593176
  },
  "aspect_ratio": {
    "count": 68070,
    "min": 1.0,
    "p01": 1.0038934716288113,
    "p05": 1.0404040404040404,
    "p25": 2.0481872024491996,
    "p50": 2.5555555555555554,
    "p75": 3.076923076923077,
    "p95": 5.325214424332692,
    "p99": 10.04631437543689,
    "max": 91.50152266525785,
    "mean": 2.7858334063156973
  }
}

### test

- Conversion backend: `fallback`
- Image check mode: `dimensions`
- Assumed image size: `(800, 800)`
- Images checked: `0`
- Images assumed: `11738`
- Label files checked: `11738`
- Objects: `124445`
- Bad image decodes: `0`
- Bad label files: `2`
- Unknown classes: `{}`
- First bad conversion: `None`
- Missing image info for labels: `0`
- Bounds checks: `{'qbox_out_of_bounds': 1322, 'qbox_out_of_bounds_examples': [{'path': '/data5/2025/ldh/OpenRSD/data/DIOR_R_dota/test/labelTxt/11736.txt', 'class_name': 'airplane'}, {'path': '/data5/2025/ldh/OpenRSD/data/DIOR_R_dota/test/labelTxt/11737.txt', 'class_name': 'airplane'}, {'path': '/data5/2025/ldh/OpenRSD/data/DIOR_R_dota/test/labelTxt/11765.txt', 'class_name': 'airplane'}, {'path': '/data5/2025/ldh/OpenRSD/data/DIOR_R_dota/test/labelTxt/11778.txt', 'class_name': 'ship'}, {'path': '/data5/2025/ldh/OpenRSD/data/DIOR_R_dota/test/labelTxt/11816.txt', 'class_name': 'stadium'}, {'path': '/data5/2025/ldh/OpenRSD/data/DIOR_R_dota/test/labelTxt/11842.txt', 'class_name': 'vehicle'}, {'path': '/data5/2025/ldh/OpenRSD/data/DIOR_R_dota/test/labelTxt/11842.txt', 'class_name': 'vehicle'}, {'path': '/data5/2025/ldh/OpenRSD/data/DIOR_R_dota/test/labelTxt/11890.txt', 'class_name': 'tenniscourt'}, {'path': '/data5/2025/ldh/OpenRSD/data/DIOR_R_dota/test/labelTxt/11890.txt', 'class_name': 'tenniscourt'}, {'path': '/data5/2025/ldh/OpenRSD/data/DIOR_R_dota/test/labelTxt/11913.txt', 'class_name': 'Expressway-Service-area'}], 'rbox_center_out_of_bounds': 4, 'rbox_center_out_of_bounds_examples': [{'path': '/data5/2025/ldh/OpenRSD/data/DIOR_R_dota/test/labelTxt/15772.txt', 'class_name': 'vehicle'}, {'path': '/data5/2025/ldh/OpenRSD/data/DIOR_R_dota/test/labelTxt/16466.txt', 'class_name': 'basketballcourt'}, {'path': '/data5/2025/ldh/OpenRSD/data/DIOR_R_dota/test/labelTxt/19359.txt', 'class_name': 'harbor'}, {'path': '/data5/2025/ldh/OpenRSD/data/DIOR_R_dota/test/labelTxt/21451.txt', 'class_name': 'vehicle'}], 'invalid_rbox_size': 2}`

RBox statistics:

{
  "qbox_area": {
    "count": 124445,
    "min": 0.0,
    "p01": 21.0,
    "p05": 44.0,
    "p25": 189.0,
    "p50": 608.0,
    "p75": 2349.5,
    "p95": 29925.0,
    "p99": 127763.59999999992,
    "max": 552510.0,
    "mean": 6560.818968218892
  },
  "rbox_width": {
    "count": 124445,
    "min": 0.0,
    "p01": 3.6055512754639896,
    "p05": 5.656854249492381,
    "p25": 12.165525060596437,
    "p50": 25.179356624028344,
    "p75": 51.66236541235796,
    "p95": 179.0,
    "p99": 512.144074187035,
    "max": 1047.7556967156036,
    "mean": 51.000450255663324
  },
  "rbox_height": {
    "count": 124445,
    "min": 0.0,
    "p01": 3.6055512754639896,
    "p05": 5.656854249492381,
    "p25": 13.0,
    "p50": 26.076809620810597,
    "p75": 57.87054518492115,
    "p95": 185.0,
    "p99": 506.85808664658884,
    "max": 884.7649405350553,
    "mean": 54.11200267035642
  },
  "rbox_area": {
    "count": 124443,
    "min": 4.0,
    "p01": 21.213203435596427,
    "p05": 46.09772228646443,
    "p25": 192.0,
    "p50": 615.0,
    "p75": 2352.0820560449556,
    "p95": 29945.40004175013,
    "p99": 127764.30002042594,
    "max": 552510.1252321082,
    "mean": 6572.141244206517
  },
  "aspect_ratio": {
    "count": 124443,
    "min": 1.0,
    "p01": 1.0,
    "p05": 1.0185185185185186,
    "p25": 1.2727272727272727,
    "p50": 2.1739130434782608,
    "p75": 2.875,
    "p95": 4.807454495365984,
    "p99": 8.968621576004441,
    "max": 67.18796871833429,
    "mean": 2.424547831317388
  }
}
