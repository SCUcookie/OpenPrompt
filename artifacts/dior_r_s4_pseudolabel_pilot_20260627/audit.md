# DIOR-R S4 Pseudo-Label Pilot Audit

## Quality Table

| Policy | Kept boxes | Precision | Recall | Hierarchy consistency | Scene consistency | Matched confidence precision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| confidence_only | 66260 | 0.917658 | 0.893257 | 1.000000 | 0.929374 | n/a |
| hierarchy_scene | 61988 | 0.928793 | 0.845806 | 1.000000 | 1.000000 | 0.955508 |
| teacher_agreement_2of3 | 64512 | 0.944274 | 0.894917 | 1.000000 | 0.938491 | 0.934508 |

## GeoReason Diagnostic Ladder

| Level | Diagnostic | Result |
| --- | --- | --- |
| R0 | class/prompt grounding quality | proxy: per-class pseudo-label precision/recall in JSON |
| R1 | confusing-class relation quality | proxy: top class-pair confusion table in JSON |
| R2 | scene-context consistency | proxy: dominant scene-group consistency score |
| R3 | final detection/pseudo-label decision quality | gate result below |

## Gate

- precision_improves_at_matched_kept_count: `True`
- no_catastrophic_false_positive_expansion: `True`
- usable_recall_without_high_conf_precision_degrade: `True`
- failure_scan_clean: `True`
- launch_s4_recommended: `True`

## Top Class-Pair Confusion

- confidence_only: [('groundtrackfield->stadium', 33), ('bridge->overpass', 13), ('vehicle->ship', 12), ('tenniscourt->groundtrackfield', 8), ('overpass->bridge', 8), ('ship->harbor', 5), ('storagetank->vehicle', 5), ('vehicle->storagetank', 4), ('stadium->groundtrackfield', 3), ('baseballfield->basketballcourt', 2)]
- hierarchy_scene: [('groundtrackfield->stadium', 31), ('bridge->overpass', 13), ('tenniscourt->groundtrackfield', 8), ('overpass->bridge', 8), ('ship->harbor', 5), ('storagetank->vehicle', 5), ('stadium->groundtrackfield', 3), ('baseballfield->basketballcourt', 2), ('basketballcourt->tenniscourt', 1), ('vehicle->basketballcourt', 1)]
- teacher_agreement_2of3: [('groundtrackfield->stadium', 19), ('vehicle->ship', 8), ('tenniscourt->groundtrackfield', 8), ('bridge->overpass', 7), ('overpass->bridge', 6), ('ship->harbor', 5), ('storagetank->vehicle', 5), ('vehicle->storagetank', 4), ('stadium->groundtrackfield', 3), ('baseballfield->basketballcourt', 2)]
