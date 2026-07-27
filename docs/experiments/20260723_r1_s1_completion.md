# 2026-07-23 R1-S1 Completion Record (A0 Extraction)

R1-S1 (+TPC) replicas completed on DIOR-R.

## Metrics (mAP)
| Replica | Epoch 4 | Epoch 8 | Epoch 12 (Final) | Best Checkpoint |
|---|---|---|---|---|
| rep3407 | 0.6389 | 0.6527 | 0.6618 | 0.6618 (E12) |
| rep4407 | 0.6380 | 0.6562 | 0.6607 | 0.6607 (E12) |
| rep5407 | 0.6430 | 0.6451 | 0.6564 | 0.6564 (E12) |

Final mean mAP: 0.6596
Best mean mAP: 0.6596

The per-class AP tables are extracted and stored in:
- `docs/experiments/perclass_ap50_r1_s1_rep3407.json`
- `docs/experiments/perclass_ap50_r1_s1_rep4407.json`
- `docs/experiments/perclass_ap50_r1_s1_rep5407.json`
