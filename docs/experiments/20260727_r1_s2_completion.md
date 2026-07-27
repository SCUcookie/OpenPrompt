# 2026-07-27 R1-S2 Completion Record

R1-S2 (+HRR) replicas have completed epoch 12 on DIOR-R.

## Metrics (mAP)
| Replica | Epoch 4 | Epoch 8 | Epoch 12 (Final) | Best Checkpoint |
|---|---|---|---|---|
| rep3407 | 0.6621 | 0.6677 | 0.6668 | 0.6677 (E8) |
| rep4407 | 0.6568 | 0.6691 | 0.6682 | 0.6691 (E8) |
| rep5407 | 0.6623 | 0.6669 | 0.6693 | 0.6693 (E12) |

The final mean mAP is ~0.6681, and the best mean mAP is ~0.6687.
This is higher than R1-S1's final mean of 0.6596.
This successfully verifies the +HRR stage for Oriented R-CNN on DIOR-R.

The per-class AP tables are extracted and stored in:
- `docs/experiments/perclass_ap50_r1_s2_rep3407.json`
- `docs/experiments/perclass_ap50_r1_s2_rep4407.json`
- `docs/experiments/perclass_ap50_r1_s2_rep5407.json`
