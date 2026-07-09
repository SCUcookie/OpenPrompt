# 2026-06-30 SOTA Audit

## Scope

Fast SOTA-facing audit for the current GeoNexus evidence. This note records
published comparators, protocol boundaries, and the decision state before any
third-party retraining or checkpoint evaluation.

Conclusion: do not claim SOTA yet. GeoNexus is competitive on DIOR-R and above
several public baselines, but current public DIOR-R rows include Strip R-CNN at
`68.70` and OrientedFormer Swin-T at `68.84 AP50`, while GeoNexus S3 best mean
is `69.79 AP50` only under the local sanitized `DIOR_R_dota/test` evaluator.
The exact DIOR-R split, preprocessing, and evaluator semantics must be matched
before a SOTA claim. DOTA2 evidence remains internal `DOTA2_1024_500/ss_val`
only and is not directly comparable to official DOTA-v2.0 test submissions.

## GeoNexus Rows

| Row | Dataset/protocol | Best single | Mean | Source | Decision |
| --- | --- | ---: | ---: | --- | --- |
| DIOR-R S3 scene adapter | sanitized `DIOR_R_dota/train_val` to `DIOR_R_dota/test`, MMRotate `DOTAMetric` AP50 | rep0 epoch 8, `0.6992` | best mean `0.6979`; final mean `0.6859` | `20260615_dior_r_geonexus_s3_scene_adapter_replicas_complete.json` | strongest local DIOR-R row; protocol match still required |
| DIOR-R S4 low-LR pseudo-label | pseudo-label train root, sanitized `DIOR_R_dota/test`, MMRotate `DOTAMetric` AP50 | `0.6935`, `0.6966`, `0.6967` | best mean `0.6956`; final mean `0.6926` | `20260629_dior_r_s4_low_lr_complete.md` | weak stabilization, not superiority |
| DOTA2 S2 loss-0 | `DOTA2_1024_500/ss_val`, MMRotate `DOTAMetric` AP50 | rep3407 epoch 1, `0.6211` | all-run best mean `0.6206`; final mean `0.6167` | `20260611_dota2_s2_loss0_replicates_analysis.md` | internal validation only; not official DOTA-v2.0 comparison |

## Public Comparators

| Method | Public source | Published row | Repo/checkpoint status | Audit read |
| --- | --- | ---: | --- | --- |
| AOPG | arXiv 2110.01931 and `jbwang1997/AOPG` | DIOR-R `64.41` mAP; DOTA `75.24`; HRSC2016 `96.22` | repo and model zoo listed; DIOR-R download referenced | GeoNexus DIOR-R S3 exceeds this historical baseline locally |
| OrientedFormer | TGRS 2024 / arXiv 2409.19648 and `wokaikaixinxin/OrientedFormer` | DIOR-R `67.28` R50, `68.84` Swin-T, `65.07` LSK-T AP50; DOTA-v2.0 `54.27` AP50 | configs and Hugging Face checkpoints listed | highest-priority runnable DIOR-R comparator |
| LSKNet | IJCV 2024 / ICCV 2023 and `zcablii/LSKNet` | DOTA-v1.0 `81.85`, FAIR1M `47.87`, HRSC2016 `98.46`; repo also points to Strip R-CNN update | official repo with configs/models for LSKNet family | useful backbone/context reference, not the cleanest DIOR-R SOTA row by itself |
| Strip R-CNN | arXiv 2501.03775 and `YXB-NKU/Strip-R-CNN` | DIOR-R `68.70`; DOTA-v1.0 `82.75`; DOTA-v1.5 `72.27`; FAIR1M `48.26`; HRSC2016 `98.70` | official repo with configs and model links | mandatory DIOR-R comparator; below GeoNexus S3 best mean, close enough to require strict protocol matching |
| PKINet | arXiv 2403.06258 | paper covers DOTA-v1.0, DOTA-v1.5, HRSC2016, DIOR-R and claims strong remote-sensing detection performance | no practical official code/checkpoint target confirmed in this quick pass | mandatory paper table row; treat as paper-only until code/checkpoint source is verified |
| PKINet-v2 | arXiv 2603.16341 | paper covers DOTA-v1.0, DOTA-v1.5, HRSC2016, DIOR-R and claims SOTA plus `3.9x` FPS acceleration over PKINet-v1 | no practical official code/checkpoint target confirmed in this quick pass | mandatory paper table row; likely newest competitive context |
| OpenRSD | ICCV 2025 / arXiv 2503.06146 | seven-dataset open-prompt RS detector; reports average precision gain over YOLO-World and `20.8 FPS` | code/models stated as to be released in source | use as open-prompt reference, not direct closed-set SOTA comparator |

## Protocol Boundaries

- DIOR-R: current GeoNexus numbers use local sanitized labels and MMRotate
  `DOTAMetric` AP50. A public SOTA statement needs the same train/test split,
  label normalization, ignored-class policy, rotated box convention, and
  evaluator behavior as the cited method.
- DOTA2: current GeoNexus numbers are from `DOTA2_1024_500/ss_val`; official
  DOTA-v2.0 rows require official test-server style submission. Do not compare
  `0.6211` AP50 to OrientedFormer's official DOTA-v2.0 `54.27` AP50 or any
  leaderboard row.
- Best checkpoint vs final checkpoint must stay separate. The DIOR-R S3 best
  mean is strong; final mean is much weaker and should not be blended into a
  single paper claim.
- S4 is not a stronger result than S3. It can be cited only as attempted
  pseudo-label stabilization unless later runs surpass S3 under the same
  checkpoint-selection rule.

## Clone And Evaluation Priority

No clone or new training is required for this audit note.

If third-party evaluation is approved, clone and smoke-check in this order:

1. `wokaikaixinxin/OrientedFormer`: highest-priority DIOR-R checkpoint
   comparator. First verify config loading, checkpoint availability, class
   order, and evaluator semantics.
2. `YXB-NKU/Strip-R-CNN`: strongest confirmed DIOR-R repo row in this audit
   besides OrientedFormer and therefore mandatory for a serious SOTA claim.
3. `zcablii/LSKNet`: useful if Strip R-CNN config/model links are easier to
   route through the LSKNet family or for backbone ablation context.
4. `jbwang1997/AOPG`: lower-priority reproducibility baseline because GeoNexus
   already exceeds the published `64.41` row locally.

For any third-party run, record repo commit, config, checkpoint URL and hash,
dataset mapping, exact command, GPU, runtime log, metrics JSON, and failure
scan. Prefer released-checkpoint evaluation over retraining.

## Source URLs Checked

- AOPG arXiv: `https://arxiv.org/abs/2110.01931`
- AOPG repo: `https://github.com/jbwang1997/AOPG`
- OrientedFormer arXiv: `https://arxiv.org/abs/2409.19648`
- OrientedFormer repo: `https://github.com/wokaikaixinxin/OrientedFormer`
- LSKNet arXiv: `https://arxiv.org/abs/2303.09030`
- LSKNet repo: `https://github.com/zcablii/LSKNet`
- Strip R-CNN arXiv: `https://arxiv.org/abs/2501.03775`
- Strip R-CNN repo: `https://github.com/YXB-NKU/Strip-R-CNN`
- PKINet arXiv: `https://arxiv.org/abs/2403.06258`
- PKINet-v2 arXiv: `https://arxiv.org/abs/2603.16341`
- OpenRSD arXiv: `https://arxiv.org/abs/2503.06146`
