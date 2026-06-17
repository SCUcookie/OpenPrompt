# 2026-06-16 Result Analysis And Claim Boundaries

## DOTA2

| Stage | Paper-facing status | Best mAP | Final mAP | Claim boundary |
| --- | --- | ---: | ---: | --- |
| S0 RoI Transformer | completed detector baseline | 0.6088 | 0.6088 | Closed-set baseline only. |
| S1 RemoteCLIP | completed positive module | 0.6177 | 0.6177 | Modest prompt-head gain over S0. |
| S2 loss-0 | repeatable early-checkpoint signal | 0.6206 mean | 0.6167 mean | Use best-checkpoint evidence only; finals are unstable and slightly below S1. |
| S3 scene adapter | clean negative-to-neutral follow-up | 0.6199 mean | 0.6151 mean | Do not claim DOTA2 scene-adapter gain. |

## DIOR-R

| Stage | Paper-facing status | Best mAP | Final mAP | Claim boundary |
| --- | --- | ---: | ---: | --- |
| S0 RoI Transformer | completed sanitized detector baseline | 0.6544 | 0.6544 | Strong cross-dataset baseline after label hygiene. |
| S1 RemoteCLIP | completed positive replicas | 0.6720 mean | 0.6720 mean | Stable prompt-head gain over S0. |
| S2 hierarchy | completed positive replicas | 0.6887 mean | 0.6856 mean | Strong hierarchy evidence across six replicas. |
| S3 scene adapter | completed best-checkpoint positive, final tied | 0.6979 mean | 0.6859 mean | Claim best-checkpoint scene-context gain only; final checkpoints are tied with S2. |
| S3 epoch-8 LR5e-5 stability | completed stability follow-up | 0.6922 mean | 0.6903 mean | Improves final stability but lowers the original S3 best signal. |

## Paper Claim Boundary

Current evidence supports a narrow, defensible claim:

- RemoteCLIP prompt integration improves DOTA2 modestly and DIOR-R clearly over
  their RoI Transformer S0 baselines.
- Hierarchy evidence is strongest on DIOR-R and only early-checkpoint-positive
  on DOTA2.
- Scene-context adaptation is strong as a DIOR-R best-checkpoint effect, but is
  not yet a reliable final-checkpoint or DOTA2-positive effect.

Current evidence does not support:

- S4, pseudo-label purification, routing, FAIR1M, or semi-supervised quality
  claims.
- A DOTA2 scene-adapter improvement claim.
- A final-checkpoint-only S3 superiority claim.
- Open-vocabulary claims without a held-out vocabulary or vocabulary-robustness
  evaluation.
