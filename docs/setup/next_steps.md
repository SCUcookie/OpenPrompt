# Next Steps

## Immediate execution

1. Install the package and run the synthetic smoke test.
2. Run `configs/experiments/baseline_synthetic.yaml` first to verify the pure baseline path.
3. Run `configs/experiments/geonexus_synthetic.yaml` for the structure-only version.
4. Run `configs/experiments/geonexus_math_synthetic.yaml` after that if you want the first math-heavy ablation.
5. Fill in real dataset paths.
6. Build a prompt bank artifact from the provided taxonomy JSON.
7. Add official rotated mAP evaluation before claiming paper-ready results.

## First strong experiment sequence

1. `OpenRSD`-like baseline only
2. `+ hierarchy bank`
3. `+ scene-context adapter`
4. `+ pseudo-label composite score`
5. `+ router`

## Best low-risk ways to beat the baseline

- strengthen the hierarchy relation matrix with confusing-class annotations from real errors
- add a confusing-class margin loss
- calibrate scene-conditioned prompt temperature
- weight pseudo-label acceptance using both semantic support and scene consistency

## Best structure innovations

- learn a query router so tiny and ambiguous proposals use stronger prompt fusion
- add a geometry-aware branch for aspect-ratio and orientation priors
- add regional scene tokens instead of only global scene features

## Best mathematical or formula innovations

- hierarchy Laplacian regularization
- scene-conditioned temperature scaling
- composite pseudo-label energy function
- asymmetric margin penalties for confusing class pairs

## Already wired in this repo

- pure baseline config without hierarchy regularization
- structure modules toggled through `model.innovations`
- scene-conditioned temperature scaling through `model.innovations.scene_temperature`
- confusing-class margin loss through `criterion.margin_weight`

## Best reviewer-facing package

- one clean failure mode: small and confusing rotated objects
- one clean method story: hierarchy + context + consistency
- one clean math story: graph or margin regularization
- one clean evaluation story: mixed prompts + cross-dataset + small-object transfer

## What to do before writing the paper

- replace the hash embedder with a stronger text encoder
- add official DOTA metrics
- verify results on at least two real datasets
- produce qualitative confusion maps and prompt-robustness plots
