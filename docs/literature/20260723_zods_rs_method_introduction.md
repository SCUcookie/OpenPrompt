# ZODS-RS: Zero-Training Oriented Detection & Segmentation for Remote Sensing — Method Introduction

- **Authors:** Zuan Gu, Tianhan Gao, Langxu Zhao
- **Source:** arXiv 2606.10769, submitted 2026-06-09 (<https://arxiv.org/abs/2606.10769>)
- **Status:** arXiv-only preprint as of 2026-07-23 — no proceedings/OpenReview
  record found; per the project's verification rule this does **not** count
  as a top-venue record. Selected here for topical fit (the user requested
  the detection+segmentation intersection in remote sensing), recency, and
  direct relevance to our paused segmentation lane.
- **One line:** a completely **training-free** pipeline that produces
  detection boxes *and* instance masks in remote sensing imagery from a
  single reference example per class, by combining frozen DINOv3 dense
  features with SAM2 proposals through robust prototype statistics,
  rotation/scale-equivariant matching, and uncertainty-aware merging.

## Why this paper, given our progress

It sits precisely on the seam between our two project halves. Our detection
route (GeoNexus-RSD) is built on *learned* class prototypes from RemoteCLIP
text prompts; ZODS-RS builds *statistical* class prototypes from one visual
exemplar — and even offers an optimal-transport bridge to text anchors,
which is structurally the training-free cousin of our TPC head. Our
segmentation lane was closed largely because every candidate required GPU
training plus a broken dependency chain; ZODS-RS requires **no training at
all**, which means it could be evaluated as an analysis-only job under the
existing route closures — the first segmentation-lane candidate that does
not need the closed route reopened. And it evaluates on FAIR1M, where we
now have our own baselines.

## Problem setting

Open-set/few-shot detection and segmentation in remote sensing usually
requires either heavy vision-language pretraining (Grounding DINO variants,
OWL-V2, Florence-2) or dataset-specific fine-tuning. The authors target the
regime where a platform (satellite or UAV) must recognize new categories
from a single marked example, with no task-specific training, while
remaining robust to the two aerial nuisances our own work also revolves
around: arbitrary rotation and extreme scale variation. The output is
horizontal boxes plus instance masks; classification is one-shot,
prototype-based.

## Method in detail

The pipeline has five stages: **Engine → Robust-PP → R-SEM → CWLA → UAM.**

### Stage 1 — Engine: frozen features, class-agnostic proposals, reference memory

Two frozen foundation models supply complementary signals. DINOv3 provides
*semantics*: multi-layer dense tokens (layers 8, 10, and the last) are
resized to a common grid and L2-normalized per pixel, giving per-pixel
descriptors $f_p^{[\ell]} \in \mathbb{R}^C$ at three depths. SAM2 provides
*geometry*: a high-recall pool of class-agnostic instance masks $\{M_i\}$,
each reduced to a box via `minAreaRect`, filtered by stability checks
(minimum area/score, morphological IoU) and oriented NMS at
$\tau_{nms}=0.50$. A reference-to-memory step collects the DINOv3 tokens
inside each user-marked reference mask (with border erosion and optional
top-$q\%$ cosine filtering) as the raw material for class prototypes.
Nothing is trained; the "model" is the memory.

### Stage 2 — Robust-PP: prototype purification

The core insight: a naive mean of reference tokens is a fragile prototype,
because aerial reference crops contain background pixels, boundary bleed,
and outlier tokens. The authors therefore build the prototype with robust
statistics:

1. **Tyler scatter estimation** — a scale-free robust covariance
   $\Sigma_{t+1} = \frac{d}{n}\sum_j \frac{x_j x_j^\top}{x_j^\top \Sigma_t^{-1} x_j}$,
   trace-normalized, ~20 iterations. Unlike the sample covariance, it is
   insensitive to token magnitude outliers.
2. **Spectral purification** — eigendecompose the scatter, keep the top
   $r{=}32$ axes through an idempotent projector $P_r$, discarding
   directions dominated by noise.
3. **Whitening + bounded-influence mean** — tokens are whitened
   ($\tilde{x}_j = P_r \hat{\Sigma}_k^{-1/2} x_j$) and averaged with
   M-estimator weights $a_j = \psi(\|\tilde{x}_j\|_2)$, so outlier tokens
   cannot drag the prototype; the result is re-projected and normalized to
   give $\hat{p}_k$.
4. **Optional OT-PP** — when text or image anchors exist (e.g., CLIP-style
   text embeddings), entropy-regularized optimal transport shrinks the
   visual prototype toward the anchor manifold with weight $\alpha$. *This
   is the direct point of contact with our TPC prompt prototypes.*
5. **Negative mining** — low-prior tokens (low norm/margin/attention) are
   aggregated into negative prototypes $\{\tilde{p}^-_{k,j}\}$ that later
   compete in the softmax to suppress look-alike backgrounds.

The class memory is the tuple
$\mathcal{B}_k = (\hat{p}_k, \hat{\Sigma}_k, \{\tilde{p}^-_{k,j}\})$.

### Stage 3 — R-SEM: rotation–scale equivariant matching

Instead of making the *features* rotation-invariant (which loses
orientation information), the matching is made *equivariant*: prototype
responses $S_k^{(s,\theta)}(p) = \langle \bar{f}_{p,(s,\theta)}, \hat{p}_k\rangle$
are computed over a grid of scales $s$ and steerable rotations $\theta$
(FFT-based, still training-free), then fused with separable Gaussian
weights centered on the reference's scale/orientation statistics:
$R_k(p) = \sum_s \sum_\theta \bar{w}^{(s,\theta)} \, \text{Up}(S_k^{(s,\theta)})(p)$.

Each SAM2 candidate mask is then scored against each class with a composite
assignment cost combining (i) the top-K mean response inside the mask, (ii)
coverage — the fraction of mask pixels above a response threshold, (iii)
aspect-ratio/compactness shape priors, and (iv) an angle-consistency
penalty between the mask's minimum-area-rectangle angle and the dominant
response angle. A **Hungarian assignment** solves the global one-to-one
mask-to-class matching per image. This is the stage that operationalizes
"oriented": orientation enters through the equivariant response stack and
the angle penalty, even though the final emitted boxes are horizontal.

### Stage 4 — CWLA: consistency-weighted layer aggregation

The three DINOv3 depths disagree in characteristic ways (early layers
texture-biased, late layers semantic-biased). Per layer, a spatial
uncertainty score (entropy/energy over classes) is computed, and layers are
fused with softmax weights $\beta_\ell = \text{softmax}(-\bar{U}^{[\ell]}/\sigma)$
($\sigma{=}0.15$): the layer that is most *confident* about an image
dominates that image's response map. A cheap, training-free alternative to
learned feature-pyramid fusion.

### Stage 5 — UAM: uncertainty-aware pixel-wise merging

Overlapping candidate masks are reconciled at the pixel level. Prior-aware
logits $L_c(p) = R_c(p)/\tau + \gamma \log A_c(p)$ produce per-pixel class
posteriors; each mask votes for its assigned class with a weight that is
*down-scaled by local pixel entropy* —
$w_j(p) \propto \pi_{k_j}(p) A_{k_j}(p)^\gamma / (1+\lambda U(p))$ —
so ambiguous pixels count less in the argmax. Open-set rejection uses both
a maximum-posterior threshold and an energy gate, and the mined negative
prototypes participate as competing softmax channels (never emitted). An
optional 10-iteration CRF sharpens boundaries without altering scores.

## Results (their protocol — not comparable to ours)

Single-scale full-image inference, one reference exemplar per class,
COCO-style HBB metrics. Cross-domain xView: 16.69 mAP$_{50:95}$ vs 13.00
for the strongest baseline (Florence-2). FAIR1M ship: 23.93 vs 20.38
(Florence-2). FAIR1M airplane is a stated weakness (2.19 vs Grounded-SAM's
3.01, though better AP$_{75}$). On their UAV set: 47.30 mAP, 31.10 mask
mIoU, +30.7 small-object AP over Grounded-SAM. **Do not compare any of
these numbers to our FAIR1M baselines** — different task (one-shot HBB,
COCO metrics, no tiling) versus our fully supervised OBB AP$_{50}$ with
tiling.

## Stated limitations

Small-object recall lags under no-tiling inference; FAIR1M airplane shows
domain-shift sensitivity; the pipeline emits **HBB only** (true OBB and
shape-aware masks are named as future work); proposal quality still fails
on weak textures and extreme scales; and many thresholds
($\tau_{nms}, \tau_{open}, \varepsilon_{open}$, …) are hand-tuned per
domain.

## Relevance to GeoNexus-RSD (route impact)

1. **Segmentation lane, zero-GPU entry (most actionable).** Because
   ZODS-RS is training-free, evaluating it (or its ideas) is an
   *analysis-only* job — it does not require reopening the closed
   segmentation-lane training route. If the lane is ever revisited, this is
   the first candidate whose cost profile fits the closure constraints.
2. **Prototype machinery cross-pollination.** Their Robust-PP chain (Tyler
   scatter → spectral projection → bounded-influence mean → negative
   prototypes) is a drop-in-appealing upgrade path for *our* prompt
   prototypes: today our $\bar{e}_c$ is a simple weighted mean of prompt
   embeddings (Eq. 1 of our manuscript); a robustified aggregation plus
   mined negative prototypes is a concrete, cheap future-work idea for
   TPC. Their OT anchoring of visual prototypes to text anchors is
   literally the inverse of our text-to-visual adaptation (SCA) — the two
   mechanisms would compose.
3. **The OBB gap is our home turf.** Their named future work — "true OBB
   and shape-aware masks" — is exactly what our stack already does well.
   A GeoNexus-style oriented head grafted onto a ZODS-like training-free
   pipeline is a plausible follow-up paper direction bridging both project
   halves.
4. **Open-set rejection mechanics** (posterior threshold + energy gate +
   negative channels) are relevant reference material if the
   open-vocabulary framing of GeoNexus is ever extended beyond the current
   fixed-vocabulary claim.
5. **Caution:** arXiv-only, three authors, no code link verified in this
   pass — treat as an idea source, not a comparator; re-check for a venue
   record and code release before citing in any manuscript.
