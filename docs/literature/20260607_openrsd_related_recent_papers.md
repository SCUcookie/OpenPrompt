# OpenRSD-Adjacent Recent Paper Tracker - 2026-06-07

This is a living tracker for GeoNexus-RSD planning. Update it before changing
the research route, adding a new module, or making a paper-facing claim about
recent remote-sensing open-vocabulary detection work.

Search scope for updates:

- Primary sources first: arXiv/e-print pages, OpenReview, CVF open access,
  AAAI/OJS, IEEE/JSTARS/IJCV official pages, and official project pages.
- Search for OpenRSD-citing papers and adjacent terms: remote sensing
  open-vocabulary detection, aerial open-world detection, visual prompt object
  detection, text prompt aerial detection, RemoteCLIP detector, pseudo-label
  denoising, and aerial visual grounding.
- Blogs, social posts, and code repositories may be used for discovery only.
  Record the primary paper link before letting the idea affect the route.

## Route Impact Summary

- DOTA2-first remains the paper route. Compare both DOTA2 GeoNexus S1 runs
  against RoI Transformer S0 `dota/mAP=0.6088`, `dota/AP50=0.6090` at first
  validation before launching S2.
- Launch DOTA2 S2 only from the better clean S1 checkpoint. Do not launch S3,
  S4, pseudo-label purification, or routing until DOTA2 S1/S2 and DIOR-R
  numeric stability are resolved.
- DIOR-R detector training is blocked after ORCNN/RoITrans NaN and RetinaNet
  `loss=inf`. The next DIOR-R work is data, rotated-box conversion, and loss
  target diagnosis, not another unchanged detector launch.
- Treat RTMDet-L as completed and deprioritized: final epoch 12
  `0.2779/0.2780`, below epoch 8 `0.3521/0.3520`.
- Recent papers support a hierarchy/context/prompt-alignment story, but they
  do not justify adding modules by habit. Each new module must map to a
  measurable DOTA2/DIOR-R gate.

## Compact Comparison Table

| Paper | Source checked | Venue/status | Relation to OpenRSD / GeoNexus | Math principle | Route impact |
|---|---:|---|---|---|---|
| OpenRSD: Towards Open-prompts for Object Detection in Remote Sensing Images | 2026-06-07 | ICCV 2025, CVF open access | Reference method and local oracle for open-prompt RS detection; GeoNexus should compare against its prompt alignment and fusion ideas without overstating reproduction. | Class prompt embedding `p_c`, region/image feature `q_i`, multimodal prompt alignment and detection losses. | Anchor paper. Use as baseline idea source; keep DOTA2 evidence separate from OpenRSD checkpoint evidence. |
| RS-MPOD: Multi-Prompt Open-Vocabulary Object Detection in Remote Sensing Images | 2026-06-07 | arXiv 2026 | Adjacent prompt ambiguity paper; emphasizes visual plus text prompts for category specification. | Text prompt `p_c^t`, visual prompt `p_c^v`, region feature `q_i`; select or fuse prompts by cosine/logit alignment under semantic ambiguity. | Supports testing hierarchy/prompt variants on confusing DOTA2 classes, but not before S1 validates. |
| DisDop: Domain-Prior Distillation for Remote Sensing Open-Vocabulary Detection | 2026-06-07 | arXiv 2026 | Adjacent teacher-prior approach using RemoteCLIP/DINOv3 style multi-level knowledge. | Multi-level student features `S_l` match teacher targets `T_l` with KL/L2 distillation and domain-prior weighting. | Future inspiration for distillation. Defer until DOTA2 S1/S2 is stable; avoid adding a teacher branch now. |
| SOAR: Semi-Supervised Open-Vocabulary Aerial Object Detection | 2026-06-07 | AAAI 2026, AAAI/OJS | Strongly adjacent to GeoNexus pseudo-label goals; denoises background/foreground priors in aerial scenes. | Foreground prior `f_s`, pseudo-label score `y_i`, denoising/consistency losses separating foreground and background evidence. | Relevant to later pseudo-label purification. Keep paused until DOTA2 S1/S2 and DIOR-R stability gates pass. |
| VK-Det: Visual Knowledge Enhanced Open-Vocabulary Aerial Object Detection | 2026-06-07 | arXiv / AAAI 2026 | Adjacent prototype method for aerial OVD; close to hierarchy/prototype prompt bank ideas. | Visual-knowledge prototype `mu_k`, region feature `q_i`, prototype-aware pseudo-label selection by similarity and confidence. | Reuse cautiously as prototype diagnostics for hierarchy. Do not switch to prototype pseudo-labeling before current gates. |
| OTA-Det: Open-Text Aerial Detection | 2026-06-07 | arXiv 2026 | Adjacent unified open-text detection and visual grounding; relevant to language granularity claims. | Text span embedding `t_j`, region feature `q_i`, dense semantic alignment and grounding/detection objectives. | Future expansion for open-text claims. Reject as current scope unless real open-text evaluation is added. |
| InstructSAM | 2026-06-07 | OpenReview / NeurIPS 2025 | Adjacent instruction-driven proposal/assignment method, useful for parsing user/text instructions and SAM2 proposal filtering. | Instruction tokens `u`, proposal masks/boxes `r_i`, count constraint `K`, assignment minimizing text-region cost under constraints. | Future proposal engine only. Do not add SAM2/proposal assignment to current detector route. |
| OS-W2S: Open-Set Detection from Weak-to-Strong Supervision for Remote Sensing | 2026-06-07 | OpenReview / ICLR 2026 submission | Adjacent weak-to-strong label engine for word/phrase/sentence aerial labels. | Word/phrase/sentence labels `w,p,s`, region feature `q_i`, cross-granularity alignment and pseudo-label promotion. | Good framing for label granularity. Future inspiration, not current launch basis. |
| CastDet | 2026-06-07 | arXiv / ECCV-IJCV line | RemoteCLIP teacher-student pseudo-label queue; adjacent to semi-supervised RS-OVD. | Teacher score `s_i^T`, student score `s_i^S`, queue `Q`, pseudo-label filtering with consistency loss. | Defer. Useful if pseudo-label route reopens after stable DOTA2/DIOR-R baselines. |
| LAE | 2026-06-07 | AAAI 2025, AAAI/OJS | Supporting context for language-aided aerial detection and efficient adaptation. | Label/text embedding `p_c`, visual feature `q_i`, alignment loss and lightweight adaptation. | Supporting citation only unless its exact mechanism is implemented and measured. |
| CoseDet | 2026-06-07 | Primary source to refresh before citation | Supporting context for open-set/open-vocabulary aerial detection. | Usually region-text contrastive matching plus pseudo-label selection; verify exact variables before citing. | Future-direction reference; do not rely on it for a current claim until refreshed. |
| SCORE | 2026-06-07 | Primary source to refresh before citation | Supporting context for remote-sensing open-vocabulary or semi-supervised detection. | Verify exact objective before using. Likely score calibration/semantic consistency around `q_i` and `p_c`. | Discovery/reference only until refreshed from primary source. |
| FLAME | 2026-06-07 | Primary source to refresh before citation | Supporting context for multimodal remote-sensing detection. | Verify exact objective before using. | Discovery/reference only until refreshed from primary source. |

## Short Math Teardown

### OpenRSD

- Core variables: class/open prompt embedding `p_c`, visual region feature
  `q_i`, image/text prompt features, detector logits `z_{i,c}`.
- Principle: align prompt-conditioned features with detector regions using
  cosine/logit alignment plus standard detection classification/regression
  objectives.
- GeoNexus use: reuse the open-prompt alignment framing and RemoteCLIP
  artifact discipline. Do not claim open-vocabulary behavior unless the
  experiment evaluates a real vocabulary-robustness or open-vocabulary split.

### RS-MPOD

- Core variables: visual prompt embedding `p_c^v`, text prompt embedding
  `p_c^t`, region feature `q_i`.
- Principle: category specification under semantic ambiguity by fusing or
  selecting visual/text prompt evidence with similarity scores.
- GeoNexus use: good motivation for hierarchy prompt disambiguation. Current
  action is analysis of DOTA2 confusing classes after S1 validation, not a new
  prompt module launch.

### DisDop

- Core variables: teacher target `T_l` from domain priors, student feature
  `S_l`, level index `l`, distillation weight `alpha_l`.
- Principle: KL/L2 multi-level distillation transfers RemoteCLIP/DINOv3-like
  domain knowledge into the detector.
- GeoNexus use: defer. It increases moving parts and should not precede stable
  DOTA2 S1/S2 evidence.

### SOAR

- Core variables: foreground prior `f_s`, background prior `b_s`,
  pseudo-label `hat y_i`, student prediction `y_i`.
- Principle: denoise semi-supervised pseudo-labels by separating foreground and
  background evidence, then enforce consistency.
- GeoNexus use: useful for the later VLM-assisted pseudo-label purification
  story. Keep S4 paused until the detector route is numerically stable.

### VK-Det

- Core variables: visual prototype `mu_k`, region feature `q_i`, pseudo-label
  confidence `s_i`, class text feature `p_c`.
- Principle: prototype-aware matching and pseudo-labeling; accept regions whose
  visual-knowledge prototype similarity and confidence agree.
- GeoNexus use: a possible hierarchy-bank diagnostic. Defer prototype
  pseudo-labeling until S1/S2 gates pass.

### OTA-Det

- Core variables: text span embedding `t_j`, region feature `q_i`, dense
  alignment target `a_{i,j}`.
- Principle: unify detection and grounding by dense semantic alignment between
  open text and aerial regions.
- GeoNexus use: reject as current scope unless the paper adds open-text
  evaluation. Cite only as future expansion for language granularity.

### InstructSAM

- Core variables: instruction representation `u`, SAM2 proposal `r_i`, desired
  count `K`, assignment variable `x_i`.
- Principle: parse instructions, generate proposals, then solve a
  count-constrained assignment/selection problem over region-instruction costs.
- GeoNexus use: future proposal or annotation tool. Not part of current
  detector training route.

### OS-W2S

- Core variables: weak labels at word/phrase/sentence levels `w`, `p`, `s`;
  strong detector region feature `q_i`; promoted pseudo-label `hat y_i`.
- Principle: weak-to-strong supervision through cross-granularity alignment
  and pseudo-label promotion.
- GeoNexus use: future label-engine inspiration. Do not turn it into a paper
  claim without measured weak-label experiments.

### CastDet

- Core variables: RemoteCLIP teacher prediction `T(q_i,p_c)`, student
  prediction `S(q_i)`, pseudo-label queue `Q`.
- Principle: teacher-student consistency with queue-filtered pseudo-labels.
- GeoNexus use: relevant to S4 only after DOTA2 S1/S2 and DIOR-R diagnosis.

## Primary Source Links

- OpenRSD ICCV 2025:
  <https://openaccess.thecvf.com/content/ICCV2025/html/Huang_OpenRSD_Towards_Open-prompts_for_Object_Detection_in_Remote_Sensing_Images_ICCV_2025_paper.html>
- LAE AAAI 2025:
  <https://ojs.aaai.org/index.php/AAAI/article/view/32672>
- SOAR AAAI 2026:
  <https://ojs.aaai.org/index.php/AAAI/article/view/37671>
- VK-Det arXiv/AAAI 2026:
  <https://arxiv.org/abs/2511.18075>
- RS-MPOD arXiv 2026:
  <https://arxiv.org/abs/2602.01954>
- DisDop arXiv 2026:
  <https://arxiv.org/abs/2605.24639>
- OTA-Det arXiv 2026:
  <https://arxiv.org/abs/2602.07827>
- InstructSAM OpenReview/NeurIPS 2025:
  <https://openreview.net/forum?id=7yRwAEWxto>
- OS-W2S OpenReview/ICLR 2026 submission:
  <https://openreview.net/forum?id=K0idbmzcgc>
- CastDet arXiv/ECCV/IJCV:
  <https://arxiv.org/abs/2311.11646>

## 2026-07-09 Refresh

Refresh triggered by the DIOR-R S3/S4 evidence closure and the TGRS
comparator-table update. Search scope: arXiv, official project/repo pages, and
one supplementary WebSearch check for RSKT-Seg/Pi-Seg segmentation sources.
Route impact summary is unchanged from 2026-06-07: none of these entries
justify a new module or route change on their own; the highest-priority
action is the RiO-DETR watch item below because it is a stronger public
DIOR-R number than any comparator already in the TGRS manuscript.

| Paper | Source checked | Venue/status | Relation to OpenRSD / GeoNexus | Route impact |
|---|---:|---|---|---|
| RiO-DETR: DETR for Real-time Oriented Object Detection | 2026-07-09 | arXiv 2603.09411, 2026-03 | Reports a DOTA-1.0/DIOR-R/FAIR-1M-2.0 real-time oriented detector; a search-engine summary cited DIOR-R AP50 75.73, but the arXiv abstract page itself gives no exact number and states code is not yet released ("Code will be made publicly available"). Materially stronger than any comparator currently in the TGRS manuscript (GeoNexus S3 best mean 69.79, OrientedFormer Swin-T confirmed 68.83). | Watch-only. Do not cite the unconfirmed 75.73 figure as fact; do not add as a comparator row until the exact number is confirmed from the paper/PDF and code/checkpoints are released. Flag in the manuscript's Discussion as an acknowledged stronger recent method so the SOTA-avoidance framing stays defensible. |
| Do Open-Vocabulary Detectors Transfer to Aerial Imagery? A Comparative Evaluation | 2026-07-09 | arXiv 2601.22164 | Benchmarks existing natural-image open-vocabulary detectors directly on aerial imagery without domain adaptation; likely documents the domain-gap failure mode GeoNexus's real-vocabulary caution is designed to avoid overclaiming. | Useful citation for the "why not claim open-vocabulary" framing already in `PROJECT_INSTRUCTIONS.md`. Read before writing that paragraph in any future paper revision, not before this pass. |
| Towards Realistic Open-Vocabulary Remote Sensing Segmentation: Benchmark and Baseline (Pi-Seg) | 2026-07-09 | arXiv 2604.15652 | Primary source for the Pi-Seg segmentation-lane target already tracked in `docs/experiments/20260707_coninfer_piseg_seg_smoke_preflight.md`; confirms the OVRSISBenchV2 benchmark this manuscript's segmentation lane depends on. | Segmentation lane only; does not affect the DOTA2/DIOR-R detection route, which stays primary. |
| RSKT-Seg (open-vocabulary RS segmentation) | 2026-07-09 | arXiv 2509.12040 | Segmentation-lane target. README confirms its OVSISBench dataset and pretrained weights are hosted on Baidu Netdisk (password `USTC`) with a OneDrive mirror; the "datasets" and "pretrained weight" OneDrive links in the README are byte-identical URLs, which looks like a copy-paste error in their README (the same failure pattern independently found in Strip R-CNN's DIOR-R link, see `PROJECT_INSTRUCTIONS.md` 2026-07-06 status) rather than a real mirror of two different resources. | Segmentation lane only. Baidu Netdisk requires a Chinese-network account/client and is not scriptable from this environment; OneDrive folder links have previously hung a `WebFetch` call for 4+ hours in this project (see 2026-07-09 status) — do not retry an automated fetch on `1drv.ms` folder-listing URLs without a strict timeout wrapper. |
| GiPL: Generative augmented iterative Pseudo-Labeling for Cross-Domain Few-Shot Object Detection | 2026-07-09 | arXiv 2605.29539 | Adjacent pseudo-labeling method for cross-domain few-shot detection; not remote-sensing-specific but structurally relevant to the closed DIOR-R S4 pseudo-label stage (teacher-agreement pooling plus generative augmentation). | Future inspiration only if a redesigned S4 attempt is ever separately approved; does not reopen the current closed S4 route. |

Sources:

- RiO-DETR arXiv: <https://arxiv.org/abs/2603.09411>
- Do Open-Vocabulary Detectors Transfer to Aerial Imagery? arXiv: <https://arxiv.org/abs/2601.22164>
- Pi-Seg / OVRSISBenchV2 arXiv: <https://arxiv.org/abs/2604.15652>
- RSKT-Seg arXiv: <https://arxiv.org/abs/2509.12040>
- GiPL arXiv: <https://arxiv.org/abs/2605.29539>
