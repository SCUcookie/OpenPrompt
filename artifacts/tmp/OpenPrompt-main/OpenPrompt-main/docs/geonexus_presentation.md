# Vision Language Guided Hierarchical Prompt Learning for Remote Sensing Object Detection

**Presenter:** [Your Name]

**Duration:** 6 minutes (≈5.5–6 min)

---

Hello. My name is [Your Name]. Today I will present our short-method paper titled “Vision Language Guided Hierarchical Prompt Learning for Remote Sensing Object Detection.” I will explain the motivation, the proposed method, the planned experimental protocol, expected contributions, and practical limitations.

Background. High-resolution remote sensing images are critical for applications such as urban analysis, traffic monitoring, maritime surveillance, and disaster assessment. Compared to natural images, aerial scenes present dense object layouts, severe scale variation, and arbitrary object orientations. Benchmarks like DOTA formalize this challenge using oriented bounding boxes. Modern oriented detectors improve localization, but they typically rely on closed, fixed category vocabularies and abundant box annotations — constraints that are costly and brittle in practice.

Motivation. Language supervision and vision-language models (VLMs) offer a path toward more flexible, taxonomy-aware detection. However, remote sensing poses three difficulties for naive prompt-based transfer: (1) small and rotated objects are visually ambiguous, (2) class labels are often fine-grained and hierarchical, and (3) pseudo labels and weak annotations can introduce semantic noise. Our goal is to combine taxonomy-aware prompts, scene-conditioned adaptation, and conservative VLM-guided pseudo-label filtering to enhance robustness for DOTA-style rotated detection.

Method overview. The proposed system has three core components.

1) Taxonomy-aware prompt bank. For each class we build a set of prompts: leaf-level names, parent-category descriptions, aliases, and contrastive prompts that explicitly exclude common confusions. Each prompt is encoded into an embedding and aggregated into a class representation via learned or fixed weights. A hierarchy regularizer encourages consistency between child and parent predictions so that a high-confidence child prediction is supported by its parent-level evidence.

2) Scene-context prompt adapter. Small or confusing objects often require scene-level cues. We extract a tile-level descriptor from the feature map and a region-level feature for each candidate; a lightweight adapter uses these to adjust prompt embeddings before similarity matching. This lets the model prefer class descriptions that are plausible given the tile context — for example, distinguishing small vehicles near runways from rooftop structures in dense urban tiles. To keep adaptation efficient, the adapter can use parameter-efficient updates such as LoRA.

3) VLM-assisted pseudo-label purification. Pseudo labels expand supervision but are noisy in dense aerial imagery. Before training on candidate pseudo boxes, we compute a purification confidence q that combines detector confidence, hierarchical consistency, VLM semantic agreement between the crop and the class prompts, and geometric plausibility. Only candidates with sufficiently high q are accepted, and their training loss is weighted by q to provide conservative, soft supervision.

Integration and optional routing. We implement the detector in an OpenPrompt/OpenRSD-style scaffold: tiled visual features align with adapted prompt embeddings, and an oriented detection head predicts rotated boxes and class scores. Routing via Gumbel-Softmax is provided as an optional extension to select among prompt branches or fusion heads, but routing is evaluated only after the core hierarchy and purification components are stabilized.

Planned evaluation. The primary benchmark is DOTA-style rotated mAP, supported by class-wise AP and confusion analysis for fine-grained categories. We will run staged ablations: baseline oriented detector, hierarchical prompts, scene-context adapter, purified pseudo supervision, and optional routing. We will also evaluate prompt robustness across alias prompts, parent-level prompts, and mixed prompt sets, and report efficiency metrics such as inference time and memory overhead introduced by prompts and VLM calls.

Expected contributions and limitations. Contributions: a taxonomy-aware prompt representation for oriented detection, a scene-context adapter for small-object disambiguation, and a conservative VLM-guided pseudo-label purification strategy coupled with a clear experimental protocol. Limitations: increased engineering and compute cost from tiling, prompt management, and VLM usage; and potential VLM unreliability on rare, domain-specific aerial categories. The paper is presented as a method and evaluation plan rather than a final benchmark claim.

Conclusion. In short, GeoNexus-RSD reframes prompt learning for remote sensing as a hierarchy- and context-aware process guided by vision-language evidence, aiming to reduce semantic noise when expanding supervision. The next step is to run the staged experiments in the OpenPrompt scaffold and report measured results. Thank you — I’m happy to answer questions.

---

*Notes:*
- Replace `[Your Name]` with your real name before presenting.
- If you want a teleprompter-style line-by-line script or an English Q&A practice list, tell me and I will produce them.
