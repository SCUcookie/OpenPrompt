# Direction Ranking

## Ranking summary

I rank the candidate directions as follows.

| Rank | Direction | Publication upside | Accuracy upside | 4 x 4090 fit | Risk |
| --- | --- | --- | --- | --- | --- |
| 1 | Hierarchy + scene-context + rotated OVD on OpenRSD | High | High | Good | Medium |
| 2 | LLM-assisted pseudo-label refinement for OpenRSD | Medium-High | Medium-High | Good | Medium |
| 3 | Open-vocabulary remote sensing segmentation | Medium | Medium | Medium | Medium-High |
| 4 | Diffusion-centered detection/segmentation paper | Medium | Unclear | Weak-Medium | High |
| 5 | Agent-centered remote sensing detection paper | Low-Medium | Low | Good | High |

## Rank 1: Hierarchy + scene-context + rotated OVD on OpenRSD

### Why it ranks first

- It matches your current baseline directly.
- It hits the most credible unresolved problem: **small, ambiguous, cross-domain rotated objects**.
- It is aligned with recent trends without becoming trend-chasing.
- It can use LLMs in a way that reviewers can accept: offline semantic structure, not vague reasoning.

### Core idea

Build a **hierarchy-aware and scene-aware prompt controller** on top of OpenRSD.

### Why reviewers may like it

- Strong baseline
- Clear gap
- Low-overhead method
- Strong cross-dataset evaluation story

## Rank 2: LLM-assisted pseudo-label refinement for OpenRSD

### Why it is still good

- ICCV 2025 already shows LLM-assisted semantic guidance can help sparsely annotated remote sensing detection.
- OpenRSD already has a self-training stage, so this extension is natural.
- It is lighter than diffusion and more defensible than agent.

### Why it is not rank 1

- It risks becoming too close to "LLM helps pseudo labels" rather than a broader open-prompt detection paper.
- It may look narrower unless you combine it with hierarchy and context.

### Best use

Use it as a **module inside the rank-1 project**, not as a separate paper at the beginning.

## Rank 3: Open-vocabulary remote sensing segmentation

### Why it is attractive

- It is fashionable.
- SegEarth-OV and SCORE show the area is real and active.
- Context helps a lot in aerial imagery.

### Why it is not first

- It is more crowded now.
- It may force you into mask-heavy pipelines and benchmarks that are farther from your current baseline.
- It is less direct than improving OpenRSD for a first strong paper.

### When to choose it instead

Choose this only if:

- you already have segmentation labels or a segmentation codebase ready
- or your OpenRSD reproduction becomes blocked

## Rank 4: Diffusion-centered paper

### Why it looks attractive

- It is still a hot keyword.
- It can help representation learning, synthetic data, or pseudo-label generation.

### Why it is risky

- The connection to **rotated open-prompt remote sensing detection** is currently less direct.
- It is easy to spend a lot of compute and get weak AP gains.
- Under 4 x 4090, it is easy to overbuild and underdeliver.

### Best use

Only use diffusion as:

- a frozen auxiliary feature prior
- or a small augmentation tool

Do not make it the main story first.

## Rank 5: Agent-centered paper

### Why it ranks last

- I did not find strong evidence that agent pipelines are a top-tier winning recipe for remote sensing detection AP right now.
- It is much harder to make benchmark gains reproducible.
- Reviewers may interpret it as pipeline complexity instead of scientific novelty.

### Best use

Agent can help your workflow later:

- data cleaning
- prompt bank construction
- experiment management

But not as the central model contribution.

## Final recommendation

If you want the best balance of:

- publication probability
- accuracy upside
- hardware feasibility
- closeness to OpenRSD

then choose:

**Rank 1, with Rank 2 embedded inside it.**

That means:

**GeoNexus-RSD = OpenRSD + hierarchy-aware prompt modeling + scene-context adaptation + smarter pseudo-label refinement**
