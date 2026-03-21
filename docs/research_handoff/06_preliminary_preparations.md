# Preliminary Preparations

This file states what has already been completed and what still remains before implementation work can begin.

## Completed in this directory

### Research framing

- The main task has been fixed as **open-prompt rotated object detection in remote sensing**.
- Semantic segmentation has been evaluated and kept as a backup, not the primary first-paper path.
- The working paper concept has been fixed as **GeoNexus-RSD**.

### Baseline analysis

- The local paper [openrsd-ICCV2025_paper.pdf](./openrsd-ICCV2025_paper.pdf) was inspected and summarized.
- OpenRSD was decomposed into:
  - prompt construction
  - alignment head
  - fusion head
  - class embeddings
  - multi-stage training
  - self-training
- Key strengths and weaknesses were documented in [01_openrsd_gap_analysis.md](./01_openrsd_gap_analysis.md).

### Direction selection

- Detection vs segmentation was evaluated.
- LLM vs agent vs diffusion was evaluated.
- The recommended path and the rejected paths were documented in [00_research_decision.md](./00_research_decision.md) and [02_direction_ranking.md](./02_direction_ranking.md).

### Paper blueprint

- The main hypothesis is defined.
- The core modules are defined.
- Primary evaluation targets and go/no-go criteria are defined.

See [03_hsc_openrsd_blueprint.md](./03_hsc_openrsd_blueprint.md).

### Execution logic

- Staged execution plan exists.
- Failure signals and pivots are defined.
- Hardware-aware training logic under 4 x 4090 is defined.

See [04_execution_roadmap.md](./04_execution_roadmap.md).

### Literature grounding

- Recent directly relevant papers were mapped.
- The literature file contains only works useful for the next step, not a broad survey.

See [05_related_work_map.md](./05_related_work_map.md).

## Not completed because external assets are missing

These are not planning gaps. These are external-resource gaps.

### Codebase

Not present in this directory:

- training code
- evaluation code
- dataset loaders
- experiment scripts

### Data

Not present in this directory:

- DOTA-v2.0
- DIOR-R
- FAIR1M-2.0
- HRSC2016
- SpaceNet
- VEDAI
- CORS
- DOSR
- SODA-A

### Environment facts

Still unknown:

- exact GPU availability
- CUDA version
- whether mixed precision and distributed training are already configured
- available storage for datasets and checkpoints

### Project management facts

Still unknown:

- target venue
- target submission date
- whether reproduction must happen before any method design
- whether the user already has a preferred codebase branch

## What is now ready for immediate use

This directory is now ready to support the following next-stage tasks:

1. baseline reproduction planning
2. codebase inspection
3. experiment prioritization
4. data and hardware audit
5. method specification refinement
6. paper-outline drafting

## What the next session should not redo

The next session should not spend time redoing:

1. broad topic brainstorming
2. deciding between detection and segmentation unless new evidence appears
3. deciding whether agent or diffusion should be the main story
4. rereading OpenRSD from scratch unless exact implementation details are needed

## Definition of preparation complete

For this directory, "preliminary preparation complete" means:

- problem statement fixed
- baseline fixed
- best direction fixed
- research risks mapped
- next-session prompt prepared
- user-input gaps isolated

That condition is now satisfied.
