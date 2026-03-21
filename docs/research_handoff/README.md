# OpenPrompt Research Workspace

This directory is prepared to let a future session start research work immediately with minimal re-discovery.

## What is in this directory

- [README.md](./README.md): entrypoint and reading order
- [00_research_decision.md](./00_research_decision.md): final direction choice
- [01_openrsd_gap_analysis.md](./01_openrsd_gap_analysis.md): what OpenRSD solves and where the gap remains
- [02_direction_ranking.md](./02_direction_ranking.md): alternative directions ranked by publication value and risk
- [03_hsc_openrsd_blueprint.md](./03_hsc_openrsd_blueprint.md): recommended paper blueprint
- [04_execution_roadmap.md](./04_execution_roadmap.md): staged plan and go/no-go gates
- [05_related_work_map.md](./05_related_work_map.md): literature map with links
- [06_preliminary_preparations.md](./06_preliminary_preparations.md): everything already prepared and what still depends on external assets
- [07_truthfulness_and_execution_protocol.md](./07_truthfulness_and_execution_protocol.md): research operating rules
- [08_user_inputs_needed.md](./08_user_inputs_needed.md): exact facts still needed from the user
- [09_next_session_prompt.md](./09_next_session_prompt.md): prompt to paste next time so work can start directly
- [10_experiment_registry_template.md](./10_experiment_registry_template.md): structured template for recording experiments
- [11_dataset_and_resource_manifest.md](./11_dataset_and_resource_manifest.md): template for datasets, hardware, paths, and licenses
- [openrsd-ICCV2025_paper.pdf](./openrsd-ICCV2025_paper.pdf): local anchor paper

## Current state

The strategic planning step is complete.

The recommended next paper direction is:

**GeoNexus-RSD: hierarchy- and context-aware open-prompt rotated detection for remote sensing**

The project is intentionally defined as:

- baseline-first
- evidence-first
- detection-first
- small-object and cross-dataset focused
- feasible under 4 x 4090

## What is already decided

1. Start from **OpenRSD**, not from a fresh architecture.
2. Prioritize **rotated object detection**, not semantic segmentation, for the first paper.
3. Use **LLM support only in a constrained, offline, semantics-first way**.
4. Do **not** make diffusion or agents the core contribution of the first paper.

## What is not yet known

These are the main unresolved external facts:

- where the actual codebase lives
- which datasets are already downloaded and usable
- whether you want a target venue and deadline fixed now
- whether you want the first stage to be pure reproduction or direct method design

See [08_user_inputs_needed.md](./08_user_inputs_needed.md).

## Reading order for a future session

If the next session has only a few minutes, read in this order:

1. [README.md](./README.md)
2. [00_research_decision.md](./00_research_decision.md)
3. [03_hsc_openrsd_blueprint.md](./03_hsc_openrsd_blueprint.md)
4. [04_execution_roadmap.md](./04_execution_roadmap.md)
5. [08_user_inputs_needed.md](./08_user_inputs_needed.md)
6. [09_next_session_prompt.md](./09_next_session_prompt.md)

## Standard next action

The default next action for a future session should be:

1. confirm the codebase path and available datasets
2. build the baseline reproduction checklist
3. prepare the hierarchy/context experimental design

## Important constraint

Accuracy and truthfulness take priority over convenience, speed, and user preference ordering.

That rule is formalized in [07_truthfulness_and_execution_protocol.md](./07_truthfulness_and_execution_protocol.md).
