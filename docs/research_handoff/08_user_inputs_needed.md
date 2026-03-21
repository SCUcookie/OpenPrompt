# User Inputs Needed

This file lists only the questions that materially affect the next research step.

## Highest-priority questions

### 1. Where is the actual codebase?

Needed because:

- the current directory contains planning files and a PDF only
- implementation and reproduction cannot start without code

What is needed:

- repository path
- branch name if relevant
- whether OpenRSD code already exists there

### 2. Which datasets are already present on disk?

Needed because:

- the experiment plan depends on real dataset availability
- dataset preparation time can dominate the schedule

What is needed:

- dataset names
- absolute paths
- annotation format
- whether rotated boxes are ready or need conversion

### 3. What is your real target venue and deadline?

Needed because:

- the correct scope depends on timeline
- a workshop paper and a top-tier conference paper should not use the same risk profile

What is needed:

- venue name
- submission deadline
- whether you care more about first submission speed or maximum paper quality

### 4. Do you want the next session to reproduce OpenRSD first, or design the new method immediately?

Needed because:

- this changes the execution order substantially

Options:

- strict reproduction first
- parallel reproduction plus method design
- direct method design if a trusted codebase already exists

## Secondary questions

### 5. What hardware is actually available now?

Needed because:

- the plan assumes up to 4 x 4090

What is needed:

- GPU count
- GPU model
- per-GPU VRAM
- storage budget

### 6. Do you already have a preferred baseline repository?

Needed because:

- starting from MMRotate, OpenMMLab, or another private codebase changes implementation cost

### 7. Are you willing to make the first paper narrower if it increases acceptance probability?

Needed because:

- the best publishable path may be a narrower detection paper rather than a larger "foundation" story

## Facts that should be written back into this directory once known

When the user answers, record the answers into:

- [11_dataset_and_resource_manifest.md](./11_dataset_and_resource_manifest.md)
- [10_experiment_registry_template.md](./10_experiment_registry_template.md)

## Suggested short reply format for the user

The user can answer in this format:

```text
Codebase: /abs/path/to/repo
Branch: main
Datasets: DOTA-v2.0=/abs/path1; DIOR-R=/abs/path2; ...
Hardware: 4x4090 24GB
Venue: ICCV 2027
Deadline: 2027-03-15
Mode: reproduction first
```
