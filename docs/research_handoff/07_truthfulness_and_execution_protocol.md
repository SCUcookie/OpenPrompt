# Truthfulness And Execution Protocol

This file defines the research operating rules for any future work started from this directory.

## Core rule

**Accuracy and truthfulness come before user preference, speed, elegance, and novelty claims.**

If there is tension between:

- what seems fashionable
- what the user initially prefers
- what the evidence supports

then follow the evidence.

## Mandatory behavior rules

### 1. Do not pretend uncertain facts are known

If a fact is unknown, state that it is unknown.

Examples:

- codebase path unknown
- dataset availability unknown
- exact OpenRSD implementation details unknown
- venue deadline unknown

### 2. Distinguish evidence from inference

Always separate:

- what is directly supported by the paper or experiment
- what is inferred as a likely next step

### 3. Baseline before novelty

Do not claim a better method before:

1. reproducing a meaningful baseline trend
2. identifying a measurable failure mode
3. defining an evaluation protocol that can test the claim

### 4. Avoid trend-driven method inflation

Do not add:

- LLM
- agent
- diffusion
- extra foundation models

unless they address a concrete failure mode and remain feasible under 4 x 4090.

### 5. Prefer lightweight, testable modules

When choosing between:

- a large fashionable module
- a smaller but measurable and defensible module

prefer the smaller one first.

### 6. Do not optimize only for headline AP

The main paper story must include at least one real difficulty:

- small objects
- fine-grained confusion
- cross-dataset transfer
- mixed-prompt robustness
- practical efficiency

### 7. Ask only when the answer matters materially

Questions to the user should be asked only when the answer changes:

- implementation path
- experimental validity
- feasibility
- publication strategy

Otherwise proceed with clearly marked assumptions.

## Execution order

Every future session should follow this order unless a strong reason is documented:

1. confirm facts
2. inspect assets
3. reproduce baseline or verify prior results
4. isolate failure mode
5. design the smallest strong intervention
6. evaluate against the right baselines
7. only then expand scope

## Publication filter

A direction is worth continued work only if it satisfies most of these:

1. clear technical gap
2. measurable improvement path
3. realistic under 4 x 4090
4. easy to compare against OpenRSD
5. unlikely to be rejected as mere prompt engineering

## Anti-patterns

Avoid these failure modes:

1. adding an LLM only to generate more prompt text
2. adding more data without explaining why OpenRSD's Million-AID result does not already weaken that idea
3. making an "agentic" paper without benchmark evidence
4. making diffusion the core story without a strong compute-justified gain
5. claiming a paper direction is best only because it is hot

## Allowed default assumptions

If missing, the future session may assume:

1. the primary first paper is still detection-first
2. OpenRSD remains the starting baseline
3. 4 x 4090 is the hard hardware ceiling
4. practical publication probability is more important than conceptual novelty alone

## Stop conditions

A future session should stop and ask the user if:

1. the available codebase does not support the proposed baseline
2. the available datasets do not match the evaluation plan
3. the target venue requires a major scope change
4. the user has a conflicting external constraint not captured here
