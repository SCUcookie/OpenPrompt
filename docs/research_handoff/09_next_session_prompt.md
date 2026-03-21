# Next Session Prompt

Paste the following prompt in the next session if you want the work to continue directly from this directory.

```text
You are continuing a remote sensing research project from the directory /opt/pangu/ldh/openprompt.

Treat that directory as the research handoff package and source of truth for planning context.

Before doing anything else:
1. Read /opt/pangu/ldh/openprompt/README.md
2. Read /opt/pangu/ldh/openprompt/00_research_decision.md
3. Read /opt/pangu/ldh/openprompt/03_hsc_openrsd_blueprint.md
4. Read /opt/pangu/ldh/openprompt/04_execution_roadmap.md
5. Read /opt/pangu/ldh/openprompt/07_truthfulness_and_execution_protocol.md
6. Read /opt/pangu/ldh/openprompt/08_user_inputs_needed.md

Research rules for this session:
- Accuracy and truthfulness are higher priority than user preference ordering.
- Do not pretend unknown facts are known.
- Distinguish direct evidence from inference.
- Baseline reproduction or verification comes before novelty claims.
- Do not make agent or diffusion the main story unless strong evidence forces that change.
- The default paper direction is GeoNexus-RSD: hierarchy- and context-aware open-prompt rotated remote sensing detection built on top of OpenRSD.
- The hard hardware ceiling is 4 x 4090 unless the user updates it.

Your first operational objective is to move from planning into executable research work.

Do this in order:
1. Audit the current filesystem for the actual codebase, dataset paths, and available assets.
2. Compare the discovered assets against /opt/pangu/ldh/openprompt/08_user_inputs_needed.md.
3. Ask the user only the minimal missing questions that materially block valid execution.
4. Create or update Markdown records in /opt/pangu/ldh/openprompt so the directory stays self-contained.
5. If a usable codebase exists, inspect it and produce an implementation-ready baseline reproduction plan.
6. If the codebase and datasets are sufficient, start the concrete research work rather than re-planning.

Expected output style:
- be direct
- challenge weak assumptions
- prioritize executable next steps
- preserve a clear record in Markdown files inside /opt/pangu/ldh/openprompt

If code is available, prefer this sequence:
- baseline audit
- reproduction checklist
- failure mode analysis
- first lightweight hierarchy/context ablation design

If code is not available, do not hallucinate implementation status. Instead, produce the smallest exact list of missing assets and ask for them.
```

## Why this prompt exists

It is designed to prevent a future session from:

- restarting from generic brainstorming
- losing the chosen direction
- overfitting to the user's first intuition
- skipping truthfulness when facts are missing

## When to update this prompt

Update this file only if one of these changes:

1. the main paper direction changes
2. the baseline changes
3. hardware constraints change
4. the codebase path becomes fixed and should be embedded here
