# AI4S Harness Structure For GeoNexus-RSD

Date: 2026-06-11

Goal: define a project-specific AI4S experiment harness inside `New/` that
wraps the current OpenRSD/MMEngine training workflow without replacing it. The
harness should automate GPU discovery, launch, monitoring, failure detection,
metric extraction, experiment-record generation, and AI-assisted diagnosis while
keeping detector training and validation source-of-truth behavior in
OpenRSD/MMEngine.

This document is a structure and implementation guide, not a claim that the
harness already exists.

## Current Repository Reality

`New/` is the local `openprompt-rs` scaffold. Its native training and evaluation
entrypoints are config-driven Python scripts:

- `scripts/train.py`
- `scripts/evaluate.py`

Those scripts are useful for the rebuilt prompt/VLM scaffold, synthetic checks,
and local GeoNexus module experiments.

`OpenRSD/` is the effective high-performance MMRotate/OpenRSD backend currently
used for paper-facing DOTA2 detector experiments. DOTA2 S0/S1/S2 runs, MMEngine
logs, `work_dirs/`, checkpoints, and `vis_data/scalars.json` files are produced
there. The harness must treat OpenRSD/MMEngine as authoritative for detector
training, validation, and metric semantics.

`New/scripts/gpu_queue_monitor.py` already provides a small queue launcher. It
loads a JSON list of jobs, waits for configured GPUs to be idle for several
polls, and launches jobs in detached `screen` sessions. Its scope is intentionally
small: it handles pending-job launch only. It does not manage full lifecycle
state, provenance, retries, metric extraction, diagnosis bundles, or final
experiment reports.

`New/docs/experiments/README.md` defines the fields required for experiment
records:

- date
- Git commit hash
- machine/server name
- GPU type and count
- dataset version, root path, and split
- config path
- exact command
- validation command
- checkpoint path outside Git
- log path outside Git
- metric implementation
- embedding backend, if used
- key metrics
- class-wise metrics for DOTA-style detection
- failure notes
- next action

The harness should generate these fields automatically from specs, registry
events, logs, metrics, and runtime environment snapshots.

## Design Patterns To Borrow

Use Hydra/OmegaConf-style config composition for repeatable experiment specs.
The project does not need a hard dependency on Hydra in the first version, but
the spec layout should allow base configs plus run-specific overrides, for
example dataset, detector family, stage, seed, GPU policy, and parent
checkpoint.

Use MLflow/W&B-style run metadata, metric history, and artifact indexing, but
store everything locally first. A local JSONL registry and small artifacts are
enough for the current server workflow and avoid introducing an external service
dependency.

Use Prefect/Airflow-style state-machine concepts for run lifecycle. The harness
only needs a light state model because the project mostly runs GPU-bound research
jobs, not a general data platform.

Use AiiDA/Snakemake/Nextflow-style provenance thinking. Every result should know
its code commit, config, dataset split, checkpoint, command, environment, parent
run, and generated outputs. A run without provenance should be treated as
diagnostic only, not paper-facing evidence.

Keep MMEngine/MMRotate as the training and evaluation source of truth for
detector runs. The harness should orchestrate commands, collect state, and parse
outputs; it should not reinterpret detector training semantics.

## Lifecycle Model

The first harness version should use a small append-only lifecycle:

| State | Meaning | Required evidence |
| --- | --- | --- |
| `planned` | Spec is valid but not launched | spec path, resolved config, command plan |
| `launched` | Process or screen was started | timestamp, screen name, PID if known, GPU |
| `accepted` | Startup reached expected progress | progress regex or first checkpoint/log marker |
| `running` | Process is alive and still making progress | PID/screen state, log cursor, GPU residency |
| `completed` | Training/evaluation finished successfully | final metric, checkpoint marker, clean failure scan |
| `failed` | Run hit a scoped failure signature or exited early | failure class, tail log, command, environment |
| `archived` | Human-facing record has been generated | docs path, metric artifact path, diagnosis if any |

The lifecycle should be append-only. Corrections should be new events rather
than in-place edits whenever possible.

## Proposed Modules

### `src/openprompt_rs/harness/specs.py`

Defines typed run specifications.

Inputs:

- YAML or JSON spec under `configs/harness/`
- optional shared defaults for dataset, backend, stage, GPU policy, and failure
  signatures

Core fields:

- run id
- stage, such as S0, S1, S2, diagnostic, or smoke
- dataset name, version, root path, and split
- config path
- backend, such as `opensrd_mmengine` or `openprompt_rs`
- seed
- GPU policy, including allowed devices, memory threshold, utilization
  threshold, stable-poll count, and excluded process names
- parent checkpoint or parent run id
- expected metrics
- startup acceptance regex
- completion markers
- failure signatures
- command template and environment overrides

Outputs:

- validated run spec object
- resolved command plan
- normalized provenance fields for registry and reporter modules

Why it exists:

Typed specs prevent manual `screen` commands from becoming undocumented
experiment state. The spec is the durable experiment intent.

Connections:

- Reads `New/configs/harness/*`
- References `OpenRSD` configs and checkpoints for MMEngine detector jobs
- References `New/configs/experiments/*` for local scaffold jobs

### `src/openprompt_rs/harness/registry.py`

Maintains a local run registry.

Inputs:

- validated spec
- lifecycle events from launcher and monitor
- metric summaries from parsers
- generated report paths from reporter

Outputs:

- append-only records in `records/harness/runs.jsonl`
- optional per-run event streams under `records/harness/events/`
- query helpers for `status`, `collect`, `report`, and `diagnose`

Why it exists:

One append-only registry makes it possible to answer "what ran, where, with
which config, and what happened" without searching screen sessions, shell
history, and logs by hand.

Connections:

- Supplies run metadata to `reporter.py`
- Supplies failure context to `diagnostics.py`
- Should be grep-friendly and recoverable after interrupted sessions

### `src/openprompt_rs/harness/gpu.py`

Wraps `nvidia-smi` GPU and process queries.

Inputs:

- allowed GPU list from a run spec
- idle memory and utilization thresholds
- process-name allow/deny policy

Outputs:

- physical GPU state: memory used, utilization, process list, process owners if
  available
- selected GPU for launch
- polling snapshots for event logs

Why it exists:

GPU availability must distinguish project training jobs from unrelated resident
processes such as `VLLM::EngineCore`. Simple memory thresholds are useful, but
they are not enough for lifecycle auditing or for avoiding accidental launches
onto occupied devices.

Connections:

- Extends the practical behavior of `scripts/gpu_queue_monitor.py`
- Feeds launch decisions in `launcher.py`
- Feeds liveness and residency checks in `monitor.py`

### `src/openprompt_rs/harness/launcher.py`

Launches jobs through `screen` or subprocess with controlled runtime settings.

Inputs:

- resolved run spec
- selected GPU
- command template
- environment overrides
- workdir and log path
- dry-run flag

Outputs:

- launch event: timestamp, command, screen name, PID if known, GPU, log path,
  workdir, environment summary
- detached `screen` session or foreground subprocess

Why it exists:

The current workflow relies on hand-written launch commands. The launcher should
replace those commands while preserving the server-compatible detached-screen
workflow already used for long detector training.

Connections:

- For OpenRSD detector jobs, launches from `OpenRSD/` and writes logs beside
  `OpenRSD/work_dirs/...` or configured external log paths
- For local scaffold jobs, launches `New/scripts/train.py` or
  `New/scripts/evaluate.py`
- Writes registry events through `registry.py`

### `src/openprompt_rs/harness/monitor.py`

Polls liveness, progress, outputs, and failure signatures.

Inputs:

- run id
- registry launch event
- log path
- workdir
- expected progress regex
- checkpoint expectations
- failure signatures
- GPU polling policy

Outputs:

- state transitions: `accepted`, `running`, `completed`, `failed`
- event snapshots under `records/harness/events/`
- failure context for diagnostics

Why it exists:

The monitor formalizes the manual checks currently used during server work:
process alive, screen active, GPU-active, no `Traceback`, no CUDA OOM, no
corrupt-image failures, no true non-finite values, and expected progress
markers.

Failure signatures should be scoped:

- `Traceback`
- CUDA OOM or out-of-memory variants
- `libpng`
- `CRC`
- `NoneType`
- `ValueError`
- `KeyboardInterrupt`
- true `nan` in losses or metrics
- true `inf` in losses or metrics

The `nan` and `inf` checks must avoid false positives from static text such as
`metainfo`.

Acceptance gates:

- Startup is accepted only after a configured progress regex, for example
  `Epoch(train) [1][  200/39007]`.
- Completion is accepted only after final metric extraction and an expected
  checkpoint marker.
- Failure is accepted immediately when a scoped failure signature appears or the
  process exits before startup acceptance.

Connections:

- Uses `gpu.py` for residency checks
- Uses `parsers.py` for progress and metrics
- Writes lifecycle events through `registry.py`
- Triggers `diagnostics.py` for failed runs

### `src/openprompt_rs/harness/parsers.py`

Parses logs, scalar files, and checkpoint markers.

Inputs:

- MMEngine logs from OpenRSD
- `vis_data/scalars.json`
- `last_checkpoint`
- checkpoint files such as `epoch_12.pth`
- local scaffold metric files such as `metrics.json`

Outputs:

- latest train progress
- validation metric history
- final key metrics
- class-wise DOTA metrics when available
- checkpoint summary
- compact failure excerpts

Why it exists:

Metrics should be extracted from structured files where possible. Logs should be
used for progress, command echo, and failure context rather than as the primary
metric source when `scalars.json` or metric JSON exists.

Connections:

- Feeds completion decisions in `monitor.py`
- Feeds generated experiment records in `reporter.py`
- Feeds metric curves in `diagnostics.py`

### `src/openprompt_rs/harness/reporter.py`

Generates compact Markdown experiment records.

Inputs:

- run registry record
- resolved spec
- parsed metrics
- parsed class-wise metrics
- environment snapshot
- command and validation command
- checkpoint and log paths
- failure notes

Outputs:

- `docs/experiments/YYYYMMDD_<run>.md`
- optional adjacent metric JSON summary

Why it exists:

The project already has a documented experiment-record standard. The reporter
enforces it and removes manual copy/paste errors from paper-facing evidence
collection.

Generated records should include every field listed in
`docs/experiments/README.md`. If a field is unavailable, the report should write
`not captured` rather than silently omitting it.

Connections:

- Reads from `registry.py` and `parsers.py`
- Writes into the existing `New/docs/experiments/` workflow
- Should preserve the policy that large checkpoints, raw logs, raw predictions,
  and datasets stay outside Git

### `src/openprompt_rs/harness/diagnostics.py`

Produces AI-readable diagnosis bundles.

Inputs:

- failed run registry event
- failed command
- tail logs
- config path and optional config diff
- GPU state at failure or latest poll
- parsed metric curve
- checkpoint state
- failure signatures

Outputs:

- compact JSON bundle under `artifacts/harness/diagnostics/`
- optional Markdown summary under `artifacts/harness/diagnostics/`
- suspected failure class
- suggested next investigation target

Why it exists:

An AI agent needs enough context to propose the next action without redoing all
manual investigation. The bundle should capture the exact run state, not just a
short error line.

Connections:

- Reads from `registry.py`, `parsers.py`, and `gpu.py`
- Can link to generated experiment records after `reporter.py` runs
- Should support diagnostic-only jobs such as DIOR-R geometry/target checks

### `scripts/harness.py`

Provides the user-facing CLI.

Subcommands:

- `validate-spec`: parse a YAML/JSON spec, resolve paths, and print a command
  plan without launching
- `launch`: select GPU, start the run, and append launch records
- `status`: print current lifecycle state, latest progress, GPU residency, and
  metric summary
- `monitor`: poll until completion or failure
- `collect`: parse metrics and checkpoint markers for an existing run
- `report`: generate the Markdown experiment record
- `diagnose`: generate diagnosis bundles for failed or suspicious runs

Why it exists:

One CLI keeps automation discoverable and avoids many single-purpose shell
scripts. It should also provide a dry-run mode so launch plans can be reviewed
without starting training.

Connections:

- Uses all harness modules under `src/openprompt_rs/harness/`
- Can eventually replace the launch-only role of `scripts/gpu_queue_monitor.py`
  while still reusing its conservative GPU-idle policy

## Data And File Structure

### `configs/harness/`

Stores declarative run specs. Examples:

- DOTA2 S1/S2 replications
- DOTA2 smoke runs
- DIOR-R diagnostics
- baseline collection jobs

Why:

Configs become the durable experiment intent, separate from generated runtime
files. They should be small enough to track in Git.

### `records/harness/runs.jsonl`

Append-only lifecycle registry.

Each line should contain:

- run id
- event type or current lifecycle state
- timestamp
- spec path and spec digest
- backend
- command
- workdir
- log path
- screen name and PID if known
- GPU assignment
- code commit
- config path and config digest
- dataset identity
- parent checkpoint or parent run
- metric summary when available
- report path when generated

Why:

JSONL is easy to append, grep, parse, back up, and recover after interrupted
sessions.

### `records/harness/events/`

Optional per-run event streams for polling snapshots and state transitions.

Why:

Detailed polling data should not clutter experiment notes, but it should remain
available for audits and failure diagnosis.

### `docs/experiments/YYYYMMDD_<run>.md`

Human-facing experiment summary generated from the registry, logs, and metrics.

Why:

This preserves the current documentation workflow while making it automatic.

### `artifacts/harness/`

Small generated summaries:

- metric tables
- config diffs
- diagnosis bundles
- command plans
- report previews

Why:

Generated but lightweight artifacts can be reviewed before deciding what belongs
in Git.

Large checkpoints, full logs, raw predictions, and datasets stay outside Git,
matching the existing policy.

## Minimal First Version

Start with a local-file harness around existing behavior:

1. Read a YAML or JSON run spec.
2. Validate required fields and resolve paths.
3. Build a command plan and support `--dry-run`.
4. Query GPUs with `nvidia-smi` and select an allowed idle GPU.
5. Launch a detached `screen` job with controlled `CUDA_VISIBLE_DEVICES`,
   `PYTHONNOUSERSITE`, `PYTHONPATH`, and `MPLCONFIGDIR` as needed.
6. Record screen name, PID when available, command, config, workdir, log path,
   selected GPU, timestamp, and code commit in `records/harness/runs.jsonl`.
7. Poll until completion or failure.
8. Extract DOTA metrics from `vis_data/scalars.json` where available.
9. Generate `docs/experiments/YYYYMMDD_<run>.md` with all required record
   fields.

The first version should target the active stage-gated GeoNexus route. It should
not open S3/S4, pseudo-labeling, FAIR1M, or DIOR-R detector training unless the
project gate explicitly allows those launches. DIOR-R diagnostic utilities can
be represented as diagnostic runs.

## Acceptance Gates

Startup acceptance:

- process or screen exists
- log file exists and is growing
- configured progress regex appears, such as
  `Epoch(train) [1][  200/39007]`

Runtime health:

- PID remains alive
- screen remains listed for detached jobs
- selected GPU shows expected process residency when GPU training is required
- log cursor advances within a configured timeout
- no scoped failure signature appears

Completion acceptance:

- process exits normally or expected final marker appears
- final metric can be parsed
- expected checkpoint marker exists, such as `last_checkpoint` or final epoch
  checkpoint
- failure scan remains clean

Failure detection:

- `Traceback`
- CUDA OOM
- `libpng`
- `CRC`
- `NoneType`
- `ValueError`
- `KeyboardInterrupt`
- true `nan`
- true `inf`
- early exit before startup acceptance
- no progress for a configured timeout

## Testing Plan

Add focused tests before using the harness for paper-facing launches:

- run spec validation tests for required fields, path resolution, GPU policy,
  parent checkpoint fields, and failure signatures
- parser tests using small fixture MMEngine logs and `scalars.json` snippets
- GPU parser tests using captured `nvidia-smi` output, including unrelated
  processes such as `VLLM::EngineCore`
- reporter tests comparing generated Markdown against the required experiment
  record fields from `docs/experiments/README.md`
- dry-run launcher tests that print command plans without starting training
- monitor tests for startup acceptance, true failure signatures, and false
  positive avoidance for text such as `metainfo`

The tests should avoid requiring a live GPU. Live GPU checks can be smoke tests
run manually on the server.

## Example Spec Shape

```yaml
run_id: dota2_s2_loss0_rep4407
stage: S2
backend: opensrd_mmengine
dataset:
  name: DOTA
  version: DOTA2_1024_500
  root: /data5/2025/temp/Dataset/DOTA2_1024_500
  split: ss_val
config: /data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/example_config.py
workdir: /data5/2025/ldh/OpenRSD
command:
  argv:
    - /data1/anaconda3/envs/zwl_mmrotate/bin/python
    - tools/bootstrap_run.py
    - tools/train.py
    - /data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/example_config.py
    - --work-dir
    - /data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/dota2_s2_loss0_rep4407
gpu_policy:
  allowed: [0, 1, 2, 3, 4, 5, 6]
  idle_memory_mib: 1000
  idle_util_percent: 10
  stable_polls: 3
  exclude_process_names:
    - VLLM::EngineCore
startup:
  progress_regex: "Epoch\\(train\\) \\[1\\].*39007"
completion:
  require_metric_keys:
    - dota/mAP
    - dota/AP50
  require_checkpoint: last_checkpoint
failure_signatures:
  - Traceback
  - CUDA out of memory
  - libpng
  - CRC
  - NoneType
  - ValueError
  - KeyboardInterrupt
  - true_nan
  - true_inf
provenance:
  parent_run: dota2_s1_main
  parent_checkpoint: /data5/2025/ldh/OpenRSD/work_dirs/geonexus_dota2/roi_trans_remoteclip_s1/epoch_12.pth
  metric_implementation: MMEngine DOTA evaluator
  embedding_backend: RemoteCLIP ViT-B-32
```

The exact schema can evolve, but it should remain explicit enough to regenerate
the command, explain the result, and connect the result to a parent run.

## Implementation Order

1. Create the package directory `src/openprompt_rs/harness/` with empty module
   boundaries and tests.
2. Implement `specs.py` and `scripts/harness.py validate-spec --dry-run`.
3. Implement `registry.py` append and query helpers.
4. Implement `parsers.py` for `scalars.json`, local `metrics.json`, checkpoint
   markers, and compact log tails.
5. Implement `reporter.py` against the current experiment-record template.
6. Implement `gpu.py` with injectable command output for tests.
7. Implement `launcher.py` with dry-run first, then detached-screen launch.
8. Implement `monitor.py` with startup acceptance and scoped failure detection.
9. Implement `diagnostics.py` for failed-run bundles.
10. Migrate the current queue-launch use case from `gpu_queue_monitor.py` only
    after the harness can perform equivalent conservative idle detection.

## Non-Goals For Version One

- No external MLflow, W&B, Prefect, Airflow, database, or dashboard dependency.
- No replacement of OpenRSD/MMEngine detector training.
- No automatic S3/S4, pseudo-labeling, FAIR1M, or DIOR-R detector launch unless
  the project gate explicitly approves those stages.
- No tracking of large checkpoints, full logs, raw predictions, or datasets in
  Git.
- No broad refactor of existing `New/scripts/train.py` or `New/scripts/evaluate.py`.

## Success Criteria

The harness is useful when a user can run one command to validate and launch a
spec, leave it detached on the server, and later run one command to obtain:

- current lifecycle state
- exact command and environment
- GPU and process history
- final metric summary
- checkpoint and log paths
- clean failure scan or diagnosis bundle
- complete Markdown experiment record under `docs/experiments/`

At that point, paper-facing detector runs become reproducible and auditable
without replacing the training backend that produced the metrics.
