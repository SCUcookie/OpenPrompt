# Experiment Records

Store small, Git-tracked experiment records here. Large outputs stay outside
Git in `outputs/`, `checkpoints/`, or server-local storage.

Each experiment record should include:

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
- embedding backend, if the run touches prompt/VLM modules
- key metrics
- class-wise metrics for DOTA-style detection
- failure notes
- next action

For paper-facing records, do not mix DOTA v1.0, DOTA v1.5, and DOTA v2
numbers in one comparison table unless the table explicitly separates dataset
versions. Near-zero scaffold detector runs are pipeline sanity checks only and
should not be used as strong detector evidence.

Suggested filename:

```text
YYYYMMDD_short_experiment_name.md
```
