# Experiment Records

Store small, Git-tracked experiment records here. Large outputs stay outside
Git in `outputs/`, `checkpoints/`, or server-local storage.

Each experiment record should include:

- date
- Git commit hash
- machine/server name
- GPU type and count
- dataset path and split
- config path
- exact command
- checkpoint path outside Git
- log path outside Git
- key metrics
- failure notes
- next action

Suggested filename:

```text
YYYYMMDD_short_experiment_name.md
```

