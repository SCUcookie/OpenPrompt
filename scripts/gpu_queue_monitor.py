#!/usr/bin/env python3
"""Launch queued single-GPU jobs when selected GPUs become idle.

Queue file format: JSON list of jobs. Each job supports:
  name: screen session/job name
  workdir: directory to run from
  command: shell command to execute after CUDA_VISIBLE_DEVICES is set
  log: path for stdout/stderr
  gpus: optional list of allowed physical GPU ids
  wait_for: optional path that must exist before the job can launch

The script intentionally uses conservative idle detection: a GPU must be below
the memory/util thresholds for N consecutive polls before a job is launched.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log_line(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{now()}] {message}\n")


def query_gpus() -> dict[int, dict[str, int]]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    output = subprocess.check_output(cmd, text=True)
    result: dict[int, dict[str, int]] = {}
    for raw_line in output.strip().splitlines():
        if not raw_line.strip():
            continue
        idx_s, mem_s, util_s = [part.strip() for part in raw_line.split(",")]
        result[int(idx_s)] = {"mem": int(mem_s), "util": int(util_s)}
    return result


def load_jobs(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        jobs = json.load(handle)
    if not isinstance(jobs, list):
        raise ValueError("queue file must contain a JSON list")
    for job in jobs:
        for key in ("name", "workdir", "command", "log"):
            if key not in job:
                raise ValueError(f"job is missing required key: {key}")
    return jobs


def save_jobs(path: Path, jobs: list[dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(jobs, handle, indent=2)
        handle.write("\n")
    tmp.replace(path)


def launch(job: dict[str, Any], gpu: int, monitor_log: Path) -> None:
    name = str(job["name"])
    workdir = Path(job["workdir"])
    log = Path(job["log"])
    log.parent.mkdir(parents=True, exist_ok=True)
    command = (
        f"cd {shell_quote(str(workdir))} && "
        f"CUDA_VISIBLE_DEVICES={gpu} MPLCONFIGDIR=/tmp/matplotlib_{name} "
        f"{job['command']} >> {shell_quote(str(log))} 2>&1"
    )
    subprocess.check_call(["screen", "-dmS", name, "bash", "-lc", command])
    log_line(monitor_log, f"launched {name} on GPU {gpu}; log={log}")


def shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", required=True, type=Path)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--gpus", default="1,2,3,4,5,6")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--idle-memory-mib", type=int, default=1000)
    parser.add_argument("--idle-util-percent", type=int, default=10)
    parser.add_argument("--stable-polls", type=int, default=3)
    args = parser.parse_args()

    allowed_gpus = [int(item) for item in args.gpus.split(",") if item.strip()]
    idle_counts = {gpu: 0 for gpu in allowed_gpus}
    log_line(args.log, f"monitor start; queue={args.queue}; gpus={allowed_gpus}")

    while True:
        try:
            jobs = load_jobs(args.queue)
            pending = [job for job in jobs if job.get("status", "pending") == "pending"]
            if not pending:
                log_line(args.log, "queue empty; exiting")
                return 0

            gpu_state = query_gpus()
            for gpu in allowed_gpus:
                state = gpu_state.get(gpu)
                if (
                    state
                    and state["mem"] <= args.idle_memory_mib
                    and state["util"] <= args.idle_util_percent
                ):
                    idle_counts[gpu] += 1
                else:
                    idle_counts[gpu] = 0

            state_text = ", ".join(
                f"{gpu}:mem={gpu_state.get(gpu, {}).get('mem', -1)}"
                f",util={gpu_state.get(gpu, {}).get('util', -1)}"
                f",idle={idle_counts[gpu]}"
                for gpu in allowed_gpus
            )
            log_line(args.log, f"poll; {state_text}; pending={len(pending)}")

            used_this_poll: set[int] = set()
            for job in pending:
                wait_for = job.get("wait_for")
                if wait_for and not Path(wait_for).exists():
                    continue
                job_gpus = [int(gpu) for gpu in job.get("gpus", allowed_gpus)]
                for gpu in job_gpus:
                    if gpu in used_this_poll:
                        continue
                    if idle_counts.get(gpu, 0) < args.stable_polls:
                        continue
                    launch(job, gpu, args.log)
                    job["status"] = "launched"
                    job["launched_at"] = now()
                    job["gpu"] = gpu
                    save_jobs(args.queue, jobs)
                    idle_counts[gpu] = 0
                    used_this_poll.add(gpu)
                    break

            time.sleep(args.poll_seconds)
        except KeyboardInterrupt:
            log_line(args.log, "monitor stopped by KeyboardInterrupt")
            return 130
        except Exception as exc:  # keep the monitor alive across transient nvidia-smi errors
            log_line(args.log, f"error: {exc!r}")
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    sys.exit(main())
