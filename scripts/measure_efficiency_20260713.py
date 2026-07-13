#!/usr/bin/env python3
"""Measure params/GFLOPs/FPS for the TGRS efficiency table (job A3).

Feeds Table VI (tab:efficiency). Must run on the server inside the MMRotate
environment on an idle GPU:

    CUDA_VISIBLE_DEVICES=0 python scripts/measure_efficiency_20260713.py \
        --config <config .py> --ckpt <checkpoint .pth> --tag baseline \
        --image <any 1024x1024 test tile> --out efficiency_baseline.json

Run once for the DIOR-R baseline (RoI Transformer epoch-52 config/ckpt) and
once for GeoNexus-RSD (scene-adapter rep0 config/ckpt). FPS is the median
over 200 timed forward passes after 20 warm-up passes, batch size 1. Record
the GPU model in the JSON so the manuscript caption can state the hardware.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--image", required=True, help="Path to a representative 1024x1024 tile")
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    import torch
    from mmdet.apis import inference_detector, init_detector

    model = init_detector(args.config, args.ckpt, device="cuda:0")
    n_params = sum(p.numel() for p in model.parameters())

    gflops = None
    try:
        from mmengine.analysis import get_model_complexity_info

        analysis = get_model_complexity_info(model, input_shape=(3, 1024, 1024))
        gflops = analysis["flops"] / 1e9 if isinstance(analysis.get("flops"), (int, float)) else analysis.get("flops_str")
    except Exception as exc:  # complexity analysis is best-effort; FPS/params still valid
        print(f"WARNING: FLOPs analysis failed ({type(exc).__name__}: {exc}); record FLOPs manually if required")

    for _ in range(args.warmup):
        inference_detector(model, args.image)
    torch.cuda.synchronize()

    times = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        inference_detector(model, args.image)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    median_s = statistics.median(times)
    result = {
        "tag": args.tag,
        "config": args.config,
        "ckpt": args.ckpt,
        "gpu": torch.cuda.get_device_name(0),
        "params_M": round(n_params / 1e6, 2),
        "gflops": gflops,
        "fps_median": round(1.0 / median_s, 2),
        "latency_ms_median": round(1000.0 * median_s, 2),
        "runs": args.runs,
    }
    print(json.dumps(result, indent=2))
    out = args.out or Path(f"efficiency_{args.tag}.json")
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
