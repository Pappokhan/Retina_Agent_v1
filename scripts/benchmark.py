#!/usr/bin/env python
"""Measure latency and throughput on THIS machine (nothing is assumed).

    python scripts/benchmark.py --dataset A --variant best --runs 30
    python scripts/benchmark.py --dataset A --image sample.png
"""
import argparse
import statistics as st
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import config  # noqa: E402
from inference import ModelRegistry  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="A")
    ap.add_argument("--variant", default="best")
    ap.add_argument("--image", type=Path, help="real image to use (default: random noise, same cost)")
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()

    img = (Image.open(args.image).convert("RGB") if args.image else
           Image.fromarray(np.random.default_rng(0).integers(0, 255, (512, 512, 3), dtype=np.uint8)))
    reg = ModelRegistry()
    t0 = time.perf_counter()
    reg.get(args.dataset, args.variant)
    print(f"device: {reg.device} | model load: {(time.perf_counter() - t0) * 1000:.0f} ms")
    for _ in range(args.warmup):
        reg.predict(img, args.dataset, args.variant)

    total, fwd = [], []
    for _ in range(args.runs):
        t = time.perf_counter()
        p = reg.predict(img, args.dataset, args.variant)
        total.append((time.perf_counter() - t) * 1000)
        fwd.append(p.ms)

    def line(name, xs):
        xs = sorted(xs)
        p95 = xs[min(len(xs) - 1, int(round(0.95 * (len(xs) - 1))))]
        print(f"{name:<22} mean {st.mean(xs):7.1f} ms | p50 {st.median(xs):7.1f} | p95 {p95:7.1f}")

    line("forward pass", fwd)
    line("preprocess + forward", total)
    print(f"throughput (one image at a time): {1000 / st.mean(total):.2f} images/s over {args.runs} runs")
    print("Live camera adds JPEG encode/decode and the network round trip on top of this.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
