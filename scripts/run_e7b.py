#!/usr/bin/env python3
"""E7b — Schwefel mechanism: optimum near the bound (c=420.97) vs. moved inward by 300
(c=120.97), with clip and reflect boundary handling, all 13 algorithms."""
from _runlib import *
from bioqpso.runner import run_job
from bioqpso.benchmarks.classical import get_schwefel_at
OUT = "results/e7b_schwefel_shift"; os.makedirs(OUT, exist_ok=True)
CS = {"near": 420.9687, "inward": 120.9687}
ALGOS = {**ALL13, "ant-nobio": {}}
def job(path, a, tag, bnd, r):
    rec = run_job(a, get_schwefel_at(CS[tag]), r, SEEDS[r], max_fes=30030,
                  boundary_mode=bnd, extra=ALGOS[a])
    rec.update(optimum=tag, boundary=bnd, phi3_hist=None)
    write_json(path, rec)
run_jobs("E7b", job, [(f"{OUT}/{a}__{t}__{b}__r{r:03d}.json", (a, t, b, r))
                      for a in ALGOS for t in CS for b in ("clip", "reflect") for r in range(30)])
