#!/usr/bin/env python3
"""E1b — classical functions with the optimum moved to o ~ U(0.8 lb, 0.8 ub)^D per run."""
from _runlib import *
from bioqpso.runner import run_job
from bioqpso.benchmarks.classical import get_shifted_problem
OUT = "results/e1b_shifted"; os.makedirs(OUT, exist_ok=True)
def job(path, a, fn, r):
    p = get_shifted_problem(fn, 30, SEEDS[r])
    rec = run_job(a, p, r, SEEDS[r], max_fes=30030, extra=ALL13[a])
    rec["shift"] = p["shift"]; rec["phi3_hist"] = None
    write_json(path, rec)
run_jobs("E1b", job, [(f"{OUT}/{a}__{f}__d30__r{r:03d}.json", (a, f, r))
                      for a in ALL13 for f in CLASSICAL for r in range(30)])
