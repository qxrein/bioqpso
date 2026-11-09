#!/usr/bin/env python3
"""E6 on the official CEC2017 code: the CEC part of E6 only (official F1, F4, F11, F21), with the
settings, seeds and budget of scripts/run_e6.py (30 runs, 30,030 FEs, D=30)."""
from _runlib import *
from bioqpso.runner import run_job
from bioqpso.benchmarks.cec2017_official import get_official_problem
from run_e6 import CFG  # noqa: E402  (the same parameter grid; importing does not run E6)
OUT = "results/e6_official"; os.makedirs(OUT, exist_ok=True)
CE = ["F1", "F4", "F11", "F21"]
def job(path, k, val, a, extra, fn, r):
    rec = run_job(a, get_official_problem(fn, 30), r, SEEDS[r], max_fes=30030, extra=extra)
    rec.update(param=k, value=val, phi3_hist=None, convergence=None)
    write_json(path, rec)
if __name__ == "__main__":
    run_jobs("E6-official", job, [(f"{OUT}/{k}__{v}__{fn}__r{r:03d}.json", (k, v, a, e, fn, r))
                                  for k, v, a, e in CFG for fn in CE for r in range(30)])
