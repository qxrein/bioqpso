#!/usr/bin/env python3
"""E3s — CEC2017 at the standard budget (10,000 D FEs). Usage: run_e3s.py D [algo,algo,...]
n=30, T = FEs/30, 30 runs. History every 100 iterations; phi3 traces are not stored."""
from _runlib import *
from bioqpso.runner import run_job
from bioqpso.benchmarks.cec2017 import CEC2017_CLASSES, get_cec_problem
D = int(sys.argv[1])
ALGOS = sys.argv[2].split(",") if len(sys.argv) > 2 else list(ALL13)
OUT = f"results/e3s_cec2017_d{D}"; os.makedirs(OUT, exist_ok=True)
FES = 10000 * D
def job(path, a, cname, r):
    fc = next(c for c in CEC2017_CLASSES if c.__name__ == cname)
    rec = run_job(a, get_cec_problem(fc, D), r, SEEDS[r], n_particles=30, max_iter=FES // 30,
                  max_fes=FES, hist_every=100, extra=ALL13[a])
    rec["phi3_hist"] = None
    write_json(path, rec)
# Cheapest functions first so that partial progress covers all algorithms evenly.
run_jobs(f"E3s-D{D}", job, [(f"{OUT}/{a}__{c.__name__}__d{D}__r{r:03d}.json", (a, c.__name__, r))
                            for r in range(30) for c in CEC2017_CLASSES for a in ALGOS])
