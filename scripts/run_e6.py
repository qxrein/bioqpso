#!/usr/bin/env python3
"""E6 — one-at-a-time sensitivity, 30 runs, 30,030 FEs, 6 classical + 4 CEC2017 functions.
CEC functions fixed in advance: one per class (unimodal F1, multimodal F4, hybrid F11,
composition F21; opfunu ids)."""
from _runlib import *
from bioqpso.runner import run_job
from bioqpso.benchmarks import get_classical_problem
from bioqpso.benchmarks.cec2017 import CEC2017_CLASSES, get_cec_problem
OUT = "results/e6_sensitivity"; os.makedirs(OUT, exist_ok=True)
CL = ["sphere", "rastrigin", "levy", "styblinski_tang", "schwefel", "griewank"]
CE = ["F12017", "F42017", "F112017", "F212017"]
FB = {"W": 20, "delta": 0.02}
SWEEPS = {
    "rho":    ("ant-fixed", "evaporation_rate", [0.05, 0.1, 0.2, 0.3, 0.5], {"phi3": 0.3}),
    "Delta":  ("ant-fixed", "pheromone_deposit", [0.5, 1.0, 2.0, 5.0, 10.0], {"phi3": 0.3}),
    "Lscout": ("bee-fixed", "L_scout", [5, 10, 15, 30, 60], {"phi3": 0.3}),
    "W":      ("ant-feedback", "W", [5, 10, 20, 30, 50], FB),
    "delta":  ("ant-feedback", "delta", [0.01, 0.02, 0.05, 0.1, 0.2], FB),
}
PAIRS = {
    "phi3bounds": ("ant-cosine", ("phi3_min", "phi3_max"),
                   [(0.05, 0.6), (0.1, 0.5), (0.15, 0.5), (0.1, 0.4), (0.2, 0.4)], {}),
    "beta":       ("ant-fixed", ("beta_init", "beta_final"),
                   [(1.0, 0.5), (1.0, 0.3), (1.0, 0.75), (0.8, 0.5), (1.2, 0.5)], {"phi3": 0.3}),
}
CFG = []
for k, (a, p, vals, base) in SWEEPS.items():
    CFG += [(k, str(v), a, {**base, p: v}) for v in vals]
for k, (a, ps, vals, base) in PAIRS.items():
    CFG += [(k, f"{v[0]}-{v[1]}", a, {**base, ps[0]: v[0], ps[1]: v[1]}) for v in vals]
def job(path, k, val, a, extra, fn, r):
    if fn in CE:
        p = get_cec_problem(next(c for c in CEC2017_CLASSES if c.__name__ == fn), 30)
    else:
        p = get_classical_problem(fn, 30)
    rec = run_job(a, p, r, SEEDS[r], max_fes=30030, extra=extra)
    rec.update(param=k, value=val, phi3_hist=None, convergence=None)
    write_json(path, rec)
if __name__ == "__main__":
    run_jobs("E6", job, [(f"{OUT}/{k}__{v}__{fn}__r{r:03d}.json", (k, v, a, e, fn, r))
                         for k, v, a, e in CFG for fn in CL + CE for r in range(30)])
