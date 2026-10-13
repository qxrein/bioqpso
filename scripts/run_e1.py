#!/usr/bin/env python3
"""Run E1 (classical benchmark) in parallel using joblib."""
import json, os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import yaml
from joblib import Parallel, delayed

from bioqpso.runner import run_job
from bioqpso.benchmarks import get_classical_problem

# ── Config ────────────────────────────────────────────────────
N_PARTICLES = 30
MAX_ITER    = 1000
MAX_FES     = N_PARTICLES * (MAX_ITER + 1)   # 30 030
N_JOBS      = 8    # parallel workers
OUT_DIR     = "results/e1_classical"
os.makedirs(OUT_DIR, exist_ok=True)

with open("config/seeds.yaml") as f:
    _s = yaml.safe_load(f)
SEEDS = [int(s) for s in _s["seeds"]]
N_RUNS = len(SEEDS)   # 30

ALGORITHMS = {
    "pso":              {},
    "qpso":             {},
    "aco_r":            {},
    "abc":              {},
    "gwo":              {},
    "woa":              {},
    "ant-fixed":        {"phi3": 0.3},
    "bee-fixed":        {"phi3": 0.3},
    "ant-linear":       {},
    "ant-cosine":       {},
    "ant-feedback":     {"W": 20, "delta": 0.02},
    "bee-per_particle": {},
}

FUNCTIONS = [
    "sphere","rosenbrock","zakharov","dixon_price","rastrigin",
    "ackley","griewank","schwefel","levy","styblinski_tang",
]

# ── Job helpers ───────────────────────────────────────────────
def _path(algo, fn, run):
    return os.path.join(OUT_DIR, f"{algo}__{fn}__d30__r{run:03d}.json")

def _done(algo, fn, run):
    p = _path(algo, fn, run)
    if not os.path.exists(p):
        return False
    try:
        with open(p) as f: d = json.load(f)
        return "final_f" in d
    except Exception:
        return False

def _serial(obj):
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, np.generic):  return obj.item()
    if isinstance(obj, dict):        return {k: _serial(v) for k,v in obj.items()}
    if isinstance(obj, (list,tuple)):return [_serial(v) for v in obj]
    return obj

def execute(algo, fn, run_idx):
    seed = SEEDS[run_idx]
    prob = get_classical_problem(fn, dim=30)
    rec  = run_job(algo, prob, run_idx=run_idx, seed=seed,
                   n_particles=N_PARTICLES, max_iter=MAX_ITER,
                   max_fes=MAX_FES, extra=ALGORITHMS[algo])
    with open(_path(algo, fn, run_idx), "w") as f:
        json.dump(_serial(rec), f)
    return (algo, fn, run_idx, rec["final_f"], rec["n_fes"])

# ── Build job list ────────────────────────────────────────────
jobs = [
    (algo, fn, r)
    for algo in ALGORITHMS
    for fn   in FUNCTIONS
    for r    in range(N_RUNS)
    if not _done(algo, fn, r)
]
print(f"E1: {len(jobs)} pending jobs  ({N_PARTICLES}p × {MAX_ITER}T × {N_RUNS}runs × "
      f"{len(FUNCTIONS)}fns × {len(ALGORITHMS)}algos = "
      f"{N_PARTICLES*(MAX_ITER+1)*N_RUNS*len(FUNCTIONS)*len(ALGORITHMS):,} total FEs)")
print(f"Estimated runtime: {len(jobs)*0.7/N_JOBS/60:.1f} min at {N_JOBS} workers")

t0 = time.time()
results = Parallel(n_jobs=N_JOBS, verbose=2)(
    delayed(execute)(a, fn, r) for a, fn, r in jobs
)
elapsed = time.time() - t0
print(f"\nE1 done: {len(results)} jobs in {elapsed/60:.1f} min")
