#!/usr/bin/env python3
"""Run E2 (controls: schedule vs fixed phi3) in parallel."""
import json, os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import yaml
from joblib import Parallel, delayed

from bioqpso.runner import run_job
from bioqpso.benchmarks import get_classical_problem

N_PARTICLES = 30
MAX_ITER    = 1000
MAX_FES     = N_PARTICLES * (MAX_ITER + 1)
N_JOBS      = 8
OUT_DIR     = "results/e2_controls"
os.makedirs(OUT_DIR, exist_ok=True)

with open("config/seeds.yaml") as f:
    _s = yaml.safe_load(f)
SEEDS  = [int(s) for s in _s["seeds"]]
N_RUNS = len(SEEDS)

# Full factorial: both leaders × all schedules/controls.
# Each entry: display_name -> (runner_algo_name, extra_params)
# display_name is used for filenames; runner_algo_name is what _make_algorithm sees.
ALGORITHMS = {
    # baselines
    "qpso":                  ("qpso",         {}),
    # nobio controls
    "ant-nobio":             ("ant-nobio",     {}),
    "bee-nobio":             ("bee-nobio",     {}),
    # fixed phi3 sweep (6 values × 2 leaders)
    "ant-fixed-p00":         ("ant-fixed",     {"phi3": 0.0}),
    "ant-fixed-p01":         ("ant-fixed",     {"phi3": 0.1}),
    "ant-fixed-p02":         ("ant-fixed",     {"phi3": 0.2}),
    "ant-fixed-p03":         ("ant-fixed",     {"phi3": 0.3}),
    "ant-fixed-p04":         ("ant-fixed",     {"phi3": 0.4}),
    "ant-fixed-p05":         ("ant-fixed",     {"phi3": 0.5}),
    "bee-fixed-p00":         ("bee-fixed",     {"phi3": 0.0}),
    "bee-fixed-p01":         ("bee-fixed",     {"phi3": 0.1}),
    "bee-fixed-p02":         ("bee-fixed",     {"phi3": 0.2}),
    "bee-fixed-p03":         ("bee-fixed",     {"phi3": 0.3}),
    "bee-fixed-p04":         ("bee-fixed",     {"phi3": 0.4}),
    "bee-fixed-p05":         ("bee-fixed",     {"phi3": 0.5}),
    # random phi3 (variation-only control)
    "ant-random":            ("ant-random",    {}),
    "bee-random":            ("bee-random",    {}),
    # scheduled variants (full factorial: both leaders)
    "ant-linear":            ("ant-linear",    {}),
    "ant-cosine":            ("ant-cosine",    {}),
    "ant-feedback":          ("ant-feedback",  {"W": 20, "delta": 0.02}),
    "bee-linear":            ("bee-linear",    {}),
    "bee-cosine":            ("bee-cosine",    {}),
    "bee-feedback":          ("bee-feedback",  {"W": 20, "delta": 0.02}),
    "bee-per_particle":      ("bee-per_particle", {}),
}

# E2 uses same 10 functions as E1
FUNCTIONS = [
    "sphere","rosenbrock","zakharov","dixon_price","rastrigin",
    "ackley","griewank","schwefel","levy","styblinski_tang",
]

def _path(display_name, fn, run):
    return os.path.join(OUT_DIR, f"{display_name}__{fn}__d30__r{run:03d}.json")

def _done(display_name, fn, run):
    p = _path(display_name, fn, run)
    if not os.path.exists(p): return False
    try:
        with open(p) as f: d = json.load(f)
        return "final_f" in d
    except Exception: return False

def _serial(obj):
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, np.generic):  return obj.item()
    if isinstance(obj, dict):        return {k:_serial(v) for k,v in obj.items()}
    if isinstance(obj, (list,tuple)):return [_serial(v) for v in obj]
    return obj

def execute(display_name, fn, run_idx):
    runner_name, extra = ALGORITHMS[display_name]
    seed = SEEDS[run_idx]
    prob = get_classical_problem(fn, dim=30)
    rec  = run_job(runner_name, prob, run_idx=run_idx, seed=seed,
                   n_particles=N_PARTICLES, max_iter=MAX_ITER,
                   max_fes=MAX_FES, extra=extra)
    # store display_name as algorithm so analysis can use it
    rec["algorithm"] = display_name
    with open(_path(display_name, fn, run_idx), "w") as f:
        json.dump(_serial(rec), f)
    return (display_name, fn, run_idx, rec["final_f"], rec["n_fes"])

jobs = [
    (display, fn, r)
    for display in ALGORITHMS
    for fn      in FUNCTIONS
    for r       in range(N_RUNS)
    if not _done(display, fn, r)
]
print(f"E2: {len(jobs)} pending jobs")
print(f"Estimated: {len(jobs)*0.7/N_JOBS/60:.1f} min at {N_JOBS} workers")

t0 = time.time()
results = Parallel(n_jobs=N_JOBS, verbose=2)(
    delayed(execute)(d,fn,r) for d,fn,r in jobs
)
print(f"\nE2 done: {len(results)} jobs in {(time.time()-t0)/60:.1f} min")
