#!/usr/bin/env python3
"""Run E3a (CEC2017 light budget, D=30) in parallel."""
import json, os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import yaml
from joblib import Parallel, delayed

import opfunu
from bioqpso.runner import run_job
from bioqpso.benchmarks.cec2017 import CEC2017_CLASSES, get_cec_problem

N_PARTICLES = 30
MAX_ITER    = 1000
MAX_FES     = N_PARTICLES * (MAX_ITER + 1)   # 30 030  (light tier)
DIM         = 30
N_JOBS      = 9
OUT_DIR     = "results/e3_cec2017_d30"
os.makedirs(OUT_DIR, exist_ok=True)

with open("config/seeds.yaml") as f:
    _s = yaml.safe_load(f)
SEEDS  = [int(s) for s in _s["seeds"]]
N_RUNS = len(SEEDS)

ALGORITHMS = {
    "lshade":           {},
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

def _path(algo, fn, run):
    return os.path.join(OUT_DIR, f"{algo}__{fn}__d{DIM}__r{run:03d}.json")

def _done(algo, fn, run):
    p = _path(algo, fn, run)
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

def execute(algo, fcls_name, run_idx):
    seed  = SEEDS[run_idx]
    fcls  = next(f for f in CEC2017_CLASSES if f.__name__ == fcls_name)
    prob  = get_cec_problem(fcls, DIM)
    extra = ALGORITHMS[algo]
    rec   = run_job(algo, prob, run_idx=run_idx, seed=seed,
                    n_particles=N_PARTICLES, max_iter=MAX_ITER,
                    max_fes=MAX_FES, extra=extra)
    with open(_path(algo, fcls_name, run_idx), "w") as f:
        json.dump(_serial(rec), f)
    return (algo, fcls_name, run_idx, rec["final_f"])

CEC_NAMES = [f.__name__ for f in CEC2017_CLASSES]   # 28 (F5 excluded)
jobs = [
    (algo, fn, r)
    for algo in ALGORITHMS
    for fn   in CEC_NAMES
    for r    in range(N_RUNS)
    if not _done(algo, fn, r)
]
print(f"E3a: {len(jobs)} pending jobs  "
      f"({len(ALGORITHMS)} algos × {len(CEC_NAMES)} funcs × {N_RUNS} runs)")
print(f"Estimated: {len(jobs)*0.7/N_JOBS/60:.1f} min at {N_JOBS} workers")

t0 = time.time()
results = Parallel(n_jobs=N_JOBS, verbose=2)(
    delayed(execute)(a,fn,r) for a,fn,r in jobs
)
print(f"\nE3a done: {len(results)} jobs in {(time.time()-t0)/60:.1f} min")
