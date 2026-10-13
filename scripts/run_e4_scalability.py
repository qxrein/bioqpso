#!/usr/bin/env python3
"""E4 — Scalability: D in {30,50,100}, 6 functions, 30 runs, FE=10000*D."""
import json, os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, yaml
from joblib import Parallel, delayed
from bioqpso.runner import run_job
from bioqpso.benchmarks import get_classical_problem

N_PARTICLES = 30; N_JOBS = 4; OUT_DIR = "results/e4_scalability"
os.makedirs(OUT_DIR, exist_ok=True)
with open("config/seeds.yaml") as f: SEEDS = [int(s) for s in yaml.safe_load(f)["seeds"]]
N_RUNS = len(SEEDS)

DIMS = [30, 50, 100]
FUNCTIONS = ["sphere","rastrigin","rosenbrock","levy","schwefel","griewank"]
ALGORITHMS = {
    "qpso":             {}, "gwo": {},
    "ant-fixed":        {"phi3": 0.3}, "ant-cosine": {},
    "ant-feedback":     {"W": 20, "delta": 0.02},
    "bee-fixed":        {"phi3": 0.3}, "bee-per_particle": {},
}

def _path(algo, fn, dim, run):
    return os.path.join(OUT_DIR, f"{algo}__{fn}__d{dim}__r{run:03d}.json")

def _done(algo, fn, dim, run):
    p = _path(algo, fn, dim, run)
    if not os.path.exists(p): return False
    try:
        with open(p) as f: d = json.load(f)
        return "final_f" in d
    except: return False

def _serial(o):
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, np.generic): return o.item()
    if isinstance(o, dict): return {k:_serial(v) for k,v in o.items()}
    if isinstance(o, (list,tuple)): return [_serial(v) for v in o]
    return o

def execute(algo, fn, dim, run_idx):
    max_fes = max(10000 * dim, 100000)
    max_iter = max_fes // N_PARTICLES
    prob = get_classical_problem(fn, dim=dim)
    extra = ALGORITHMS[algo]
    rec = run_job(algo, prob, run_idx=run_idx, seed=SEEDS[run_idx],
                  n_particles=N_PARTICLES, max_iter=max_iter,
                  max_fes=max_fes, extra=extra)
    with open(_path(algo, fn, dim, run_idx), "w") as f:
        json.dump(_serial(rec), f)
    return (algo, fn, dim, run_idx, rec["final_f"])

jobs = [(a,fn,d,r) for a in ALGORITHMS for fn in FUNCTIONS
        for d in DIMS for r in range(N_RUNS) if not _done(a,fn,d,r)]
print(f"E4: {len(jobs)} jobs"); t0=time.time()
Parallel(n_jobs=N_JOBS, verbose=2)(delayed(execute)(*j) for j in jobs)
print(f"E4 done in {(time.time()-t0)/60:.1f} min")
