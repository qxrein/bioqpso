#!/usr/bin/env python3
"""E7 — Schwefel diagnosis: boundary mode × algorithm, 30 runs."""
import json, os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, yaml
from joblib import Parallel, delayed
from bioqpso.runner import run_job
from bioqpso.benchmarks.classical import schwefel

N_PARTICLES = 30; MAX_ITER = 1000; N_JOBS = 8
OUT_DIR = "results/e7_schwefel"; os.makedirs(OUT_DIR, exist_ok=True)
with open("config/seeds.yaml") as f: SEEDS = [int(s) for s in yaml.safe_load(f)["seeds"]]
N_RUNS = len(SEEDS)

ALGORITHMS = {"qpso":{}, "ant-nobio":{}, "ant-fixed":{"phi3":0.3},
              "bee-fixed":{"phi3":0.3}, "ant-cosine":{}}
BOUNDARY_MODES = ["clip", "reflect", "random_reinit", "absorb"]

def _path(algo, bmode, run):
    return os.path.join(OUT_DIR, f"{algo}__bnd-{bmode}__d30__r{run:03d}.json")

def _done(algo, bmode, run):
    p = _path(algo, bmode, run)
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

def execute(algo, bmode, run_idx):
    prob = {"func": schwefel, "lb": -500.0, "ub": 500.0, "dim": 30, "name": "schwefel"}
    rec = run_job(algo, prob, run_idx=run_idx, seed=SEEDS[run_idx],
                  n_particles=N_PARTICLES, max_iter=MAX_ITER,
                  max_fes=N_PARTICLES*(MAX_ITER+1),
                  boundary_mode=bmode, extra=ALGORITHMS[algo])
    rec["boundary_mode"] = bmode
    with open(_path(algo, bmode, run_idx), "w") as f:
        json.dump(_serial(rec), f)
    return (algo, bmode, run_idx, rec["final_f"])

jobs = [(a,b,r) for a in ALGORITHMS for b in BOUNDARY_MODES
        for r in range(N_RUNS) if not _done(a,b,r)]
print(f"E7: {len(jobs)} jobs"); t0=time.time()
Parallel(n_jobs=N_JOBS, verbose=2)(delayed(execute)(*j) for j in jobs)
print(f"E7 done in {(time.time()-t0)/60:.1f} min")
