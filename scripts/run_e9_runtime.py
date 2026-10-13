#!/usr/bin/env python3
"""E9 — Runtime benchmark: 10 reps per (algo, function), OMP_NUM_THREADS=1."""
import json, os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np
from bioqpso.runner import run_job
from bioqpso.benchmarks import get_classical_problem

OUT_DIR = "results/e9_runtime"; os.makedirs(OUT_DIR, exist_ok=True)
N_PARTICLES = 30; MAX_ITER = 1000; MAX_FES = N_PARTICLES*(MAX_ITER+1)
N_REPS = 10

ALGORITHMS = {
    "pso":{}, "qpso":{}, "aco_r":{}, "abc":{}, "gwo":{}, "woa":{},
    "ant-fixed":{"phi3":0.3}, "bee-fixed":{"phi3":0.3},
    "ant-linear":{}, "ant-cosine":{}, "ant-feedback":{"W":20,"delta":0.02},
    "bee-per_particle":{},
}
FUNCTIONS = ["sphere","rastrigin","levy","schwefel"]

results = {}
for fn in FUNCTIONS:
    prob = get_classical_problem(fn, dim=30)
    results[fn] = {}
    for algo, extra in ALGORITHMS.items():
        times = []
        best_vals = []
        fe_counts = []
        for rep in range(N_REPS):
            t0 = time.perf_counter()
            r = run_job(algo, prob, run_idx=rep, seed=1000+rep,
                        n_particles=N_PARTICLES, max_iter=MAX_ITER,
                        max_fes=MAX_FES, extra=extra)
            elapsed = time.perf_counter() - t0
            times.append(elapsed)
            best_vals.append(r["final_f"])
            fe_counts.append(r["n_fes"])
        med = float(np.median(times))
        iqr = float(np.percentile(times,75) - np.percentile(times,25))
        per1k = med / (np.mean(fe_counts)/1000)
        results[fn][algo] = {
            "median_s": med, "iqr_s": iqr,
            "per_1k_fe_s": per1k,
            "mean_fe": float(np.mean(fe_counts)),
        }
        print(f"  {algo:22s} / {fn:15s}  median={med:.3f}s  IQR={iqr:.3f}s  "
              f"per1kFE={per1k:.4f}s")

out_path = os.path.join(OUT_DIR, "runtime_results.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved → {out_path}")
