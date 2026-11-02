#!/usr/bin/env python3
"""E9 — Runtime: 10 reps per (algorithm, function), one process, one BLAS thread.
Must run on an otherwise idle machine."""
from _runlib import *
from bioqpso.runner import run_job
from bioqpso.benchmarks import get_classical_problem
OUT = "results/e9_runtime"; os.makedirs(OUT, exist_ok=True)
results = {}
for fn in ["sphere", "rastrigin", "levy", "schwefel"]:
    prob = get_classical_problem(fn, dim=30)
    results[fn] = {}
    for algo, extra in ALL13.items():
        times = []
        for rep in range(10):
            t0 = time.perf_counter()
            r = run_job(algo, prob, rep, 1000 + rep, max_fes=30030, extra=extra)
            times.append(time.perf_counter() - t0)
        med = float(np.median(times))
        results[fn][algo] = {
            "median_s": med,
            "iqr_s": float(np.percentile(times, 75) - np.percentile(times, 25)),
            "per_1k_fe_s": med / (r["n_fes"] / 1000),
        }
        print(f"{algo:18s} {fn:10s} {med:.3f}s", flush=True)
json.dump(results, open(f"{OUT}/runtime_results.json", "w"), indent=2)
