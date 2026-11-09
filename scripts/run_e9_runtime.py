#!/usr/bin/env python3
"""E9 — Runtime: 10 reps per (algorithm, function), one process, one BLAS thread.
Must run on an otherwise idle machine. Algorithms are interleaved in a shuffled order within
each rep, so that drift in machine speed falls on all of them alike. One warm-up pass is
discarded. A GWO profile, run after the timings, shows where its extra time goes."""
from _runlib import *
import cProfile, pstats
from bioqpso.runner import run_job
from bioqpso.benchmarks import get_classical_problem
OUT = "results/e9_runtime"; os.makedirs(OUT, exist_ok=True)
FNS, REPS = ["sphere", "rastrigin", "levy", "schwefel"], 10
probs = {fn: get_classical_problem(fn, dim=30) for fn in FNS}
for fn in FNS:
    for algo, extra in ALL13.items():
        run_job(algo, probs[fn], 0, 999, max_fes=3030, extra=extra)
times = {fn: {a: [] for a in ALL13} for fn in FNS}
n_fes = {fn: {} for fn in FNS}
order = np.random.default_rng(20261103)
for rep in range(REPS):
    pairs = [(fn, a) for fn in FNS for a in ALL13]
    for k in order.permutation(len(pairs)):
        fn, algo = pairs[k]
        t0 = time.perf_counter()
        r = run_job(algo, probs[fn], rep, 1000 + rep, max_fes=30030, extra=ALL13[algo])
        times[fn][algo].append(time.perf_counter() - t0)
        n_fes[fn][algo] = r["n_fes"]
    print(f"rep {rep} done, load {os.getloadavg()[0]:.2f}", flush=True)
results = {fn: {a: {
    "times_s": times[fn][a],
    "median_s": float(np.median(times[fn][a])),
    "iqr_s": float(np.percentile(times[fn][a], 75) - np.percentile(times[fn][a], 25)),
    "per_1k_fe_s": float(np.median(times[fn][a])) / (n_fes[fn][a] / 1000),
} for a in ALL13} for fn in FNS}
json.dump(results, open(f"{OUT}/runtime_results.json", "w"), indent=2)


def profile(algo, fn):
    pr = cProfile.Profile()
    pr.enable()
    run_job(algo, probs[fn], 0, 1000, max_fes=30030, extra=ALL13[algo])
    pr.disable()
    st = pstats.Stats(pr)
    total = st.total_tt
    rows = sorted(((v[2], k[2]) for k, v in st.stats.items()), reverse=True)[:12]
    return {"total_s": total, "calls": sum(v[0] for v in st.stats.values()),
            "top_self": [{"name": n, "share": s / total} for s, n in rows]}


json.dump({a: profile(a, "sphere") for a in ("gwo", "qpso")}, open(f"{OUT}/profile.json", "w"), indent=2)
print("E9 finished", flush=True)
