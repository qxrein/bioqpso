#!/usr/bin/env python3
"""Run PSO/GWO/WOA/ABC under the settings of their original papers
(30 agents, 500 iterations, D=30, 30 runs) for docs/BASELINE_CHECK.md."""
import json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, yaml
from joblib import Parallel, delayed
from bioqpso.runner import run_job
from bioqpso.benchmarks import get_classical_problem

with open("config/seeds.yaml") as f:
    SEEDS = [int(s) for s in yaml.safe_load(f)["seeds"]]
ALGOS = ["pso", "gwo", "woa", "abc"]
FUNCS = ["sphere", "rastrigin", "ackley", "griewank"]
T = 500

def one(a, f, s):
    return run_job(a, get_classical_problem(f, 30), 0, s, n_particles=30,
                   max_iter=T, max_fes=30 * (T + 1))["final_f"]

vals = Parallel(n_jobs=8)(delayed(one)(a, f, s) for a in ALGOS for f in FUNCS for s in SEEDS)
it = iter(vals); out = {}
for a in ALGOS:
    out[a] = {}
    for f in FUNCS:
        v = np.array([next(it) for _ in SEEDS])
        out[a][f] = {"mean": float(v.mean()), "std": float(v.std()), "median": float(np.median(v))}
        print(f"{a:5s}{f:12s} mean={v.mean():.4g} std={v.std():.3g} median={np.median(v):.3g}")
os.makedirs("results/baseline_check", exist_ok=True)
json.dump(out, open("results/baseline_check/original_settings.json", "w"), indent=1)

# ABC under Karaboga & Basturk (2007): colony 125, 1000 cycles, D=30, for two limits.
def abc_one(f, s, limit):
    n, T = 125, 1000
    return run_job("abc", get_classical_problem(f, 30), 0, s, n_particles=n, max_iter=T,
                   max_fes=2 * n * T + n, extra={"limit": limit})["final_f"]

out["abc_ks2007"] = {}
for limit in (30, 62 * 30):
    out["abc_ks2007"][str(limit)] = {}
    for f in ["griewank", "rastrigin", "rosenbrock", "ackley", "schwefel"]:
        v = np.array(Parallel(n_jobs=8)(delayed(abc_one)(f, s, limit) for s in SEEDS))
        out["abc_ks2007"][str(limit)][f] = {"mean": float(v.mean()), "std": float(v.std())}
        print(f"abc limit={limit:5d} {f:12s} mean={v.mean():.4g} std={v.std():.3g}")
# ACO_R sanity after the sigma fix, main budget
out["aco_r_main"] = {}
for f in ["sphere", "rastrigin", "ackley"]:
    v = np.array(Parallel(n_jobs=8)(delayed(lambda f, s: run_job("aco_r", get_classical_problem(f, 30), 0, s, max_fes=30030)["final_f"])(f, s) for s in SEEDS))
    out["aco_r_main"][f] = {"mean": float(v.mean()), "std": float(v.std())}
    print(f"aco_r main {f:12s} mean={v.mean():.4g} std={v.std():.3g}")
json.dump(out, open("results/baseline_check/original_settings.json", "w"), indent=1)
