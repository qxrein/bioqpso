"""Shared helpers for the experiment scripts: resumable per-run JSON jobs."""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np, yaml
from joblib import Parallel, delayed

with open(ROOT / "config/seeds.yaml") as f:
    SEEDS = [int(s) for s in yaml.safe_load(f)["seeds"]]
N_JOBS = int(os.environ.get("BIOQPSO_JOBS", "9"))


def serial(o):
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, np.generic): return o.item()
    if isinstance(o, dict): return {k: serial(v) for k, v in o.items() if not k.startswith("_")}
    if isinstance(o, (list, tuple)): return [serial(v) for v in o]
    return o


def is_done(path) -> bool:
    if not os.path.exists(path):
        return False
    try:
        with open(path) as f:
            return "final_f" in json.load(f)
    except Exception:
        return False


def write_json(path, rec):
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(serial(rec), f)
    os.replace(tmp, path)


def run_jobs(name, fn, jobs):
    """jobs: list of (path, args). Skip finished paths; fn(path, *args) writes the file."""
    todo = [(p, a) for p, a in jobs if not is_done(p)]
    print(f"{name}: {len(todo)} pending of {len(jobs)}", flush=True)
    t0 = time.time()
    Parallel(n_jobs=N_JOBS, verbose=1, batch_size=1)(delayed(fn)(p, *a) for p, a in todo)
    print(f"{name} done in {(time.time()-t0)/60:.1f} min", flush=True)


FULL12 = {
    "pso": {}, "qpso": {}, "aco_r": {}, "abc": {}, "gwo": {}, "woa": {},
    "ant-fixed": {"phi3": 0.3}, "bee-fixed": {"phi3": 0.3},
    "ant-linear": {}, "ant-cosine": {},
    "ant-feedback": {"W": 20, "delta": 0.02}, "bee-per_particle": {},
}
ALL13 = {**FULL12, "lshade": {}}
CLASSICAL = ["sphere", "rosenbrock", "zakharov", "dixon_price", "rastrigin",
             "ackley", "griewank", "schwefel", "levy", "styblinski_tang"]
