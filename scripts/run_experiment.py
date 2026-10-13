#!/usr/bin/env python3
"""
run_experiment.py — parallel experiment runner.

Usage:
    python scripts/run_experiment.py --config config/e1_classical.yaml \\
           --output results/e1 --n_jobs 4

Skips already-finished (algorithm, function, dim, run) tuples.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bioqpso.runner import run_job
from bioqpso.benchmarks import get_classical_problem, CEC2017_CLASSES, get_cec_problem


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def load_seeds(seed_file: str = "config/seeds.yaml") -> list[int]:
    with open(seed_file) as f:
        cfg = yaml.safe_load(f)
    return [int(s) for s in cfg["seeds"]]


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def result_path(out_dir: str, algorithm: str, function: str, dim: int, run: int) -> str:
    fn = f"{algorithm}__{function}__d{dim}__r{run:03d}.json"
    return os.path.join(out_dir, fn)


def already_done(out_dir: str, algorithm: str, function: str, dim: int, run: int) -> bool:
    p = result_path(out_dir, algorithm, function, dim, run)
    if not os.path.exists(p):
        return False
    try:
        with open(p) as f:
            d = json.load(f)
        return "final_f" in d and "n_fes" in d
    except Exception:
        return False


def _to_serial(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _to_serial(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serial(v) for v in obj]
    return obj


# ─────────────────────────────────────────────────────────────
# Job generation
# ─────────────────────────────────────────────────────────────

def generate_jobs(cfg: dict, seeds: list[int], out_dir: str,
                  n_runs: int | None = None) -> list[dict]:
    """Return a list of pending job dicts."""
    experiment = cfg.get("experiment", "exp")
    n_particles = int(cfg.get("n_particles", 30))
    max_iter = int(cfg.get("max_iter", 1000))
    max_fes = cfg.get("max_fes", None)
    if max_fes is not None:
        max_fes = int(max_fes)
    boundary = cfg.get("boundary", "clip")
    hist_every = int(cfg.get("hist_every", 10))
    n_runs_cfg = n_runs if n_runs is not None else len(seeds)

    algorithms: dict = cfg.get("algorithms", {})
    functions: list[str] = cfg.get("functions", [])
    dims: list[int] = [int(d) for d in cfg.get("dims", [30])]

    # CEC2017 mode
    cec_exclude = set(cfg.get("cec_exclude", ["F52017"]))
    is_cec = cfg.get("experiment", "").startswith("e3")

    jobs = []
    for algo_name, extra in algorithms.items():
        extra = extra or {}
        if is_cec:
            # CEC: use CEC2017 problems
            for dim in dims:
                for fcls in CEC2017_CLASSES:
                    if fcls.__name__ in cec_exclude:
                        continue
                    fn_name = fcls.__name__
                    for run_idx in range(n_runs_cfg):
                        if already_done(out_dir, algo_name, fn_name, dim, run_idx):
                            continue
                        jobs.append(dict(
                            algorithm_name=algo_name,
                            problem_spec=("cec", fcls, dim),
                            run_idx=run_idx,
                            seed=seeds[run_idx],
                            n_particles=n_particles,
                            max_iter=max_iter,
                            max_fes=max_fes,
                            boundary_mode=boundary,
                            hist_every=hist_every,
                            extra=extra,
                            out_dir=out_dir,
                        ))
        else:
            for fn_name in functions:
                for dim in dims:
                    for run_idx in range(n_runs_cfg):
                        if already_done(out_dir, algo_name, fn_name, dim, run_idx):
                            continue
                        jobs.append(dict(
                            algorithm_name=algo_name,
                            problem_spec=("classical", fn_name, dim),
                            run_idx=run_idx,
                            seed=seeds[run_idx],
                            n_particles=n_particles,
                            max_iter=max_iter,
                            max_fes=max_fes,
                            boundary_mode=boundary,
                            hist_every=hist_every,
                            extra=extra,
                            out_dir=out_dir,
                        ))
    return jobs


def execute_job(job: dict) -> dict:
    """Execute a single job and save the result."""
    spec = job["problem_spec"]
    if spec[0] == "classical":
        _, fn_name, dim = spec
        problem = get_classical_problem(fn_name, dim)
    else:
        _, fcls, dim = spec
        problem = get_cec_problem(fcls, dim)

    rec = run_job(
        algorithm_name=job["algorithm_name"],
        problem=problem,
        run_idx=job["run_idx"],
        seed=job["seed"],
        n_particles=job["n_particles"],
        max_iter=job["max_iter"],
        max_fes=job["max_fes"],
        boundary_mode=job["boundary_mode"],
        hist_every=job["hist_every"],
        extra=job["extra"],
    )

    p = result_path(job["out_dir"],
                    rec["algorithm"], rec["function"],
                    rec["dim"], rec["run"])
    with open(p, "w") as f:
        json.dump(_to_serial(rec), f)
    return rec


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", default="config/seeds.yaml")
    parser.add_argument("--n_jobs", type=int, default=1)
    parser.add_argument("--n_runs", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seeds = load_seeds(args.seeds)
    out_dir = args.output
    os.makedirs(out_dir, exist_ok=True)

    jobs = generate_jobs(cfg, seeds, out_dir, args.n_runs)
    print(f"[run_experiment] {len(jobs)} pending jobs  "
          f"(config={args.config}, out={out_dir})")

    if not jobs:
        print("[run_experiment] Nothing to do — all jobs already complete.")
        return

    t0 = time.time()
    if args.n_jobs == 1:
        for k, job in enumerate(jobs):
            rec = execute_job(job)
            if (k + 1) % 50 == 0 or k == 0:
                elapsed = time.time() - t0
                rate = (k + 1) / elapsed
                eta = (len(jobs) - k - 1) / rate if rate > 0 else 0
                print(f"  {k+1}/{len(jobs)}  {rec['algorithm']} / "
                      f"{rec['function']} / d{rec['dim']} / r{rec['run']} "
                      f"f={rec['final_f']:.3e}  FE={rec['n_fes']}  "
                      f"ETA={eta/60:.1f}min")
    else:
        try:
            from joblib import Parallel, delayed
        except ImportError:
            print("joblib not found; falling back to sequential.")
            for job in jobs:
                execute_job(job)
            return

        results = Parallel(n_jobs=args.n_jobs, verbose=5)(
            delayed(execute_job)(job) for job in jobs
        )
        print(f"[run_experiment] Done. {len(results)} jobs in "
              f"{(time.time()-t0)/60:.1f} min.")


if __name__ == "__main__":
    main()
