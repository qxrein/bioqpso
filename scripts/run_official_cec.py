#!/usr/bin/env python3
"""CEC2017 on the official code (Amendment 2 of docs/DECISION_RULE.md).
Usage: run_official_cec.py e3 | e2c | e3s30 | e3s50

e3:    13 algorithms, D=30, 30 particles, 30,030 FEs (as E3).
e2c:   the five extra control settings of E2c at the E3 budget; the other settings are the E3 runs
       (same seeds and code, hence the same runs).
e3s30, e3s50: 13 algorithms at 10,000*D FEs, T = FEs/30, history every 100 iterations (as E3s).
29 functions (F1, F3-F30), 30 runs, the seeds of config/seeds.yaml, frozen settings."""
from _runlib import *
from bioqpso.runner import run_job
from bioqpso.benchmarks.cec2017_official import NAMES, get_official_problem

E2C = {
    "ant-nobio": ("ant-nobio", {}), "bee-nobio": ("bee-nobio", {}),
    "ant-fixed-p01": ("ant-fixed", {"phi3": 0.1}), "ant-fixed-p05": ("ant-fixed", {"phi3": 0.5}),
    "ant-random": ("ant-random", {}),
}


def job_e3(path, a, fn, r):
    rec = run_job(a, get_official_problem(fn, 30), r, SEEDS[r], n_particles=30, max_iter=1000,
                  max_fes=30030, extra=ALL13[a])
    write_json(path, rec)


def job_e2c(path, disp, fn, r):
    algo, extra = E2C[disp]
    rec = run_job(algo, get_official_problem(fn, 30), r, SEEDS[r], max_fes=30030, extra=extra)
    rec["algorithm"] = disp; rec["phi3_hist"] = None
    write_json(path, rec)


def job_e3s(path, a, fn, r, D):
    fes = 10000 * D
    rec = run_job(a, get_official_problem(fn, D), r, SEEDS[r], n_particles=30, max_iter=fes // 30,
                  max_fes=fes, hist_every=100, extra=ALL13[a])
    rec["phi3_hist"] = None
    write_json(path, rec)


if __name__ == "__main__":
    what = sys.argv[1]
    if what == "e3":
        out = "results/e3_official_d30"; os.makedirs(out, exist_ok=True)
        run_jobs("E3-official", job_e3, [(f"{out}/{a}__{fn}__d30__r{r:03d}.json", (a, fn, r))
                                         for r in range(30) for fn in NAMES for a in ALL13])
    elif what == "e2c":
        out = "results/e2c_official"; os.makedirs(out, exist_ok=True)
        run_jobs("E2c-official", job_e2c, [(f"{out}/{d}__{fn}__d30__r{r:03d}.json", (d, fn, r))
                                           for r in range(30) for fn in NAMES for d in E2C])
    elif what in ("e3s30", "e3s50"):
        D = int(what[3:]); out = f"results/e3s_official_d{D}"; os.makedirs(out, exist_ok=True)
        run_jobs(f"E3s-official-D{D}", job_e3s, [(f"{out}/{a}__{fn}__d{D}__r{r:03d}.json", (a, fn, r, D))
                                                 for r in range(30) for fn in NAMES for a in ALL13])
    else:
        sys.exit(__doc__)
