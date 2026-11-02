#!/usr/bin/env python3
"""E2c — controls on CEC2017 (D=30, 30 runs, 30,030 FEs). Settings already present in
E3 (qpso, ant-fixed, ant-linear, ant-cosine, ant-feedback, bee-fixed, bee-per_particle)
are reused from results/e3_cec2017_d30: same seeds and code, hence the same runs."""
from _runlib import *
from bioqpso.runner import run_job
from bioqpso.benchmarks.cec2017 import CEC2017_CLASSES, get_cec_problem
OUT = "results/e2c_controls_cec"; os.makedirs(OUT, exist_ok=True)
NEW = {
    "ant-nobio": ("ant-nobio", {}), "bee-nobio": ("bee-nobio", {}),
    "ant-fixed-p01": ("ant-fixed", {"phi3": 0.1}), "ant-fixed-p05": ("ant-fixed", {"phi3": 0.5}),
    "ant-random": ("ant-random", {}),
}
def job(path, disp, cname, r):
    fc = next(c for c in CEC2017_CLASSES if c.__name__ == cname)
    algo, extra = NEW[disp]
    rec = run_job(algo, get_cec_problem(fc, 30), r, SEEDS[r], max_fes=30030, extra=extra)
    rec["algorithm"] = disp; rec["phi3_hist"] = None
    write_json(path, rec)
run_jobs("E2c", job, [(f"{OUT}/{d}__{c.__name__}__d30__r{r:03d}.json", (d, c.__name__, r))
                      for d in NEW for c in CEC2017_CLASSES for r in range(30)])
