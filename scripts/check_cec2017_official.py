"""Compare opfunu 1.0.4 CEC2017 with the official C code (cec17_test_func.cpp).

Usage: python scripts/check_cec2017_official.py <dir with compiled driver and input_data/>

The driver is a small main() around the official cec17_test_func (see docs/CEC2017_CHECK.md).
Five fixed random points per function in [-100, 100]^D, D = 30 and 50. opfunu F_k is official
F_{k+1} for k >= 2. Both biases differ, so we compare errors f - f_bias. A second comparison
loads the official data files into opfunu (shift, rotation and shuffle of official F_{k+1}),
which separates differences in the data from differences in the formula.
Writes results/analysis/cec2017_official_check.json.
"""
import inspect
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import opfunu

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "analysis" / "cec2017_official_check.json"
N_POINTS = 5
SEED = 20261106


def official(build, fn, x):
    inp = " ".join(f"{v:.17g}" for v in x.ravel())
    r = subprocess.run([str(build / "driver"), str(fn), str(x.shape[1]), str(x.shape[0])],
                       input=inp, capture_output=True, text=True, cwd=build, check=True)
    return np.array([float(v) for v in r.stdout.split()])


def main():
    build = Path(sys.argv[1]).resolve()
    rng = np.random.default_rng(SEED)
    out = {"seed": SEED, "n_points": N_POINTS, "opfunu": opfunu.__version__, "functions": {}}
    for D in (30, 50):
        x = rng.uniform(-100, 100, size=(N_POINTS, D))
        for k in range(1, 30):
            off = 1 if k == 1 else k + 1
            f = getattr(opfunu.cec_based, f"F{k}2017")(ndim=D)
            ours = np.array([f.evaluate(xi) for xi in x]) - f.f_bias
            ref = official(build, off, x) - 100.0 * off
            rel = np.abs(ours - ref) / np.maximum(np.abs(ref), 1e-12)
            own = {n: p.default for n, p in inspect.signature(type(f).__init__).parameters.items()
                   if isinstance(p.default, str)}
            kw = {n: re.sub(r"_(\d+)(_D)?$", lambda m: f"_{off}{m.group(2) or ''}", p.default)
                  for n, p in inspect.signature(type(f).__init__).parameters.items()
                  if isinstance(p.default, str)}
            g = type(f)(ndim=D, **kw)
            ours_od = np.array([g.evaluate(xi) for xi in x]) - g.f_bias
            rel_od = np.abs(ours_od - ref) / np.maximum(np.abs(ref), 1e-12)
            # the optimum of the official instance, evaluated by opfunu
            xo = np.loadtxt(build / "input_data" / f"shift_data_{off}.txt").ravel()[:D]
            at_opt = float(f.evaluate(xo) - f.f_bias)
            out["functions"].setdefault(f"F{k}", {})[f"D{D}"] = {
                "official": off, "opfunu_data": own, "max_rel_diff": float(rel.max()),
                "max_rel_diff_official_data": float(rel_od.max()),
                "opfunu_err": ours.tolist(), "official_err": ref.tolist(),
                "opfunu_err_at_official_optimum": at_opt}
    OUT.write_text(json.dumps(out, indent=1))
    for k, v in out["functions"].items():
        print(k, v["D30"]["official"], *(f"{v[d][m]:.3g}" for m in ("max_rel_diff", "max_rel_diff_official_data")
                                          for d in ("D30", "D50")))


if __name__ == "__main__":
    main()
