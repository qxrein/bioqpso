"""Our L-SHADE on the official CEC2017 suite (E3s) against published competition results.

Usage: python scripts/check_lshade_official.py <jSO rawResults dir> <LSHADE_SPACMA results dir>

Both come from the official CEC2017 repository (P-N-Suganthan/CEC2017-BoundContrained,
Codes-of-Top-Methods-and-results.zip): jSO-SOURCE-RESULTS-29May2017.tar.gz/rawResults
(Brest et al. 2017) and Results for all papers/Paper ID 17051/LSHADE_SPACMA_Results_2.zip
(Mohamed et al. 2017). Each file <alg>_<k>_<D>.txt holds 14 rows of errors f - f* at the
competition checkpoints by 51 runs; the last row is the error at 10,000*D FEs, our E3s budget.
Errors below 1e-8 count as 0, as in the CEC2017 rules. A function is flagged if our mean error is
better than jSO's by more than one order of magnitude or worse by more than two (means floored at 1e-8).
Writes results/analysis/lshade_reference_check.json (the extracted reference errors included).
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "analysis" / "lshade_reference_check.json"
FUNCS = [1] + list(range(3, 31))
EPS = 1e-8


def cec(e):
    e = np.asarray(e, float)
    return np.where(e < EPS, 0.0, e)


def final_errors(directory, prefix, k, D):
    return cec(np.loadtxt(Path(directory) / f"{prefix}_{k}_{D}.txt")[-1])


def ours(k, D):
    v = [json.load(open(p))["final_f"] for p in glob.glob(str(ROOT / f"results/e3s_official_d{D}/lshade__F{k}__d{D}__r*.json"))]
    return cec(np.array(v) - 100.0 * k)


def main():
    jso_dir, spacma_dir = sys.argv[1:3]
    out = {}
    for D in (30, 50):
        for k in FUNCS:
            o = ours(k, D)
            if len(o) != 30:
                sys.exit(f"E3s D={D} F{k}: {len(o)} L-SHADE runs")
            j, s = final_errors(jso_dir, "jSO", k, D), final_errors(spacma_dir, "LSHADE_SPACMA", k, D)
            ratio = max(o.mean(), EPS) / max(j.mean(), EPS)
            flag = "better by >10x" if ratio < 0.1 else "worse by >100x" if ratio > 100 else ""
            out[f"F{k}_D{D}"] = {"ours_mean": o.mean(), "ours_median": float(np.median(o)), "ours_std": o.std(),
                                 "jso_mean": j.mean(), "jso_median": float(np.median(j)), "jso_std": j.std(),
                                 "spacma_mean": s.mean(), "spacma_median": float(np.median(s)), "spacma_std": s.std(),
                                 "ratio_to_jso": ratio, "flag": flag, "jso": j.tolist(), "spacma": s.tolist()}
    OUT.write_text(json.dumps(out, indent=1))
    print("| F | D | ours mean | ours median | jSO mean | jSO median | SPACMA mean | SPACMA median | ours/jSO | flag |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for key, v in out.items():
        k, D = key.split("_D")
        print(f"| {k} | {D} | {v['ours_mean']:.3g} | {v['ours_median']:.3g} | {v['jso_mean']:.3g} | {v['jso_median']:.3g} | "
              f"{v['spacma_mean']:.3g} | {v['spacma_median']:.3g} | {v['ratio_to_jso']:.3g} | {v['flag']} |")
    print("flagged:", [k for k, v in out.items() if v["flag"]])

if __name__ == "__main__":
    main()
