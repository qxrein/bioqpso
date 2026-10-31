#!/usr/bin/env python3
"""Generate CEC2017 standard-deviation table from E3 results."""
import json, os, sys
import numpy as np

RESULTS_DIR = "results/e3_cec2017_d30"
OUT = "paper/generated/cec2017_std.tex"

ALGO_DISPLAY = {
    "pso":"PSO", "qpso":"QPSO", "aco_r":"ACO$_R$", "abc":"ABC",
    "gwo":"GWO", "woa":"WOA", "ant-fixed":"Ant", "bee-fixed":"Bee",
    "ant-linear":"Lin", "ant-cosine":"Cos", "ant-feedback":"Fb",
    "bee-per_particle":"PP",
}
ALGO_ORDER = list(ALGO_DISPLAY.keys())
FN_ORDER = sorted([
    "F12017","F22017","F32017","F42017","F62017","F72017","F82017","F92017",
    "F102017","F112017","F122017","F132017","F142017","F152017","F162017",
    "F172017","F182017","F192017","F202017","F212017","F222017","F232017",
    "F242017","F252017","F262017","F272017","F282017","F292017",
])
FN_SHORT = {f: f.replace("2017","") for f in FN_ORDER}

# Load all per-run results
data = {}
for fn in os.listdir(RESULTS_DIR):
    if not fn.endswith(".json"): continue
    with open(os.path.join(RESULTS_DIR, fn)) as f:
        r = json.load(f)
    key = (r["algorithm"], r["function"])
    data.setdefault(key, []).append(r["final_f"])

algos = [a for a in ALGO_ORDER if any((a, fn) in data for fn in FN_ORDER)]
fns   = [f for f in FN_ORDER if any((a, f) in data for a in algos)]

# Split into two halves
fns_a = fns[:14]
fns_b = fns[14:]

def make_half(fn_list, outname):
    col_spec = "l" + "r" * len(algos)
    header = "Fn & " + " & ".join(ALGO_DISPLAY[a] for a in algos)
    lines = [r"\begin{tabular}{" + col_spec + "}", r"\toprule",
             header + r" \\", r"\midrule"]
    for fn in fn_list:
        cells = []
        for a in algos:
            vals = data.get((a, fn), [])
            if not vals:
                cells.append("---")
            else:
                std = np.std(vals)
                if std < 1e-300:
                    cells.append("$<10^{-300}$")
                elif std >= 1e4 or (std < 1e-3 and std > 0):
                    s = f"{std:.2e}"
                    mant, exp = s.split("e")
                    cells.append(f"${mant}\\times10^{{{int(exp)}}}$")
                else:
                    cells.append(f"${std:.4f}$")
        lines.append(FN_SHORT.get(fn, fn) + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(outname, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {outname}")

make_half(fns_a, OUT.replace(".tex", "_a.tex"))
make_half(fns_b, OUT.replace(".tex", "_b.tex"))
print("Done")
