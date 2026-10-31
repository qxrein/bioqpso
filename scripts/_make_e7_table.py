#!/usr/bin/env python3
"""Generate E7 Schwefel table from results/e7_schwefel/."""
import json, os, numpy as np

RESULTS_DIR = "results/e7_schwefel"
OUT = "paper/generated/e7_schwefel.tex"

ALGOS = ["qpso", "ant-nobio", "ant-fixed", "bee-fixed", "ant-cosine"]
ALGO_DISPLAY = {
    "qpso": "QPSO", "ant-nobio": "no-bio", "ant-fixed": "AntBio",
    "bee-fixed": "BeeBio", "ant-cosine": "Ant-Cos",
}
BMODES = ["clip", "reflect", "random_reinit", "absorb"]
BMODE_DISPLAY = {"clip": "Clip", "reflect": "Reflect",
                 "random_reinit": "Random", "absorb": "Absorb"}

data = {}
for fn in os.listdir(RESULTS_DIR):
    if not fn.endswith(".json"): continue
    with open(os.path.join(RESULTS_DIR, fn)) as f:
        r = json.load(f)
    algo = r["algorithm"]
    bmode = r.get("boundary_mode", "clip")
    data[(algo, bmode)] = data.get((algo, bmode), [])
    data[(algo, bmode)].append(r["final_f"])

col_spec = "l" + "r" * len(BMODES)
header = "Algorithm & " + " & ".join(BMODE_DISPLAY[b] for b in BMODES)
lines = [r"\begin{tabular}{" + col_spec + "}", r"\toprule",
         header + r" \\", r"\midrule"]

for algo in ALGOS:
    cells = []
    for bmode in BMODES:
        vals = data.get((algo, bmode), [])
        if not vals:
            cells.append("---")
        else:
            m = np.mean(vals)
            s = np.std(vals)
            if m >= 1e4:
                cells.append(f"${m:.0f}$")
            elif m >= 1:
                cells.append(f"${m:.1f}$")
            else:
                s_str = f"{m:.2e}"
                mant, exp = s_str.split("e")
                cells.append(f"${mant}\\times10^{{{int(exp)}}}$")
    lines.append(ALGO_DISPLAY[algo] + " & " + " & ".join(cells) + r" \\")

lines += [r"\bottomrule", r"\end{tabular}"]
with open(OUT, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"Written {OUT}")
