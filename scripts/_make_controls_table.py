#!/usr/bin/env python3
"""Generate paper/generated/controls_fixed_sweep.tex from E2 results.

Table: ant leader rows = QPSO, no-bio, fixed phi3=0.1..0.5, random, Linear, Cosine, Feedback
Columns: key functions (Sphere, Rastrigin, Levy, Styblinski-Tang, Schwefel, Griewank)
"""
import json, os, sys
import numpy as np

RESULTS_DIR = "results/e2_controls"
OUT = "paper/generated/controls_fixed_sweep.tex"

# Algorithm display names and E2 keys
ROWS = [
    ("QPSO",              "qpso"),
    ("no-bio ($\\phi_3=0$)", "ant-nobio"),
    ("$\\phi_3=0.1$",     "ant-fixed-p01"),
    ("$\\phi_3=0.2$",     "ant-fixed-p02"),
    ("$\\phi_3=0.3$",     "ant-fixed-p03"),
    ("$\\phi_3=0.4$",     "ant-fixed-p04"),
    ("$\\phi_3=0.5$",     "ant-fixed-p05"),
    ("random $\\phi_3$",  "ant-random"),
    ("Linear",            "ant-linear"),
    ("Cosine",            "ant-cosine"),
    ("Feedback",          "ant-feedback"),
]

FUNCTIONS = ["sphere", "rastrigin", "levy", "styblinski_tang", "schwefel", "griewank"]
FN_SHORT = {
    "sphere": "Sphere", "rastrigin": "Rastrigin", "levy": "Levy",
    "styblinski_tang": "Styb.--Tang", "schwefel": "Schwefel", "griewank": "Griewank",
}

# Load all per-run results
data = {}  # (algo_key, func) -> list of final_f
for fn in os.listdir(RESULTS_DIR):
    if not fn.endswith(".json"):
        continue
    with open(os.path.join(RESULTS_DIR, fn)) as f:
        r = json.load(f)
    key = (r["algorithm"], r["function"])
    data.setdefault(key, []).append(r["final_f"])

# Build table
col_spec = "l" + "r" * len(FUNCTIONS)
header = "Setting & " + " & ".join(FN_SHORT[f] for f in FUNCTIONS)

lines = [
    "\\begin{tabular}{" + col_spec + "}",
    "\\toprule",
    header + " \\\\",
    "\\midrule",
]

for display, algo_key in ROWS:
    cells = []
    for fn in FUNCTIONS:
        vals = data.get((algo_key, fn), [])
        if not vals:
            cells.append("---")
        else:
            m = np.mean(vals)
            if abs(m) < 1e-300:
                cells.append("$<10^{-300}$")
            elif abs(m) >= 1e4 or (abs(m) < 1e-3 and m != 0):
                s = f"{m:.2e}"
                mant, exp = s.split("e")
                cells.append(f"${mant}\\times10^{{{int(exp)}}}$")
            else:
                cells.append(f"${m:.4f}$")
    lines.append(display + " & " + " & ".join(cells) + " \\\\")

lines += ["\\bottomrule", "\\end{tabular}"]

with open(OUT, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"Written {OUT}")
