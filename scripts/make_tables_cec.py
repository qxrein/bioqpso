#!/usr/bin/env python3
"""Generate CEC2017 LaTeX tables from E3 analysis results."""
import json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

OUT_DIR = "paper/generated"
os.makedirs(OUT_DIR, exist_ok=True)

ANALYSIS = "results/analysis/e3_cec2017_d30"
RESULTS  = "results/e3_cec2017_d30"

ALGO_DISPLAY = {
    "pso":"PSO", "qpso":"QPSO", "aco_r":r"ACO$_R$", "abc":"ABC",
    "gwo":"GWO", "woa":"WOA", "ant-fixed":"AntBio", "bee-fixed":"BeeBio",
    "ant-linear":"Ant-Lin", "ant-cosine":"Ant-Cos",
    "ant-feedback":"Ant-FB", "bee-per_particle":"Bee-PP",
}
ALGO_ORDER = list(ALGO_DISPLAY.keys())

# CEC2017 function display names (abbreviated)
FN_DISPLAY = {
    "F12017":"F1","F22017":"F2","F32017":"F3","F42017":"F4",
    "F62017":"F6","F72017":"F7","F82017":"F8","F92017":"F9",
    "F102017":"F10","F112017":"F11","F122017":"F12","F132017":"F13",
    "F142017":"F14","F152017":"F15","F162017":"F16","F172017":"F17",
    "F182017":"F18","F192017":"F19","F202017":"F20","F212017":"F21",
    "F222017":"F22","F232017":"F23","F242017":"F24","F252017":"F25",
    "F262017":"F26","F272017":"F27","F282017":"F28","F292017":"F29",
}

def _fmt(v):
    if not np.isfinite(v): return "---"
    if v <= 0: return r"$\mathbf{0}$"
    s = f"{v:.2e}"; m, e = s.split("e"); exp = int(e)
    return f"${m}\\times10^{{{exp}}}$"

def _bold(s): return r"\textbf{" + s + "}"

# Load stats
stats_df = pd.read_csv(f"{ANALYSIS}/stats.csv")
ranks_df = pd.read_csv(f"{ANALYSIS}/ranks.csv")
with open(f"{ANALYSIS}/summary.json") as f:
    summary = json.load(f)

algos = [a for a in ALGO_ORDER if a in stats_df["algorithm"].unique()]
fns   = sorted([f for f in stats_df["function"].unique() if f != "F52017"])

# --- CEC2017 ranks table ---
lines = [r"\begin{tabular}{lr}", r"\toprule",
         r"Algorithm & Avg.\ Rank \\", r"\midrule"]
for _, row in ranks_df.iterrows():
    a = row["algorithm"]
    d = ALGO_DISPLAY.get(a, a)
    r = row["mean_rank"]
    lines.append(f"{d} & {r:.2f} " + r"\\")
lines += [r"\midrule",
          r"\multicolumn{2}{l}{" +
          f"Friedman $\\chi^2={summary['friedman_chi2']:.2f}$, "
          f"$p={summary['friedman_p']:.2e}$" + r"} \\",
          r"\multicolumn{2}{l}{" +
          f"Nemenyi CD$={summary['nemenyi_CD']:.3f}$ "
          f"($k={summary['k_for_CD']}$, $N={summary['N_for_CD']}$)" + r"} \\",
          r"\bottomrule", r"\end{tabular}"]
with open(f"{OUT_DIR}/cec2017_ranks.tex", "w") as f:
    f.write("\n".join(lines) + "\n")
print("Wrote cec2017_ranks.tex")

# --- Compact mean table (first 14 functions) ---
fns_a = fns[:14]
pivot = stats_df.pivot_table(index="function", columns="algorithm",
                             values="mean", aggfunc="first")

def make_half_table(fn_list, outname):
    col_spec = "l" + "r" * len(algos)
    header = " & ".join(["Fn"] + [ALGO_DISPLAY.get(a, a) for a in algos])
    rows = [r"\begin{tabular}{" + col_spec + "}", r"\toprule",
            header + r" \\", r"\midrule"]
    for fn in fn_list:
        if fn not in pivot.index: continue
        vals = [pivot.loc[fn, a] if a in pivot.columns else np.nan for a in algos]
        finite = [v for v in vals if np.isfinite(v)]
        best = min(finite) if finite else np.nan
        cells = []
        for a, v in zip(algos, vals):
            s = _fmt(v)
            if np.isfinite(v) and abs(v - best) / (abs(best) + 1e-300) < 0.001:
                s = _bold(s)
            cells.append(s)
        rows.append(FN_DISPLAY.get(fn, fn) + " & " + " & ".join(cells) + r" \\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    with open(f"{OUT_DIR}/{outname}", "w") as f:
        f.write("\n".join(rows) + "\n")
    print(f"Wrote {outname}")

make_half_table(fns[:14], "cec2017_mean_a.tex")
make_half_table(fns[14:], "cec2017_mean_b.tex")

# --- numbers.tex additions ---
with open(f"{OUT_DIR}/numbers.tex", "a") as f:
    f.write(f"\n% CEC2017 numbers\n")
    f.write(f"\\newcommand{{\\numCECFriedmanChiSq}}{{{summary['friedman_chi2']:.2f}}}\n")
    f.write(f"\\newcommand{{\\numCECFriedmanP}}{{{summary['friedman_p']:.2e}}}\n")
    f.write(f"\\newcommand{{\\numCECNemenyiCD}}{{{summary['nemenyi_CD']:.3f}}}\n")
    f.write(f"\\newcommand{{\\numCECNemenyiK}}{{{summary['k_for_CD']}}}\n")
    f.write(f"\\newcommand{{\\numCECNemenyiN}}{{{summary['N_for_CD']}}}\n")
    best_algo = ranks_df.iloc[0]["algorithm"]
    best_rank = ranks_df.iloc[0]["mean_rank"]
    f.write(f"\\newcommand{{\\numCECBestAlgo}}{{{ALGO_DISPLAY.get(best_algo, best_algo)}}}\n")
    f.write(f"\\newcommand{{\\numCECBestRank}}{{{best_rank:.2f}}}\n")
    qpso_rank = float(ranks_df[ranks_df["algorithm"]=="qpso"]["mean_rank"].iloc[0])
    f.write(f"\\newcommand{{\\numCECQPSORank}}{{{qpso_rank:.2f}}}\n")
print("Updated numbers.tex with CEC2017 macros")
print()
print("CEC2017 summary:")
print(f"  Friedman chi2={summary['friedman_chi2']:.2f}  p={summary['friedman_p']:.2e}")
print(f"  CD={summary['nemenyi_CD']:.3f}  k={summary['k_for_CD']}  N={summary['N_for_CD']}")
print(f"  Best: {best_algo} (rank {best_rank:.2f})  QPSO rank {qpso_rank:.2f}")
