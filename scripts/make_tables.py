#!/usr/bin/env python3
"""
make_tables.py — Generate LaTeX fragments from analysis results.

Outputs to paper/generated/*.tex and paper/generated/numbers.tex.
Run after analyse.py has produced results/analysis/e1_classical/stats.csv etc.
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

OUT_DIR = "paper/generated"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Algorithm display names ─────────────────────────────────────────────────
ALGO_DISPLAY = {
    "pso":              "PSO",
    "qpso":             "QPSO",
    "aco_r":            "ACO$_R$",
    "abc":              "ABC",
    "gwo":              "GWO",
    "woa":              "WOA",
    "ant-fixed":        "AntBio",
    "bee-fixed":        "BeeBio",
    "ant-linear":       "Ant-Lin",
    "ant-cosine":       "Ant-Cos",
    "ant-feedback":     "Ant-FB",
    "bee-per_particle": "Bee-PP",
}

FUNCTION_DISPLAY = {
    "sphere":          "Sphere",
    "rosenbrock":      "Rosenbrock",
    "zakharov":        "Zakharov",
    "dixon_price":     "Dixon-Price",
    "rastrigin":       "Rastrigin",
    "ackley":          "Ackley",
    "griewank":        "Griewank",
    "schwefel":        "Schwefel",
    "levy":            "Levy",
    "styblinski_tang": "Styblinski-Tang",
}

ALGO_ORDER = list(ALGO_DISPLAY.keys())
FN_ORDER   = list(FUNCTION_DISPLAY.keys())


def _fmt(v: float, is_negative_best: bool = False) -> str:
    """Format a mean value in scientific notation."""
    if not np.isfinite(v):
        return "---"
    if abs(v) < 1e-300:
        return r"$<$1e-300"
    if abs(v) >= 1e4 or (abs(v) < 1e-3 and abs(v) > 0):
        s = f"{v:.2e}"
        # LaTeX: 1.23e+04 → 1.23\times10^{4}
        m, e = s.split("e")
        exp = int(e)
        return f"${m}\\times10^{{{exp}}}$"
    return f"${v:.4f}$"


def _bold(s: str) -> str:
    return r"\textbf{" + s + "}"


def make_classical_table(stats_csv: str, wilcoxon_csv: str,
                          out_mean: str, out_std: str):
    """Generate classical benchmark mean and std tables."""
    stats  = pd.read_csv(stats_csv)
    wilcox = pd.read_csv(wilcoxon_csv)

    algos = [a for a in ALGO_ORDER if a in stats["algorithm"].unique()]
    fns   = [f for f in FN_ORDER   if f in stats["function"].unique()]

    # Build pivot tables
    mean_pivot = stats.pivot_table(index="function", columns="algorithm",
                                    values="mean", aggfunc="first")
    std_pivot  = stats.pivot_table(index="function", columns="algorithm",
                                    values="std",  aggfunc="first")

    # Wilcoxon Holm p for bold marking
    wdf = wilcoxon_csv if isinstance(wilcoxon_csv, pd.DataFrame) else pd.read_csv(wilcoxon_csv)
    holm_pivot = wdf.pivot_table(index="function", columns="algorithm",
                                  values="wilcoxon_holm", aggfunc="first")

    col_spec = "l" + "r" * len(algos)
    header   = " & ".join(["Function"] + [ALGO_DISPLAY.get(a, a) for a in algos])

    def build_table(pivot, caption_frag):
        lines = []
        lines.append(r"\begin{tabular}{" + col_spec + "}")
        lines.append(r"\toprule")
        lines.append(header + r" \\")
        lines.append(r"\midrule")
        for fn in fns:
            if fn not in pivot.index:
                continue
            row_vals = [pivot.loc[fn, a] if a in pivot.columns else np.nan
                        for a in algos]
            # Find best (min mean; for styblinski_tang min = most negative = best)
            finite_vals = [v for v in row_vals if np.isfinite(v)]
            best_val = min(finite_vals) if finite_vals else np.nan

            cells = []
            for a, v in zip(algos, row_vals):
                s = _fmt(v, is_negative_best=(fn == "styblinski_tang"))
                # Bold if this is the best value AND p<0.05 vs QPSO (or is QPSO itself)
                is_best = np.isfinite(v) and abs(v - best_val) / (abs(best_val) + 1e-300) < 0.001
                if is_best:
                    s = _bold(s)
                cells.append(s)
            fn_disp = FUNCTION_DISPLAY.get(fn, fn)
            lines.append(fn_disp + " & " + " & ".join(cells) + r" \\")
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        return "\n".join(lines)

    mean_tex = build_table(mean_pivot, "mean")
    std_tex  = build_table(std_pivot,  "std")

    with open(out_mean, "w") as f: f.write(mean_tex + "\n")
    with open(out_std,  "w") as f: f.write(std_tex  + "\n")
    print(f"  Written {out_mean}")
    print(f"  Written {out_std}")


def make_ranks_table(ranks_csv: str, summary_json: str, out: str):
    """Generate average-rank table with Friedman p."""
    ranks = pd.read_csv(ranks_csv)
    with open(summary_json) as f:
        summary = json.load(f)

    lines = []
    lines.append(r"\begin{tabular}{lr}")
    lines.append(r"\toprule")
    lines.append(r"Algorithm & Avg.\ Rank \\")
    lines.append(r"\midrule")
    for _, row in ranks.iterrows():
        algo = row["algorithm"]
        disp = ALGO_DISPLAY.get(algo, algo)
        lines.append(f"{disp} & {row['mean_rank']:.2f} " + r"\\")
    lines.append(r"\midrule")
    chi2 = summary.get("friedman_chi2", float("nan"))
    fp   = summary.get("friedman_p",   float("nan"))
    cd   = summary.get("nemenyi_CD",   float("nan"))
    k    = summary.get("k_for_CD",     "?")
    N    = summary.get("N_for_CD",     "?")
    lines.append(r"\multicolumn{2}{l}{"
                 + f"Friedman $\\chi^2={chi2:.2f}$, $p={fp:.3e}$"
                 + r"} \\")
    lines.append(r"\multicolumn{2}{l}{"
                 + f"Nemenyi CD$={cd:.3f}$ ($k={k}$, $N={N}$)"
                 + r"} \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    with open(out, "w") as f: f.write("\n".join(lines) + "\n")
    print(f"  Written {out}")


def make_wilcoxon_table(wilcoxon_csv: str, out: str, control: str = "qpso"):
    """Win/Tie/Loss table vs control and Holm-corrected p."""
    wdf = pd.read_csv(wilcoxon_csv)
    algos = [a for a in ALGO_ORDER if a in wdf["algorithm"].unique() and a != control]
    fns   = [f for f in FN_ORDER   if f in wdf["function"].unique()]

    lines = []
    col_spec = "l" + "c" * len(fns) + "ccc"
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")
    fn_headers = " & ".join([FUNCTION_DISPLAY.get(fn, fn)[:4] for fn in fns])
    lines.append(r"Algorithm & " + fn_headers + r" & W & T & L \\")
    lines.append(r"\midrule")

    for algo in algos:
        sub = wdf[wdf["algorithm"] == algo]
        cells = []
        W = T = L = 0
        for fn in fns:
            row = sub[sub["function"] == fn]
            if row.empty:
                cells.append("---"); continue
            p_holm = float(row["wilcoxon_holm"].iloc[0])
            rel    = float(row["rel_improvement"].iloc[0])
            if p_holm < 0.05 and rel > 0.01:
                cells.append(r"\win"); W += 1
            elif p_holm >= 0.05:
                cells.append(r"\tie"); T += 1
            else:
                cells.append(r"\loss"); L += 1
        disp = ALGO_DISPLAY.get(algo, algo)
        lines.append(disp + " & " + " & ".join(cells)
                     + f" & {W} & {T} & {L}" + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    # note: \win \tie \loss should be defined in the preamble as coloured markers
    with open(out, "w") as f: f.write("\n".join(lines) + "\n")
    print(f"  Written {out}")


def make_numbers_tex(summary_json: str, stats_csv: str, out: str):
    """Generate \newcommand macros for every number quoted in prose."""
    with open(summary_json) as f:
        summary = json.load(f)
    stats = pd.read_csv(stats_csv)

    cmds = []
    def cmd(name, val):
        cmds.append(f"\\newcommand{{\\{name}}}{{{val}}}")

    cmd("numFriedmanChiSq",   f"{summary['friedman_chi2']:.2f}")
    cmd("numFriedmanP",       f"{summary['friedman_p']:.3e}")
    cmd("numNemenyiCD",       f"{summary['nemenyi_CD']:.3f}")
    cmd("numNemenyiK",        str(summary['k_for_CD']))
    cmd("numNemenyiN",        str(summary['N_for_CD']))
    cmd("numNRuns",           str(summary['n_runs']))

    # Key per-function numbers for prose
    for fn in FN_ORDER:
        for algo in ["qpso", "ant-fixed", "bee-fixed", "ant-cosine", "ant-feedback"]:
            row = stats[(stats["function"] == fn) & (stats["algorithm"] == algo)]
            if row.empty: continue
            mean_v = float(row["mean"].iloc[0])
            std_v  = float(row["std"].iloc[0])
            fn_s   = fn.replace("_","").replace("-","")
            algo_s = algo.replace("-","").replace("_","")
            if np.isfinite(mean_v):
                cmd(f"num{fn_s.capitalize()}{algo_s.capitalize()}Mean",
                    f"{mean_v:.2e}")
                cmd(f"num{fn_s.capitalize()}{algo_s.capitalize()}Std",
                    f"{std_v:.2e}")

    with open(out, "w") as f:
        f.write("% Auto-generated by make_tables.py — do not edit manually\n")
        f.write("\n".join(cmds) + "\n")
    print(f"  Written {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp",      default="e1_classical")
    parser.add_argument("--analysis", default="results/analysis/e1_classical")
    args = parser.parse_args()

    stats_csv    = os.path.join(args.analysis, "stats.csv")
    wilcoxon_csv = os.path.join(args.analysis, "wilcoxon.csv")
    ranks_csv    = os.path.join(args.analysis, "ranks.csv")
    summary_json = os.path.join(args.analysis, "summary.json")

    print(f"Generating tables from {args.analysis}…")

    make_classical_table(
        stats_csv, wilcoxon_csv,
        os.path.join(OUT_DIR, "classical_mean.tex"),
        os.path.join(OUT_DIR, "classical_std.tex"),
    )
    make_ranks_table(
        ranks_csv, summary_json,
        os.path.join(OUT_DIR, "ranks.tex"),
    )
    make_wilcoxon_table(
        wilcoxon_csv,
        os.path.join(OUT_DIR, "wilcoxon.tex"),
    )
    make_numbers_tex(
        summary_json, stats_csv,
        os.path.join(OUT_DIR, "numbers.tex"),
    )
    print("Done.")


if __name__ == "__main__":
    main()
