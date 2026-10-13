#!/usr/bin/env python3
"""
make_figures.py — Generate all manuscript figures (PDF + PNG, 300 dpi).

Colour-blind-safe palette; identical colour+marker per algorithm across all figures.
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

OUT_DIR = "paper/figures"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Colour-blind-safe palette (Wong 2011) ──────────────────────────────────
PALETTE = {
    "pso":              ("#E69F00", "o",  "-"),
    "qpso":             ("#56B4E9", "s",  "-"),
    "aco_r":            ("#009E73", "^",  "--"),
    "abc":              ("#F0E442", "D",  "--"),
    "gwo":              ("#0072B2", "v",  "-"),
    "woa":              ("#D55E00", "P",  "-."),
    "ant-fixed":        ("#CC79A7", "h",  "-"),
    "bee-fixed":        ("#000000", "H",  "--"),
    "ant-linear":       ("#56B4E9", "x",  ":"),
    "ant-cosine":       ("#009E73", "+",  ":"),
    "ant-feedback":     ("#D55E00", "*",  ":"),
    "bee-per_particle": ("#CC79A7", "p",  "-."),
}
ALGO_ORDER = list(PALETTE.keys())

FUNCTION_DISPLAY = {
    "sphere":          "Sphere",   "rosenbrock": "Rosenbrock",
    "zakharov":        "Zakharov", "dixon_price": "Dixon-Price",
    "rastrigin":       "Rastrigin","ackley":      "Ackley",
    "griewank":        "Griewank", "schwefel":    "Schwefel",
    "levy":            "Levy",     "styblinski_tang": "Styblinski-Tang",
}

ALGO_DISPLAY = {
    "pso":"PSO", "qpso":"QPSO", "aco_r":"ACO$_R$", "abc":"ABC",
    "gwo":"GWO", "woa":"WOA", "ant-fixed":"AntBio", "bee-fixed":"BeeBio",
    "ant-linear":"Ant-Lin", "ant-cosine":"Ant-Cos",
    "ant-feedback":"Ant-FB", "bee-per_particle":"Bee-PP",
}


def _save(fig, name: str, dpi: int = 300):
    p_png = os.path.join(OUT_DIR, name + ".png")
    p_pdf = os.path.join(OUT_DIR, name + ".pdf")
    fig.savefig(p_png, dpi=dpi, bbox_inches="tight")
    fig.savefig(p_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {p_png}")


# ── 1. Convergence curves ──────────────────────────────────────────────────

def plot_convergence(results_dir: str, top_n: int = 6):
    """Small-multiples convergence plots, log-y, FEs on x-axis."""
    # Load mean convergence histories per (algo, function)
    records = {}
    for fn_file in Path(results_dir).glob("*.json"):
        with open(fn_file) as f:
            r = json.load(f)
        key = (r["algorithm"], r["function"])
        if key not in records:
            records[key] = []
        records[key].append(r.get("convergence", []))

    fns = list(FUNCTION_DISPLAY.keys())
    # Always show QPSO + GWO + top-4 by rank on E1
    priority = ["gwo", "qpso", "abc", "ant-fixed", "ant-feedback", "ant-cosine",
                "bee-fixed", "ant-linear"]
    show_algos = [a for a in priority if a in ALGO_DISPLAY][:top_n + 2]

    hist_every = 10
    n_cols = 5; n_rows = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 7))
    axes = axes.flatten()

    for ax_idx, fn in enumerate(fns):
        ax = axes[ax_idx]
        for algo in show_algos:
            runs = records.get((algo, fn), [])
            if not runs: continue
            L = min(len(r) for r in runs)
            arr = np.array([r[:L] for r in runs], dtype=float)
            mean = np.mean(arr, axis=0)
            std  = np.std(arr, axis=0)
            x    = np.arange(L) * hist_every * 30   # FEs
            col, marker, ls = PALETTE.get(algo, ("#999999","o","-"))
            ax.semilogy(x, np.abs(mean) + 1e-300, color=col, ls=ls,
                        label=ALGO_DISPLAY.get(algo, algo), lw=1.3)
            # ±1 std shading
            lo = np.abs(mean - std) + 1e-300
            hi = np.abs(mean + std) + 1e-300
            ax.fill_between(x, lo, hi, color=col, alpha=0.10)
        ax.set_title(FUNCTION_DISPLAY.get(fn, fn), fontsize=9)
        ax.set_xlabel("FEs", fontsize=7)
        ax.set_ylabel("Fitness", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, which="both", ls="--", alpha=0.3)

    # Legend in last axis if spare
    if len(fns) < len(axes):
        legend_ax = axes[len(fns)]
        handles = [
            plt.Line2D([0],[0], color=PALETTE.get(a,("#999","o","-"))[0],
                       ls=PALETTE.get(a,("#999","o","-"))[2],
                       label=ALGO_DISPLAY.get(a,a))
            for a in show_algos
        ]
        legend_ax.legend(handles=handles, fontsize=7, loc="center")
        legend_ax.axis("off")

    fig.tight_layout()
    _save(fig, "convergence_classical")


# ── 2. CD diagram ──────────────────────────────────────────────────────────

def plot_cd_diagram(ranks_csv: str, summary_json: str, name: str):
    """Simple horizontal CD diagram."""
    ranks_df = pd.read_csv(ranks_csv)
    with open(summary_json) as f:
        summary = json.load(f)

    cd = summary.get("nemenyi_CD", 1.0)
    algos = ranks_df["algorithm"].tolist()
    rank_vals = ranks_df["mean_rank"].tolist()
    algo_disp = [ALGO_DISPLAY.get(a, a) for a in algos]

    fig, ax = plt.subplots(figsize=(9, max(3, len(algos) * 0.4)))
    y_positions = list(range(len(algos)))

    for y, (algo, r) in enumerate(zip(algos, rank_vals)):
        col, marker, _ = PALETTE.get(algo, ("#555555", "o", "-"))
        ax.barh(y, r, color=col, alpha=0.8, height=0.6)
        ax.text(r + 0.05, y, f"{r:.2f}", va="center", fontsize=8)

    # CD bar on the first algorithm
    best_rank = min(rank_vals)
    ax.annotate("", xy=(best_rank + cd, len(algos) - 0.5),
                xytext=(best_rank, len(algos) - 0.5),
                arrowprops=dict(arrowstyle="<->", color="red", lw=1.5))
    ax.text(best_rank + cd / 2, len(algos) - 0.3,
            f"CD={cd:.2f}", ha="center", color="red", fontsize=8)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(algo_disp, fontsize=9)
    ax.set_xlabel("Average Rank", fontsize=10)
    ax.set_title(f"Critical Difference Diagram (Nemenyi, α=0.05)", fontsize=10)
    ax.invert_yaxis()
    ax.grid(True, axis="x", ls="--", alpha=0.4)
    fig.tight_layout()
    _save(fig, name)


# ── 3. Fixed φ3 heat-map ──────────────────────────────────────────────────

def plot_phi3_heatmap(e2_results_dir: str):
    """Heat-map: function × φ3 for ant leader; log10 mean fitness."""
    phi3_vals = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    phi3_map  = {0.0: "ant-fixed-p00", 0.1: "ant-fixed-p01", 0.2: "ant-fixed-p02",
                 0.3: "ant-fixed-p03", 0.4: "ant-fixed-p04", 0.5: "ant-fixed-p05"}
    fns = list(FUNCTION_DISPLAY.keys())

    # Load
    records: dict[tuple, list[float]] = {}
    for fn_file in Path(e2_results_dir).glob("*.json"):
        with open(fn_file) as f:
            r = json.load(f)
        records.setdefault((r["algorithm"], r["function"]), []).append(r["final_f"])

    mat = np.full((len(fns), len(phi3_vals)), np.nan)
    for j, phi3 in enumerate(phi3_vals):
        algo = phi3_map[phi3]
        for i, fn in enumerate(fns):
            vals = records.get((algo, fn), [])
            if vals:
                mat[i, j] = np.log10(np.mean(vals) + 1e-300)

    # Normalise per function
    norm_mat = mat.copy()
    for i in range(len(fns)):
        row = mat[i]
        finite = row[np.isfinite(row)]
        if len(finite):
            rng = finite.max() - finite.min()
            norm_mat[i] = (row - finite.min()) / (rng + 1e-300)

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(norm_mat, aspect="auto", cmap="RdYlGn_r",
                   vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks(range(len(phi3_vals)))
    ax.set_xticklabels([f"φ₃={v}" for v in phi3_vals], fontsize=9)
    ax.set_yticks(range(len(fns)))
    ax.set_yticklabels([FUNCTION_DISPLAY[fn] for fn in fns], fontsize=9)
    ax.set_xlabel("Fixed φ₃ value", fontsize=10)
    ax.set_title("Normalised log₁₀ mean fitness (Ant leader, fixed φ₃ sweep)\n"
                 "Green = low (better), Red = high (worse)", fontsize=9)
    plt.colorbar(im, ax=ax, label="Normalised log₁₀ fitness")
    fig.tight_layout()
    _save(fig, "phi3_heatmap")


# ── 4. φ3 traces ───────────────────────────────────────────────────────────

def plot_phi3_traces(e1_results_dir: str):
    """φ3(t) mean ± IQR for scheduled variants on Levy and Schwefel."""
    trace_algos = ["ant-linear", "ant-cosine", "ant-feedback"]
    target_fns  = ["levy", "schwefel", "sphere", "rastrigin"]

    records: dict[tuple, list] = {}
    for fn_file in Path(e1_results_dir).glob("*.json"):
        with open(fn_file) as f:
            r = json.load(f)
        if r["algorithm"] not in trace_algos: continue
        phi3 = r.get("phi3_hist")
        if phi3 is None: continue
        records.setdefault((r["algorithm"], r["function"]), []).append(phi3)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    axes = axes.flatten()
    for ax_idx, fn in enumerate(target_fns):
        ax = axes[ax_idx]
        for algo in trace_algos:
            runs = records.get((algo, fn), [])
            if not runs: continue
            L = min(len(r) for r in runs)
            arr = np.array([r[:L] for r in runs], dtype=float)
            mean = np.mean(arr, axis=0)
            p25  = np.percentile(arr, 25, axis=0)
            p75  = np.percentile(arr, 75, axis=0)
            x    = np.arange(L)
            col, marker, ls = PALETTE.get(algo, ("#555","o","-"))
            ax.plot(x, mean, color=col, ls=ls,
                    label=ALGO_DISPLAY.get(algo, algo), lw=1.4)
            ax.fill_between(x, p25, p75, color=col, alpha=0.15)
        ax.axhline(0.3, color="gray", ls="--", lw=1.0, label="Fixed φ₃=0.3")
        ax.set_title(FUNCTION_DISPLAY.get(fn, fn), fontsize=9)
        ax.set_xlabel("Iteration", fontsize=8)
        ax.set_ylabel("φ₃(t)", fontsize=8)
        ax.set_ylim(0.05, 0.55)
        ax.legend(fontsize=7)
        ax.grid(True, ls="--", alpha=0.3)
    fig.tight_layout()
    _save(fig, "phi3_traces")


# ── 5. Scalability ─────────────────────────────────────────────────────────

def plot_scalability(e4_dir: str):
    """Log-log fitness vs D for 6 functions × subset of algorithms."""
    dims = [30, 50, 100]
    fns  = ["sphere", "rastrigin", "rosenbrock", "levy", "schwefel", "griewank"]
    show_algos = ["qpso", "gwo", "ant-fixed", "ant-cosine", "ant-feedback",
                  "bee-fixed", "bee-per_particle"]

    records: dict[tuple, list[float]] = {}
    for fn_file in Path(e4_dir).glob("*.json"):
        with open(fn_file) as f:
            r = json.load(f)
        records.setdefault((r["algorithm"], r["function"], r["dim"]), []).append(r["final_f"])

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    for ax_idx, fn in enumerate(fns):
        ax = axes[ax_idx]
        for algo in show_algos:
            y_vals = []
            for d in dims:
                vals = records.get((algo, fn, d), [])
                y_vals.append(np.mean(vals) if vals else np.nan)
            if any(np.isfinite(v) for v in y_vals):
                col, marker, ls = PALETTE.get(algo, ("#555","o","-"))
                ax.semilogy(dims, [abs(v)+1e-300 for v in y_vals],
                            color=col, ls=ls, marker=marker, ms=5,
                            label=ALGO_DISPLAY.get(algo, algo), lw=1.3)
        ax.set_title(FUNCTION_DISPLAY.get(fn, fn), fontsize=9)
        ax.set_xlabel("D", fontsize=8)
        ax.set_ylabel("Mean fitness", fontsize=8)
        ax.set_xticks(dims)
        ax.legend(fontsize=6, ncol=2)
        ax.grid(True, ls="--", alpha=0.3)
    fig.tight_layout()
    _save(fig, "scalability")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--e1",  default="results/e1_classical")
    parser.add_argument("--e2",  default="results/e2_controls")
    parser.add_argument("--e4",  default="results/e4_scalability")
    parser.add_argument("--analysis", default="results/analysis/e1_classical")
    args = parser.parse_args()

    print("Generating figures…")

    if os.path.isdir(args.e1):
        plot_convergence(args.e1)
        plot_phi3_traces(args.e1)

    analysis_ranks = os.path.join(args.analysis, "ranks.csv")
    analysis_summ  = os.path.join(args.analysis, "summary.json")
    if os.path.exists(analysis_ranks):
        plot_cd_diagram(analysis_ranks, analysis_summ, "cd_classical")

    if os.path.isdir(args.e2):
        plot_phi3_heatmap(args.e2)

    if os.path.isdir(args.e4):
        plot_scalability(args.e4)

    print("Figures done.")


if __name__ == "__main__":
    main()
