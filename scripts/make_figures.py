#!/usr/bin/env python3
"""
make_figures.py — all manuscript figures (PDF + PNG), sized for the printed text
width so that 7 pt in the file is 7 pt on paper. Colour plus line style and marker
identify an algorithm, with one fixed style per algorithm across all figures.
"""
from __future__ import annotations
import glob, json, os, sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy import stats

OUT = "paper/figures"
os.makedirs(OUT, exist_ok=True)
WIDTH = 4.8  # inches, LNCS text width
plt.rcParams.update({
    "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "legend.fontsize": 7, "axes.linewidth": 0.5, "pdf.fonttype": 42,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.minor.visible": False,
    "ytick.minor.visible": False, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.linewidth": 0.3, "grid.color": "#d9d9d9", "axes.axisbelow": True,
})
STYLE = {  # colour, line style, marker
    "qpso":      ("#4F46B8", "-",  "o"),
    "gwo":       ("#0072B2", "-",  "v"),
    "lshade":    ("#882255", "--", "X"),
    "abc":       ("#E69F00", "--", "D"),
    "woa":       ("#56B4E9", ":",  "P"),
    "ant-fixed": ("#D55E00", "-",  "s"),
    "ant-cosine":("#CC79A7", "--", "^"),
    "ant-feedback": ("#CC79A7", ":", "*"),
    "bee-fixed": ("#009E73", "-",  "h"),
    "bee-per_particle": ("#009E73", ":", "p"),
}
NAME = {"qpso": "QPSO", "gwo": "GWO", "lshade": "L-SHADE", "abc": "ABC", "woa": "WOA",
        "ant-fixed": "AntBio", "ant-cosine": "Ant-Cos", "ant-feedback": "Ant-FB",
        "bee-fixed": "BeeBio", "bee-per_particle": "Bee-PP"}
FN = {"sphere": "Sphere", "rosenbrock": "Rosenbrock", "zakharov": "Zakharov", "dixon_price": "Dixon–Price",
      "rastrigin": "Rastrigin", "ackley": "Ackley", "griewank": "Griewank", "schwefel": "Schwefel",
      "levy": "Levy", "styblinski_tang": "Styblinski–Tang"}
ST_OPT = -39.16616570377142 * 30


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT}/{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  saved", name)


def shared_legend(fig, algos, ncol=4, y=1.0):
    handles = [plt.Line2D([0], [0], color=STYLE[a][0], ls=STYLE[a][1], marker=STYLE[a][2], ms=3.5, lw=1.0,
                          label=NAME[a]) for a in algos]
    fig.legend(handles=handles, loc="lower center", ncol=ncol, frameon=False,
               bbox_to_anchor=(0.5, y), handlelength=2.6, columnspacing=1.2)


def fig_convergence():
    """Median error curves; the median is not dominated by single diverged runs."""
    algos = ["qpso", "gwo", "woa", "abc", "lshade", "ant-fixed", "ant-cosine", "bee-fixed"]
    curves = defaultdict(list)
    for a in algos:
        for f in FN:
            for p in glob.glob(f"results/e1_classical/{a}__{f}__d30__r*.json"):
                curves[a, f].append(json.load(open(p))["convergence"])
    fig, axes = plt.subplots(4, 3, figsize=(WIDTH, 5.2), sharex=True)
    axes = axes.ravel()
    for ax, f in zip(axes, FN):
        fstar = ST_OPT if f == "styblinski_tang" else 0.0
        for a in algos:
            L = min(len(c) for c in curves[a, f])
            arr = np.array([c[:L] for c in curves[a, f]], dtype=float) - fstar
            med = np.median(arr, axis=0)
            x = np.arange(L) * 300 + 30
            col, ls, mk = STYLE[a]
            floor = 1e-30
            y = np.maximum(med, floor)
            ax.plot(x, y, color=col, ls=ls, lw=0.9, marker=mk, ms=2.8, markevery=(L // 6, L // 3))
        ax.set_yscale("log")
        ax.set_ylim(bottom=1e-30 if ax.get_ylim()[0] < 1e-28 else None)
        ax.set_title(FN[f], pad=2)
        ax.set_xticks([0, 15000, 30000]); ax.set_xticklabels(["0", "15k", "30k"])
        ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(numticks=4))
        ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    for ax in axes[len(FN):]:
        ax.set_visible(False)
    for ax in axes[6:9]:
        ax.tick_params(labelbottom=True)
    axes[9].set_xlabel("Function evaluations")
    for ax in axes[[0, 3, 6, 9]]:
        ax.set_ylabel(r"Median $f-f^*$")
    fig.tight_layout(pad=0.3, h_pad=0.6, w_pad=0.4)
    # The two unused panels hold the shared legend.
    handles = [plt.Line2D([0], [0], color=STYLE[a][0], ls=STYLE[a][1], marker=STYLE[a][2], ms=3.5, lw=1.0,
                          label=NAME[a]) for a in algos]
    pos = axes[10].get_position()
    fig.legend(handles=handles, ncol=2, frameon=False, handlelength=2.6, columnspacing=1.2,
               loc="center", bbox_to_anchor=((pos.x0 + axes[11].get_position().x1) / 2, (pos.y0 + pos.y1) / 2))
    save(fig, "convergence_classical")


SEQ = ListedColormap(["#08519c", "#3182bd", "#6baed6", "#bdd7e7", "#eff3ff", "#f7f7f7"])


def rank_cells(ax, R, xl, yl, cmap=SEQ, title=None, n=None):
    n = n or R.shape[1]
    ax.imshow(R, cmap=ListedColormap(SEQ.colors[:n]) if n < 6 else cmap, vmin=0.5, vmax=n + 0.5, aspect="auto")
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.set_xticks(range(len(xl))); ax.set_xticklabels(xl)
    ax.set_yticks(range(len(yl))); ax.set_yticklabels(yl)
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            v = R[i, j]
            t = f"{v:.0f}" if float(v).is_integer() else f"{v:.1f}"
            ax.text(j, i, t, ha="center", va="center", fontsize=6.5,
                    color="white" if v <= 2 else "#222222")
    if title:
        ax.set_title(title, pad=3)


def fig_phi3():
    """Per-function rank of the fixed-phi3 settings; no-bio is the true phi3 = 0 run."""
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.3), sharey=True)
    fns = list(FN)
    for ax, L, title in zip(axes, ("ant", "bee"), ("Ant leader", "Bee leader")):
        keys = [f"{L}-nobio"] + [f"{L}-fixed-p0{i}" for i in range(1, 6)]
        M = np.zeros((len(fns), len(keys)))
        for i, f in enumerate(fns):
            for j, k in enumerate(keys):
                M[i, j] = np.mean([json.load(open(p))["final_f"]
                                   for p in glob.glob(f"results/e2_controls/{k}__{f}__d30__r*.json")])
        R = np.apply_along_axis(stats.rankdata, 1, M)
        rank_cells(ax, R, ["no-bio", "0.1", "0.2", "0.3", "0.4", "0.5"], [FN[f] for f in fns], title=title)
        ax.set_xlabel(r"Fixed $\phi_3$")
    fig.tight_layout(pad=0.3, w_pad=0.8)
    save(fig, "phi3_heatmap")


def fig_scalability():
    dims = [30, 50, 100]
    fns = ["sphere", "rastrigin", "rosenbrock", "levy", "schwefel", "griewank"]
    algos = ["qpso", "gwo", "ant-fixed", "ant-cosine", "ant-feedback", "bee-fixed", "bee-per_particle"]
    fig, axes = plt.subplots(2, 3, figsize=(WIDTH, 3.5))
    for ax, f in zip(axes.ravel(), fns):
        vals = {}
        for a in algos:
            vals[a] = [np.mean([json.load(open(p))["final_f"] for p in
                                glob.glob(f"results/e4_scalability/{a}__{f}__d{D}__r*.json")]) for D in dims]
        nz = [v for a in algos for v in vals[a] if v > 1e-250]
        lo, hi = np.log10(min(nz)), np.log10(max(nz))
        pad = max((hi - lo) * 0.08, 0.3)
        band = 10 ** (lo - 3 * pad)  # separate "= 0" band below the data
        any_zero = False
        for a in algos:
            col, ls, mk = STYLE[a]
            y = np.array(vals[a], dtype=float)
            zero = y <= 1e-250
            any_zero = any_zero or zero.any()
            ax.plot(np.array(dims)[~zero], y[~zero], color=col, ls=ls, marker=mk, ms=3, lw=0.9)
            ax.plot(np.array(dims)[zero], [band] * zero.sum(), ls="none", color=col, marker=mk, ms=3,
                    clip_on=False)
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_xticks(dims); ax.set_xticklabels(dims)
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(numticks=5))
        ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        bottom = band / 10 ** pad if any_zero else 10 ** (lo - pad)
        ax.set_ylim(bottom, 10 ** (hi + pad))
        if any_zero:
            boundary = 10 ** (lo - 1.5 * pad)
            ax.axhline(boundary, color="#888888", lw=0.5, ls=(0, (2, 2)))
            ax.text(101, band, "= 0", ha="left", va="center", fontsize=7, clip_on=False)
        ax.set_title(FN[f], pad=2)
    for ax in axes[1]:
        ax.set_xlabel("$D$")
    for ax in axes[:, 0]:
        ax.set_ylabel("Mean best $f$")
    fig.subplots_adjust(left=0.12, right=0.97, top=0.88, bottom=0.11, hspace=0.42, wspace=0.42)
    shared_legend(fig, algos, ncol=4, y=0.92)
    save(fig, "scalability")


def fig_sensitivity():
    p6 = "results/analysis/e6_params.json"
    if not os.path.exists(p6):
        print("  E6 parameters missing; skipped")
        return
    params = json.load(open(p6))["params"]
    fns = ["sphere", "rastrigin", "levy", "styblinski_tang", "schwefel", "griewank", "F12017", "F42017", "F112017", "F212017"]
    lab = [FN[f][:6] if f in FN else f.replace("2017", "") for f in fns]
    titles = {"rho": r"$\rho$ (AntBio)", "Delta": r"$\Delta$ (AntBio)", "Lscout": r"$L_{\mathrm{scout}}$ (BeeBio)",
              "W": r"$W$ (Ant-FB)", "delta": r"$\delta$ (Ant-FB)",
              "phi3bounds": r"$[\phi_3^{\min},\phi_3^{\max}]$ (Ant-Cos)", "beta": r"$\beta$ start–end (AntBio)"}
    data = defaultdict(list)
    for p in glob.glob("results/e6_sensitivity/*.json"):
        r = json.load(open(p))
        data[r["param"], r["value"], r["function"]].append(r["final_f"])
    fig, axes = plt.subplots(3, 3, figsize=(WIDTH, 5.6), sharey=True)
    axes = axes.ravel()
    for ax, (p, t) in zip(axes, titles.items()):
        vals = params[p]
        M = np.array([[np.mean(data[p, v, f]) for v in vals] for f in fns])
        R = np.apply_along_axis(stats.rankdata, 1, M)
        xl = [v.replace("-", "\n") for v in vals]
        rank_cells(ax, R, xl, lab, title=t, n=5)
        ax.tick_params(axis="x", labelsize=6.5)
    for ax in axes[7:]:
        ax.set_visible(False)
    for ax in axes[4:7]:
        ax.tick_params(labelleft=True)
    fig.tight_layout(pad=0.3, h_pad=0.8, w_pad=0.5)
    save(fig, "sensitivity")


if __name__ == "__main__":
    import matplotlib.ticker  # noqa: F401
    if glob.glob("results/e1_classical/lshade__sphere*"):
        fig_convergence()
    fig_phi3()
    fig_scalability()
    fig_sensitivity()
