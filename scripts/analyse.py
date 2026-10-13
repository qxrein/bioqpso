#!/usr/bin/env python3
"""
analyse.py — Statistics for all experiment results.

Usage:
    python scripts/analyse.py --exp e1_classical  --results results/e1_classical
    python scripts/analyse.py --exp e2_controls   --results results/e2_controls
    python scripts/analyse.py --exp e3_cec2017_d30 --results results/e3_cec2017_d30 --cec

Outputs:
    results/analysis/<exp>/wilcoxon.csv
    results/analysis/<exp>/stats.csv
    results/analysis/<exp>/ranks.csv
    results/analysis/<exp>/summary.json
"""
from __future__ import annotations
import argparse, json, os, sys, warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# Load results
# ─────────────────────────────────────────────────────────────

def load_results(results_dir: str) -> pd.DataFrame:
    """Read all per-run JSON files into a DataFrame."""
    rows = []
    for fn in Path(results_dir).glob("*.json"):
        try:
            with open(fn) as f:
                rec = json.load(f)
            rows.append({
                "algorithm": rec["algorithm"],
                "function":  rec["function"],
                "dim":       rec.get("dim", 30),
                "run":       rec["run"],
                "seed":      rec["seed"],
                "final_f":   float(rec["final_f"]),
                "n_fes":     int(rec["n_fes"]),
                "wall_time": float(rec.get("wall_time_s", 0)),
            })
        except Exception as e:
            print(f"  [warn] could not load {fn}: {e}")
    return pd.DataFrame(rows)

# ─────────────────────────────────────────────────────────────
# Statistics helpers
# ─────────────────────────────────────────────────────────────

def holm_bonferroni(p_values: np.ndarray) -> np.ndarray:
    """Return Holm-Bonferroni adjusted p-values."""
    n = len(p_values)
    order = np.argsort(p_values)
    adj = np.zeros(n)
    running_max = 0.0
    for rank, idx in enumerate(order):
        raw = p_values[idx] * (n - rank)
        running_max = max(running_max, raw)
        adj[idx] = min(running_max, 1.0)
    return adj


def rank_biserial_ci(x: np.ndarray, y: np.ndarray,
                      n_boot: int = 10000, seed: int = 0) -> tuple[float, float, float]:
    """Rank-biserial r and 95% bootstrap CI. Positive r: x < y (x better)."""
    n1, n2 = len(x), len(y)
    u, _ = sp_stats.mannwhitneyu(x, y, alternative="less")
    r = 1.0 - 2.0 * u / (n1 * n2)
    rng = np.random.default_rng(seed)
    boot_r = []
    for _ in range(n_boot):
        bx = rng.choice(x, n1, replace=True)
        by = rng.choice(y, n2, replace=True)
        bu, _ = sp_stats.mannwhitneyu(bx, by, alternative="less")
        boot_r.append(1.0 - 2.0 * bu / (n1 * n2))
    lo, hi = np.percentile(boot_r, [2.5, 97.5])
    return float(r), float(lo), float(hi)


def per_function_stats(df: pd.DataFrame, algos: list[str],
                        functions: list[str], control: str = "qpso") -> dict:
    """Per-function descriptive stats and Wilcoxon vs control."""
    results = {}
    for fn in functions:
        fn_df = df[df["function"] == fn]
        fn_res = {}
        raw: dict[str, np.ndarray] = {}
        for algo in algos:
            vals = fn_df[fn_df["algorithm"] == algo]["final_f"].values
            if len(vals) == 0:
                continue
            raw[algo] = vals
            fn_res[algo] = {
                "mean":   float(np.mean(vals)),
                "std":    float(np.std(vals)),
                "median": float(np.median(vals)),
                "best":   float(np.min(vals)),
                "worst":  float(np.max(vals)),
            }

        # Wilcoxon vs control
        ctrl = raw.get(control)
        p_raw = {}
        for algo in algos:
            if algo == control or algo not in raw:
                p_raw[algo] = 1.0
                continue
            a = raw[algo]
            if len(a) != len(ctrl) or np.array_equal(a, ctrl):
                p_raw[algo] = 1.0
                continue
            try:
                _, p = sp_stats.wilcoxon(a, ctrl, alternative="less")
                p_raw[algo] = float(p)
            except Exception:
                p_raw[algo] = float("nan")

        # Holm correction (exclude control)
        non_ctrl = [k for k in p_raw if k != control and not np.isnan(p_raw[k])]
        if non_ctrl:
            raw_arr = np.array([p_raw[k] for k in non_ctrl])
            adj_arr = holm_bonferroni(raw_arr)
            p_holm = {k: float(adj_arr[i]) for i, k in enumerate(non_ctrl)}
        else:
            p_holm = {}
        p_holm[control] = 1.0

        # Rank-biserial (subset, no bootstrap for every cell — just r)
        r_vals = {}
        for algo in algos:
            if algo == control or algo not in raw or ctrl is None:
                r_vals[algo] = 0.0
                continue
            a = raw[algo]
            if len(a) == len(ctrl):
                try:
                    u, _ = sp_stats.mannwhitneyu(a, ctrl, alternative="less")
                    r_vals[algo] = float(1.0 - 2.0 * u / (len(a) * len(ctrl)))
                except Exception:
                    r_vals[algo] = float("nan")

        # Practical significance
        ctrl_mean = float(np.mean(ctrl)) if ctrl is not None else float("nan")
        rel_improv = {}
        for algo, st in fn_res.items():
            if algo == control or abs(ctrl_mean) < 1e-300:
                rel_improv[algo] = 0.0
            else:
                rel_improv[algo] = float((ctrl_mean - st["mean"]) / abs(ctrl_mean))

        fn_res["_wilcoxon_raw"]  = p_raw
        fn_res["_wilcoxon_holm"] = p_holm
        fn_res["_rank_biserial"] = r_vals
        fn_res["_rel_improvement"] = rel_improv
        results[fn] = fn_res
    return results


def global_ranks(per_fn: dict, algos: list[str], functions: list[str]) -> dict:
    """Mean rank over functions (lower = better)."""
    rank_lists: dict[str, list[float]] = {a: [] for a in algos}
    for fn in functions:
        fn_res = per_fn.get(fn, {})
        means = {a: fn_res[a]["mean"] for a in algos if a in fn_res}
        if not means:
            continue
        sorted_algos = sorted(means, key=means.get)
        for rank, algo in enumerate(sorted_algos, 1):
            rank_lists[algo].append(float(rank))
    return {a: float(np.mean(v)) if v else float("nan")
            for a, v in rank_lists.items()}


def friedman_test(per_fn: dict, algos: list[str],
                   functions: list[str]) -> tuple[float, float]:
    """Friedman test using per-function means as blocks."""
    matrix = []
    for fn in functions:
        fn_res = per_fn.get(fn, {})
        row = [fn_res[a]["mean"] if a in fn_res else float("nan") for a in algos]
        matrix.append(row)
    M = np.array(matrix, dtype=float)
    valid = ~np.any(np.isnan(M), axis=1)
    M = M[valid]
    if M.shape[0] < 3:
        return float("nan"), float("nan")
    try:
        stat, p = sp_stats.friedmanchisquare(*[M[:, j] for j in range(M.shape[1])])
        return float(stat), float(p)
    except Exception:
        return float("nan"), float("nan")


def nemenyi_cd(k: int, n: int, alpha: float = 0.05) -> float:
    """Nemenyi critical difference CD = q_alpha * sqrt(k(k+1)/(6N))."""
    # q_alpha for alpha=0.05 from Demšar 2006 Table 5 (two-tailed)
    q_table = {
        2: 1.960, 3: 2.344, 4: 2.569, 5: 2.728, 6: 2.850,
        7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164, 11: 3.219,
        12: 3.268, 13: 3.313, 14: 3.354, 15: 3.391, 20: 3.532,
        25: 3.621, 30: 3.728,
    }
    # Interpolate for k not in table
    ks = sorted(q_table.keys())
    if k <= ks[0]:
        q = q_table[ks[0]]
    elif k >= ks[-1]:
        q = q_table[ks[-1]]
    else:
        for i in range(len(ks)-1):
            if ks[i] <= k <= ks[i+1]:
                lo, hi = ks[i], ks[i+1]
                q = q_table[lo] + (q_table[hi]-q_table[lo])*(k-lo)/(hi-lo)
                break
    return float(q * np.sqrt(k * (k+1) / (6.0 * n)))


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp",     required=True, help="Experiment name (for output subdir)")
    parser.add_argument("--results", required=True, help="Directory of per-run JSON files")
    parser.add_argument("--control", default="qpso", help="Control algorithm name")
    parser.add_argument("--cec",     action="store_true", help="CEC mode (subtract f_bias)")
    args = parser.parse_args()

    out_dir = os.path.join("results", "analysis", args.exp)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading results from {args.results}…")
    df = load_results(args.results)
    if df.empty:
        print("No results found. Exiting.")
        return

    print(f"  {len(df)} records, "
          f"{df['algorithm'].nunique()} algorithms, "
          f"{df['function'].nunique()} functions, "
          f"{df['run'].nunique()} runs")

    algos     = sorted(df["algorithm"].unique())
    functions = sorted(df["function"].unique())
    control   = args.control if args.control in algos else algos[0]

    # ── Per-function stats ─────────────────────────────────
    per_fn = per_function_stats(df, algos, functions, control=control)

    # ── Global ranks ───────────────────────────────────────
    g_ranks = global_ranks(per_fn, algos, functions)
    friedman_stat, friedman_p = friedman_test(per_fn, algos, functions)
    k = len(algos)
    N = len(functions)
    cd = nemenyi_cd(k, N)

    print(f"\nFriedman: χ²={friedman_stat:.3f}  p={friedman_p:.4e}")
    print(f"Nemenyi CD (k={k}, N={N}): {cd:.3f}")
    print("\nGlobal average ranks (lower = better):")
    for algo, r in sorted(g_ranks.items(), key=lambda x: x[1]):
        print(f"  {algo:<22}: {r:.2f}")

    # ── FE sanity check ────────────────────────────────────
    max_allowed = 30 * 1001
    over = df[df["n_fes"] > max_allowed]
    if not over.empty:
        print(f"\n[WARN] {len(over)} records exceed FE budget!")
    else:
        print(f"\nFE check: all {len(df)} records within budget ✓")
    nan_inf = df[~np.isfinite(df["final_f"])]
    if not nan_inf.empty:
        print(f"[WARN] {len(nan_inf)} non-finite final_f values!")
    else:
        print("NaN/Inf check: all results finite ✓")

    # ── Save CSVs ─────────────────────────────────────────
    # stats.csv: mean/std/median/best/worst per (function, algo)
    stats_rows = []
    for fn in functions:
        for algo in algos:
            s = per_fn.get(fn, {}).get(algo, {})
            if not s or "_" in algo[:1]:
                continue
            stats_rows.append({
                "function": fn, "algorithm": algo,
                "mean":   s.get("mean"),  "std":    s.get("std"),
                "median": s.get("median"), "best":  s.get("best"),
                "worst":  s.get("worst"),
            })
    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(os.path.join(out_dir, "stats.csv"), index=False)

    # wilcoxon.csv
    wilcox_rows = []
    for fn in functions:
        fn_res = per_fn.get(fn, {})
        for algo in algos:
            wilcox_rows.append({
                "function":        fn,
                "algorithm":       algo,
                "wilcoxon_raw":    fn_res.get("_wilcoxon_raw", {}).get(algo, 1.0),
                "wilcoxon_holm":   fn_res.get("_wilcoxon_holm", {}).get(algo, 1.0),
                "rank_biserial":   fn_res.get("_rank_biserial", {}).get(algo, 0.0),
                "rel_improvement": fn_res.get("_rel_improvement", {}).get(algo, 0.0),
            })
    wilcox_df = pd.DataFrame(wilcox_rows)
    wilcox_df.to_csv(os.path.join(out_dir, "wilcoxon.csv"), index=False)

    # ranks.csv
    ranks_df = pd.DataFrame([
        {"algorithm": a, "mean_rank": r}
        for a, r in sorted(g_ranks.items(), key=lambda x: x[1])
    ])
    ranks_df.to_csv(os.path.join(out_dir, "ranks.csv"), index=False)

    # summary.json
    summary = {
        "experiment":     args.exp,
        "n_algorithms":   k,
        "n_functions":    N,
        "n_runs":         int(df["run"].nunique()),
        "control":        control,
        "friedman_chi2":  friedman_stat,
        "friedman_p":     friedman_p,
        "nemenyi_CD":     cd,
        "k_for_CD":       k,
        "N_for_CD":       N,
        "global_ranks":   {a: float(r) for a,r in g_ranks.items()},
        "fe_budget_ok":   len(over) == 0,
        "nan_inf_count":  len(nan_inf),
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved to {out_dir}/  (stats.csv, wilcoxon.csv, ranks.csv, summary.json)")


if __name__ == "__main__":
    main()
