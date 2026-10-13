#!/usr/bin/env python3
"""
E8 — Feature selection with proper nested CV (parallel, practical budget).

Protocol:
- Outer: stratified 5-fold × 6 repeats = 30 outer evaluations per (algo, dataset)
- Inner fitness: 5-fold k-NN accuracy on outer-train split only
- Outer-test accuracy: (a) k-NN and (b) RBF-SVM
- Fitness: 0.7*(1-acc_inner) + 0.3*|S|/D
- Datasets: Breast Cancer (D=30), Wine (D=13), Ionosphere (D=34),
            Madelon-synthetic (D=500, limited folds for compute reasons)
- Baselines: all-features, MI top-D//4
- FS budget: n=30, T=100 (3030 FEs) for speed; D=500 uses T=50
- Parallelism: each (algo, fold) is one job via joblib
"""
from __future__ import annotations
import json, os, sys, time, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

import numpy as np
from scipy import stats
from sklearn.datasets import load_breast_cancer, load_wine, make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.feature_selection import mutual_info_classif
from joblib import Parallel, delayed
import yaml

from bioqpso.budget import FEBudget
from bioqpso.boundary import BoundaryHandler
from bioqpso.runner import _make_algorithm

OUT_DIR = "results/e8_feature_selection"
os.makedirs(OUT_DIR, exist_ok=True)

with open("config/seeds.yaml") as f:
    SEEDS = [int(s) for s in yaml.safe_load(f)["seeds"]]

N_PARTICLES     = 30
N_OUTER_FOLDS   = 5
N_OUTER_REPEATS = 6   # → 30 outer folds
N_INNER_FOLDS   = 3   # 3-fold inner CV for speed
N_JOBS          = 6   # parallel workers

ALGORITHMS = {
    "qpso":             {},
    "pso":              {},
    "aco_r":            {},
    "abc":              {},
    "gwo":              {},
    "woa":              {},
    "ant-fixed":        {"phi3": 0.3},
    "bee-fixed":        {"phi3": 0.3},
    "ant-cosine":       {},
    "ant-feedback":     {"W": 20, "delta": 0.02},
}


def load_datasets():
    ds = {}
    bc = load_breast_cancer()
    ds["breast_cancer"] = {
        "X": bc.data.astype(float), "y": bc.target,
        "name": "Breast Cancer", "max_iter": 50}
    wi = load_wine()
    ds["wine"] = {
        "X": wi.data.astype(float), "y": wi.target,
        "name": "Wine", "max_iter": 50}
    # Ionosphere via synthetic fallback
    X_ion, y_ion = make_classification(
        n_samples=351, n_features=34, n_informative=10,
        n_redundant=10, n_repeated=0, random_state=0)
    ds["ionosphere"] = {
        "X": X_ion.astype(float), "y": y_ion,
        "name": "Ionosphere-synthetic", "max_iter": 50}
    # High-dim: synthetic with 500 features, 5 informative (fewer folds)
    X_mad, y_mad = make_classification(
        n_samples=600, n_features=500, n_informative=5,
        n_redundant=10, n_clusters_per_class=1, random_state=42)
    ds["madelon_synth"] = {
        "X": X_mad.astype(float), "y": y_mad,
        "name": "Madelon-synthetic (D=500)", "max_iter": 20,
        "n_repeats": 2}   # 10 outer folds only (compute-limited)
    return ds


def _run_one_fold(algo, extra, X_tr, y_tr, X_te, y_te, D, max_iter, seed):
    """Run one optimizer on one outer fold and return test accuracies."""
    max_fes = N_PARTICLES * (max_iter + 1)

    def fitness(x):
        mask = np.asarray(x) >= 0.5
        n = int(mask.sum())
        if n == 0:
            return 1.0
        try:
            acc = float(np.mean(cross_val_score(
                KNeighborsClassifier(n_neighbors=5),
                X_tr[:, mask], y_tr, cv=N_INNER_FOLDS, scoring="accuracy")))
        except Exception:
            acc = 0.0
        return 0.7 * (1.0 - acc) + 0.3 * n / D

    rng = np.random.default_rng(seed)
    bh  = BoundaryHandler(0.0, 1.0, rng=rng)
    bud = FEBudget(fitness, max_fes, soft=True)
    opt = _make_algorithm(algo, bud, 0.0, 1.0, D,
                          N_PARTICLES, max_iter, rng, bh,
                          hist_every=max_iter, extra=extra)
    opt._initialize()
    opt.run()
    gbest = opt.gbest_pos.copy()

    mask = gbest >= 0.5
    n_sel = int(mask.sum())
    if n_sel == 0:
        return 0.0, 0.0, 0

    X_tr_s, X_te_s = X_tr[:, mask], X_te[:, mask]
    knn_acc = float(KNeighborsClassifier(n_neighbors=5).fit(
        X_tr_s, y_tr).score(X_te_s, y_te))
    svm_acc = float(SVC(kernel="rbf", C=1.0, gamma="scale").fit(
        X_tr_s, y_tr).score(X_te_s, y_te))
    return knn_acc, svm_acc, n_sel


def _baseline_fold(X_tr, y_tr, X_te, y_te, D):
    """Evaluate all-features and MI top-k baselines on one fold."""
    results = {}
    # all features
    knn = KNeighborsClassifier(n_neighbors=5).fit(X_tr, y_tr).score(X_te, y_te)
    svm = SVC(kernel="rbf", C=1.0, gamma="scale").fit(X_tr, y_tr).score(X_te, y_te)
    results["all_features"] = (knn, svm, D)
    # MI top-D//4
    mi = mutual_info_classif(X_tr, y_tr, random_state=42)
    k = max(1, D // 4)
    top = np.argsort(mi)[-k:]
    knn2 = KNeighborsClassifier(n_neighbors=5).fit(X_tr[:, top], y_tr).score(X_te[:, top], y_te)
    svm2 = SVC(kernel="rbf", C=1.0, gamma="scale").fit(X_tr[:, top], y_tr).score(X_te[:, top], y_te)
    results["mi_topk"] = (knn2, svm2, k)
    return results


def holm(p_dict):
    names = list(p_dict)
    raw = np.array([p_dict[n] for n in names], dtype=float)
    n = len(raw)
    order = np.argsort(raw)
    adj, running = np.zeros(n), 0.0
    for rank, idx in enumerate(order):
        running = max(running, raw[idx] * (n - rank))
        adj[idx] = min(running, 1.0)
    return {names[i]: float(adj[i]) for i in range(n)}


def run_e8():
    print("=== E8: Feature Selection (Nested CV, parallel) ===\n")
    datasets = load_datasets()
    all_results = {}

    for ds_name, ds_info in datasets.items():
        X_full = ds_info["X"]
        y_full = ds_info["y"]
        D = X_full.shape[1]
        max_iter = ds_info["max_iter"]
        n_reps   = ds_info.get("n_repeats", N_OUTER_REPEATS)
        print(f"--- {ds_info['name']} (n={X_full.shape[0]}, D={D}, T={max_iter}, folds={n_reps*N_OUTER_FOLDS}) ---")
        t0 = time.time()

        # Build list of folds
        folds = []
        for rep in range(n_reps):
            skf = StratifiedKFold(n_splits=N_OUTER_FOLDS, shuffle=True,
                                  random_state=SEEDS[rep % len(SEEDS)])
            for train_idx, test_idx in skf.split(X_full, y_full):
                folds.append((X_full[train_idx], y_full[train_idx],
                               X_full[test_idx],  y_full[test_idx]))

        n_folds = len(folds)  # 30

        # Baselines (sequential, fast)
        bl_data = {"all_features": {"knn": [], "svm": [], "n_sel": []},
                   "mi_topk":      {"knn": [], "svm": [], "n_sel": []}}
        for X_tr, y_tr, X_te, y_te in folds:
            br = _baseline_fold(X_tr, y_tr, X_te, y_te, D)
            for bname, (k, s, ns) in br.items():
                bl_data[bname]["knn"].append(k)
                bl_data[bname]["svm"].append(s)
                bl_data[bname]["n_sel"].append(ns)

        # Algorithms (parallel over folds)
        algo_data = {a: {"knn": [], "svm": [], "n_sel": []} for a in ALGORITHMS}
        jobs = [(a, extra, fi)
                for a, extra in ALGORITHMS.items()
                for fi in range(n_folds)]

        print(f"  {len(jobs)} jobs ({len(ALGORITHMS)} algos × {n_folds} folds)  "
              f"ETA~{len(jobs)*0.8/N_JOBS/60:.0f}min")

        def _job(a, extra, fi):
            X_tr, y_tr, X_te, y_te = folds[fi]
            seed = SEEDS[fi % len(SEEDS)]
            return a, fi, _run_one_fold(a, extra, X_tr, y_tr, X_te, y_te,
                                         D, max_iter, seed)

        raw = Parallel(n_jobs=N_JOBS, verbose=0)(
            delayed(_job)(a, extra, fi) for a, extra, fi in jobs)

        for a, fi, (knn_a, svm_a, n_sel) in raw:
            algo_data[a]["knn"].append(knn_a)
            algo_data[a]["svm"].append(svm_a)
            algo_data[a]["n_sel"].append(n_sel)

        # Statistics
        qpso_knn = np.array(algo_data["qpso"]["knn"])
        p_raw = {}
        for a in ALGORITHMS:
            v = np.array(algo_data[a]["knn"])
            if a != "qpso" and len(v) == len(qpso_knn) and not np.array_equal(v, qpso_knn):
                try:
                    _, p = stats.wilcoxon(v, qpso_knn, alternative="two-sided")
                    p_raw[a] = float(p)
                except Exception:
                    p_raw[a] = float("nan")
        p_holm = holm(p_raw)

        # Print
        print(f"\n  {'Algorithm':22s}  {'kNN mean':>9s}  {'SVM mean':>9s}  {'#feat':>6s}  {'Holm p':>8s}")
        print("  " + "-"*65)
        for a in ALGORITHMS:
            v = algo_data[a]
            ph = p_holm.get(a, 1.0)
            sig = "*" if ph < 0.05 else ""
            print(f"  {a:22s}  {np.mean(v['knn']):>9.4f}  {np.mean(v['svm']):>9.4f}"
                  f"  {np.mean(v['n_sel']):>6.1f}  {ph:>8.4f}{sig}")
        print(f"  {'all_features':22s}  {np.mean(bl_data['all_features']['knn']):>9.4f}"
              f"  {np.mean(bl_data['all_features']['svm']):>9.4f}  {D:>6d}  {'baseline':>8s}")
        print(f"  {'mi_topk (D//4)':22s}  {np.mean(bl_data['mi_topk']['knn']):>9.4f}"
              f"  {np.mean(bl_data['mi_topk']['svm']):>9.4f}"
              f"  {np.mean(bl_data['mi_topk']['n_sel']):>6.1f}  {'baseline':>8s}")

        elapsed = time.time() - t0
        print(f"\n  Done in {elapsed/60:.1f} min")

        all_results[ds_name] = {
            "display_name": ds_info["name"],
            "n_samples": int(X_full.shape[0]),
            "n_features": D,
            "max_iter": max_iter,
            "n_outer_folds": n_folds,
            "algorithms": {
                a: {
                    "knn_mean": float(np.mean(algo_data[a]["knn"])),
                    "knn_std":  float(np.std(algo_data[a]["knn"])),
                    "svm_mean": float(np.mean(algo_data[a]["svm"])),
                    "svm_std":  float(np.std(algo_data[a]["svm"])),
                    "n_sel_mean": float(np.mean(algo_data[a]["n_sel"])),
                    "knn_per_fold": [float(x) for x in algo_data[a]["knn"]],
                    "wilcoxon_p_holm": p_holm.get(a, 1.0),
                }
                for a in ALGORITHMS
            },
            "baselines": {
                b: {"knn_mean": float(np.mean(bl_data[b]["knn"])),
                    "knn_std":  float(np.std(bl_data[b]["knn"])),
                    "svm_mean": float(np.mean(bl_data[b]["svm"])),
                    "svm_std":  float(np.std(bl_data[b]["svm"])),
                    "n_sel_mean": float(np.mean(bl_data[b]["n_sel"]))}
                for b in bl_data
            },
        }

        # Save incrementally
        _save(all_results)

    print("\n=== E8 complete ===")
    return all_results


def _save(results):
    def _ser(o):
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, np.generic): return o.item()
        if isinstance(o, dict): return {k: _ser(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)): return [_ser(v) for v in o]
        return o
    with open(f"{OUT_DIR}/e8_results.json", "w") as f:
        json.dump(_ser(results), f, indent=2)


if __name__ == "__main__":
    t0 = time.time()
    run_e8()
    print(f"Total: {(time.time()-t0)/60:.1f} min")
