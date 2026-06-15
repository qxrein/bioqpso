import json
import time
from pathlib import Path

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

from .problems import make_feature_selection_problem
from .optimizers import ALGORITHMS_TO_TEST
from .plots import plot_convergence


def _evaluate_feature_selection_solution(x, X, y):
    x = np.asarray(x)
    mask = x >= 0.5
    selected = int(np.sum(mask))
    if selected == 0:
        return 0.0, 0

    X_sel = X[:, mask]
    clf = KNeighborsClassifier(n_neighbors=5)
    scores = cross_val_score(clf, X_sel, y, cv=5)
    mean_acc = float(np.mean(scores))
    return mean_acc, selected


def _print_feature_selection_table(dataset_label, fs_stats):
    print(f"\n--- Feature Selection Results ({dataset_label}) ---")
    header = (
        f"{'Algorithm':<16} | {'Mean Acc':<12} | {'Std Acc':<12} | "
        f"{'Avg Features':<14} | {'Avg Time (s)':<12}"
    )
    print(header)
    print("-" * len(header))
    for name, data in fs_stats.items():
        print(
            f"{name:<16} | "
            f"{data['MeanAcc']:<12.4f} | "
            f"{data['StdAcc']:<12.4f} | "
            f"{data['AvgFeatures']:<14.2f} | "
            f"{data['Time (s)']:<12.4f}"
        )


def _save_feature_selection_json(output_json, dataset_label, fs_stats, n_runs, max_iter, n_particles):
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "dataset": dataset_label,
        "n_runs": n_runs,
        "max_iter": max_iter,
        "n_particles": n_particles,
        "results": {
            algo: {
                "MeanAcc": float(data["MeanAcc"]),
                "StdAcc": float(data["StdAcc"]),
                "AvgFeatures": float(data["AvgFeatures"]),
                "Time (s)": float(data["Time (s)"]),
            }
            for algo, data in fs_stats.items()
        },
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved feature selection results to {output_json}")


def run_feature_selection_experiment(
    dataset_label="Breast Cancer",
    fs_problem=None,
    output_json=None,
    plot_name=None,
    n_runs=10,
    max_iter=200,
    n_particles=30,
):
    print(f"\n--- Running Feature Selection Experiment ({dataset_label}) ---")

    if fs_problem is None:
        fs_problem = make_feature_selection_problem()

    fs_problem = fs_problem.copy()
    plot_name = plot_name or dataset_label.lower().replace(" ", "_")
    fs_problem["name"] = plot_name

    X_fs = fs_problem["X"]
    y_fs = fs_problem["y"]

    fs_stats = {}
    fs_histories = {}

    for name, config in ALGORITHMS_TO_TEST.items():
        print(f"Running {name} on feature selection...")
        algo_class = config["class"]
        params = config["params"]

        run_accuracies = []
        run_features = []
        run_histories = []

        start_time = time.time()
        for _ in range(n_runs):
            optimizer = algo_class(
                problem=fs_problem,
                n_particles=n_particles,
                max_iter=max_iter,
                **params,
            )
            _, history = optimizer.run()
            run_histories.append(history)

            acc, n_feat = _evaluate_feature_selection_solution(
                optimizer.gbest_pos, X_fs, y_fs
            )
            run_accuracies.append(acc)
            run_features.append(n_feat)

        end_time = time.time()

        fs_stats[name] = {
            "MeanAcc": np.mean(run_accuracies),
            "StdAcc": np.std(run_accuracies),
            "AvgFeatures": np.mean(run_features),
            "Time (s)": (end_time - start_time) / n_runs,
        }
        fs_histories[name] = np.mean(run_histories, axis=0)

    _print_feature_selection_table(dataset_label, fs_stats)

    if output_json is not None:
        _save_feature_selection_json(
            output_json, dataset_label, fs_stats, n_runs, max_iter, n_particles
        )

    plot_convergence(fs_histories, plot_name)
