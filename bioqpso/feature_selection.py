import time
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


def run_feature_selection_experiment():
    print("\n--- Running Feature Selection Experiment (Breast Cancer) ---")
    fs_problem = make_feature_selection_problem()
    fs_problem["name"] = "feature_selection"

    N_RUNS_FS = 10
    MAX_ITER_FS = 200
    N_PARTICLES_FS = 30

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
        for _ in range(N_RUNS_FS):
            optimizer = algo_class(
                problem=fs_problem,
                n_particles=N_PARTICLES_FS,
                max_iter=MAX_ITER_FS,
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
            "Time (s)": (end_time - start_time) / N_RUNS_FS,
        }
        fs_histories[name] = np.mean(run_histories, axis=0)

    print("\n--- Feature Selection Results (Breast Cancer) ---")
    header = f"{'Algorithm':<16} | {'Mean Acc':<12} | {'Std Acc':<12} | {'Avg Features':<14} | {'Avg Time (s)':<12}"
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

    plot_convergence(fs_histories, "feature_selection")

