import json
import time
from pathlib import Path

import numpy as np
from scipy import stats


def run_experiment(problem, algorithms, n_runs=30, max_iter=1000, n_particles=50):
    print("\n--- Running Experiment ---")
    print(f"Problem: {problem['name']} (D={problem['D']})")
    print(f"Runs: {n_runs}, Iterations: {max_iter}, Particles: {n_particles}\n")

    results_stats = {}
    results_histories = {}
    results_raw_bests = {}

    for name, config in algorithms.items():
        print(f"Running {name}...")
        algo_class = config["class"]
        params = config["params"]

        run_bests = []
        run_histories = []

        start_time = time.time()
        for _ in range(n_runs):
            optimizer = algo_class(
                problem=problem,
                n_particles=n_particles,
                max_iter=max_iter,
                **params,
            )
            best_val, history = optimizer.run()
            run_bests.append(best_val)
            run_histories.append(history)

        end_time = time.time()

        results_stats[name] = {
            "Best": np.min(run_bests),
            "Worst": np.max(run_bests),
            "Mean": np.mean(run_bests),
            "StdDev": np.std(run_bests),
            "Time (s)": (end_time - start_time) / n_runs,
        }

        results_histories[name] = np.mean(run_histories, axis=0)
        results_raw_bests[name] = run_bests

    return results_stats, results_histories, results_raw_bests


def print_results_table(stats_data):
    print("\n--- Experimental Results ---")
    header = f"{'Algorithm':<16} | {'Mean':<12} | {'StdDev':<12} | {'Best':<12} | {'Worst':<12} | {'Avg Time (s)':<12}"
    print(header)
    print("-" * len(header))
    for name, data in stats_data.items():
        print(
            f"{name:<16} | {data['Mean']:<12.2e} | {data['StdDev']:<12.2e} | "
            f"{data['Best']:<12.2e} | {data['Worst']:<12.2e} | {data['Time (s)']:<12.4f}"
        )


def run_statistical_analysis(results_raw_bests, control_name="QPSO"):
    print("\n--- Statistical Significance (Wilcoxon & Friedman) ---")
    print(f"(Comparing against control: {control_name})")

    if control_name not in results_raw_bests:
        print(f"Control algorithm '{control_name}' not in results. Skipping.")
        return {}

    control_data = np.array(results_raw_bests[control_name])
    algo_names = list(results_raw_bests.keys())

    try:
        friedman_args = [np.array(results_raw_bests[name]) for name in algo_names]
        friedman_stat, friedman_p = stats.friedmanchisquare(*friedman_args)
    except ValueError:
        friedman_stat, friedman_p = np.nan, np.nan

    header = (
        f"{'Algorithm':<16} | {'Wilcoxon p-value':<18} | "
        f"{'Significant (p<0.05)?':<22} | {'Friedman chi2':<14} | {'Friedman p':<12}"
    )
    print(header)
    print("-" * len(header))

    p_values = {}

    for name, data in results_raw_bests.items():
        data = np.array(data)

        if name == control_name:
            wilcoxon_p = 1.0
            significant = "Control"
        else:
            if np.array_equal(data, control_data):
                wilcoxon_p = 1.0
            else:
                try:
                    _, wilcoxon_p = stats.wilcoxon(
                        data, control_data, alternative="less"
                    )
                except ValueError:
                    wilcoxon_p = np.nan

            significant = "YES" if (not np.isnan(wilcoxon_p) and wilcoxon_p < 0.05) else "No"

        p_values[name] = wilcoxon_p

        print(
            f"{name:<16} | "
            f"{wilcoxon_p:<18.4e} | "
            f"{significant:<22} | "
            f"{friedman_stat:<14.4f} | "
            f"{friedman_p:<12.4e}"
        )

    return p_values


def save_experiment_results(stats_data, raw_bests, output_dir, problem_name):
    """Write summary stats and per-run bests to *output_dir*."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "problem": problem_name,
        "stats": {
            algo: {k: float(v) if np.isscalar(v) else v for k, v in data.items()}
            for algo, data in stats_data.items()
        },
        "raw_bests": {
            algo: [float(v) for v in values] for algo, values in raw_bests.items()
        },
    }

    out_path = output_dir / f"results_{problem_name}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved experiment results to {out_path}")

