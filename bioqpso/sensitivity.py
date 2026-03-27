import json
import os

import numpy as np
import matplotlib.pyplot as plt

from .problems import PROBLEMS
from .optimizers import AntBioQPSO


def _to_serializable(obj):
    """Convert numpy types to plain Python for JSON dumping."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    return obj


def run_sensitivity_analysis(output_dir=None):
    print("\n--- Running Sensitivity Analysis (AntBioQPSO on sphere) ---")

    N_RUNS = 10
    MAX_ITER = 500
    N_PARTICLES = 30

    evaporation_rates = [0.05, 0.1, 0.2, 0.3, 0.5]
    pheromone_deposits = [0.5, 1.0, 2.0, 5.0, 10.0]
    phi3_values = [0.1, 0.2, 0.3, 0.4, 0.5]

    problem = PROBLEMS["sphere"].copy()
    problem["name"] = "sphere"

    def run_for_params(phi1, phi2, phi3, evaporation_rate, pheromone_deposit):
        run_bests = []
        for _ in range(N_RUNS):
            optimizer = AntBioQPSO(
                problem=problem,
                n_particles=N_PARTICLES,
                max_iter=MAX_ITER,
                phi1=phi1,
                phi2=phi2,
                phi3=phi3,
                beta=0.5,
                evaporation_rate=evaporation_rate,
                pheromone_deposit=pheromone_deposit,
            )
            best_val, _ = optimizer.run()
            run_bests.append(best_val)
        run_bests = np.array(run_bests)
        return np.mean(run_bests), np.std(run_bests)

    base_phi1, base_phi2, base_phi3 = 0.4, 0.3, 0.3
    base_evaporation_rate = 0.1
    base_pheromone_deposit = 1.0

    evap_means, evap_stds = [], []
    for evap in evaporation_rates:
        mean_val, std_val = run_for_params(
            base_phi1, base_phi2, base_phi3, evap, base_pheromone_deposit
        )
        evap_means.append(mean_val)
        evap_stds.append(std_val)

    depo_means, depo_stds = [], []
    for depo in pheromone_deposits:
        mean_val, std_val = run_for_params(
            base_phi1, base_phi2, base_phi3, base_evaporation_rate, depo
        )
        depo_means.append(mean_val)
        depo_stds.append(std_val)

    phi3_means, phi3_stds = [], []
    for phi3 in phi3_values:
        remaining = 1.0 - phi3
        scale = remaining / (base_phi1 + base_phi2)
        phi1 = base_phi1 * scale
        phi2 = base_phi2 * scale
        mean_val, std_val = run_for_params(
            phi1, phi2, phi3, base_evaporation_rate, base_pheromone_deposit
        )
        phi3_means.append(mean_val)
        phi3_stds.append(std_val)

    print("\nSensitivity: evaporation_rate")
    header = f"{'evaporation_rate':<18} | {'Mean':<12} | {'StdDev':<12}"
    print(header)
    print("-" * len(header))
    for v, m, s in zip(evaporation_rates, evap_means, evap_stds):
        print(f"{v:<18.3f} | {m:<12.2e} | {s:<12.2e}")

    print("\nSensitivity: pheromone_deposit")
    header = f"{'pheromone_deposit':<18} | {'Mean':<12} | {'StdDev':<12}"
    print(header)
    print("-" * len(header))
    for v, m, s in zip(pheromone_deposits, depo_means, depo_stds):
        print(f"{v:<18.3f} | {m:<12.2e} | {s:<12.2e}")

    print("\nSensitivity: phi3 (bio-leader weight)")
    header = f"{'phi3':<18} | {'Mean':<12} | {'StdDev':<12}"
    print(header)
    print("-" * len(header))
    for v, m, s in zip(phi3_values, phi3_means, phi3_stds):
        print(f"{v:<18.3f} | {m:<12.2e} | {s:<12.2e}")

    plt.figure(figsize=(12, 10))

    markers = ["o", "s", "^"]

    plt.subplot(3, 1, 1)
    plt.errorbar(
        evaporation_rates,
        evap_means,
        yerr=evap_stds,
        fmt="-o",
        capsize=4,
        color="black",
        ecolor="black",
    )
    plt.title("Sensitivity of AntBioQPSO to evaporation_rate (sphere)")
    plt.xlabel("evaporation_rate")
    plt.ylabel("Best Fitness")
    plt.grid(True, ls="--", alpha=0.5)

    plt.subplot(3, 1, 2)
    plt.errorbar(
        pheromone_deposits,
        depo_means,
        yerr=depo_stds,
        fmt="--s",
        capsize=4,
        color="black",
        ecolor="black",
    )
    plt.title("Sensitivity of AntBioQPSO to pheromone_deposit (sphere)")
    plt.xlabel("pheromone_deposit")
    plt.ylabel("Best Fitness")
    plt.grid(True, ls="--", alpha=0.5)

    plt.subplot(3, 1, 3)
    plt.errorbar(
        phi3_values,
        phi3_means,
        yerr=phi3_stds,
        fmt="-.^",
        capsize=4,
        color="black",
        ecolor="black",
    )
    plt.title("Sensitivity of AntBioQPSO to phi3 (sphere)")
    plt.xlabel("phi3")
    plt.ylabel("Best Fitness")
    plt.grid(True, ls="--", alpha=0.5)

    plt.tight_layout()
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, "sensitivity_analysis.png")
    else:
        save_path = "sensitivity_analysis.png"

    plt.savefig(save_path)
    print(f"\nSaved sensitivity analysis plot to {save_path}")
    plt.close()

    if output_dir is not None:
        results_path = os.path.join(output_dir, "sensitivity_results.json")
        sensitivity_results = {
            "problem_name": "sphere",
            "n_runs": N_RUNS,
            "max_iter": MAX_ITER,
            "n_particles": N_PARTICLES,
            "evaporation_rates": evaporation_rates,
            "pheromone_deposits": pheromone_deposits,
            "phi3_values": phi3_values,
            "evaporation_rate_results": {"means": evap_means, "stds": evap_stds},
            "pheromone_deposit_results": {"means": depo_means, "stds": depo_stds},
            "phi3_results": {"means": phi3_means, "stds": phi3_stds},
        }
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(_to_serializable(sensitivity_results), f, indent=2)
        print(f"Saved sensitivity results to {results_path}")

