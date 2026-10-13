#!/usr/bin/env python3
"""Generate supplementary outputs requested by reviewers."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bioqpso import (
    generate_extended_stats_report,
    run_feature_selection_experiment,
    run_phi_ratio_ablation,
    run_scalability_study,
    run_scheduling_ablation,
)
from bioqpso.problems import (
    make_ionosphere_feature_selection_problem,
    make_wine_feature_selection_problem,
)
from bioqpso.reviewer_experiments import (
    compute_complexity_table,
    patch_levy_linearada_stats,
    rerun_linearada_levy_timing,
    run_diversity_experiment,
    run_extended_sensitivity,
)


def main():
    output_dir = os.path.join("outputs", "adabioqpso")
    os.makedirs(output_dir, exist_ok=True)

    print("=== Task 1: Re-run LinearAda Levy timing ===")
    levy_stats = rerun_linearada_levy_timing(n_runs=30)
    with open(os.path.join(output_dir, "levy_linearada_rerun.json"), "w") as f:
        json.dump(levy_stats, f, indent=2)
    patch_levy_linearada_stats(output_dir, levy_stats)
    print(json.dumps(levy_stats, indent=2))

    print("\n=== Task 2: Extended sensitivity analysis ===")
    run_extended_sensitivity(output_dir)

    print("\n=== Task 3: Diversity traces (Sphere, Schwefel) ===")
    run_diversity_experiment(output_dir, problems=("sphere", "schwefel"), n_runs=30)

    print("\n=== Task 4: Complexity table ===")
    complexity = compute_complexity_table(output_dir)
    print(json.dumps(complexity, indent=2))

    print("\n=== Extended statistics from existing runs ===")
    report = generate_extended_stats_report(output_dir)
    print(json.dumps(report["average_ranks"], indent=2))

    print("\n=== Phi ratio ablation ===")
    run_phi_ratio_ablation(output_dir)

    print("\n=== Scheduling ablation ===")
    run_scheduling_ablation(output_dir)

    print("\n=== Scalability study ===")
    run_scalability_study(output_dir)

    print("\n=== Additional feature selection datasets ===")
    run_feature_selection_experiment(
        output_dir=output_dir,
        fs_problem=make_wine_feature_selection_problem(),
        plot_name="feature_selection_wine",
    )
    run_feature_selection_experiment(
        output_dir=output_dir,
        fs_problem=make_ionosphere_feature_selection_problem(),
        plot_name="feature_selection_ionosphere",
    )


if __name__ == "__main__":
    main()
