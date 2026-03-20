import numpy as np
import matplotlib.pyplot as plt
import time

from bioqpso import run_experiment, PROBLEMS, ALGORITHMS_TO_TEST


def run_all_experiments(n_runs=30, max_iter=1000, n_particles=30):
    results = {}
    for problem_name, problem_config in PROBLEMS.items():
        problem = problem_config.copy()
        problem['name'] = problem_name

        stats, histories, _ = run_experiment(
            problem=problem,
            algorithms=ALGORITHMS_TO_TEST,
            n_runs=n_runs,
            max_iter=max_iter,
            n_particles=n_particles,
        )
        results[problem_name] = stats
    return results


def make_bar_charts(results):
    plt.style.use('default')
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 13,
            "axes.labelsize": 13,
            "legend.fontsize": 11,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "axes.linewidth": 1.2,
        }
    )

    algorithms = list(ALGORITHMS_TO_TEST.keys())
    functions = list(results.keys())

    # Black-and-white friendly: gray bars + hatching patterns
    hatches = ['/', '\\\\', 'x', '-', '+', 'o']
    colors = ['0.2'] * len(algorithms)

    # Combined bar chart across all functions
    x = np.arange(len(functions))
    bar_width = 0.8 / len(algorithms)

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, algo in enumerate(algorithms):
        mean_values = [results[f][algo]['Mean'] for f in functions]
        bars = ax.bar(
            x + i * bar_width - (len(algorithms) - 1) * bar_width / 2,
            mean_values,
            width=bar_width,
            label=algo,
            color=colors[i % len(colors)],
            edgecolor='black',
            linewidth=0.8,
        )
        for b in bars:
            b.set_hatch(hatches[i % len(hatches)])

    ax.set_xlabel('Benchmark Functions', fontweight='bold')
    ax.set_ylabel('Accuracy (Mean Value)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f.capitalize() for f in functions], fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, which='both', linestyle=':', linewidth=0.5, alpha=0.8)
    ax.legend(frameon=True, edgecolor='black', loc='upper right')

    plt.tight_layout()
    plt.savefig('accuracy_bar_graph.png', dpi=400, bbox_inches='tight')
    plt.close()

    # Individual per-function bar charts
    for func in functions:
        fig, ax = plt.subplots(figsize=(8, 5))
        mean_values = [results[func][algo]['Mean'] for algo in algorithms]
        x = np.arange(len(algorithms))

        bars = ax.bar(
            x,
            mean_values,
            width=0.6,
            color='0.8',
            edgecolor='black',
            linewidth=0.8,
        )
        for i, b in enumerate(bars):
            b.set_hatch(hatches[i % len(hatches)])

        ax.set_xlabel('Algorithms', fontweight='bold')
        ax.set_ylabel('Accuracy (Mean Value)', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(algorithms, fontweight='bold', rotation=15)
        ax.set_yscale('log')
        ax.grid(True, which='both', linestyle=':', linewidth=0.5, alpha=0.8)
        ax.set_title(f'{func.capitalize()} Function', fontweight='bold')

        plt.tight_layout()
        plt.savefig(f'bar_{func}.png', dpi=400, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    start = time.time()
    results = run_all_experiments()
    make_bar_charts(results)
    print(f"Bar charts generated in {time.time() - start:.2f} seconds.")
