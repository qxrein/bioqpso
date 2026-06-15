from pathlib import Path

import matplotlib.pyplot as plt


def plot_convergence(histories, problem_name, output_dir=None):
    print(f"\n--- Plotting Convergence ({problem_name}) ---")
    plt.figure(figsize=(12, 8))

    # Black-and-white friendly styles
    linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 5))]
    markers = ["o", "s", "^", "D", "v", "x"]

    for idx, (name, history) in enumerate(histories.items()):
        ls = linestyles[idx % len(linestyles)]
        marker = markers[idx % len(markers)]
        plt.plot(
            history,
            label=name,
            linestyle=ls,
            marker=marker,
            markevery=max(len(history) // 20, 1),
            linewidth=1.5,
        )

    plt.title(f"Mean Convergence Curve ({problem_name})")
    plt.xlabel("Iteration")
    plt.ylabel("Best Fitness (Log Scale)")

    # Styblinski-Tang and some other benchmarks can be negative; plain `log` fails then.
    try:
        import numpy as _np

        all_values = _np.concatenate([_np.asarray(h, dtype=float) for h in histories.values()])
        y_min = float(_np.min(all_values))
        y_max = float(_np.max(all_values))

        if y_min <= 0:
            # Signed log scale supports negative/zero values and still resembles log behavior.
            span = max(abs(y_min), abs(y_max), 1.0)
            linthresh = max(span * 1e-6, 1e-12)
            plt.yscale("symlog", linthresh=linthresh)
        else:
            plt.yscale("log")
    except Exception:
        # Fallback to plain log if anything goes wrong; it will only warn/fail for non-positive values.
        plt.yscale("log")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)

    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"convergence_{problem_name}.png"
    else:
        out_path = Path(f"convergence_{problem_name}.png")

    plt.savefig(out_path)
    print(f"Saved convergence plot to {out_path}")
    plt.close()

