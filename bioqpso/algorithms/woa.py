"""woa.py — Whale Optimization Algorithm (Mirjalili & Lewis 2016).
Own NumPy implementation — no mealpy dependency.
"""
from __future__ import annotations

import numpy as np

from .base import BaseOptimizer


class WOA(BaseOptimizer):
    """Whale Optimization Algorithm (standard formulation).

    b controls the spiral shape (default 1.0).
    """

    def __init__(self, *args, b: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.b = b

    def run(self) -> tuple[float, list[float]]:
        self._initialize()

        for t in range(self.max_iter):
            if self.func.exhausted():
                break
            a = 2.0 - 2.0 * t / max(self.max_iter - 1, 1)
            a2 = -1.0 - t / max(self.max_iter - 1, 1)  # from −1 to −2

            for i in range(self.n_particles):
                # A, C, l and p are per-agent scalars, as in the reference code.
                A = 2.0 * a * self.rng.random() - a
                C = 2.0 * self.rng.random()
                l = (a2 - 1.0) * self.rng.random() + 1.0
                p = self.rng.random()

                if p < 0.5:
                    if abs(A) < 1.0:
                        # Shrinking encircling
                        D = np.abs(C * self.gbest_pos - self.x[i])
                        self.x[i] = self.boundary(self.gbest_pos - A * D)
                    else:
                        # Random search; the reference code draws the leader per dimension
                        idx = self.rng.integers(self.n_particles, size=self.dim)
                        x_rand = self.x[idx, np.arange(self.dim)]
                        D = np.abs(C * x_rand - self.x[i])
                        self.x[i] = self.boundary(x_rand - A * D)
                else:
                    # Spiral bubble-net attack
                    D_prime = np.abs(self.gbest_pos - self.x[i])
                    self.x[i] = self.boundary(
                        D_prime * np.exp(self.b * l) * np.cos(2.0 * np.pi * l)
                        + self.gbest_pos
                    )

            # Evaluate after all moves: gbest is fixed during the move loop (reference code).
            for i in range(self.n_particles):
                f_val = self.func(self.x[i])
                self._update_bests(i, f_val)
            self._record(t)

        return self.gbest_val, self.convergence_history
