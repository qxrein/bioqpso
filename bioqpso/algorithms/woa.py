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
                r = self.rng.random()
                A = 2.0 * a * self.rng.random(self.dim) - a
                C = 2.0 * self.rng.random(self.dim)
                p = self.rng.random()
                l = self.rng.uniform(a2, 1.0, self.dim)

                if p < 0.5:
                    if np.linalg.norm(A) < 1.0:
                        # Shrinking encircling
                        D = np.abs(C * self.gbest_pos - self.x[i])
                        self.x[i] = self.boundary(self.gbest_pos - A * D)
                    else:
                        # Random search (exploration)
                        rand_idx = int(self.rng.integers(self.n_particles))
                        x_rand = self.x[rand_idx]
                        D = np.abs(C * x_rand - self.x[i])
                        self.x[i] = self.boundary(x_rand - A * D)
                else:
                    # Spiral bubble-net attack
                    D_prime = np.abs(self.gbest_pos - self.x[i])
                    self.x[i] = self.boundary(
                        D_prime * np.exp(self.b * l) * np.cos(2.0 * np.pi * l)
                        + self.gbest_pos
                    )

                f_val = self.func(self.x[i])
                self._update_bests(i, f_val)
            self._record(t)

        return self.gbest_val, self.convergence_history
