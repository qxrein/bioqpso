"""aco_r.py — ACO_R: Ant Colony Optimisation for continuous domains
(Socha & Dorigo 2008)."""
from __future__ import annotations

import numpy as np

from .base import BaseOptimizer


class ACO_R(BaseOptimizer):
    """Ant Colony Optimisation for Real-valued search spaces.

    Parameters
    ----------
    q : float   Locality parameter (default 0.1).
    xi : float  Pheromone evaporation speed (default 0.85).
    """

    def __init__(self, *args, q: float = 0.1, xi: float = 0.85, **kwargs):
        super().__init__(*args, **kwargs)
        self.q = q
        self.xi = xi

    def run(self) -> tuple[float, list[float]]:
        self._initialize()

        # Archive is the initial population; n_fes already spent in _initialize.
        archive = self.x.copy()
        # Re-use already-evaluated values from _initialize() — no extra FEs.
        archive_f = self.pbest_val.copy()

        m = self.n_particles
        ranks = np.arange(1, m + 1, dtype=float)
        weights = (1.0 / (self.q * m * np.sqrt(2 * np.pi))) * np.exp(
            -((ranks - 1) ** 2) / (2 * (self.q * m) ** 2)
        )
        weights /= weights.sum()

        for t in range(self.max_iter):
            if self.func.exhausted():
                break
            # Sort archive by fitness
            idx = np.argsort(archive_f)
            archive = archive[idx]
            archive_f = archive_f[idx]

            sigma = self.xi * np.abs(archive[-1] - archive[0]) + 1e-300

            new_sols = np.empty((m, self.dim))
            new_f = np.empty(m)
            for j in range(m):
                k = int(self.rng.choice(m, p=weights))
                center = archive[k]
                candidate = self.boundary(center + self.rng.standard_normal(self.dim) * sigma)
                new_sols[j] = candidate
                new_f[j] = self.func(candidate)

            combined = np.vstack([archive, new_sols])
            combined_f = np.concatenate([archive_f, new_f])
            best_idx = np.argsort(combined_f)[:m]
            archive = combined[best_idx]
            archive_f = combined_f[best_idx]

            # Update gbest
            if archive_f[0] < self.gbest_val:
                self.gbest_val = float(archive_f[0])
                self.gbest_pos = archive[0].copy()

            self._record(t)

        return self.gbest_val, self.convergence_history
