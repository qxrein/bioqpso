"""gwo.py — Grey Wolf Optimizer (Mirjalili et al. 2014).
Own NumPy implementation — no mealpy dependency.
"""
from __future__ import annotations

import numpy as np

from .base import BaseOptimizer


class GWO(BaseOptimizer):
    """Grey Wolf Optimizer.

    α is the best wolf, β the second-best, δ the third-best.
    a decreases from 2 to 0 over T iterations.
    """

    def run(self) -> tuple[float, list[float]]:
        self._initialize()

        # Sort initial population to identify α, β, δ
        order = np.argsort(self.pbest_val)
        alpha_pos = self.pbest_pos[order[0]].copy()
        alpha_score = float(self.pbest_val[order[0]])
        beta_pos = self.pbest_pos[order[1]].copy() if self.n_particles > 1 else alpha_pos.copy()
        beta_score = float(self.pbest_val[order[1]]) if self.n_particles > 1 else alpha_score
        delta_pos = self.pbest_pos[order[2]].copy() if self.n_particles > 2 else alpha_pos.copy()
        delta_score = float(self.pbest_val[order[2]]) if self.n_particles > 2 else alpha_score

        for t in range(self.max_iter):
            if self.func.exhausted():
                break
            a = 2.0 - 2.0 * t / max(self.max_iter - 1, 1)

            for i in range(self.n_particles):
                for leader_pos in (alpha_pos, beta_pos, delta_pos):
                    A = 2.0 * a * self.rng.random(self.dim) - a
                    C = 2.0 * self.rng.random(self.dim)
                    D = np.abs(C * leader_pos - self.x[i])
                    # accumulate position update (averaged over 3 leaders below)

                A1 = 2.0 * a * self.rng.random(self.dim) - a
                C1 = 2.0 * self.rng.random(self.dim)
                D_alpha = np.abs(C1 * alpha_pos - self.x[i])
                X1 = alpha_pos - A1 * D_alpha

                A2 = 2.0 * a * self.rng.random(self.dim) - a
                C2 = 2.0 * self.rng.random(self.dim)
                D_beta = np.abs(C2 * beta_pos - self.x[i])
                X2 = beta_pos - A2 * D_beta

                A3 = 2.0 * a * self.rng.random(self.dim) - a
                C3 = 2.0 * self.rng.random(self.dim)
                D_delta = np.abs(C3 * delta_pos - self.x[i])
                X3 = delta_pos - A3 * D_delta

                self.x[i] = self.boundary((X1 + X2 + X3) / 3.0)
                f_val = self.func(self.x[i])
                self._update_bests(i, f_val)

                # Update leader hierarchy
                if f_val < alpha_score:
                    delta_pos, delta_score = beta_pos.copy(), beta_score
                    beta_pos, beta_score = alpha_pos.copy(), alpha_score
                    alpha_pos, alpha_score = self.x[i].copy(), f_val
                elif f_val < beta_score:
                    delta_pos, delta_score = beta_pos.copy(), beta_score
                    beta_pos, beta_score = self.x[i].copy(), f_val
                elif f_val < delta_score:
                    delta_pos, delta_score = self.x[i].copy(), f_val

            self._record(t)

        return self.gbest_val, self.convergence_history
