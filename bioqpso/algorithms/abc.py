"""abc.py — Artificial Bee Colony (Karaboga 2005)."""
from __future__ import annotations

import numpy as np

from .base import BaseOptimizer


class ABC(BaseOptimizer):
    """Artificial Bee Colony algorithm.

    Parameters
    ----------
    limit : int
        Abandonment limit (default None = SN * D, the usual ABC setting).  A
        food source is abandoned and re-scouted when its trial counter exceeds
        limit.  Scout re-evaluations count toward the FE budget.
    """

    def __init__(self, *args, limit: int | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.limit = limit if limit is not None else (self.n_particles // 2) * self.dim

    def run(self) -> tuple[float, list[float]]:
        self._initialize()
        n_food = self.n_particles // 2
        # Use the first n_food particles as food sources (already evaluated)
        food = self.x[:n_food].copy()
        food_f = self.pbest_val[:n_food].copy()
        trials = np.zeros(n_food, dtype=int)

        if food_f.min() < self.gbest_val:
            bi = int(np.argmin(food_f))
            self.gbest_val = float(food_f[bi])
            self.gbest_pos = food[bi].copy()

        for t in range(self.max_iter):
            if self.func.exhausted():
                break

            # --- Employed bees ---
            for i in range(n_food):
                k = int(self.rng.integers(n_food))
                while k == i:
                    k = int(self.rng.integers(n_food))
                d = int(self.rng.integers(self.dim))
                phi = self.rng.uniform(-1.0, 1.0)
                candidate = food[i].copy()
                candidate[d] += phi * (candidate[d] - food[k][d])
                candidate = self.boundary(candidate)
                f_val = self.func(candidate)
                if f_val < food_f[i]:
                    food[i] = candidate
                    food_f[i] = f_val
                    trials[i] = 0
                    if f_val < self.gbest_val:
                        self.gbest_val = f_val
                        self.gbest_pos = candidate.copy()
                else:
                    trials[i] += 1

            # --- Onlooker bees ---
            fitness = 1.0 / (1.0 + food_f - food_f.min() + 1e-300)
            probs = fitness / fitness.sum()
            for _ in range(n_food):
                i = int(self.rng.choice(n_food, p=probs))
                k = int(self.rng.integers(n_food))
                while k == i:
                    k = int(self.rng.integers(n_food))
                d = int(self.rng.integers(self.dim))
                phi = self.rng.uniform(-1.0, 1.0)
                candidate = food[i].copy()
                candidate[d] += phi * (candidate[d] - food[k][d])
                candidate = self.boundary(candidate)
                f_val = self.func(candidate)
                if f_val < food_f[i]:
                    food[i] = candidate
                    food_f[i] = f_val
                    trials[i] = 0
                    if f_val < self.gbest_val:
                        self.gbest_val = f_val
                        self.gbest_pos = candidate.copy()
                else:
                    trials[i] += 1

            # --- Scout bees (count FEs) ---
            for i in range(n_food):
                if trials[i] > self.limit:
                    food[i] = self.rng.uniform(self.lb, self.ub, self.dim)
                    food_f[i] = self.func(food[i])
                    trials[i] = 0
                    if food_f[i] < self.gbest_val:
                        self.gbest_val = float(food_f[i])
                        self.gbest_pos = food[i].copy()

            self._record(t)

        return self.gbest_val, self.convergence_history
