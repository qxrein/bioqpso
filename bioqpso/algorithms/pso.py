"""pso.py — Standard PSO (Shi & Eberhart 1998)."""
from __future__ import annotations

import numpy as np

from .base import BaseOptimizer


class PSO(BaseOptimizer):
    """Particle Swarm Optimisation.

    Parameters
    ----------
    w : float   Inertia weight (default 0.729).
    c1 : float  Cognitive coefficient (default 1.494).
    c2 : float  Social coefficient (default 1.494).
    """

    def __init__(self, *args, w: float = 0.729, c1: float = 1.494,
                 c2: float = 1.494, **kwargs):
        super().__init__(*args, **kwargs)
        self.w = w
        self.c1 = c1
        self.c2 = c2

    def run(self) -> tuple[float, list[float]]:
        self._initialize()
        v = np.zeros((self.n_particles, self.dim))

        for t in range(self.max_iter):
            if self.func.exhausted():
                break
            for i in range(self.n_particles):
                r1 = self.rng.random(self.dim)
                r2 = self.rng.random(self.dim)
                v[i] = (self.w * v[i]
                        + self.c1 * r1 * (self.pbest_pos[i] - self.x[i])
                        + self.c2 * r2 * (self.gbest_pos - self.x[i]))
                self.x[i] = self.boundary(self.x[i] + v[i])
                f_val = self.func(self.x[i])
                self._update_bests(i, f_val)
            self._record(t)

        return self.gbest_val, self.convergence_history
