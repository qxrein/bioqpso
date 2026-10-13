"""qpso.py — Quantum-behaved PSO (Sun et al. 2004/2011)."""
from __future__ import annotations

import numpy as np

from .base import BaseOptimizer


class QPSO(BaseOptimizer):
    """QPSO with linearly decreasing contraction-expansion coefficient β.

    β decreases from beta_init (default 1.0) to beta_final (default 0.5)
    over T iterations.  Standard formulation: β ∈ [0.5, 1.0] recommended.
    """

    def __init__(self, *args,
                 beta_init: float = 1.0,
                 beta_final: float = 0.5,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.beta_init = beta_init
        self.beta_final = beta_final

    def run(self) -> tuple[float, list[float]]:
        self._initialize()

        for t in range(self.max_iter):
            if self.func.exhausted():
                break
            # β schedule
            beta = (self.beta_init
                    - (self.beta_init - self.beta_final) * t / max(self.max_iter - 1, 1))

            # Mean of personal bests (mbest)
            mbest = np.mean(self.pbest_pos, axis=0)

            for i in range(self.n_particles):
                phi = self.rng.random(self.dim)
                # Local attractor between pbest and gbest
                p = phi * self.pbest_pos[i] + (1.0 - phi) * self.gbest_pos

                u = self.rng.random(self.dim)
                u = np.where(u == 0.0, 1e-300, u)   # avoid log(0)
                L = beta * np.abs(mbest - self.x[i])

                sign = np.where(self.rng.random(self.dim) > 0.5, 1.0, -1.0)
                self.x[i] = self.boundary(p + sign * L * (-np.log(u)))
                f_val = self.func(self.x[i])
                self._update_bests(i, f_val)
            self._record(t)

        return self.gbest_val, self.convergence_history
