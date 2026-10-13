"""
base.py — BaseOptimizer with FE budget, explicit RNG, boundary handler,
          and convergence-history recording.
"""
from __future__ import annotations

import numpy as np

from ..budget import FEBudget
from ..boundary import BoundaryHandler


class BaseOptimizer:
    """Common initialisation and bookkeeping for all optimizers.

    Parameters
    ----------
    func : FEBudget
        Wrapped objective (counts FEs; raises/returns inf when exhausted).
    lb, ub : float
        Scalar bounds applied identically to all dimensions.
    dim : int
        Problem dimensionality.
    n_particles : int
    max_iter : int
    rng : np.random.Generator
        Child RNG stream for this run.  All random draws go through here.
    boundary : BoundaryHandler
        Shared boundary handler (default clip).
    hist_every : int
        Record gbest every this many iterations (default 10).
    """

    def __init__(
        self,
        func: FEBudget,
        lb: float,
        ub: float,
        dim: int,
        n_particles: int,
        max_iter: int,
        rng: np.random.Generator,
        boundary: BoundaryHandler | None = None,
        hist_every: int = 10,
    ):
        self.func = func
        self.lb = float(lb)
        self.ub = float(ub)
        self.dim = dim
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.rng = rng
        self.boundary = boundary or BoundaryHandler(lb, ub, mode="clip", rng=rng)
        self.hist_every = hist_every

        # Set by _initialize()
        self.x: np.ndarray | None = None
        self.pbest_pos: np.ndarray | None = None
        self.pbest_val: np.ndarray | None = None
        self.gbest_pos: np.ndarray | None = None
        self.gbest_val: float = np.inf
        self.convergence_history: list[float] = []

    # ─────────────────────────────────────────────────────────
    # Initialisation
    # ─────────────────────────────────────────────────────────

    def _initialize(self):
        """Uniform random initial population; evaluate and set pbest/gbest."""
        self.x = self.rng.uniform(self.lb, self.ub, (self.n_particles, self.dim))
        self.pbest_pos = self.x.copy()
        self.pbest_val = np.array([self.func(p) for p in self.pbest_pos])

        best_idx = int(np.argmin(self.pbest_val))
        self.gbest_val = float(self.pbest_val[best_idx])
        self.gbest_pos = self.pbest_pos[best_idx].copy()
        self.convergence_history = [self.gbest_val]

    # ─────────────────────────────────────────────────────────
    # Personal / global best update
    # ─────────────────────────────────────────────────────────

    def _update_bests(self, i: int, f_val: float) -> bool:
        """Update pbest[i] and gbest. Returns True if pbest improved."""
        improved = False
        if f_val < self.pbest_val[i]:
            self.pbest_val[i] = f_val
            self.pbest_pos[i] = self.x[i].copy()
            improved = True
            if f_val < self.gbest_val:
                self.gbest_val = f_val
                self.gbest_pos = self.x[i].copy()
        return improved

    # ─────────────────────────────────────────────────────────
    # History recording
    # ─────────────────────────────────────────────────────────

    def _record(self, t: int):
        """Append gbest to convergence history every hist_every iterations."""
        if t % self.hist_every == 0:
            self.convergence_history.append(self.gbest_val)

    # ─────────────────────────────────────────────────────────
    # Abstract run
    # ─────────────────────────────────────────────────────────

    def run(self) -> tuple[float, list[float]]:
        """Run the optimizer.  Returns (gbest_val, convergence_history)."""
        raise NotImplementedError
