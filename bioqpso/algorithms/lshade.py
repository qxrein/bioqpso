"""lshade.py — L-SHADE (Tanabe & Fukunaga, CEC 2014).

Success-history parameter adaptation with linear population-size reduction.
Own NumPy implementation in the shared code path; every evaluation goes
through the same FEBudget as the other algorithms.
"""
from __future__ import annotations

import numpy as np

from .base import BaseOptimizer


class LSHADE(BaseOptimizer):
    """L-SHADE with published defaults.

    Parameters
    ----------
    r_ninit : float  N_init = round(r_ninit * D) (default 18).
    n_min : int      Final population size (default 4).
    r_arc : float    Archive size = round(r_arc * N) (default 2.6).
    p : float        Fraction of the population used as p-best (default 0.11).
    H : int          Success-history memory size (default 6).

    Out-of-bounds components use the original midpoint repair under the
    default 'clip' setting; any other boundary mode (Schwefel diagnosis only)
    goes through the shared handler instead.
    """

    def __init__(self, *args, r_ninit: float = 18.0, n_min: int = 4,
                 r_arc: float = 2.6, p: float = 0.11, H: int = 6, **kwargs):
        super().__init__(*args, **kwargs)
        self.r_ninit = r_ninit
        self.n_min = n_min
        self.r_arc = r_arc
        self.p = p
        self.H = H
        # Same FE spacing of the convergence history as the other algorithms.
        self._hist_fe = self.hist_every * self.n_particles
        self.n_particles = max(int(round(r_ninit * self.dim)), n_min)

    def _repair(self, v: np.ndarray, x: np.ndarray) -> np.ndarray:
        if self.boundary.mode != "clip":
            return np.array([self.boundary(row) for row in v])
        v = np.where(v < self.lb, (self.lb + x) / 2.0, v)
        return np.where(v > self.ub, (self.ub + x) / 2.0, v)

    def _sample_params(self, m_cr, m_f, n):
        r = self.rng.integers(self.H, size=n)
        cr = self.rng.normal(m_cr[r], 0.1)
        cr = np.where(np.isnan(m_cr[r]), 0.0, np.clip(cr, 0.0, 1.0))
        f = np.empty(n)
        for i in range(n):
            while True:
                v = m_f[r[i]] + 0.1 * np.tan(np.pi * (self.rng.random() - 0.5))
                if v > 0.0:
                    f[i] = min(v, 1.0)
                    break
        return cr, f

    def run(self) -> tuple[float, list[float]]:
        self._initialize()
        pop = self.x
        fit = self.pbest_val.copy()
        n_init = self.n_particles
        max_fes = self.func.max_fes
        arc = np.empty((0, self.dim))
        m_cr = np.full(self.H, 0.5)
        m_f = np.full(self.H, 0.5)
        k = 0
        next_hist = self._hist_fe

        while not self.func.exhausted():
            n = len(pop)
            cr, f = self._sample_params(m_cr, m_f, n)
            order = np.argsort(fit)
            n_p = max(int(round(self.p * n)), 2)

            union = np.vstack([pop, arc]) if len(arc) else pop
            pbest = pop[order[self.rng.integers(n_p, size=n)]]
            r1 = np.empty(n, dtype=int)
            r2 = np.empty(n, dtype=int)
            for i in range(n):
                r1[i] = self.rng.integers(n)
                while r1[i] == i:
                    r1[i] = self.rng.integers(n)
                r2[i] = self.rng.integers(len(union))
                while r2[i] == i or r2[i] == r1[i]:
                    r2[i] = self.rng.integers(len(union))

            v = pop + f[:, None] * (pbest - pop) + f[:, None] * (pop[r1] - union[r2])
            v = self._repair(v, pop)
            mask = self.rng.random((n, self.dim)) <= cr[:, None]
            mask[np.arange(n), self.rng.integers(self.dim, size=n)] = True
            trial = np.where(mask, v, pop)

            new_pop = pop.copy()
            new_fit = fit.copy()
            s_cr, s_f, s_df, lost = [], [], [], []
            for i in range(n):
                f_val = self.func(trial[i])
                if f_val <= fit[i]:
                    new_pop[i] = trial[i]
                    new_fit[i] = f_val
                    if f_val < fit[i]:
                        lost.append(pop[i].copy())
                        s_cr.append(cr[i])
                        s_f.append(f[i])
                        s_df.append(fit[i] - f_val)
                    if f_val < self.gbest_val:
                        self.gbest_val = float(f_val)
                        self.gbest_pos = trial[i].copy()
                while self.func.n_fes >= next_hist:
                    self.convergence_history.append(self.gbest_val)
                    next_hist += self._hist_fe
                if self.func.exhausted():
                    break
            pop, fit = new_pop, new_fit
            if lost:
                arc = np.vstack([arc, np.array(lost)])

            # Success-history update (weighted Lehmer means)
            if s_cr:
                w = np.array(s_df) / np.sum(s_df)
                s_cr_a, s_f_a = np.array(s_cr), np.array(s_f)
                if np.isnan(m_cr[k]) or s_cr_a.max() == 0.0:
                    m_cr[k] = np.nan
                else:
                    m_cr[k] = np.sum(w * s_cr_a ** 2) / np.sum(w * s_cr_a)
                m_f[k] = np.sum(w * s_f_a ** 2) / np.sum(w * s_f_a)
                k = (k + 1) % self.H

            # Linear population-size reduction
            n_next = int(round(((self.n_min - n_init) / max_fes) * self.func.n_fes + n_init))
            n_next = max(n_next, self.n_min)
            if n_next < len(pop):
                keep = np.argsort(fit)[:n_next]
                pop, fit = pop[keep], fit[keep]
            arc_max = int(round(self.r_arc * len(pop)))
            if len(arc) > arc_max:
                arc = arc[self.rng.choice(len(arc), arc_max, replace=False)]

        return self.gbest_val, self.convergence_history
