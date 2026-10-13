"""
bioqpso.py — BioQPSO family: AntBioQPSO and BeeBioQPSO with any schedule.

Three-term attractor:
    q = (φ1·p + φ2·g + φ3·b) / (φ1+φ2+φ3)
where p = personal best, g = global best, b = ring-neighbour leader.

Leaders:
  Ant: pheromone-based (highest pheromone in ring → leader)
  Bee: fitness-based  (best pbest in ring → leader)

Schedules are injected via a schedule callable (from bioqpso.schedules).
"""
from __future__ import annotations

import numpy as np

from .base import BaseOptimizer
from ..schedules import (
    FixedSchedule, redistribute_phi, PerParticleSchedule, NoBioSchedule,
    PHI3_MIN, PHI3_MAX,
)


class _BioQPSOBase(BaseOptimizer):
    """Shared BioQPSO logic for both leader types.

    Parameters
    ----------
    schedule : callable(t, T, gbest_val=...) -> float
        φ3 schedule.
    phi1_ratio, phi2_ratio : float
        Ratio in which remaining weight (1-φ3) is split. Default 4:3.
    beta_init, beta_final : float
        β contraction-expansion schedule (same as QPSO).
    """

    def __init__(
        self,
        *args,
        schedule=None,
        phi1_ratio: float = 4.0,
        phi2_ratio: float = 3.0,
        beta_init: float = 1.0,
        beta_final: float = 0.5,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.schedule = schedule if schedule is not None else FixedSchedule(0.3)
        self.phi1_ratio = phi1_ratio
        self.phi2_ratio = phi2_ratio
        self.beta_init = beta_init
        self.beta_final = beta_final

        # Whether the schedule is per-particle
        self._is_per_particle = isinstance(self.schedule, PerParticleSchedule)
        # Whether the bio-leader is disabled
        self._is_nobio = isinstance(self.schedule, NoBioSchedule)

        # Per-run diagnostics
        self.phi3_history: list[float] = []
        self.trigger_count: int = 0

    # ─────────────────────────────────────────────────────────
    # Neighbour-leader update (implemented by subclass)
    # ─────────────────────────────────────────────────────────

    def _init_leader_state(self):
        raise NotImplementedError

    def _get_local_leader(self, i: int) -> np.ndarray:
        raise NotImplementedError

    def _update_leader_state(self, i: int, pbest_improved: bool):
        raise NotImplementedError

    # ─────────────────────────────────────────────────────────
    # Run loop
    # ─────────────────────────────────────────────────────────

    def run(self) -> tuple[float, list[float]]:
        self._initialize()
        self._init_leader_state()
        self.schedule.reset()
        self.phi3_history = []
        self.trigger_count = 0

        for t in range(self.max_iter):
            if self.func.exhausted():
                break

            beta = (self.beta_init
                    - (self.beta_init - self.beta_final) * t / max(self.max_iter - 1, 1))
            mbest = np.mean(self.pbest_pos, axis=0)

            # Global φ3 (used for non-per-particle schedules)
            if not self._is_per_particle:
                phi3_global = self.schedule(t, self.max_iter, gbest_val=self.gbest_val)
                phi1_g, phi2_g, phi3_g = redistribute_phi(
                    phi3_global, self.phi1_ratio, self.phi2_ratio)
                self.phi3_history.append(phi3_global)
                if hasattr(self.schedule, 'trigger_count'):
                    self.trigger_count = self.schedule.trigger_count

            for i in range(self.n_particles):
                # Per-particle φ3 if applicable
                if self._is_per_particle:
                    stag_lim = getattr(self, 'stagnation_limit', 15)
                    stag_i = getattr(self, '_stag_counter', np.zeros(self.n_particles))[i]
                    phi3_i = self.schedule.per_particle(int(stag_i), stag_lim)
                    phi1_i, phi2_i, phi3_i = redistribute_phi(
                        phi3_i, self.phi1_ratio, self.phi2_ratio)
                else:
                    phi1_i, phi2_i, phi3_i = phi1_g, phi2_g, phi3_g

                p = self.pbest_pos[i]
                g = self.gbest_pos
                b = self._get_local_leader(i)

                # Three-term attractor (normalised)
                if self._is_nobio:
                    # φ3 = 0: standard two-term QPSO attractor
                    q = phi1_i * p + phi2_i * g
                else:
                    q = phi1_i * p + phi2_i * g + phi3_i * b

                # QPSO position update
                u = self.rng.random(self.dim)
                u = np.where(u == 0.0, 1e-300, u)
                L = beta * np.abs(mbest - self.x[i])
                sign = np.where(self.rng.random(self.dim) > 0.5, 1.0, -1.0)
                self.x[i] = self.boundary(q + sign * L * (-np.log(u)))
                f_val = self.func(self.x[i])
                improved = self._update_bests(i, f_val)
                self._update_leader_state(i, improved)

            if self._is_per_particle:
                stag = getattr(self, '_stag_counter', np.zeros(self.n_particles))
                self.phi3_history.append(float(np.mean(
                    [self.schedule.per_particle(int(s), getattr(self, 'stagnation_limit', 15))
                     for s in stag])))

            self._record(t)

        return self.gbest_val, self.convergence_history


# ─────────────────────────────────────────────────────────────────
# Ant leader (pheromone-based ring neighbour)
# ─────────────────────────────────────────────────────────────────

class AntBioQPSO(_BioQPSOBase):
    """BioQPSO with ant-pheromone ring-neighbourhood leader.

    Parameters
    ----------
    evaporation_rate : float  Pheromone evaporation ρ (default 0.1).
    pheromone_deposit : float Δ deposited on pbest improvement (default 1.0).
    """

    def __init__(
        self,
        *args,
        evaporation_rate: float = 0.1,
        pheromone_deposit: float = 1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.evaporation_rate = evaporation_rate
        self.pheromone_deposit = pheromone_deposit
        self._pheromone: np.ndarray | None = None

    def _init_leader_state(self):
        self._pheromone = np.ones(self.n_particles)

    def _get_local_leader(self, i: int) -> np.ndarray:
        left = (i - 1) % self.n_particles
        right = (i + 1) % self.n_particles
        nbs = [left, i, right]
        best_local = int(np.argmax(self._pheromone[nbs]))
        return self.pbest_pos[nbs[best_local]].copy()

    def _update_leader_state(self, i: int, pbest_improved: bool):
        self._pheromone *= (1.0 - self.evaporation_rate)
        if pbest_improved:
            self._pheromone[i] += self.pheromone_deposit

    def run(self) -> tuple[float, list[float]]:
        self._initialize()
        self._init_leader_state()
        self.schedule.reset()
        self.phi3_history = []
        self.trigger_count = 0

        for t in range(self.max_iter):
            if self.func.exhausted():
                break

            beta = (self.beta_init
                    - (self.beta_init - self.beta_final) * t / max(self.max_iter - 1, 1))
            mbest = np.mean(self.pbest_pos, axis=0)

            # Evaporate pheromones once per iteration
            self._pheromone *= (1.0 - self.evaporation_rate)

            if not self._is_per_particle:
                phi3_global = self.schedule(t, self.max_iter, gbest_val=self.gbest_val)
                phi1_g, phi2_g, phi3_g = redistribute_phi(
                    phi3_global, self.phi1_ratio, self.phi2_ratio)
                self.phi3_history.append(phi3_global)
                if hasattr(self.schedule, 'trigger_count'):
                    self.trigger_count = self.schedule.trigger_count

            for i in range(self.n_particles):
                if self._is_per_particle:
                    stag_lim = getattr(self, 'stagnation_limit', 15)
                    stag_i = getattr(self, '_stag_counter', np.zeros(self.n_particles))[i]
                    phi3_i = self.schedule.per_particle(int(stag_i), stag_lim)
                    phi1_i, phi2_i, phi3_i = redistribute_phi(
                        phi3_i, self.phi1_ratio, self.phi2_ratio)
                else:
                    phi1_i, phi2_i, phi3_i = phi1_g, phi2_g, phi3_g

                p = self.pbest_pos[i]
                g = self.gbest_pos
                b = self._get_local_leader(i)

                if self._is_nobio:
                    q = phi1_i * p + phi2_i * g
                else:
                    q = phi1_i * p + phi2_i * g + phi3_i * b

                u = self.rng.random(self.dim)
                u = np.where(u == 0.0, 1e-300, u)
                L = beta * np.abs(mbest - self.x[i])
                sign = np.where(self.rng.random(self.dim) > 0.5, 1.0, -1.0)
                self.x[i] = self.boundary(q + sign * L * (-np.log(u)))
                f_val = self.func(self.x[i])
                improved = self._update_bests(i, f_val)
                if improved:
                    self._pheromone[i] += self.pheromone_deposit

            if self._is_per_particle:
                self.phi3_history.append(float(np.mean(
                    [self.schedule.per_particle(0, 15) for _ in range(self.n_particles)])))

            self._record(t)

        return self.gbest_val, self.convergence_history


# ─────────────────────────────────────────────────────────────────
# Bee leader (fitness-based ring neighbour + scout phase)
# ─────────────────────────────────────────────────────────────────

class BeeBioQPSO(_BioQPSOBase):
    """BioQPSO with bee-inspired ring-neighbourhood leader.

    Particles that stagnate for L_scout iterations are re-initialised
    (scout phase).  Scout FEs count toward the budget.

    Parameters
    ----------
    L_scout : int  Stagnation limit before re-init (default 15).
    """

    def __init__(self, *args, L_scout: int = 15, **kwargs):
        super().__init__(*args, **kwargs)
        self.stagnation_limit = L_scout   # renamed from old 'stagnation_limit'
        self._stag_counter: np.ndarray | None = None

    def _init_leader_state(self):
        self._stag_counter = np.zeros(self.n_particles, dtype=int)

    def _get_local_leader(self, i: int) -> np.ndarray:
        left = (i - 1) % self.n_particles
        right = (i + 1) % self.n_particles
        nbs = [left, i, right]
        best_local = int(np.argmin(self.pbest_val[nbs]))
        return self.pbest_pos[nbs[best_local]].copy()

    def _update_leader_state(self, i: int, pbest_improved: bool):
        if pbest_improved:
            self._stag_counter[i] = 0
        else:
            self._stag_counter[i] += 1

    def run(self) -> tuple[float, list[float]]:
        self._initialize()
        self._init_leader_state()
        self.schedule.reset()
        self.phi3_history = []
        self.trigger_count = 0

        for t in range(self.max_iter):
            if self.func.exhausted():
                break

            beta = (self.beta_init
                    - (self.beta_init - self.beta_final) * t / max(self.max_iter - 1, 1))
            mbest = np.mean(self.pbest_pos, axis=0)

            if not self._is_per_particle:
                phi3_global = self.schedule(t, self.max_iter, gbest_val=self.gbest_val)
                phi1_g, phi2_g, phi3_g = redistribute_phi(
                    phi3_global, self.phi1_ratio, self.phi2_ratio)
                self.phi3_history.append(phi3_global)
                if hasattr(self.schedule, 'trigger_count'):
                    self.trigger_count = self.schedule.trigger_count

            for i in range(self.n_particles):
                if self._is_per_particle:
                    phi3_i = self.schedule.per_particle(
                        int(self._stag_counter[i]), self.stagnation_limit)
                    phi1_i, phi2_i, phi3_i = redistribute_phi(
                        phi3_i, self.phi1_ratio, self.phi2_ratio)
                else:
                    phi1_i, phi2_i, phi3_i = phi1_g, phi2_g, phi3_g

                p = self.pbest_pos[i]
                g = self.gbest_pos
                b = self._get_local_leader(i)

                if self._is_nobio:
                    q = phi1_i * p + phi2_i * g
                else:
                    q = phi1_i * p + phi2_i * g + phi3_i * b

                u = self.rng.random(self.dim)
                u = np.where(u == 0.0, 1e-300, u)
                L = beta * np.abs(mbest - self.x[i])
                sign = np.where(self.rng.random(self.dim) > 0.5, 1.0, -1.0)
                self.x[i] = self.boundary(q + sign * L * (-np.log(u)))
                f_val = self.func(self.x[i])
                improved = self._update_bests(i, f_val)
                self._update_leader_state(i, improved)

            # Scout phase (counts FEs)
            for i in range(self.n_particles):
                if self._stag_counter[i] > self.stagnation_limit:
                    self.x[i] = self.rng.uniform(self.lb, self.ub, self.dim)
                    f_val = self.func(self.x[i])
                    self.pbest_pos[i] = self.x[i].copy()
                    self.pbest_val[i] = f_val
                    self._stag_counter[i] = 0
                    if f_val < self.gbest_val:
                        self.gbest_val = f_val
                        self.gbest_pos = self.x[i].copy()

            if self._is_per_particle:
                self.phi3_history.append(float(np.mean(
                    [self.schedule.per_particle(int(s), self.stagnation_limit)
                     for s in self._stag_counter])))

            self._record(t)

        return self.gbest_val, self.convergence_history
