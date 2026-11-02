"""
schedules.py — φ3 schedule functions for BioQPSO.

Every schedule is a pure callable: schedule(t, T) -> float
where t ∈ [0, T-1] is the current iteration and T is max_iter.

Constraints enforced here:
  - output is clipped to [PHI3_MIN, PHI3_MAX]
  - φ3 >= 0 always

Caller is responsible for keeping φ1+φ2+φ3 == 1 (via redistribute_phi).
"""
from __future__ import annotations

import numpy as np

# Global bounds used across all schedules
PHI3_MIN: float = 0.1
PHI3_MAX: float = 0.5


def _clip(v: float) -> float:
    """Clip to [PHI3_MIN, PHI3_MAX] using pure Python (np.clip on scalars is ~10x slower)."""
    return max(PHI3_MIN, min(PHI3_MAX, float(v)))


def redistribute_phi(phi3: float,
                     phi1_base: float = 4.0,
                     phi2_base: float = 3.0) -> tuple[float, float, float]:
    """Return (phi1, phi2, phi3) summing to 1 with phi1:phi2 = phi1_base:phi2_base."""
    phi3 = _clip(phi3)
    remaining = 1.0 - phi3
    total_base = phi1_base + phi2_base
    phi1 = remaining * phi1_base / total_base
    phi2 = remaining * phi2_base / total_base
    return phi1, phi2, phi3


# ─────────────────────────────────────────────────────────────
# Fixed schedule (no adaptation)
# ─────────────────────────────────────────────────────────────

class FixedSchedule:
    """φ3 is constant throughout the run."""

    def __init__(self, phi3: float = 0.3):
        self.phi3_const = _clip(phi3)

    def __call__(self, t: int, T: int, **_) -> float:
        return self.phi3_const

    def reset(self):
        pass  # stateless


# ─────────────────────────────────────────────────────────────
# Linear decay (high → low bio-influence)
# ─────────────────────────────────────────────────────────────

class LinearSchedule:
    """φ3 decreases linearly from phi3_max to phi3_min over T iterations."""

    def __init__(self, phi3_max: float = PHI3_MAX, phi3_min: float = PHI3_MIN):
        self.phi3_max = phi3_max
        self.phi3_min = phi3_min

    def __call__(self, t: int, T: int, **_) -> float:
        if T <= 1:
            return self.phi3_max
        phi3 = self.phi3_max - (self.phi3_max - self.phi3_min) * (t / (T - 1))
        return _clip(phi3)

    def reset(self):
        pass


# ─────────────────────────────────────────────────────────────
# Cosine annealing
# ─────────────────────────────────────────────────────────────

class CosineSchedule:
    """φ3 follows a cosine curve from phi3_max to phi3_min."""

    def __init__(self, phi3_max: float = PHI3_MAX, phi3_min: float = PHI3_MIN):
        self.phi3_max = phi3_max
        self.phi3_min = phi3_min

    def __call__(self, t: int, T: int, **_) -> float:
        if T <= 1:
            return self.phi3_max
        phi3 = self.phi3_min + 0.5 * (self.phi3_max - self.phi3_min) * (
            1.0 + np.cos(np.pi * t / (T - 1))
        )
        return _clip(phi3)

    def reset(self):
        pass


# ─────────────────────────────────────────────────────────────
# Feedback (stagnation-driven)
# ─────────────────────────────────────────────────────────────

class FeedbackSchedule:
    """Increase φ3 when global best stagnates for W iterations; decrease otherwise.

    Parameters
    ----------
    W : int       Stagnation window (default 20).
    delta : float Step size for φ3 adjustment (default 0.02).
    phi3_0 : float Initial φ3 (default midpoint of [min, max]).
    """

    def __init__(self, W: int = 20, delta: float = 0.02,
                 phi3_0: float | None = None):
        self.W = W
        self.delta = delta
        self.phi3_0 = phi3_0 if phi3_0 is not None else 0.5 * (PHI3_MIN + PHI3_MAX)
        self._phi3 = _clip(self.phi3_0)
        self._stag = 0
        self._prev_gbest: float | None = None
        self.trigger_count: int = 0  # for diagnostics

    def __call__(self, t: int, T: int, gbest_val: float | None = None, **_) -> float:
        if gbest_val is not None:
            if self._prev_gbest is None or gbest_val < self._prev_gbest:
                self._stag = 0
                self._prev_gbest = gbest_val
            else:
                self._stag += 1

        if self._stag > self.W:
            self._phi3 = _clip(self._phi3 + self.delta)
            self.trigger_count += 1
        else:
            self._phi3 = _clip(self._phi3 - self.delta)

        return self._phi3

    def reset(self):
        self._phi3 = _clip(self.phi3_0)
        self._stag = 0
        self._prev_gbest = None
        self.trigger_count = 0


# ─────────────────────────────────────────────────────────────
# Random φ3 (control: does variation alone explain gains?)
# ─────────────────────────────────────────────────────────────

class RandomSchedule:
    """φ3(t) ~ U[phi3_min, phi3_max] independently each iteration."""

    def __init__(self, rng: np.random.Generator | None = None):
        self._rng = rng

    def __call__(self, t: int, T: int, **_) -> float:
        rng = self._rng if self._rng is not None else np.random.default_rng()
        return float(rng.uniform(PHI3_MIN, PHI3_MAX))

    def reset(self):
        pass


# ─────────────────────────────────────────────────────────────
# NoBio control (φ3 = 0, no neighbour leader at all)
# ─────────────────────────────────────────────────────────────

class NoBioSchedule:
    """φ3 = 0: the bio-leader term is completely removed.
    φ1:φ2 = 4:3 (fixed).  This isolates bio-leader from fixed-weights effect.
    """

    def __call__(self, t: int, T: int, **_) -> float:
        return 0.0

    def reset(self):
        pass


# ─────────────────────────────────────────────────────────────
# Per-particle schedule (phi3 proportional to stagnation counter)
# ─────────────────────────────────────────────────────────────

class PerParticleSchedule:
    """Each particle gets its own φ3 based on its stagnation counter.

    This is stateful per-particle; the schedule is queried with particle_idx
    and stagnation_counter. At the swarm level a mean φ3 is logged.

    Usage: call schedule.per_particle(i, stag_i, stag_limit) -> phi3_i
    """

    def __init__(self):
        pass

    def __call__(self, t: int, T: int, **_) -> float:
        # Returns a nominal value; actual per-particle value from per_particle()
        return 0.5 * (PHI3_MIN + PHI3_MAX)

    def per_particle(self, stag_i: int, stag_limit: int) -> float:
        denom = max(float(stag_limit), 1.0)
        frac = float(np.clip(stag_i / denom, 0.0, 1.0))
        return _clip(PHI3_MIN + (PHI3_MAX - PHI3_MIN) * frac)

    def reset(self):
        pass


# ─────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────

def make_schedule(name: str, **kwargs):
    """Return a schedule instance by name.

    Names: 'fixed', 'linear', 'cosine', 'feedback', 'random', 'nobio',
           'per_particle'.
    """
    name = name.lower()
    if name == "fixed":
        return FixedSchedule(**kwargs)
    elif name == "linear":
        return LinearSchedule(**kwargs)
    elif name == "cosine":
        return CosineSchedule(**kwargs)
    elif name == "feedback":
        return FeedbackSchedule(**kwargs)
    elif name == "random":
        return RandomSchedule(**kwargs)
    elif name == "nobio":
        return NoBioSchedule()
    elif name == "per_particle":
        return PerParticleSchedule()
    else:
        raise ValueError(f"Unknown schedule name {name!r}")
