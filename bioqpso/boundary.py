"""
boundary.py — shared boundary handlers.

All algorithms import and use one of these handlers so boundary handling
is identical across the whole suite.
"""
from __future__ import annotations

import numpy as np


class BoundaryHandler:
    """Apply boundary correction to a position vector.

    Parameters
    ----------
    lb, ub : float
        Scalar lower / upper bounds (same for all dimensions).
    mode : str
        One of 'clip', 'reflect', 'random_reinit', 'absorb'.
    rng : np.random.Generator
        Required for 'random_reinit'; ignored otherwise.
    """

    MODES = ("clip", "reflect", "random_reinit", "absorb")

    def __init__(self, lb: float, ub: float, mode: str = "clip",
                 rng: np.random.Generator | None = None):
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got {mode!r}")
        self.lb = float(lb)
        self.ub = float(ub)
        self.mode = mode
        self.rng = rng

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).copy()
        if self.mode == "clip":
            return np.clip(x, self.lb, self.ub)
        elif self.mode == "reflect":
            return _reflect(x, self.lb, self.ub)
        elif self.mode == "absorb":
            return np.clip(x, self.lb, self.ub)  # absorb = clip (velocity zeroed by caller)
        elif self.mode == "random_reinit":
            out_mask = (x < self.lb) | (x > self.ub)
            if out_mask.any() and self.rng is not None:
                x[out_mask] = self.rng.uniform(self.lb, self.ub, int(out_mask.sum()))
            else:
                x = np.clip(x, self.lb, self.ub)
            return x
        return x  # unreachable


def _reflect(x: np.ndarray, lb: float, ub: float) -> np.ndarray:
    """Reflect out-of-bounds components back into [lb, ub]."""
    span = ub - lb
    if span <= 0:
        return np.clip(x, lb, ub)
    result = x.copy()
    for i in range(len(result)):
        v = result[i]
        if v < lb:
            excess = lb - v
            bounces, rem = divmod(excess, span)
            result[i] = lb + rem if int(bounces) % 2 == 0 else ub - rem
        elif v > ub:
            excess = v - ub
            bounces, rem = divmod(excess, span)
            result[i] = ub - rem if int(bounces) % 2 == 0 else lb + rem
    return result
