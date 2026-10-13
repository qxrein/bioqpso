"""
budget.py — FE-counting wrapper and budget enforcement.

Every objective function used in experiments is wrapped with FEBudget.
Algorithms must call budget.exhausted() to check termination.
"""
from __future__ import annotations

import numpy as np


class FEBudget:
    """Wrap an objective function and count every evaluation.

    Parameters
    ----------
    func : callable
        The raw objective function f(x) -> float.
    max_fes : int
        Hard budget (inclusive). After max_fes calls, any further call
        raises BudgetExhausted (or returns np.inf if soft=True).
    soft : bool
        If True, return np.inf once exhausted instead of raising.
    """

    def __init__(self, func, max_fes: int, soft: bool = False):
        self._func = func
        self.max_fes = max_fes
        self.soft = soft
        self.n_fes: int = 0

    def __call__(self, x) -> float:
        if self.n_fes >= self.max_fes:
            if self.soft:
                return np.inf
            raise BudgetExhausted(
                f"FE budget of {self.max_fes} exhausted (n_fes={self.n_fes})"
            )
        self.n_fes += 1
        return float(self._func(x))

    def exhausted(self) -> bool:
        return self.n_fes >= self.max_fes

    def remaining(self) -> int:
        return max(0, self.max_fes - self.n_fes)

    def reset(self):
        self.n_fes = 0


class BudgetExhausted(RuntimeError):
    pass
