"""
classical.py — 10 classical benchmark functions.

All return f(x) with global minimum = 0 (except Styblinski-Tang which
has a known negative minimum; minimisation is still towards −∞ direction,
so we do NOT shift it — the reported values are the raw function values).

Schwefel: optimum at x_i = 420.9687 for all i, f(x*) = 0.
"""
from __future__ import annotations

import numpy as np


# ─────────────────────────────────────────────────────────────
# Function implementations
# ─────────────────────────────────────────────────────────────

def sphere(x: np.ndarray) -> float:
    return float(np.sum(x ** 2))


def rosenbrock(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (x[:-1] - 1.0) ** 2))


def zakharov(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    i = np.arange(1, len(x) + 1, dtype=float)
    s1 = np.sum(x ** 2)
    s2 = np.sum(0.5 * i * x)
    return float(s1 + s2 ** 2 + s2 ** 4)


def dixon_price(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    i = np.arange(2, len(x) + 1, dtype=float)
    return float((x[0] - 1.0) ** 2 + np.sum(i * (2.0 * x[1:] ** 2 - x[:-1]) ** 2))


def rastrigin(x: np.ndarray) -> float:
    D = len(x)
    return float(10 * D + np.sum(x ** 2 - 10.0 * np.cos(2.0 * np.pi * x)))


def ackley(x: np.ndarray) -> float:
    D = len(x)
    s1 = np.sum(x ** 2)
    s2 = np.sum(np.cos(2.0 * np.pi * x))
    return float(-20.0 * np.exp(-0.2 * np.sqrt(s1 / D))
                 - np.exp(s2 / D) + 20.0 + np.e)


def griewank(x: np.ndarray) -> float:
    idx = np.arange(1, len(x) + 1, dtype=float)
    return float(np.sum(x ** 2) / 4000.0
                 - np.prod(np.cos(x / np.sqrt(idx))) + 1.0)


def schwefel(x: np.ndarray) -> float:
    """Global optimum at x_i ≈ 420.9687 → f = 0."""
    x = np.asarray(x, dtype=float)
    D = len(x)
    return float(418.9829 * D - np.sum(x * np.sin(np.sqrt(np.abs(x)))))


def levy(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    w = 1.0 + (x - 1.0) / 4.0
    t1 = np.sin(np.pi * w[0]) ** 2
    t3 = (w[-1] - 1.0) ** 2 * (1.0 + np.sin(2.0 * np.pi * w[-1]) ** 2)
    t2 = np.sum((w[:-1] - 1.0) ** 2 * (1.0 + 10.0 * np.sin(np.pi * w[:-1] + np.pi) ** 2))
    return float(t1 + t2 + t3)


def styblinski_tang(x: np.ndarray) -> float:
    """Global minimum ≈ −39.166·D (negative).  Minimisation is correct."""
    x = np.asarray(x, dtype=float)
    return float(0.5 * np.sum(x ** 4 - 16.0 * x ** 2 + 5.0 * x))


# ─────────────────────────────────────────────────────────────
# Problem catalogue
# ─────────────────────────────────────────────────────────────

CLASSICAL_PROBLEMS: dict[str, dict] = {
    "sphere":          {"func": sphere,          "lb": -100.0,    "ub": 100.0,   "f_opt": 0.0},
    "rosenbrock":      {"func": rosenbrock,       "lb": -2.048,    "ub": 2.048,   "f_opt": 0.0},
    "zakharov":        {"func": zakharov,         "lb": -5.0,      "ub": 10.0,    "f_opt": 0.0},
    "dixon_price":     {"func": dixon_price,      "lb": -10.0,     "ub": 10.0,    "f_opt": 0.0},
    "rastrigin":       {"func": rastrigin,        "lb": -5.12,     "ub": 5.12,    "f_opt": 0.0},
    "ackley":          {"func": ackley,           "lb": -32.768,   "ub": 32.768,  "f_opt": 0.0},
    "griewank":        {"func": griewank,         "lb": -600.0,    "ub": 600.0,   "f_opt": 0.0},
    "schwefel":        {"func": schwefel,         "lb": -500.0,    "ub": 500.0,   "f_opt": 0.0},
    "levy":            {"func": levy,             "lb": -10.0,     "ub": 10.0,    "f_opt": 0.0},
    "styblinski_tang": {"func": styblinski_tang,  "lb": -5.0,      "ub": 5.0,     "f_opt": None},
}

# Correct global optimum for styblinski_tang at D=30
# f* = −39.16616570377…× D; at x_i = −2.903534027771 for all i
_ST_X_OPT = -2.903534027771
_ST_F_OPT_PER_DIM = 0.5 * (_ST_X_OPT**4 - 16*_ST_X_OPT**2 + 5*_ST_X_OPT)  # ≈ -39.1662


def get_classical_problem(name: str, dim: int = 30) -> dict:
    """Return a problem dict with 'func', 'lb', 'ub', 'dim', 'f_opt', 'name'."""
    if name not in CLASSICAL_PROBLEMS:
        raise KeyError(f"Unknown classical problem {name!r}. "
                       f"Available: {list(CLASSICAL_PROBLEMS)}")
    p = CLASSICAL_PROBLEMS[name].copy()
    p["dim"] = dim
    p["name"] = name
    if name == "styblinski_tang":
        p["f_opt"] = _ST_F_OPT_PER_DIM * dim
    return p
