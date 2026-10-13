"""
cec2017.py — CEC2017 benchmark wrapper.

F5 (opfunu F52017) is excluded due to the near-flat landscape bug
documented in docs/CEC2017_CHECK.md.  All other 28 functions are used.

Values reported are raw f(x) (which equals f_bias at the optimum).
"""
from __future__ import annotations

import opfunu

# Functions to use (all 29 except F52017)
CEC_EXCLUDE = {"F52017"}

_all_cec2017 = sorted(
    [f for f in opfunu.get_cec_based_functions() if "2017" in f.__name__],
    key=lambda f: f.__name__,
)
CEC2017_CLASSES = [f for f in _all_cec2017 if f.__name__ not in CEC_EXCLUDE]


def get_cec_problem(func_cls, dim: int = 30) -> dict:
    """Instantiate a CEC2017 function and return a problem dict.

    Parameters
    ----------
    func_cls : class
        One of the CEC2017_CLASSES.
    dim : int
        Dimensionality (must be in func_cls.dim_supported).
    """
    f = func_cls(ndim=dim)
    return {
        "func": f.evaluate,
        "lb": float(f.bounds[0, 0]),
        "ub": float(f.bounds[0, 1]),
        "dim": dim,
        "f_opt": float(f.f_bias),
        "name": f.__class__.__name__,
        "f_bias": float(f.f_bias),
        "_instance": f,   # kept for reproducibility checks
    }
