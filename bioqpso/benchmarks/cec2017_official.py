"""
cec2017_official.py — the official CEC2017 functions through ctypes.

The C code in cec17_official/ is the official cec17_test_func.cpp with the patches listed in its
header (portable includes, "%lf", data directory). The official files are not in the repository;
scripts/fetch_cec2017_official.sh downloads, verifies and patches them. It is compiled on first use into
cec17_official/_build/ with the system C++ compiler. The suite has 29 functions, F1 and F3-F30;
F2 was removed from CEC2017 by its organizers. Values are raw f(x), equal to 100*k at the optimum
of function k. Only the D = 30 and D = 50 data are installed.

The official code keeps the data of the last function in global state, so one process evaluates
one (function, D) pair at a time efficiently; switching reloads the data files.
"""
from __future__ import annotations

import ctypes
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent / "cec17_official"
DATA = HERE / "input_data"
BUILD = HERE / "_build"
LIB = BUILD / "libcec17.so"
SOURCES = [HERE / "cec17_test_func.cpp", HERE / "cec17_capi.cpp"]
FUNCS = [1] + list(range(3, 31))
NAMES = [f"F{k}" for k in FUNCS]
DIMS = (30, 50)

_lib = None


def _build():
    """Compile the shared library; the rename is atomic, so parallel workers may race safely."""
    BUILD.mkdir(exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=BUILD, suffix=".so")
    os.close(fd)
    cxx = os.environ.get("CXX", "c++")
    subprocess.run([cxx, "-O2", "-shared", "-fPIC", "-w", "-o", tmp, *map(str, SOURCES)], check=True)
    os.replace(tmp, LIB)


def _load():
    global _lib
    if _lib is None:
        if not SOURCES[0].exists() or not DATA.exists():
            raise FileNotFoundError("The official CEC2017 code is not installed (it is not redistributed here); "
                                    "run scripts/fetch_cec2017_official.sh")
        if not LIB.exists() or LIB.stat().st_mtime < max(s.stat().st_mtime for s in SOURCES):
            _build()
        lib = ctypes.CDLL(str(LIB))
        lib.cec17_set_dir.argtypes = [ctypes.c_char_p]
        lib.cec17_eval.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                   ctypes.c_int, ctypes.c_int, ctypes.c_int]
        lib.cec17_set_dir(str(DATA).encode())
        _lib = lib
    return _lib


def evaluate(k: int, x) -> np.ndarray:
    """Official F_k at the rows of x (shape (m, D) or (D,)); returns m values."""
    if k not in FUNCS:
        raise ValueError(f"F{k} is not in the CEC2017 suite")
    x = np.ascontiguousarray(np.atleast_2d(x), dtype=np.float64)
    m, D = x.shape
    if D not in DIMS:
        raise ValueError(f"only D in {DIMS} is vendored")
    f = np.empty(m)
    _load().cec17_eval(x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                       f.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), D, m, k)
    return f


def optimum(k: int, D: int) -> np.ndarray:
    """The global optimum of F_k, where f = 100 k: the shift vector o, except for F9. The official
    Levy function has its minimum at z = M(x - o) = 1, so its optimum is o + M^{-1} 1 (the official
    matrices are not all orthogonal, hence the solve)."""
    o = np.loadtxt(DATA / f"shift_data_{k}.txt").ravel()[:D]
    if k == 9:
        o = o + np.linalg.solve(np.loadtxt(DATA / f"M_9_D{D}.txt"), np.ones(D))
    return o


class OfficialFunction:
    """Callable F_k at dimension D, with the attributes the runner expects."""

    def __init__(self, k: int, D: int):
        if k not in FUNCS or D not in DIMS:
            raise ValueError((k, D))
        self.k, self.D = k, D
        self.f_bias = 100.0 * k
        self.name = f"F{k}"

    def __call__(self, x) -> float:
        return float(evaluate(self.k, x)[0])


def get_official_problem(name: str, dim: int = 30) -> dict:
    """Problem dict in the format of benchmarks.cec2017.get_cec_problem; name is "F1".."F30"."""
    f = OfficialFunction(int(name[1:]), dim)
    return {"func": f, "lb": -100.0, "ub": 100.0, "dim": dim, "f_opt": f.f_bias,
            "name": f.name, "f_bias": f.f_bias, "_instance": f}
