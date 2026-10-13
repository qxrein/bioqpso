"""
runner.py — Run one (algorithm, problem, seed) job and return a result record.

Each job is fully self-contained: given the same inputs it returns the same
outputs bit-for-bit (deterministic through np.random.default_rng(seed)).

Result record keys
------------------
algorithm : str
function   : str
dim        : int
run        : int
seed       : int
final_f    : float
n_fes      : int
convergence: list[float]    (gbest every hist_every iterations)
phi3_hist  : list[float] | None
wall_time_s: float
config_hash: str            (sha256 of the algorithm config dict)
"""
from __future__ import annotations

import hashlib
import json
import time
from typing import Any

import numpy as np

from .budget import FEBudget
from .boundary import BoundaryHandler
from .algorithms import (
    PSO, QPSO, ACO_R, ABC, GWO, WOA, AntBioQPSO, BeeBioQPSO,
)
from .schedules import (
    make_schedule, FixedSchedule, redistribute_phi, PHI3_MIN, PHI3_MAX,
)


# ─────────────────────────────────────────────────────────────
# Algorithm registry
# ─────────────────────────────────────────────────────────────

def _make_algorithm(name: str, func: FEBudget, lb: float, ub: float,
                    dim: int, n_particles: int, max_iter: int,
                    rng: np.random.Generator,
                    boundary: BoundaryHandler,
                    hist_every: int = 10,
                    extra: dict | None = None):
    """Instantiate an algorithm by registry name.

    Name format:  <leader>-<schedule>   e.g. 'ant-fixed', 'bee-cosine',
                  or a baseline: 'pso', 'qpso', 'aco_r', 'abc', 'gwo', 'woa'.

    For BioQPSO names the schedule is built from extra params if provided.
    """
    extra = extra or {}
    common = dict(func=func, lb=lb, ub=ub, dim=dim, n_particles=n_particles,
                  max_iter=max_iter, rng=rng, boundary=boundary,
                  hist_every=hist_every)

    name_lower = name.lower()

    # ── Baselines ──────────────────────────────────────────
    if name_lower == "pso":
        return PSO(**common,
                   w=extra.get("w", 0.729),
                   c1=extra.get("c1", 1.494),
                   c2=extra.get("c2", 1.494))

    if name_lower == "qpso":
        return QPSO(**common,
                    beta_init=extra.get("beta_init", 1.0),
                    beta_final=extra.get("beta_final", 0.5))

    if name_lower == "aco_r":
        return ACO_R(**common,
                     q=extra.get("q", 0.1),
                     xi=extra.get("xi", 0.85))

    if name_lower == "abc":
        return ABC(**common, limit=extra.get("limit", 30))

    if name_lower == "gwo":
        return GWO(**common)

    if name_lower == "woa":
        return WOA(**common, b=extra.get("b", 1.0))

    # ── BioQPSO family ─────────────────────────────────────
    # Name format: "ant-<schedule>" or "bee-<schedule>"
    parts = name_lower.split("-", 1)
    if len(parts) != 2 or parts[0] not in ("ant", "bee"):
        raise ValueError(f"Unknown algorithm name {name!r}. Expected "
                         f"'pso'/'qpso'/'aco_r'/'abc'/'gwo'/'woa' "
                         f"or 'ant-<schedule>'/'bee-<schedule>'.")
    leader, sched_name = parts

    # Build schedule
    sched_kwargs: dict[str, Any] = {}
    if sched_name == "fixed":
        phi3_val = float(extra.get("phi3", 0.3))
        sched_kwargs = {"phi3": phi3_val}
    elif sched_name == "linear":
        sched_kwargs = {
            "phi3_max": extra.get("phi3_max", PHI3_MAX),
            "phi3_min": extra.get("phi3_min", PHI3_MIN),
        }
    elif sched_name == "cosine":
        sched_kwargs = {
            "phi3_max": extra.get("phi3_max", PHI3_MAX),
            "phi3_min": extra.get("phi3_min", PHI3_MIN),
        }
    elif sched_name == "feedback":
        sched_kwargs = {
            "W": extra.get("W", 20),
            "delta": extra.get("delta", 0.02),
            "phi3_0": extra.get("phi3_0", None),
        }
    elif sched_name == "random":
        sched_kwargs = {"rng": rng}
    elif sched_name in ("nobio", "per_particle"):
        sched_kwargs = {}
    else:
        raise ValueError(f"Unknown schedule name {sched_name!r}")

    schedule = make_schedule(sched_name, **sched_kwargs)

    bio_common = dict(
        **common,
        schedule=schedule,
        phi1_ratio=extra.get("phi1_ratio", 4.0),
        phi2_ratio=extra.get("phi2_ratio", 3.0),
        beta_init=extra.get("beta_init", 1.0),
        beta_final=extra.get("beta_final", 0.5),
    )

    if leader == "ant":
        return AntBioQPSO(
            **bio_common,
            evaporation_rate=extra.get("evaporation_rate", 0.1),
            pheromone_deposit=extra.get("pheromone_deposit", 1.0),
        )
    else:  # bee
        return BeeBioQPSO(
            **bio_common,
            L_scout=extra.get("L_scout", 15),
        )


# ─────────────────────────────────────────────────────────────
# Single-job runner
# ─────────────────────────────────────────────────────────────

def _config_hash(d: dict) -> str:
    s = json.dumps(d, sort_keys=True, default=str)
    return hashlib.sha256(s.encode()).hexdigest()[:12]


def run_job(
    algorithm_name: str,
    problem: dict,
    run_idx: int,
    seed: int,
    n_particles: int = 30,
    max_iter: int = 1000,
    max_fes: int | None = None,
    boundary_mode: str = "clip",
    hist_every: int = 10,
    extra: dict | None = None,
) -> dict:
    """Execute one (algorithm, problem, run) job.

    Parameters
    ----------
    algorithm_name : str   Registry name (e.g. 'qpso', 'ant-cosine').
    problem : dict         Must have keys: func (callable), lb, ub, dim, name.
    run_idx : int          Run index (0-based); used in the result record.
    seed : int             RNG seed for this run.
    n_particles : int
    max_iter : int
    max_fes : int | None   If None, defaults to n_particles*(max_iter+1).
    boundary_mode : str    'clip', 'reflect', 'random_reinit', 'absorb'.
    hist_every : int       Record convergence every this many iterations.
    extra : dict           Algorithm-specific hyperparameters.

    Returns
    -------
    dict with keys: algorithm, function, dim, run, seed, final_f, n_fes,
                    convergence, phi3_hist, wall_time_s, config_hash.
    """
    extra = extra or {}

    lb = float(problem["lb"])
    ub = float(problem["ub"])
    dim = int(problem["dim"])
    func_raw = problem["func"]

    if max_fes is None:
        max_fes = n_particles * (max_iter + 1)

    rng = np.random.default_rng(seed)
    bh = BoundaryHandler(lb, ub, mode=boundary_mode, rng=rng)
    budget = FEBudget(func_raw, max_fes, soft=True)

    algo = _make_algorithm(
        algorithm_name, budget, lb, ub, dim,
        n_particles, max_iter, rng, bh, hist_every, extra,
    )

    t0 = time.perf_counter()
    final_f, convergence = algo.run()
    wall_time = time.perf_counter() - t0

    phi3_hist = getattr(algo, "phi3_history", None)

    cfg = dict(algorithm=algorithm_name, n_particles=n_particles,
               max_iter=max_iter, max_fes=max_fes,
               boundary=boundary_mode, extra=extra)

    return {
        "algorithm":   algorithm_name,
        "function":    problem["name"],
        "dim":         dim,
        "run":         run_idx,
        "seed":        seed,
        "final_f":     float(final_f),
        "n_fes":       int(budget.n_fes),
        "convergence": convergence,
        "phi3_hist":   phi3_hist,
        "wall_time_s": wall_time,
        "config_hash": _config_hash(cfg),
    }
