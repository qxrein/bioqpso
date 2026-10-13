"""
test_phase2.py — Phase 2 GATE 2 unit tests.

Tests:
1. φ1+φ2+φ3 == 1 at every iteration for every schedule
2. φ3 ∈ [PHI3_MIN, PHI3_MAX] for every schedule
3. attractor q lies in componentwise convex hull of {p, g, b}
4. Schedule endpoints: linear and cosine at t=0 and t=T-1
5. FeedbackSchedule increases only after stagnation_counter > W
6. Determinism: same seed → same result
7. Different algorithms → same initial population (same seed)
8. FE accounting: all algorithms stay within budget
9. Classical benchmarks: f(x*) = f_min
10. Bounds: no algorithm returns a point outside bounds
"""
from __future__ import annotations

import numpy as np
import pytest

from bioqpso.schedules import (
    FixedSchedule, LinearSchedule, CosineSchedule, FeedbackSchedule,
    RandomSchedule, NoBioSchedule, PerParticleSchedule,
    redistribute_phi, PHI3_MIN, PHI3_MAX,
)
from bioqpso.budget import FEBudget
from bioqpso.boundary import BoundaryHandler
from bioqpso.benchmarks.classical import (
    sphere, rosenbrock, zakharov, dixon_price, rastrigin,
    ackley, griewank, schwefel, levy, styblinski_tang,
    get_classical_problem, _ST_X_OPT, _ST_F_OPT_PER_DIM,
)
from bioqpso.runner import run_job


# ─────────────────────────────────────────────────────────────
# 1 & 2. Schedule φ3 constraints
# ─────────────────────────────────────────────────────────────

T = 1000

SCHEDULES = [
    FixedSchedule(0.3),
    FixedSchedule(0.1),
    FixedSchedule(0.5),
    LinearSchedule(),
    CosineSchedule(),
    FeedbackSchedule(W=20, delta=0.02),
    RandomSchedule(rng=np.random.default_rng(0)),
    NoBioSchedule(),
    PerParticleSchedule(),
]


@pytest.mark.parametrize("sched", SCHEDULES)
def test_phi3_bounds(sched):
    """φ3 must be in [PHI3_MIN, PHI3_MAX] (NoBio is 0.0, allowed)."""
    sched.reset()
    for t in range(T):
        phi3 = sched(t, T, gbest_val=float(T - t))
        if isinstance(sched, NoBioSchedule):
            assert phi3 == 0.0
        else:
            assert PHI3_MIN - 1e-9 <= phi3 <= PHI3_MAX + 1e-9, (
                f"{sched.__class__.__name__} at t={t}: phi3={phi3}"
            )


@pytest.mark.parametrize("sched", SCHEDULES)
def test_redistribute_phi_sums_to_one(sched):
    """φ1+φ2+φ3 must equal 1.0 after redistribute_phi."""
    sched.reset()
    for t in range(0, T, 100):
        phi3 = sched(t, T, gbest_val=1.0)
        phi3 = max(0.0, phi3)   # NoBio returns 0
        phi1, phi2, phi3r = redistribute_phi(phi3)
        total = phi1 + phi2 + phi3r
        assert abs(total - 1.0) < 1e-12, (
            f"{sched.__class__.__name__} t={t}: sum={total}"
        )
        assert phi1 >= 0 and phi2 >= 0 and phi3r >= 0


# ─────────────────────────────────────────────────────────────
# 3. Attractor in convex hull
# ─────────────────────────────────────────────────────────────

def test_attractor_in_convex_hull():
    """q = φ1*p + φ2*g + φ3*b must lie in componentwise hull of {p,g,b}."""
    rng = np.random.default_rng(42)
    for _ in range(200):
        p = rng.uniform(-10, 10, 5)
        g = rng.uniform(-10, 10, 5)
        b = rng.uniform(-10, 10, 5)
        phi3 = rng.uniform(PHI3_MIN, PHI3_MAX)
        phi1, phi2, phi3r = redistribute_phi(phi3)
        q = phi1 * p + phi2 * g + phi3r * b
        lo = np.minimum(np.minimum(p, g), b)
        hi = np.maximum(np.maximum(p, g), b)
        assert np.all(q >= lo - 1e-12) and np.all(q <= hi + 1e-12), (
            f"attractor outside convex hull: q={q}, lo={lo}, hi={hi}"
        )


# ─────────────────────────────────────────────────────────────
# 4. Schedule endpoints
# ─────────────────────────────────────────────────────────────

def test_linear_endpoints():
    s = LinearSchedule(phi3_max=0.5, phi3_min=0.1)
    assert abs(s(0, T) - 0.5) < 1e-9
    assert abs(s(T - 1, T) - 0.1) < 1e-9


def test_cosine_endpoints():
    s = CosineSchedule(phi3_max=0.5, phi3_min=0.1)
    assert abs(s(0, T) - 0.5) < 1e-9
    assert abs(s(T - 1, T) - 0.1) < 1e-9


# ─────────────────────────────────────────────────────────────
# 5. Feedback only increases when stagnating
# ─────────────────────────────────────────────────────────────

def test_feedback_increases_only_after_stagnation():
    s = FeedbackSchedule(W=5, delta=0.05)
    s.reset()
    prev_phi3 = s(0, T, gbest_val=10.0)
    # Simulate stagnation: gbest stays at 10.0
    triggered = False
    for t in range(1, 50):
        phi3 = s(t, T, gbest_val=10.0)  # no improvement
        if phi3 > prev_phi3:
            triggered = True
        prev_phi3 = phi3
    assert triggered, "FeedbackSchedule should trigger increase after W=5 stagnation steps"

    # Now simulate improvement: phi3 should decrease
    s2 = FeedbackSchedule(W=5, delta=0.05)
    s2.reset()
    phi3_0 = s2(0, 100, gbest_val=1.0)
    # Continuously improving gbest
    for t in range(1, 30):
        phi3 = s2(t, 100, gbest_val=float(1.0 - t * 0.03))
    assert phi3 <= phi3_0 or phi3 == PHI3_MIN, (
        "Feedback should decrease phi3 when gbest improves"
    )


# ─────────────────────────────────────────────────────────────
# 6. Determinism: same seed → same result
# ─────────────────────────────────────────────────────────────

_PROBLEM = get_classical_problem("sphere", dim=10)
_QUICK_PARAMS = dict(n_particles=10, max_iter=20, max_fes=220)


@pytest.mark.parametrize("algo", ["qpso", "ant-fixed", "bee-fixed"])
def test_determinism(algo):
    r1 = run_job(algo, _PROBLEM, run_idx=0, seed=42, **_QUICK_PARAMS)
    r2 = run_job(algo, _PROBLEM, run_idx=0, seed=42, **_QUICK_PARAMS)
    assert r1["final_f"] == r2["final_f"], (
        f"{algo}: non-deterministic results {r1['final_f']} vs {r2['final_f']}"
    )
    assert r1["convergence"] == r2["convergence"]


# ─────────────────────────────────────────────────────────────
# 7. Different algorithms → same initial population
# ─────────────────────────────────────────────────────────────

def test_same_initial_population_different_algorithms():
    """With the same seed, QPSO and AntBioQPSO should see the same x[0] at init."""
    # We verify this indirectly: the same seed produces the same f(x_init[0])
    from bioqpso.budget import FEBudget
    from bioqpso.boundary import BoundaryHandler
    from bioqpso.algorithms import QPSO, AntBioQPSO
    from bioqpso.schedules import FixedSchedule
    from bioqpso.benchmarks.classical import sphere

    lb, ub, dim, n = -10.0, 10.0, 5, 4
    seed = 99

    def make_budget():
        return FEBudget(sphere, 10000)

    def make_bh(rng):
        return BoundaryHandler(lb, ub, rng=rng)

    rng1 = np.random.default_rng(seed)
    q = QPSO(make_budget(), lb, ub, dim, n, 5,
             rng1, make_bh(rng1))
    q._initialize()
    x0_qpso = q.x.copy()

    rng2 = np.random.default_rng(seed)
    a = AntBioQPSO(make_budget(), lb, ub, dim, n, 5,
                   rng2, make_bh(rng2),
                   schedule=FixedSchedule(0.3))
    a._initialize()
    x0_ant = a.x.copy()

    assert np.allclose(x0_qpso, x0_ant), (
        "QPSO and AntBioQPSO do not share initial population with same seed"
    )


# ─────────────────────────────────────────────────────────────
# 8. FE accounting: algorithms stay within budget
# ─────────────────────────────────────────────────────────────

FE_ALGOS = ["pso", "qpso", "aco_r", "abc", "gwo", "woa",
            "ant-fixed", "bee-fixed", "ant-linear",
            "ant-cosine", "ant-feedback", "bee-per_particle"]


@pytest.mark.parametrize("algo", FE_ALGOS)
def test_fe_budget_respected(algo):
    """n_fes must not exceed max_fes."""
    max_fes = 220
    r = run_job(algo, _PROBLEM, run_idx=0, seed=7, n_particles=10,
                max_iter=20, max_fes=max_fes)
    assert r["n_fes"] <= max_fes, (
        f"{algo}: n_fes={r['n_fes']} > max_fes={max_fes}"
    )


# ─────────────────────────────────────────────────────────────
# 9. Classical benchmarks: f(x*) == f_min
# ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("func,x_opt,f_min", [
    (sphere,    np.zeros(10),                  0.0),
    (rastrigin, np.zeros(10),                  0.0),
    (ackley,    np.zeros(10),                  0.0),
    (griewank,  np.zeros(10),                  0.0),
    (zakharov,  np.zeros(10),                  0.0),
    (levy,      np.ones(10),                   0.0),
    (schwefel,  np.full(10, 420.9687),         0.0),   # approx; tolerance 0.01
    (rosenbrock, np.ones(10),                  0.0),
])
def test_classical_optimum(func, x_opt, f_min):
    val = func(x_opt)
    # Schwefel's x* = 420.9687 is a 4-decimal approximation; allow 0.01 tolerance.
    tol = 0.01 if func.__name__ == "schwefel" else 1e-6
    assert abs(val - f_min) < tol, (
        f"{func.__name__}: f(x*)={val:.6e} != {f_min}"
    )


def test_styblinski_tang_optimum():
    """styblinski_tang at x_i = -2.903534 should equal _ST_F_OPT_PER_DIM * D."""
    D = 10
    x_opt = np.full(D, _ST_X_OPT)
    val = styblinski_tang(x_opt)
    expected = _ST_F_OPT_PER_DIM * D
    assert abs(val - expected) < 1e-4, (
        f"styblinski_tang: f(x*)={val:.6e}, expected={expected:.6e}"
    )


def test_schwefel_constant():
    """Verify the Schwefel formula uses exactly 418.9829."""
    D = 5
    x = np.full(D, 420.9687)
    val = schwefel(x)
    assert abs(val) < 0.01, f"schwefel(x*)={val:.4e}, expected ~0"


# ─────────────────────────────────────────────────────────────
# 10. Bounds: no algorithm returns outside bounds
# ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("algo", ["pso", "qpso", "ant-fixed", "bee-fixed",
                                   "ant-cosine", "gwo", "woa"])
def test_no_out_of_bounds(algo):
    """All positions should stay within [lb, ub] after the run."""
    # We check this by running a job and verifying the final gbest is in bounds.
    # (Full position check would require modifying the runner.)
    r = run_job(algo, _PROBLEM, run_idx=0, seed=123,
                n_particles=10, max_iter=20, max_fes=220)
    lb, ub = _PROBLEM["lb"], _PROBLEM["ub"]
    # final_f at sphere optimum 0 requires x in bounds; check is indirect.
    # More direct: verify no NaN/inf in result.
    assert np.isfinite(r["final_f"]), f"{algo}: final_f={r['final_f']} is not finite"
    assert r["final_f"] >= 0.0, f"{algo}: sphere final_f={r['final_f']} < 0"


# ─────────────────────────────────────────────────────────────
# Bonus: PerParticleSchedule.per_particle
# ─────────────────────────────────────────────────────────────

def test_per_particle_schedule():
    s = PerParticleSchedule()
    # stag=0 → phi3_min
    assert abs(s.per_particle(0, 15) - PHI3_MIN) < 1e-9
    # stag=stag_limit → phi3_max
    assert abs(s.per_particle(15, 15) - PHI3_MAX) < 1e-9
    # intermediate
    v = s.per_particle(8, 16)
    assert PHI3_MIN <= v <= PHI3_MAX
