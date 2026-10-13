"""
Phase 1 GATE 1 — CEC2017 validity unit tests.

Tests:
1. f(x_global) == f_bias for all 29 CEC2017 functions (D=30)
2. f(random) > f_bias for all functions
3. Bounds are [-100, 100]^D for all functions
4. D=30 is in dim_supported for all functions
5. All non-F5 functions have std > 10 over 50 random points (not flat)
6. F5 is effectively flat (std < 5) — documenting the known defect
"""
import numpy as np
import pytest
import opfunu

CEC2017_CLASSES = sorted(
    [f for f in opfunu.get_cec_based_functions() if "2017" in f.__name__],
    key=lambda f: f.__name__,
)
NON_FLAT_CLASSES = [f for f in CEC2017_CLASSES if "F52017" not in f.__name__]


@pytest.fixture(params=CEC2017_CLASSES, ids=lambda f: f.__name__)
def cec_func(request):
    return request.param(ndim=30)


@pytest.fixture(params=NON_FLAT_CLASSES, ids=lambda f: f.__name__)
def cec_func_non_flat(request):
    return request.param(ndim=30)


class TestCEC2017Basic:
    def test_f_opt_equals_f_bias(self, cec_func):
        """f(x*) must equal f_bias to within 1e-4."""
        fopt = cec_func.evaluate(cec_func.x_global)
        assert abs(fopt - cec_func.f_bias) < 1e-4, (
            f"{cec_func.__class__.__name__}: f(x*)={fopt}, f_bias={cec_func.f_bias}"
        )

    def test_random_points_above_f_bias(self, cec_func):
        """f(random) >= f_bias — the optimum is a global minimum."""
        np.random.seed(0)
        pts = np.random.uniform(-100, 100, (20, 30))
        vals = [cec_func.evaluate(p) for p in pts]
        assert all(v >= cec_func.f_bias - 1e-6 for v in vals), (
            f"{cec_func.__class__.__name__}: some random points below f_bias "
            f"({min(vals):.4f} < {cec_func.f_bias})"
        )

    def test_bounds_are_minus100_to_100(self, cec_func):
        """All functions use bounds [-100, 100]."""
        assert float(cec_func.bounds[0, 0]) == pytest.approx(-100.0)
        assert float(cec_func.bounds[0, 1]) == pytest.approx(100.0)

    def test_dim30_supported(self, cec_func):
        """D=30 must be in dim_supported."""
        dim_sup = getattr(cec_func, "dim_supported", [30])
        assert 30 in dim_sup, (
            f"{cec_func.__class__.__name__}: D=30 not in dim_supported={dim_sup}"
        )


class TestCEC2017NotFlat:
    def test_non_flat_functions(self, cec_func_non_flat):
        """All CEC2017 functions except F5 must have std > 10 over 50 random points."""
        np.random.seed(42)
        pts = np.random.uniform(-100, 100, (50, 30))
        vals = [cec_func_non_flat.evaluate(p) for p in pts]
        std_val = np.std(vals)
        assert std_val > 10.0, (
            f"{cec_func_non_flat.__class__.__name__} appears flat: "
            f"std={std_val:.4f} over 50 random points"
        )


class TestF5Flat:
    def test_f5_is_flat(self):
        """Document the known F52017 defect: std < 5 over the full domain."""
        f5 = opfunu.cec_based.F52017(ndim=30)
        np.random.seed(42)
        pts = np.random.uniform(-100, 100, (100, 30))
        vals = [f5.evaluate(p) for p in pts]
        std_val = np.std(vals)
        # The defect: function is essentially constant (std << 1)
        assert std_val < 5.0, (
            f"F52017 appears to be fixed — std={std_val:.4f}, expected < 5.0"
        )
        # Also confirm f_bias check still passes (so x_global is still the optimum)
        fopt = f5.evaluate(f5.x_global)
        assert abs(fopt - f5.f_bias) < 1e-4
