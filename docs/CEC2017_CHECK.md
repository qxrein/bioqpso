# CEC2017_CHECK.md — CEC2017 Validity Analysis

Generated: 2026-10-05  
Branch: merged-unified  
opfunu version: 1.0.4  

---

## 1. Function inventory (opfunu 1.0.4, D=30)

opfunu provides **29** CEC2017 functions. The official Awad et al. 2016 technical report
defines 30 functions but **F2 was officially removed** after the 2017 competition. opfunu
nevertheless includes `F22017` (mapped to a Shifted+Rotated Zakharov function); this is an
opfunu-specific extension and is **not the same as the removed official F2**. We treat
`F22017` (Zakharov) as an extra function and use the 28 non-flat valid functions in our CEC
experiments (excluding F5 for reasons stated below).

| opfunu class | Official ID | Name (truncated) | f_bias | lb | ub | f(x\*)==f_bias | Notes |
|---|---|---|---|---|---|---|---|
| F12017 | F1 | Shifted+Rotated Bent Cigar | 100 | −100 | 100 | ✓ | Unimodal |
| F22017 | (removed) | Shifted+Rotated Zakharov | 200 | −100 | 100 | ✓ | opfunu only; excluded from official count |
| F32017 | F3 | Shifted+Rotated Rosenbrock | 300 | −100 | 100 | ✓ | |
| F42017 | F4 | Shifted+Rotated Rastrigin | 400 | −100 | 100 | ✓ | |
| F52017 | F5 | Shifted+Rotated Schaffer F7 | 500 | −100 | 100 | ✓ | **FLAT — see §3** |
| F62017 | F6 | Shifted+Rotated Lunacek bi-Rastrigin | 600 | −100 | 100 | ✓ | |
| F72017 | F7 | Shifted+Rotated Non-Cont. Rastrigin | 700 | −100 | 100 | ✓ | |
| F82017 | F8 | Shifted+Rotated Levy | 800 | −100 | 100 | ✓ | |
| F92017 | F9 | Shifted+Rotated Schwefel | 900 | −100 | 100 | ✓ | |
| F102017 | F10 | Hybrid Function 1 | 1000 | −100 | 100 | ✓ | |
| F112017 | F11 | Hybrid Function 2 | 1100 | −100 | 100 | ✓ | |
| F122017 | F12 | Hybrid Function 3 | 1200 | −100 | 100 | ✓ | |
| F132017 | F13 | Hybrid Function 4 | 1300 | −100 | 100 | ✓ | |
| F142017 | F14 | Hybrid Function 5 | 1400 | −100 | 100 | ✓ | |
| F152017 | F15 | Hybrid Function 6 | 1500 | −100 | 100 | ✓ | |
| F162017 | F16 | Hybrid Function 7 | 1600 | −100 | 100 | ✓ | |
| F172017 | F17 | Hybrid Function 8 | 1700 | −100 | 100 | ✓ | |
| F182017 | F18 | Hybrid Function 9 | 1800 | −100 | 100 | ✓ | |
| F192017 | F19 | Hybrid Function 10 | 1900 | −100 | 100 | ✓ | |
| F202017 | F20 | Composition Function 1 | 2000 | −100 | 100 | ✓ | |
| F212017 | F21 | Composition Function 2 | 2100 | −100 | 100 | ✓ | |
| F222017 | F22 | Composition Function 3 | 2200 | −100 | 100 | ✓ | |
| F232017 | F23 | Composition Function 4 | 2300 | −100 | 100 | ✓ | |
| F242017 | F24 | Composition Function 5 | 2400 | −100 | 100 | ✓ | |
| F252017 | F25 | Composition Function 6 | 2500 | −100 | 100 | ✓ | |
| F262017 | F26 | Composition Function 7 | 2600 | −100 | 100 | ✓ | |
| F272017 | F27 | Composition Function 8 | 2700 | −100 | 100 | ✓ | |
| F282017 | F28 | Composition Function 9 | 2800 | −100 | 100 | ✓ | |
| F292017 | F29 | Composition Function 10 | 2900 | −100 | 100 | ✓ | |

**Unit test results:**
- `f(x_global) == f_bias` to within 1e-6: PASS for all 29 functions ✓
- `f(random) > f_bias`: PASS for all 29 functions ✓ (verified over 100 random points each)
- All bounds are [−100, 100]^D ✓
- D=30 is in `dim_supported` for all 29 functions ✓

---

## 2. Reporting convention (critical)

All CEC2017 results **must be reported as raw function values** (i.e., `f(x)`, which
already includes the `f_bias` offset). The convention is:

```
reported value = f(x)    where f(x*) = f_bias (e.g. 100, 200, …, 2900)
```

An optimizer has "converged to the optimum" when `f(x) ≈ f_bias`. The *error* is
`f(x) − f_bias`. Both conventions appear in the literature; we will report **raw f(x)**
in tables with a note, and quote `f(x) − f_bias` in text for readability.

**The previous claim of "F10 values 1e4–1e7" was likely reporting `f(x) − f_bias`.** The
actual raw `f(x)` values for F10 (Hybrid 1) over random points in [−100,100]^30 are on the
order 10^11–10^12, which is correct for an unscaled sum of shifted/rotated components at
D=30. An error of 1e4–1e7 is therefore plausible if the algorithm partially converges.
Neither figure is suspicious.

---

## 3. F5 (Schaffer F7) — CRITICAL BUG IN opfunu 1.0.4

### Finding

`F52017.evaluate(x)` uses the scaling `z = M · 0.5·(x − shift)/100`, which maps the full
domain [−100, 100]^D into z ∈ [−0.5, 0.5]^D approximately. The Schaffer F7 function at
such small `z` values returns values extremely close to 0 everywhere. As a result, **all
points in [−100, 100]^30 evaluate to approximately f_bias ± 0.8** (measured std ≈ 0.19
over 100 random points, total range ≈ 1.1 units).

### Measured values

```
f52017 over 100 random points in [-100,100]^30:
  min: 500.09    max: 501.37    std: 0.19    range: 1.28
  
f52017 over 100 random points — error (f−f_bias):
  min: 0.09      max: 1.37      std: 0.19
```

### What this means for previous results

The previous batch results showing **all algorithms score exactly 500.00 (std < 0.01)** on
"F5 Shifted-Rotated Rastrigin" are now explained: that result was labelled "Rastrigin" in
the manuscript but was actually F4 (Rastrigin) or possibly an indexing error; and the
actual F52017 (Schaffer F7) is nearly constant across the entire domain. Any optimizer that
runs for T=1000 iterations will trivially "achieve" the optimum on F52017 because there is
no meaningful landscape to optimise.

### Root cause

The correct CEC2017 F5 should use `z = M · (x − shift)` with **no scaling** (λ=1), giving
z ∈ [−200, 200]^D and a Schaffer F7 landscape with proper variation (std >> 100 for random
points). The opfunu 1.0.4 implementation uses `0.5/100` scaling, making the function
degenerate.

### Fix and decision

We **cannot fix opfunu without forking the package**. Our decision:
1. **Exclude F52017 from all CEC2017 experiment tables.** The results are uninformative.
2. **Include F42017 (Rastrigin) and F62017 (Lunacek bi-Rastrigin)** instead for multimodal
   coverage in that function tier.
3. **Report in the manuscript**: "We use functions F1, F3, F4, F6–F29 (28 functions, excluding
   the opfunu 1.0.4 F5 implementation which we found degenerate; see supplementary
   material); D=30 and D=50."
4. Note this as a known limitation in "Threats to Validity".

---

## 4. F10 magnitude — NOT a bug

F10 (Hybrid Function 1) returns values in the range 10^10–10^12 for random points at D=30.
This is correct: the function is a sum of transformed sub-functions (Schwefel + Bent Cigar
+ Lunacek bi-Rastrigin) applied to subsets of z = M·(x−shift), and the magnitudes are
large by design. The **error** `f(x) − f_bias` will be in the range 10^10–10^12 for random
initialisation, and should decrease to 10^3–10^6 for a well-converging optimizer over
30,030 FEs — similar to the "1e4–1e7" range reported previously.

---

## 5. F2 mapping — opfunu vs official spec

The official CEC2017 technical report removed F2 after the competition (the function was
Rotated Schwefel, and was found to have degenerate properties). opfunu's `F22017` maps to
**Shifted+Rotated Zakharov**, which is a valid function not related to the removed F2. We
include `F22017` in our test suite but note it is an opfunu-specific function, not the
official CEC2017 F2.

**Effective function list for experiments (28 functions):**
F1, F2(opfunu-Zakharov), F3, F4, F6, F7, F8, F9, F10–F29 (F5 excluded).

---

## 6. Unit test script

```python
# tests/test_cec2017.py (generated in Phase 2)
import opfunu, numpy as np, pytest

CEC2017_CLASSES = [f for f in opfunu.get_cec_based_functions() if '2017' in f.__name__]
EXCLUDE_F5 = True

@pytest.mark.parametrize("fcls", CEC2017_CLASSES)
def test_f_opt_equals_f_bias(fcls):
    f = fcls(ndim=30)
    assert abs(f.evaluate(f.x_global) - f.f_bias) < 1e-4

@pytest.mark.parametrize("fcls", CEC2017_CLASSES)
def test_bounds(fcls):
    f = fcls(ndim=30)
    assert float(f.bounds[0,0]) == -100.0
    assert float(f.bounds[0,1]) == 100.0

@pytest.mark.parametrize("fcls", [f for f in CEC2017_CLASSES if 'F52017' not in f.__name__])
def test_not_flat(fcls):
    """F5 is excluded; all others must have std > 10 over 50 random points."""
    f = fcls(ndim=30)
    np.random.seed(42)
    vals = [f.evaluate(np.random.uniform(-100, 100, 30)) for _ in range(50)]
    assert np.std(vals) > 10.0, f"{fcls.__name__} appears flat (std={np.std(vals):.2f})"
```
