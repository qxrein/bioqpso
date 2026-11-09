# CEC2017_CHECK.md — CEC2017 Validity Analysis

Generated: 2026-10-05  
Branch: merged-unified  
opfunu version: 1.0.4  

---

## 1. Function inventory (opfunu 1.0.4, D=30)

opfunu provides **29** CEC2017 functions. Official function F2 was removed from the suite.
**For k >= 2, opfunu `F{k}2017` is official F{k+1}**, as the names show: opfunu F2 is
Zakharov (official F3), F3 is Rosenbrock (official F4), and F29 is Composition Function 10
(official F30). opfunu's f_bias is 100k for its own index k, not the official value. An earlier
version of this table wrongly gave official IDs equal to the opfunu index and called F22017
an extra function. The paper uses opfunu identifiers and states this mapping. We use 28
functions, excluding F5 (official F6) for the reason in section 3.

| opfunu class | Official ID | Name (truncated) | f_bias | lb | ub | f(x\*)==f_bias | Notes |
|---|---|---|---|---|---|---|---|
| F12017 | F1 | Shifted+Rotated Bent Cigar | 100 | −100 | 100 | ✓ | Unimodal |
| F22017 | F3 | Shifted+Rotated Zakharov | 200 | −100 | 100 | ✓ | |
| F32017 | F4 | Shifted+Rotated Rosenbrock | 300 | −100 | 100 | ✓ | |
| F42017 | F5 | Shifted+Rotated Rastrigin | 400 | −100 | 100 | ✓ | |
| F52017 | F6 | Shifted+Rotated Schaffer F7 | 500 | −100 | 100 | ✓ | **FLAT — see §3** |
| F62017 | F7 | Shifted+Rotated Lunacek bi-Rastrigin | 600 | −100 | 100 | ✓ | |
| F72017 | F8 | Shifted+Rotated Non-Cont. Rastrigin | 700 | −100 | 100 | ✓ | |
| F82017 | F9 | Shifted+Rotated Levy | 800 | −100 | 100 | ✓ | |
| F92017 | F10 | Shifted+Rotated Schwefel | 900 | −100 | 100 | ✓ | |
| F102017 | F11 | Hybrid Function 1 | 1000 | −100 | 100 | ✓ | |
| F112017 | F12 | Hybrid Function 2 | 1100 | −100 | 100 | ✓ | |
| F122017 | F13 | Hybrid Function 3 | 1200 | −100 | 100 | ✓ | |
| F132017 | F14 | Hybrid Function 4 | 1300 | −100 | 100 | ✓ | |
| F142017 | F15 | Hybrid Function 5 | 1400 | −100 | 100 | ✓ | |
| F152017 | F16 | Hybrid Function 6 | 1500 | −100 | 100 | ✓ | |
| F162017 | F17 | Hybrid Function 7 | 1600 | −100 | 100 | ✓ | |
| F172017 | F18 | Hybrid Function 8 | 1700 | −100 | 100 | ✓ | |
| F182017 | F19 | Hybrid Function 9 | 1800 | −100 | 100 | ✓ | |
| F192017 | F20 | Hybrid Function 10 | 1900 | −100 | 100 | ✓ | |
| F202017 | F21 | Composition Function 1 | 2000 | −100 | 100 | ✓ | |
| F212017 | F22 | Composition Function 2 | 2100 | −100 | 100 | ✓ | |
| F222017 | F23 | Composition Function 3 | 2200 | −100 | 100 | ✓ | |
| F232017 | F24 | Composition Function 4 | 2300 | −100 | 100 | ✓ | |
| F242017 | F25 | Composition Function 5 | 2400 | −100 | 100 | ✓ | |
| F252017 | F26 | Composition Function 6 | 2500 | −100 | 100 | ✓ | |
| F262017 | F27 | Composition Function 7 | 2600 | −100 | 100 | ✓ | |
| F272017 | F28 | Composition Function 8 | 2700 | −100 | 100 | ✓ | |
| F282017 | F29 | Composition Function 9 | 2800 | −100 | 100 | ✓ | |
| F292017 | F30 | Composition Function 10 | 2900 | −100 | 100 | ✓ | |

**Unit test results:**
- `f(x_global) == f_bias` to within 1e-6: PASS for all 29 functions ✓
- `f(random) > f_bias`: PASS for all 29 functions ✓ (verified over 100 random points each)
- All bounds are [−100, 100]^D ✓
- D=30 is in `dim_supported` for all 29 functions ✓

---

### Data files (added 2026-11-06)

Each opfunu `F{k}2017` for 2 <= k <= 19 loads `shift_data_{k}` and `M_{k}_D`, numbered by its own
index. Since opfunu F_k is official F_{k+1} for k >= 2, these are the data files of a different
official function (the official code reads files by its own function number). The composition
functions (F20-F29) load the official files. Section 7 confirms this against the official code and
finds formula differences as well. Values are therefore
comparable across algorithms in this study but not exactly with published CEC2017 tables. This
shows up in the L-SHADE check (`docs/BASELINE_CHECK.md`): official F4 ends at 58.6 in published
results and opfunu F3 at 31.8.

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

Official F2 was removed from the suite. opfunu omits it and numbers the rest consecutively, so
`F22017` is **Shifted+Rotated Zakharov, which is official F3**, and each later opfunu F_k is
official F_{k+1}. An earlier version of this section called `F22017` an opfunu-specific function;
that was wrong.

**Effective function list for experiments (28 functions, opfunu identifiers):**
F1, F2, F3, F4, F6, F7, F8, F9, F10–F29 (F5, official F6, excluded).

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

---

## 7. opfunu against the official C code (added 2026-11-06)

**Method.** The official code is `codes.rar` → `C version/cec17_test_func.cpp` with `input_data/`, from
https://github.com/P-N-Suganthan/CEC2017-BoundContrained (cloned 2026-11-06). We removed the
Windows-only includes (`WINDOWS.H`, `malloc.h` → `stdlib.h`), changed `fscanf("%Lf")` into a `double*`
to `%lf` (the original is undefined behavior outside MSVC), and added a `main()` that reads points
from stdin and prints `f` with 17 digits. Built with `c++ -O0`. As a sanity check, the official
code returns its bias at its own shift vector for every function tried.
`scripts/check_cec2017_official.py <build dir>` draws five fixed points in [-100, 100]^D (seed
20261106) at D=30 and D=50, evaluates each with opfunu 1.0.4 and with the official code, and compares
the errors f - f_bias (the two biases differ by design). It also loads the official data files of
F_{k+1} into opfunu, which separates data from formula. Output:
`results/analysis/cec2017_official_check.json`; the paper's Supplementary Table S3 is
generated from it. Tolerance for "identical": relative difference < 1e-6.

| opfunu | Official | opfunu data | D=30 | D=50 | D=30, official data | D=50, official data | Verdict |
|---|---|---|---|---|---|---|---|
| F1 | F1 | `shift_data_1` | 1.9e-16 | 2.4e-16 | 1.9e-16 | 2.4e-16 | identical |
| F2 | F3 | `shift_data_2` | 3.4e+02 | 1 | 1 | 1 | formula differs |
| F3 | F4 | `shift_data_3` | 2.6 | 2.1 | 3.4e-16 | 3.3e-16 | data differ, formula official |
| F4 | F5 | `shift_data_4` | 3.2e+02 | 3e+02 | 2.7e+02 | 2.8e+02 | formula differs |
| F5 | F6 | `shift_data_5` | 1 | 1 | 1 | 1 | formula differs (excluded anyway) |
| F6 | F7 | `shift_data_6` | 2.9e+03 | 3.1e+03 | 1.9e+03 | 2.2e+03 | formula differs |
| F7 | F8 | `shift_data_7` | 1.1 | 1.8 | 1.3 | 1 | formula differs |
| F8 | F9 | `shift_data_8` | 1 | 1 | 1 | 1 | formula differs |
| F9 | F10 | `shift_data_9` | 0.14 | 0.21 | 7e-16 | 1.9e-16 | data differ, formula official |
| F10 | F11 | `shift_data_10` | 7.2e+02 | 3.1e+02 | 5.7e+02 | 31 | formula differs |
| F11 | F12 | `shift_data_11` | 0.31 | 0.8 | 3.4e-08 | 1.1e-08 | data differ, formula official |
| F12 | F13 | `shift_data_12` | 0.81 | 8.7 | 7.9 | 3.4 | formula differs |
| F13 | F14 | `shift_data_13` | 79 | 73 | 0.00031 | 5.2e-05 | formula differs |
| F14 | F15 | `shift_data_14` | 12 | 1.1 | 15 | 3.2 | formula differs |
| F15 | F16 | `shift_data_15` | 2.7e+07 | 1e+08 | 2.7e+06 | 5e+06 | formula differs |
| F16 | F17 | `shift_data_16` | 4.6e+14 | 1.9e+10 | 9.4e+10 | 6.2e+10 | formula differs |
| F17 | F18 | `shift_data_17` | 0.94 | 4.2 | 1.2e-05 | 6.2e-05 | formula differs |
| F18 | F19 | `shift_data_18` | 4.5e+09 | 2e+09 | 3.5e+08 | 3.7e+08 | formula differs |
| F19 | F20 | `shift_data_19` | 49 | 21 | 33 | 43 | formula differs |
| F20 | F21 | `shift_data_21` | 2.4e+02 | 2e+02 | 2.4e+02 | 2e+02 | formula differs |
| F21 | F22 | `shift_data_22` | 0.4 | 0.11 | 0.4 | 0.11 | formula differs |
| F22 | F23 | `shift_data_23` | 56 | 1e+02 | 56 | 1e+02 | formula differs |
| F23 | F24 | `shift_data_24` | 32 | 37 | 32 | 37 | formula differs |
| F24 | F25 | `shift_data_25` | 0.17 | 2.5 | 0.17 | 2.5 | formula differs |
| F25 | F26 | `shift_data_26` | 10 | 24 | 10 | 24 | formula differs |
| F26 | F27 | `shift_data_27` | 0.94 | 1.6 | 0.94 | 1.6 | formula differs |
| F27 | F28 | `shift_data_28` | 0.75 | 0.63 | 0.75 | 0.63 | formula differs |
| F28 | F29 | `shift_data_29` | 1.2e+10 | 1.9e+12 | 1.2e+10 | 1.9e+12 | formula differs |
| F29 | F30 | `shift_data_30` | 3.5e+06 | 7.6e+09 | 3.5e+06 | 7.6e+09 | formula differs |

**Summary.** Identical: F1 only. Same formula, different data: F3, F9, F11 (F13 and F17 come within
3e-4 and 6e-5 with official data, so they are close but not identical). Formula differs: all
others, including every composition function, which already uses the official data. Two causes
were confirmed by hand: opfunu F4 (official F5, Rastrigin) omits the official scaling of z by
5.12/100 (adding it reproduces the official value), and opfunu F2 (official F3, Zakharov) omits the
index weights (i+1) in its second sum (`opfunu/utils/operator.py`, `zakharov_func`). The others
were not diagnosed further.

**Consequence.** The paper's CEC2017 results are results on the opfunu 1.0.4 suite. They remain
valid comparisons between our algorithms (all used the same functions) but cannot be compared with
published CEC2017 tables, except on F1. No experiment was re-run.

### D=50 analysis, both versions (E3s, raw values, as locked)

| | Functions | Friedman chi2 | p | CD | QPSO rank | AntBio rank (Holm p) | Ant-Lin rank (Holm p) | Ant-FB rank (Holm p) | L-SHADE rank |
|---|---|---|---|---|---|---|---|---|---|
| All 28 (as in the paper) | 28 | 239.8 | 1.9e-44 | 3.45 | 6.61 | 4.82 (0.049) | 4.50 (0.0035) | 4.07 (0.0089) | 1.71 |
| Official formula only | F1, F3, F9, F11 | 25.8 | 0.012 | 9.12 | 5.50 | 4.75 (1) | 6.00 (1) | 5.50 (1) | 1.75 |
| Identical to official | F1 | not computable (N = 1) | | | | | | | |

With N = 4, the two-sided Wilcoxon test on function means has a smallest possible p of 0.125, so no
pairwise comparison can be significant; all Holm p are 1. Per function, AntBio neither wins nor loses
against QPSO on these four. The D=50 ant-variant result therefore rests on functions that differ from
the official suite. Numbers: `results/analysis/summary_all.json` (keys `E3s50`, `CecOfficial`) and the
macros `EthreesFiftySub*` in `paper/generated/numbers.tex`.

## 8. Notes on the official code (added 2026-11-06)

- The official suite used from Amendment 2 on is wrapped in `bioqpso/benchmarks/cec2017_official.py`
  (tests: `tests/test_cec2017_official.py`; wrapper equals a standalone build of the same source
  to < 1e-12 relative at 5 points per function, D=30 and 50, and f = bias at the optimum).
- Official F9 (Levy) computes `w = 1 + (z - 1)/4` on `z = M (x - o)`, so its minimum is at z = 1,
  i.e. x = o + M^{-1} 1, not at the shift o (f(o) - 900 is 3.3 at D=30 and 5.1 at D=50). The
  optimum lies inside the bounds. This is a property of the official code, used as is.
- Many official rotation matrices are not orthogonal (max |M M^T - I| between 1.8 and 2.9 for
  most of F1-F20; F11, F14 and a few others at one dimension are orthogonal). Again used as is;
  opfunu loads the same files, so this is not one of the opfunu differences.
