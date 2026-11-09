# Draft GitHub issue for opfunu (for the author to file)

**Title:** CEC2017 functions differ from the official C code on 27 of 29 functions (data files and formulas)

**Body:**

opfunu 1.0.4, `opfunu.cec_based.cec2017`. I compared every `F{k}2017` with the official CEC2017
C code (`cec17_test_func.cpp` and `input_data/` from
https://github.com/P-N-Suganthan/CEC2017-BoundContrained, `codes.rar`, "C version") at 5 random
points in [-100, 100]^D, for D = 30 and D = 50, comparing f(x) - bias. Since opfunu omits the
deleted F2, opfunu `F{k}2017` corresponds to official F(k+1) for k >= 2.

Only F1 agrees (relative difference ~1e-16). The differences have three sources.

1. **Data files numbered by the opfunu index.** For 2 <= k <= 19, `F{k}2017` loads
   `shift_data_{k}`, `M_{k}_D` (and `shuffle_data_{k}_D`), which are the files of official F(k),
   not of official F(k+1). With the official files of F(k+1) passed in (`f_shift`, `f_matrix`,
   `f_shuffle`), F3 (Rosenbrock), F9 (Schwefel) and F11 (Hybrid 2) then match the official code
   to ~1e-15. F13 and F17 come within ~3e-4 and ~6e-5. The composition functions F20-F29 already
   load the official files.
2. **Formula differences** (they remain with the official data):
   - `F22017` (Zakharov, official F3): `operator.zakharov_func` uses `sum(0.5 * x)` in the second
     term; the official code uses `sum(0.5 * (i+1) * z_i)`.
   - `F42017` (Rastrigin, official F5): opfunu uses `z = M (x - o)`; the official code scales the
     shift by 5.12/100 first (`sr_func(..., 5.12/100.0, ...)`). Adding the scaling reproduces the
     official value.
   - `F52017` (Schaffer F7, official F6): opfunu scales by 0.5/100; the official code uses 1.0. This
     makes the opfunu function nearly constant over the domain (std ~0.2 over random points).
   - F4, F6, F7, F8, F10, F12, F14-F16, F18, F19 and all composition functions differ by factors
     from ~1 to ~1e10 even with official data; I did not trace each one.

Largest relative difference of f - bias over the 5 points (opfunu data / official data loaded):

| opfunu | official | D=30 | D=50 | D=30 official data | D=50 official data |
|---|---|---|---|---|---|
| F1 | F1 | 1.9e-16 | 2.4e-16 | 1.9e-16 | 2.4e-16 |
| F2 | F3 | 3.4e+02 | 1.0 | 1.0 | 1.0 |
| F3 | F4 | 2.6 | 2.1 | 3.4e-16 | 3.3e-16 |
| F4 | F5 | 3.2e+02 | 3.0e+02 | 2.7e+02 | 2.8e+02 |
| F5 | F6 | 1.0 | 1.0 | 1.0 | 1.0 |
| F9 | F10 | 0.14 | 0.21 | 7.0e-16 | 1.9e-16 |
| F11 | F12 | 0.31 | 0.80 | 3.4e-08 | 1.2e-08 |
| F20 | F21 | 2.4e+02 | 2.0e+02 | 2.4e+02 | 2.0e+02 |

(Full table for all 29 functions available on request.)

**Reproduce.** Build the official code with a small `main()` that reads points from stdin
(on macOS/Linux remove `<WINDOWS.H>`, use `<stdlib.h>` for `<malloc.h>`, and `%lf` instead of
`%Lf`, as the official readme says), then:

```python
import subprocess, numpy as np, opfunu

def official(k, X, exe="./driver"):   # driver: argv = k D m; stdin = points; prints f
    r = subprocess.run([exe, str(k), str(X.shape[1]), str(len(X))], text=True, check=True,
                       input=" ".join(f"{v:.17g}" for v in X.ravel()), capture_output=True)
    return np.array(r.stdout.split(), float)

X = np.random.default_rng(0).uniform(-100, 100, (5, 30))
for k in range(1, 30):
    off = 1 if k == 1 else k + 1
    f = getattr(opfunu.cec_based, f"F{k}2017")(ndim=30)
    ours = np.array([f.evaluate(x) for x in X]) - f.f_bias
    ref = official(off, X) - 100 * off
    print(f"F{k} (official F{off}): max rel diff {np.max(np.abs(ours - ref) / np.abs(ref)):.2g}")
```

Environment: opfunu 1.0.4, numpy 1.26 / Python 3.11, macOS (arm64), Apple clang.
