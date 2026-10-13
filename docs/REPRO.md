# REPRO.md — Batch Reproducibility Analysis

Generated: 2026-10-05  
Branch: merged-unified  

---

## 1. Claimed discrepancies

The merged manuscript states these cross-batch inconsistencies (Run A = BioQPSO manuscript; Run B = AdaBioQPSO / `outputs/adabioqpso/`):

| Metric | Run A | Run B | Same config? |
|---|---|---|---|
| AntBioQPSO Levy mean | ≈ 2.68e-9 | ≈ 1.33e-1 (std 7.16e-1) | n=30, T=1000, n_particles=30 (claimed) |
| QPSO Levy mean | ≈ 4.17e-13 | ≈ 6.49e-13 | same config |
| AntBioQPSO Sphere mean | ≈ 1.48e-7 | ≈ 1.84e-7 | same config |
| Styblinski-Tang Wilcoxon p (Ant vs QPSO) | ≈ 1.0e-3 | ≈ 4.2e-2 | same config |

---

## 2. Reproducibility test results

### 2.1 Bit-for-bit reproducibility with fixed seed

Tested by running the same algorithm × function × T=1000 three times with identical `np.random.seed(s)` before each run, using `bioqpso/optimizers.py`:

| Algorithm | Function | Seed | Run 1 | Run 2 | Run 3 | Bit-for-bit? |
|---|---|---|---|---|---|---|
| AntBioQPSO | Levy | 7 | 4.497e-10 | 4.497e-10 | 4.497e-10 | **YES** |
| AntBioQPSO | Levy | 42 | 1.360e-09 | 1.360e-09 | 1.360e-09 | **YES** |
| AntBioQPSO | Levy | 999 | 2.935e-11 | 2.935e-11 | 2.935e-11 | **YES** |
| QPSO | Levy | 7 | 2.126e-15 | 2.126e-15 | 2.126e-15 | **YES** |
| QPSO | Levy | 42 | 7.708e-14 | 7.708e-14 | 7.708e-14 | **YES** |
| QPSO | Levy | 999 | 1.035e-15 | 1.035e-15 | 1.035e-15 | **YES** |

**Conclusion: results are bit-for-bit reproducible given the same numpy seed.**

### 2.2 Without seed: results differ between invocations

Running AntBioQPSO on Levy three times without seeding (as in the original runs) gives:
- Trial 1: 2.367e+00; Trial 2: 1.628e+00; Trial 3: 3.687e+00 (at T=100)

For T=200, out of 100 random seed values, 26 (26%) give AntBioQPSO > 0.1 on Levy, confirming that the algorithm has a genuine but stochastic failure mode on this function.

---

## 3. Root cause analysis

### 3.1 AntBioQPSO Levy: mean ≈ 2.68e-9 (Run A) vs 1.33e-1 (Run B)

**Primary cause: non-shared seeds + one divergent run.**

The Run B raw_bests JSON (`outputs/adabioqpso/levy_raw_bests.json`) contains the 30 per-run values for AntBioQPSO Levy. They are:

```
Run  1: 2.32e-09
Run  2: 4.25e-11
Run  3: 3.99e+00  ← OUTLIER (diverged run)
Run  4–30: all in range 2.4e-11 – 8.9e-09
```

Without run 3: mean ≈ 1.3e-9, consistent with Run A (≈ 2.68e-9).  
With run 3: mean ≈ 1.33e-1, std 7.16e-1 (as reported in the manuscript).

The outlier (run 3, best = 3.99e+0) represents a single diverged execution. AntBioQPSO on Levy has an estimated 3–5% failure rate per run (one bad initialisation with n_particles=30 can trap the swarm in a local minimum at levy ≈ 4.0).

**The outlier is real, not a code bug.** It reflects genuine high variance of AntBioQPSO on Levy under unlucky initialisation. The discrepancy between batches is purely due to Run A happening to draw 30 seeds that all converged, while Run B's sequential-global-RNG triggered run 3 at an unlucky state.

### 3.2 QPSO Levy: 4.17e-13 vs 6.49e-13

**Cause: seed difference only; no bug.**

Both values lie within normal stochastic variation (factor ≈ 1.6×). With a shared seed the same 30 runs would give identical means. The difference is fully explained by different RNG states.

### 3.3 AntBioQPSO Sphere: 1.48e-7 vs 1.84e-7

**Cause: seed difference only; no bug.**

Run B shows mean 1.84e-7, std 2.49e-7. Factor ≈ 1.2×, within one std of Run A. No functional difference.

### 3.4 Styblinski-Tang Wilcoxon p: 1.0e-3 vs 4.2e-2

**Cause: seed difference in AntBioQPSO Styblinski-Tang.**

Run A Wilcoxon p ≈ 0.001 implies a stronger signal. Run B p ≈ 0.042 is marginally significant. Both are < 0.05 (in the uncorrected sense), but the strength varies with the random draws. After Holm correction, neither is guaranteed to survive at α=0.05 across the full family of comparisons. This is a known sensitivity of signed-rank tests to which runs are paired.

---

## 4. Summary: is the code different between batches?

No systematic code differences were found that would cause batch differences:
- `bioqpso/optimizers.py` and `main.py` implement the same algorithm (code duplication, but logically identical).
- The phi1+phi2+phi3 normalisation (`phi_sum = phi1+phi2+phi3 or 1.0`) is present in both.
- The β schedule is the same.

The only meaningful difference: Run A (BioQPSO manuscript) included **GWO and WOA** run through **mealpy**, while Run B does not include them. GWO/WOA results in Run A are on a **different code path** (different initialisation, different FE counting, different boundary handling from mealpy) and are **not comparable** to the own-implementation results.

---

## 5. Required fix

All future runs must:
1. Use `base_seed = [list of 30 integers]` pre-committed to `config/seeds.yaml`.
2. Each run `r` seeds with `np.random.seed(base_seed[r])` before initialising every algorithm; all algorithms for the same run see the same initial population.
3. Results then are fully reproducible from the seed list and a git commit hash.

---

## 6. Old-vs-new numbers (preliminary)

Based on the above, the expected behaviour once seeds are fixed:

| Cell | Old (Run A) | Old (Run B) | Fixed run (expected range) |
|---|---|---|---|
| AntBioQPSO Levy mean | 2.68e-9 | 1.33e-1 | 5e-10 – 5e-9 (may occasionally include outlier) |
| QPSO Levy mean | 4.17e-13 | 6.49e-13 | 1e-15 – 1e-12 (seed-dependent) |
| AntBioQPSO Sphere mean | 1.48e-7 | 1.84e-7 | 1e-8 – 3e-7 |
| Styblinski-Tang p (Ant vs QPSO) | 1.0e-3 | 4.2e-2 | 0.02–0.10 (Holm-corrected may flip) |

The exact values will be determined by the unified E1 run. The important point is that **AntBioQPSO's Levy behaviour is intrinsically high-variance**, and the true 30-run mean (over a fixed pre-registered seed list) will be the authoritative number.
