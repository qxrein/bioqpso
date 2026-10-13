# AUDIT.md — Phase 1 Code Audit

Generated: 2026-10-05  
Branch: merged-unified  
Python: 3.11.14 (`.venv`)  

---

## 1. Repository layout

```
bioqpso/
  __init__.py            re-exports from submodules; no GWO/WOA/AdaBio variants
  optimizers.py          BaseOptimizer, PSO, QPSO, ACO, ABC, AntBioQPSO, BeeBioQPSO
  problems.py            sphere/rastrigin/ackley/griewank/rosenbrock/schwefel/levy/
                         zakharov/styblinski_tang/dixon_price + make_feature_selection_problem
  experiments.py         run_experiment(), print_results_table(), run_statistical_analysis()
  plots.py               plot_convergence() only; no bar/CD/heat-map functions
  sensitivity.py         run_sensitivity_analysis() (AntBioQPSO × sphere only)
  feature_selection.py   run_feature_selection_experiment() (Breast Cancer only)

main.py                  ALSO defines PSO,QPSO,ACO,ABC,AntBioQPSO,BeeBioQPSO AND four
                         AdaBio variants: AdaptiveBioQPSO (=FeedbackAda),
                         LinearAdaBioQPSO, CosineAdaBioQPSO, FeedbackAdaBioQPSO,
                         PerParticleAdaBioQPSO.  These are NOT exported from bioqpso/.
generate_revision_extras.py  post-hoc scripts (extended stats, diversity, wine/iono FS,
                              scalability D=30/50/100, phi3 plots)
rerun_linearada_levy.py  one-off re-run of LinearAda/Levy to investigate timing anomaly
bar.py                   (content not read; appears to be bar-chart helper)
scripts/reviewer_outputs.py  (not read yet)

outputs/adabioqpso/      result JSONs + PNGs from the existing (Batch B) run
results/                 older PNGs from iter1 and main run (Batch A likely)
figures/                 LaTeX/PDF flowchart
manuscript.tex           single merged manuscript (llncs, 946 lines)
```

---

## 2. Algorithm inventory

| Name in code | Location | Description |
|---|---|---|
| PSO | optimizers.py + main.py (duplicate) | Standard PSO, w=0.729, c1=c2=1.494 |
| QPSO | optimizers.py + main.py | Sun et al. QPSO, β: 1.0→0.5 linear (see §4 below) |
| ACO (ACO_R) | optimizers.py + main.py | Ant Colony Optimisation for continuous; q=0.1, ξ=0.85; archive of size n_particles |
| ABC | optimizers.py + main.py | Standard ABC; limit=30; food sources = n_particles//2 |
| AntBioQPSO | optimizers.py + main.py | 3-term attractor q=(φ1·p+φ2·g+φ3·b)/(φ1+φ2+φ3); ant ring-neighbour leader via pheromone; ρ=0.1, Δ=1.0 |
| BeeBioQPSO | optimizers.py + main.py | Same 3-term attractor; bee ring-neighbour leader via pbest fitness; scout limit L=15 |
| AdaptiveBioQPSO (=FeedbackAda) | main.py only | Same as FeedbackAda (duplicate; not in bioqpso pkg) |
| LinearAdaBioQPSO | main.py only | φ3 linear decay 0.5→0.1; ant leader |
| CosineAdaBioQPSO | main.py only | φ3 cosine decay 0.5→0.1; ant leader |
| FeedbackAdaBioQPSO | main.py only | φ3 feedback: stagnation counter > W (=20) → increase by δ (=0.02); W=20, δ=0.02 |
| PerParticleAdaBioQPSO | main.py only | Per-particle φ3 ∝ stagnation_counter/stagnation_limit; bee leader |
| GWO | **ABSENT** from own code; only in mealpy (.venv) |
| WOA | **ABSENT** from own code; only in mealpy (.venv) |

**Critical gap**: GWO and WOA appear in the manuscript tables (Run A) but have no own NumPy implementation in the repo. They were run via mealpy in the first batch. Results are therefore on a **different code path** from all other algorithms and are NOT reproducible from this repo alone.

---

## 3. Benchmark functions

| Name | Bounds | D | Optimum x* | f(x*) |
|---|---|---|---|---|
| sphere | [-100, 100] | 30 | 0^D | 0 |
| rastrigin | [-5.12, 5.12] | 30 | 0^D | 0 |
| ackley | [-32.768, 32.768] | 30 | 0^D | 0 |
| griewank | [-600, 600] | 30 | 0^D | 0 |
| rosenbrock | [-2.048, 2.048] | 30 | 1^D | 0 |
| schwefel | [-500, 500] | 30 | 420.9687^D | 0 (f=418.9829·D − Σ x_i·sin(√|x_i|)) |
| levy | [-10, 10] | 30 | 1^D | 0 |
| zakharov | [-5, 10] | 30 | 0^D | 0 |
| styblinski_tang | [-5, 5] | 30 | ≈−2.903534^D | ≈−39.166·D |
| dixon_price | [-10, 10] | 30 | analytically defined | 0 |

**Note**: styblinski_tang has a **negative** global optimum (≈ −39.166·D for D=30 ≈ −1174.99). The manuscript treats convergence toward this as minimisation, which is correct. The convergence plot uses symlog scale.

**Schwefel**: optimum is at x_i = 420.9687 for all i (approximately; 420.9687 ≈ 418.9829·(2/π)·some_constant — exact is x* such that sin(√x*) + (√x*/2)·cos(√x*) = 0). The code uses `418.9829 * D − Σ x_i·sin(√|x_i|)` which is correct; minimum = 0 when all x_i = 420.9687.

---

## 4. QPSO β parameter — critical note

In `bioqpso/optimizers.py` and `main.py`:
```python
current_beta = 1.0 - (1.0 - self.beta) * (t / self.max_iter)
```
With `beta=0.5` (the default), this computes:
- t=0: β(0) = 1.0 − 0.5·0 = **1.0**
- t=T−1: β(T−1) = 1.0 − 0.5·(T−1)/T ≈ **0.5** (for large T)

So β actually **starts at 1.0** and **ends at ~0.5**, i.e., the schedule is β: 1.0→0.5 (linear decrease). This is correct for Sun et al. QPSO (large contraction coefficient early, small late). The parameter name `beta` in the constructor is the **final** β value, not the initial or mean.

For the standard QPSO β = 0.75 constant, the constructor should use `beta=0.75` and the schedule should not change β. Currently the code always uses a schedule, which is slightly non-standard but well-defined and consistent across all algorithms.

---

## 5. Seed discipline

**Current state: NO shared seeds across algorithms or runs.**

- `BaseOptimizer._initialize()` uses `np.random.rand(n_particles, D)` — the global numpy RNG.
- No `np.random.seed()` is set anywhere in the package.
- `main.py` also uses the global RNG without seeding.
- Each run of `run_experiment()` starts from wherever the global RNG happens to be.

**Consequence**: Results are NOT reproducible between separate script invocations. The same config run twice will give different results. The two batches (Run A, Run B) used different RNG states and possibly different code versions; it is impossible to determine whether algorithm differences or RNG differences drive the observed numerical discrepancies.

---

## 6. Iteration vs. Function Evaluation (FE) counting

**Current state: iterations are counted, NOT FEs; no FE budget enforcer exists.**

| Phase | FEs per iteration | Notes |
|---|---|---|
| `_initialize()` | n_particles evaluations | Initial population evaluation counts but is NOT tracked |
| PSO | n_particles per iter | Total ≈ n·(T+1) = 30·1001 = 30,030 for n=30, T=1000 |
| QPSO | n_particles per iter | Same as PSO |
| AntBioQPSO | n_particles per iter | Does not count scout re-evaluations separately (none needed — no scout phase) |
| BeeBioQPSO | n_particles per iter PLUS scout phase | **Scout re-evaluations are uncounted.** BeeBioQPSO fires scouts whenever `stagnation_counter[i] > stagnation_limit=15`. This uses an additional `self.func(self.x[i])` call that is not counted in any budget. Scouts therefore give BeeBioQPSO **extra function evaluations** vs. all other algorithms — a systematic unfair advantage. |
| ACO | 2×n_particles per iter (archive eval at init + n per iter) | In `run()`, the archive is initialised from `self.x` but re-evaluated: `archive_f = np.array([self.func(p) for p in archive])` — this is n_particles extra FEs on top of the n_particles from `_initialize()`, then n new solutions per iteration. Total ≈ 2·n + n·T. |
| ABC | Employed: n_food; Onlooker: n_food; Scout: occasional (per trial) | food_sources = n//2; each iteration: ~n/2 employed + ~n/2 onlooker + occasional scouts. Total FEs ≈ n·T roughly. |

**Summary of FE inequality**: BeeBioQPSO gets extra scout evaluations; ACO gets extra archive-init evaluations. No algorithm is stopped at a hard FE budget. This needs to be fixed for a fair comparison.

---

## 7. Personal best / global best initialisation

In `BaseOptimizer._initialize()`:
```python
self.pbest_val = np.array([self.func(p) for p in self.pbest_pos])
```
These n_particles evaluations ARE performed but are NOT counted toward any budget. They set the initial personal bests. This is consistent across all algorithms that call `super()._initialize()`. However, the total FEs consumed are n_particles (init) + n_particles × max_iter (per-iter) for PSO/QPSO/BioQPSO, which is n·(T+1).

ACO additionally evaluates the archive: `archive_f = np.array([self.func(p) for p in archive])` — another n_particles FEs immediately after `_initialize()`. So ACO total ≈ 2n + n·T.

---

## 8. Boundary handling

All algorithms use:
```python
self.x[i] = np.clip(self.x[i], self.min_bound, self.max_bound)
```
This is **clip (projection)** boundary handling applied to every particle after each position update. It is applied consistently to all algorithms in the main run loop. However:
- **ACO candidates** are also clipped: `candidate = np.clip(candidate, ...)` ✓
- **ABC candidates** are clipped ✓
- **BeeBioQPSO scouts** do NOT clip: `self.x[i] = self.min_bound + (self.max_bound - self.min_bound) * np.random.rand(self.D)` — this generates within bounds by construction ✓

No reflect, absorb, or random-reinit handlers exist. The Schwefel optimum (≈420.97) lies near the boundary of [−500, 500]; clip biases particles toward the boundary wall and may affect Schwefel performance.

---

## 9. Package versions (.venv)

| Package | Version |
|---|---|
| Python | 3.11.14 |
| numpy | 1.26.0 |
| scipy | 1.16.3 |
| scikit-learn | 1.8.0 |
| opfunu | 1.0.4 |
| matplotlib | (installed, version not checked) |
| mealpy | (installed; used for GWO/WOA in Run A only) |

---

## 10. Issues requiring fixes before re-running

1. **No shared seeds** → add `np.random.default_rng(seed)` per run, passed explicitly; same initial population for all algorithms on the same (function, run) pair.
2. **No FE budget** → add FEBudget wrapper; stop at `n*(T+1)=30030`; count scout FEs in BeeBioQPSO; fix ACO double-init.
3. **GWO/WOA not implemented** → write own NumPy implementations.
4. **AdaBio variants only in main.py** → move to `bioqpso/` package with schedules.py.
5. **ACO extra init eval** → either remove the extra `archive_f` line (use `pbest_val` from `_initialize()`) or count it.
6. **BeeBioQPSO scouts uncounted** → count scout FEs toward budget.
7. **opfunu F52017 bug** → F5 (Schaffer F7) is essentially flat in D=30 (std ≈ 0.16 over all of [−100,100]^30 due to 0.5/100 scaling); report as a known limitation; workaround: subtract f_bias and report error = f − f_bias, or note in manuscript that F5 results are meaningless.
8. **opfunu F102017 magnitude** → F10 Hybrid shows values 10^10–10^12 which is correct for unscaled sums of rotated/shifted components at D=30; not a bug but needs to be reported as raw function values (not shifted by f_bias).

---

## 11. Levy discrepancy root cause (preview — see REPRO.md)

Run A reported AntBioQPSO Levy mean ≈ 2.68e-9; Run B (outputs/adabioqpso) shows mean ≈ 1.33e-1. The raw bests from Run B contain a single outlier: run 3 = 3.99e+0. Without that outlier the mean is ≈ 1.3e-9, consistent with Run A. The root cause is non-shared seeds and the occasional divergence of AntBioQPSO on Levy under unlucky initialisation (not seeded, so a different random initial population triggered a divergent run). Details in REPRO.md.
