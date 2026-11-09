# BASELINE_CHECK.md: frozen baseline parameters and checks against the original papers

Frozen by Amendment 1 of `docs/DECISION_RULE.md`: no algorithm's settings change after the
commit that contains this file. "Source status" says how each value was checked.

- **Read**: read in the primary source (PDF).
- **Secondary**: only seen attributed in another document; the primary text was not read.
- **Study choice**: set by this study; the source does not prescribe it, or it differs.

## 1. Frozen parameters (n = 30 agents, FE budget per experiment)

| Algorithm | Parameter | Value | Source | Source status |
|---|---|---|---|---|
| L-SHADE | r_Ninit, r_arc, p, H | 18, 2.6, 0.11, 6 | Tanabe & Fukunaga 2014, Table II (tuned settings) | Read |
| L-SHADE | N_min | 4 | Tanabe & Fukunaga 2014, Sec. on LPSR | Read |
| L-SHADE | boundary repair | midpoint, Eq. (4) | Tanabe & Fukunaga 2014 | Read |
| GWO | a | 2 to 0 linearly | Mirjalili et al. 2014 | Read |
| GWO | hierarchy update | once per iteration | Mirjalili et al. 2014, Fig. 6 | Read |
| WOA | a, b | 2 to 0 linearly; b = 1 | Mirjalili & Lewis 2016 (b: EvoloPy) | Read |
| WOA | p threshold | 0.5 | Mirjalili & Lewis 2016, Eq. (2.6) | Read |
| WOA | l | (a2-1)r+1, a2 from -1 to -2 | EvoloPy WOA.py; the paper says l in [-1,1] | Read; follows code, not paper |
| WOA | A, C scalar per agent; per-dimension random leader | as in code | EvoloPy WOA.py; see section 3 | Read; follows code, not paper |
| ABC | SN | n/2 = 15 | Karaboga & Basturk 2007 | Read |
| ABC | limit | SN*D = 450 at D=30 | Attributed to Karaboga in later papers | **Secondary**; value not stated in Karaboga & Basturk 2007 |
| ACO_R | xi | 0.85 | Socha & Dorigo 2008 | **Secondary** |
| ACO_R | q, k, m | 0.1, 30, 30 | Pre-existing project values, kept | **Study choice**; published values not read |
| ACO_R | sigma | xi*sum_e\|s_e-s_l\|/(k-1) | Socha & Dorigo 2008 | **Secondary** |
| PSO | w, c1, c2 | 0.729, 1.494, 1.494 | Constriction values usually credited to Clerc & Kennedy 2002 | **Secondary** |
| PSO | v_max | 0.2(ub-lb) | Specified for this study | **Study choice**; Eberhart & Shi 2000 (abstract) instead limit velocity to the variable's dynamic range |
| QPSO | beta | 1.0 to 0.5 linearly | Credited to Sun et al. in later work | **Secondary** |
| BioQPSO family | phi1:phi2 = 4:3; phi3 = 0.3 or [0.1, 0.5]; rho = 0.1; Delta = 1; L_scout = 15; W = 20; delta = 0.02; phi3(0) = 0.3 | | This study | Study choice |

**The "Secondary" and "Study choice" rows need the author's check before submission.** They
are frozen as they stand: if a primary source disagrees, that is reported, not re-tuned.

## 2. Rerun at the original papers' settings

D=30, 30 agents, 500 iterations, 30 runs; same 30 seeds for both code versions. Outputs:
`results/baseline_check/gwo_woa_prefix_original_settings.json` (commit 58b79d1) and
`results/baseline_check/original_settings.json` (current). Published values read from the
results tables of the two papers. F1 Sphere, F9 Rastrigin, F10 Ackley, F11 Griewank.

| Function | GWO published | GWO old code | GWO current | WOA published | WOA old code | WOA current |
|---|---|---|---|---|---|---|
| Sphere | 6.59e-28 | 7.52e-80 | 1.03e-30 | 1.41e-30 | 1.50e-14 | 1.41e-71 |
| Rastrigin | 0.311 | 8.39 | 9.01 | 0 | 93.9 | 0 |
| Ackley | 1.06e-13 | 1.02e-14 | 3.51e-14 | 7.40 | 3.29e-8 | 3.29e-15 |
| Griewank | 4.49e-3 | 3.38e-3 | 5.61e-3 | 2.89e-4 | 0.0128 | 7.38e-3 |

What this does and does not show:

- **GWO:** the current code is much closer on Sphere and somewhat closer on Ackley. It is
  no closer on Griewank, and neither version reproduces Rastrigin (old slightly closer).
- **WOA:** the current code matches Rastrigin and is slightly closer on Griewank. On Sphere
  it is *further* from the published mean than the old code (41 orders of magnitude below it,
  against 16 above). Neither version reproduces the published Ackley value.
- **Outcomes do not establish the WOA change.** The case for both changes is the code
  comparison in section 3, not these tables. Published tables are noisy evidence: the GWO
  table itself lists std 47.4 on Rastrigin next to a mean of 0.311.
- Results at the main budget (30,030 FEs) are not evidence about correctness, and are not used as such.

## 3. Exact code that differed

Old lines are from commit 58b79d1. Sources: GWO paper Fig. 6; WOA paper Eq.
(2.3)-(2.6) and its pseudocode; EvoloPy (`EvoloPy/optimizers/{GWO,WOA}.py`, read 2026-11-02),
the Python framework from Mirjalili's group. The authors' original MATLAB was not read.

### GWO (`bioqpso/algorithms/gwo.py`)

Dead code, old lines 36-39, draws six unused random vectors per wolf. Results are
unaffected in distribution, but it costs CPU time:

```python
for leader_pos in (alpha_pos, beta_pos, delta_pos):
    A = 2.0 * a * self.rng.random(self.dim) - a
    C = 2.0 * self.rng.random(self.dim)
    D = np.abs(C * leader_pos - self.x[i])
```

Hierarchy timing, old lines 58-70, inside the per-wolf loop:

```python
f_val = self.func(self.x[i])
self._update_bests(i, f_val)
if f_val < alpha_score: ...   # leaders replaced before the next wolf moves
```

The paper (Fig. 6) moves all agents first, then "Calculate the fitness of all search agents;
Update Xα, Xβ, and Xδ". EvoloPy GWO.py has the same order. **This is a direct conflict with the
paper**, so the change is justified on the source alone.

### WOA (`bioqpso/algorithms/woa.py`)

Old lines 32-33, 35 and 38:

```python
A = 2.0 * a * self.rng.random(self.dim) - a
C = 2.0 * self.rng.random(self.dim)
l = self.rng.uniform(a2, 1.0, self.dim)
if np.linalg.norm(A) < 1.0:
```

Old lines 44-45 (one random agent) and 56-57 (evaluation and gbest update per agent):

```python
rand_idx = int(self.rng.integers(self.n_particles))
x_rand = self.x[rand_idx]
...
f_val = self.func(self.x[i])
self._update_bests(i, f_val)
```

Reference, EvoloPy WOA.py:

```python
A = 2 * a * r1 - a  # Eq. (2.3) in the paper        (r1 is a scalar)
C = 2 * r2          # Eq. (2.4) in the paper
l = (a2 - 1) * random.random() + 1
if abs(A) >= 1: rand_leader_index = math.floor(SearchAgents_no * random.random())   # per dimension
```

Comparison with the paper, item by item:

| Item | Old code | Paper | EvoloPy | Conflict with the paper's equations? |
|---|---|---|---|---|
| Evaluate after all moves | No | Yes (pseudocode) | Yes | **Yes**; old code conflicted |
| A, C | Vector | Written as vectors, Eq. (2.3), (2.4) | Scalar | **No**: old code was consistent with the notation |
| Test on A | `norm(A) < 1` | `\|A\| < 1`, undefined for a vector | `abs(A)` scalar | **Ambiguous**: the norm is one reading of the notation |
| Random agent | Whole agent | "a random search agent" | Redrawn per dimension | **No**: old code followed the paper; current code follows EvoloPy |
| l | Vector | Random number in [-1,1] | Scalar on [a2,1] | Partly; the current code follows EvoloPy |

So, for WOA, only the evaluation order is a conflict with the paper. **The other changes make
the code match the reference implementation, not correct an error against the equations.**
This is a judgement call and is frozen as it stands. Calling the earlier WOA code an
"error" is accurate only for the evaluation order.

## 4. ABC at the settings of Karaboga & Basturk (2007)

Colony 125 (SN=62), 1000 cycles, D=30, 30 runs. Published values are from Table 1.

| Function | ABC1, 1000 cycles | ABC2, 2000 cycles | here, limit=SN*D | here, limit=30 |
|---|---|---|---|---|
| Griewank | 2.87e-9 | 0 | 2.54e-8 | 3.28e-7 |
| Rastrigin | 0.0339 | 0 | 1.66e-8 | 2.71 |
| Rosenbrock | 0.220 | 0.0201 | 15.2 | 19.1 |
| Ackley | 3e-12 | 0 | 1.06e-5 | 2.09e-5 |
| Schwefel | 147 | 3.82e-4 | 3.85e-4 | 690 |

The run has 1000 cycles, so ABC1 is the comparable column. Rosenbrock and Ackley are not
reproduced. The paper does not state its limit in the text we checked, so this table does not
establish SN*D by itself; limit=30 was a study setting and not an error against any equation.

## 5. Other changed baselines

- **ACO_R:** old code used `sigma = xi*|archive[-1]-archive[0]|` for all ants. Its correct form
  is taken from secondary descriptions of Socha & Dorigo 2008, whose full text was not read.
- **PSO:** no velocity clamp before; now clamped as specified for this study. This is a
  setting, not a correction against an equation.
- **L-SHADE:** checked on E3s (D=30, 10,000*D FEs, 30 runs) against SCSS-L-SHADE, an L-SHADE
  variant, in Table S1 of the supplement of Zhang et al. 2023 (10.1109/TEVC.2022.3220424; plain
  L-SHADE on CEC2017 was not found in an accessible source; the original paper uses CEC2014).
  Mean errors, ours vs published: official F1 (opfunu F1) 0 vs 0; official F5 (opfunu F4) 6.26 vs
  6.86; official F7 (opfunu F6) 36.3 vs 38.0. Official F4 (opfunu F3): ours 31.8 with std ~1e-13,
  published 58.6 with std ~3e-14. Both are a single attractor reached by every run; the value
  differs because opfunu loads `shift_data_3`/`M_3_D` for its F3, which are by our reading the
  files of official F3, so the instance differs. Not evidence of a fault in L-SHADE; no change.
  **Update 2026-11-06:** the comparison with the official C code (`docs/CEC2017_CHECK.md`, section 7)
  confirms the F3 explanation (with official F4 data loaded, opfunu F3 matches the official code to
  1e-15) and shows that opfunu F4 and F6 use formulas that differ from official F5 and F7. Their
  agreement with the published values (6.26 vs 6.86, 36.3 vs 38.0) is therefore not a valid check.
  Only F1 is identical to the official function, and there L-SHADE matches (0 vs 0). The paper now
  states the check on F1 only.

Sources: Mirjalili et al. 2014 (10.1016/j.advengsoft.2013.12.007); Mirjalili & Lewis 2016
(10.1016/j.advengsoft.2016.01.008); Karaboga & Basturk 2007 (10.1007/s10898-007-9149-x);
Tanabe & Fukunaga 2014 (10.1109/CEC.2014.6900380); Socha & Dorigo 2008
(10.1016/j.ejor.2006.06.046); Clerc & Kennedy 2002 (10.1109/4235.985692); Eberhart & Shi 2000
(10.1109/CEC.2000.870279); EvoloPy: https://github.com/7ossam81/EvoloPy

## L-SHADE on the official CEC2017 code (added 2026-11-09)

**Settings** (`bioqpso/algorithms/lshade.py`, no overrides in the runner) match Tanabe & Fukunaga
2014: N_init = round(18 D), H = 6, archive size round(2.6 N) shrinking with N, p = 0.11 (at least 2),
N_min = 4; memories start at 0.5; CR ~ N(M_CR, 0.1) clipped to [0, 1] with the terminal value giving
CR = 0; F ~ Cauchy(M_F, 0.1), regenerated if <= 0 and capped at 1; weighted Lehmer means; linear
population size reduction; midpoint bound repair.

**Comparison.** `scripts/check_lshade_official.py` compares our E3s L-SHADE (30 runs, 10,000 D FEs)
with the raw competition results of jSO (Brest et al. 2017, an improved L-SHADE) and LSHADE-SPACMA
(Mohamed et al. 2017), 51 runs each at the same budget, from the official CEC2017 repository
(`Codes-of-Top-Methods-and-results.zip`). jSO's raw files reproduce its published table (F4, D=30:
mean 58.67). No accessible source with plain L-SHADE on CEC2017 was found. Errors below 1e-8 count
as 0. Flag rule (set by the author): our mean better than jSO's by more than 10x, or worse by more
than 100x. Output: `results/analysis/lshade_reference_check.json`.

| F | D | ours mean | ours median | jSO mean | jSO median | SPACMA mean | SPACMA median | ours/jSO | flag |
|---|---|---|---|---|---|---|---|---|---|
| F1 | 30 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |  |
| F3 | 30 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |  |
| F4 | 30 | 58.6 | 58.6 | 58.7 | 58.6 | 58.6 | 58.6 | 0.998 |  |
| F5 | 30 | 6.69 | 6.31 | 8.56 | 8.02 | 3.45 | 2.98 | 0.781 |  |
| F6 | 30 | 0 | 0 | 6.04e-09 | 0 | 0 | 0 | 1 |  |
| F7 | 30 | 37.7 | 37.8 | 38.9 | 39.1 | 33.8 | 33.6 | 0.968 |  |
| F8 | 30 | 7.87 | 8.05 | 9.09 | 8.96 | 3.2 | 2.98 | 0.866 |  |
| F9 | 30 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |  |
| F10 | 30 | 1.47e+03 | 1.45e+03 | 1.53e+03 | 1.49e+03 | 1.44e+03 | 1.44e+03 | 0.962 |  |
| F11 | 30 | 38.8 | 61.8 | 3.04 | 1.99 | 17.8 | 5.97 | 12.8 |  |
| F12 | 30 | 997 | 986 | 170 | 139 | 615 | 558 | 5.85 |  |
| F13 | 30 | 17.7 | 18.9 | 14.8 | 15.9 | 14.6 | 15.8 | 1.19 |  |
| F14 | 30 | 19.9 | 21.4 | 21.8 | 21.2 | 23.4 | 23 | 0.912 |  |
| F15 | 30 | 3.97 | 3.62 | 1.09 | 0.779 | 4.46 | 4.17 | 3.64 |  |
| F16 | 30 | 38.1 | 34.1 | 78.9 | 25.9 | 25.2 | 18.7 | 0.482 |  |
| F17 | 30 | 33.8 | 34.9 | 32.9 | 34.6 | 30.4 | 30.5 | 1.03 |  |
| F18 | 30 | 22.6 | 22.3 | 20.4 | 20.6 | 23.4 | 23.3 | 1.11 |  |
| F19 | 30 | 6.28 | 6.28 | 4.5 | 4.09 | 10.3 | 10.4 | 1.4 |  |
| F20 | 30 | 30 | 30.9 | 29.4 | 29.3 | 83.8 | 58.5 | 1.02 |  |
| F21 | 30 | 208 | 208 | 209 | 209 | 207 | 206 | 0.995 |  |
| F22 | 30 | 100 | 100 | 100 | 100 | 100 | 100 | 1 |  |
| F23 | 30 | 353 | 353 | 351 | 351 | 355 | 356 | 1.01 |  |
| F24 | 30 | 428 | 428 | 426 | 427 | 429 | 429 | 1 |  |
| F25 | 30 | 387 | 387 | 387 | 387 | 387 | 387 | 1 |  |
| F26 | 30 | 994 | 996 | 920 | 930 | 953 | 955 | 1.08 |  |
| F27 | 30 | 506 | 506 | 497 | 495 | 505 | 505 | 1.02 |  |
| F28 | 30 | 318 | 300 | 309 | 300 | 311 | 300 | 1.03 |  |
| F29 | 30 | 434 | 434 | 434 | 433 | 445 | 443 | 1 |  |
| F30 | 30 | 1.99e+03 | 1.97e+03 | 1.97e+03 | 1.97e+03 | 2.01e+03 | 1.98e+03 | 1.01 |  |
| F1 | 50 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |  |
| F3 | 50 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |  |
| F4 | 50 | 81.3 | 90.1 | 56.2 | 28.5 | 29.4 | 28.5 | 1.45 |  |
| F5 | 50 | 11.4 | 11.6 | 16.4 | 16.2 | 5.99 | 5.97 | 0.692 |  |
| F6 | 50 | 8.35e-08 | 0 | 1.09e-06 | 3.11e-07 | 0 | 0 | 0.0763 | better by >10x |
| F7 | 50 | 64.2 | 64.1 | 66.5 | 66.6 | 57 | 57 | 0.965 |  |
| F8 | 50 | 11.4 | 11.4 | 17 | 17 | 5.81 | 5.97 | 0.674 |  |
| F9 | 50 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |  |
| F10 | 50 | 3.18e+03 | 3.19e+03 | 3.14e+03 | 3.23e+03 | 3.49e+03 | 3.47e+03 | 1.01 |  |
| F11 | 50 | 49.5 | 47.1 | 27.9 | 28.5 | 32.2 | 33.1 | 1.77 |  |
| F12 | 50 | 2.39e+03 | 2.34e+03 | 1.68e+03 | 1.65e+03 | 1.56e+03 | 1.52e+03 | 1.42 |  |
| F13 | 50 | 73.3 | 65.7 | 30.6 | 37.3 | 37.1 | 40 | 2.4 |  |
| F14 | 50 | 31.3 | 31 | 25 | 24.4 | 29.4 | 29 | 1.25 |  |
| F15 | 50 | 48.3 | 46.4 | 23.9 | 23.4 | 30.4 | 29.2 | 2.02 |  |
| F16 | 50 | 392 | 377 | 451 | 477 | 335 | 302 | 0.87 |  |
| F17 | 50 | 190 | 211 | 283 | 260 | 277 | 230 | 0.67 |  |
| F18 | 50 | 51.4 | 50.4 | 24.3 | 23.9 | 32.4 | 31.3 | 2.12 |  |
| F19 | 50 | 30.6 | 27.6 | 14.1 | 13.9 | 21.7 | 21.7 | 2.17 |  |
| F20 | 50 | 166 | 157 | 140 | 113 | 168 | 111 | 1.18 |  |
| F21 | 50 | 214 | 214 | 219 | 219 | 215 | 211 | 0.979 |  |
| F22 | 50 | 2.48e+03 | 3.36e+03 | 1.49e+03 | 100 | 1.38e+03 | 100 | 1.67 |  |
| F23 | 50 | 433 | 433 | 430 | 430 | 441 | 442 | 1.01 |  |
| F24 | 50 | 511 | 511 | 507 | 507 | 513 | 511 | 1.01 |  |
| F25 | 50 | 481 | 480 | 481 | 480 | 481 | 480 | 1 |  |
| F26 | 50 | 1.2e+03 | 1.2e+03 | 1.13e+03 | 1.13e+03 | 1.14e+03 | 1.15e+03 | 1.06 |  |
| F27 | 50 | 537 | 534 | 511 | 512 | 532 | 529 | 1.05 |  |
| F28 | 50 | 465 | 459 | 460 | 459 | 460 | 459 | 1.01 |  |
| F29 | 50 | 348 | 349 | 363 | 364 | 392 | 372 | 0.959 |  |
| F30 | 50 | 6.59e+05 | 6.4e+05 | 6.01e+05 | 5.9e+05 | 6.68e+05 | 6.62e+05 | 1.1 |  |

**Result.** One flag: F6 at D=50 (ours 8.4e-8, jSO 1.1e-6), where both are at the 1e-8 floor, so it
is not a meaningful difference. The ratio of our mean to jSO's lies between 0.5 and 2 on 25 of 29
functions at D=30 and 24 of 29 at D=50. Larger gaps, all in jSO's favor and below the flag: F11
(12.8x), F12 (5.9x) and F15 (3.6x) at D=30; F13, F15, F18 and F19 (2.0-2.4x) at D=50. These are
hybrid functions; jSO differs from L-SHADE in its mutation weights and its p, F and CR rules, so a
gap in jSO's favor is plausible, and none reaches the flag. Our L-SHADE is consistent with the published L-SHADE family on the official suite.
No change.
