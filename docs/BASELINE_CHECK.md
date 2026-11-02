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
- **L-SHADE:** checked on E3s results against published ranges; opfunu F3 is official F4.
  Secondary value only so far: F4 mean error 5.86e+01 at D=30 (the original paper uses CEC2014).

Sources: Mirjalili et al. 2014 (10.1016/j.advengsoft.2013.12.007); Mirjalili & Lewis 2016
(10.1016/j.advengsoft.2016.01.008); Karaboga & Basturk 2007 (10.1007/s10898-007-9149-x);
Tanabe & Fukunaga 2014 (10.1109/CEC.2014.6900380); Socha & Dorigo 2008
(10.1016/j.ejor.2006.06.046); Clerc & Kennedy 2002 (10.1109/4235.985692); Eberhart & Shi 2000
(10.1109/CEC.2000.870279); EvoloPy: https://github.com/7ossam81/EvoloPy
