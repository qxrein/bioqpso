# SCHWEFEL.md: Schwefel diagnosis (E7, E7b)

Numbers below are from `paper/generated/e7_schwefel.tex` and `e7b_schwefel.tex`, on the corrected
code. An earlier version of this file drew conclusions from the faulty no-bio control (a mean
of about 9,200, against 7,700 now). Those conclusions, including the claim that a pheromone leader
creates a self-reinforcing attractor, had no support and are withdrawn.

## E7: boundary handler (mean fitness; D=30, 30 runs, 30,030 FEs)

| Boundary | QPSO | no-bio (ant) | AntBio | BeeBio | Ant-Cos |
|---|---|---|---|---|---|
| clip | 4280 | 7700 | 8050 | 8590 | 8260 |
| reflect | 1800 | 7310 | 7670 | 6060 | 8060 |
| random re-init. | 8180 | 8840 | 9080 | 8930 | 9140 |

- The leader is not the cause: the no-bio control is also poor, and AntBio does not differ
  significantly from it (decision-rule test, `bioleader_credit.tex`).
- Reflection helps QPSO and BeeBio much more than the ant variants.

## E7b: hypothesis and result

**Hypothesis:** the BioQPSO family's gap to QPSO exists because the optimum (c = 420.97) lies near
the bound. **Test:** move it inward to c = 120.97, with a CEC-style penalty outside the domain.

| Ratio of mean to QPSO | near, clip | inward, clip | near, reflect | inward, reflect |
|---|---|---|---|---|
| AntBio | 1.88 | 0.846 | 4.27 | 2.20 |
| BeeBio | 2.01 | 1.43 | 3.37 | 3.24 |
| no-bio (ant) | 1.80 | 2.13 | 4.07 | 4.71 |

Under clipping, the family members significantly worse than QPSO fall from 6 of 6 (near) to 1 of 6
(inward).

**Verdict:** supported under clipping, the handling used in the main experiments. Rejected as a
complete explanation, since the gap remains under reflection and the no-bio control stays worse
inward under both handlers.

## Permitted manuscript claims

- The leader does not cause the failure.
- Under clipping, most of the family's gap to QPSO depends on the optimum being near the bound.
- Do not claim a mechanism for the remaining gap under reflection, which was not tested.
