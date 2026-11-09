# Cover letter: BioQPSO manuscript, SN Computer Science

*Draft for the author to edit and sign. Fill in the bracketed journal names and dates.*

## Prior dissemination and submission history

An earlier version of this work is available as a preprint (Research Square, rs-9178752). Earlier
versions were submitted to [journal, month year] and [journal, month year] and rejected. The
reviewers' main concerns are listed below with how this manuscript addresses them. The manuscript
is not under review elsewhere, and no part of it has been published in a journal.

## Disclosure of corrections

Earlier drafts and the preprint (rs-9178752) contained implementation errors, and some
baselines used non-standard settings. All were corrected before the results in this
submission were produced, and all numbers in this manuscript come from the corrected code.
Algorithms whose code path did not change were not re-run; spot checks reproduced stored
results exactly.

- **No-bio control.** A clip on the weight phi3 made the no-bio attractor weights sum to
  0.9 rather than 1, which shrank particles towards the origin. Conclusions about the
  bio-leader in earlier drafts relied on this control and are withdrawn.
- **CEC2017 implementation.** Earlier CEC2017 runs, including the preprint's, used the `opfunu`
  library. Compared with the official C code, it agrees on only one of the 28 functions we used,
  in data or formula. All CEC2017 experiments were repeated on the official code with the same
  algorithms, settings, seeds and budgets. The manuscript reports only the official runs; the
  `opfunu` results and the comparison are in Supplementary Section S3.
- **GWO.** The leader hierarchy was updated inside the agent loop, not once per iteration as
  in the original pseudocode, and a dead code path consumed random numbers.
- **WOA.** Evaluation took place inside the agent loop, unlike the original pseudocode. The
  remaining changes (scalar A and C, per-dimension random leader) align the code with a
  reference implementation from the authors' group; they do not correct a conflict with
  the published equations.
- **ACO_R.** The sampling width used a shared worst-to-best distance instead of the
  per-solution mean distance.
- **ABC.** The abandonment limit was 30; it is now SN*D. This was a non-standard setting,
  rather than a coding error.
- **PSO.** A velocity clamp was added. This is also a setting, rather than a coding error.

`docs/BASELINE_CHECK.md` lists the code lines, the sources and a comparison at the original
papers' settings. It also records which comparisons do not support a change.

## Decision rules

The study uses a decision rule specified before the analysis (`docs/DECISION_RULE.md`). The
file timestamp is 2026-10-05, before the first experiment log of the same day. Its first git
commit is dated 2026-10-14, after those first experiments, so git alone does not prove the
order. The rule for CEC2017 (at least 15 of 28 functions) was added on 2026-11-02, before any
CEC2017 control result existed, and restated as 15 of 29 for the official suite on 2026-11-06,
before any run on the official code. Because of the corrections above, some experiments were
re-run after the rule was written. `docs/DEVIATIONS.md` lists every departure.

## What is new relative to the preprint

The preprint (rs-9178752) compared fixed-weight BioQPSO with baselines. This manuscript differs in
its question, code and conclusions.

- It decomposes the attractor into random weights, fixed weights, scouts, leader and schedule,
  with a control for each (Section 3.4, Table 6).
- It adds adaptive schedules for phi3 and tests them against the best fixed value.
- It adds L-SHADE, shifted classical functions, controls on CEC2017, a standard-budget CEC2017
  study at D=30 and D=50, a test of the Schwefel failure, and a sensitivity analysis.
- All results were regenerated with corrected code, and CEC2017 uses the official code. Its
  conclusions are largely negative. At the standard CEC2017 budget, some ant-leader variants are
  better than QPSO on function means under Holm-corrected Wilcoxon tests (one at D=30, two at D=50)
  but not under the Nemenyi test; we report this as weak evidence only.
- It checks its L-SHADE against the CEC2017 competition results of jSO and LSHADE-SPACMA at the same
  budget (Section 4.5 and Supplementary Table S11).

## How previous reviewer concerns are addressed

- **Novelty.** The method's novelty is incremental, and we measure it with controls rather than
  assert it (Sections 3.4, 4.4 and 4.5).
- **Statistics.** Friedman and Nemenyi across functions, Holm-corrected Wilcoxon tests, effect sizes
  with confidence intervals, and a stated decision rule with its dates (Section 4.1 and Supplementary Section S6).
- **Fixed vs adaptive phi3.** Each schedule is tested against the best fixed value and the default;
  none is credited on any function (Sections 4.4 and 4.5).
- **Hallucinated references.** Each reference was checked against a DOI or publisher page, and
  `references.bib` records where. Any entry that could not be checked is marked there.
- **Baselines.** Settings are frozen and sourced per value in `docs/BASELINE_CHECK.md`, L-SHADE is
  included, and origin bias is tested directly with shifted functions (Sections 4.1 and 4.3).
- **Scalability.** D = 30, 50 and 100 at 10,000 D evaluations (Section 4.7), and CEC2017 at D = 30
  and 50 at the standard budget (Section 4.5).

## Supplementary material

The sensitivity analysis, the inward-optimum Schwefel experiment, the `opfunu` discrepancy with the
superseded `opfunu` results, the L-SHADE check, all per-function CEC2017 tables and all standard
deviations are in a separate supplementary PDF.
