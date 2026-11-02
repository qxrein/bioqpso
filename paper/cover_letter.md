# Cover letter: BioQPSO manuscript, SN Computer Science

*Draft. Sections marked "to complete" are written once the final results are available.*

## Disclosure of corrections

Earlier drafts and the preprint (rs-9178752) contained implementation errors, and some
baselines used non-standard settings. All were corrected before the results in this
submission were produced, and all numbers in this manuscript come from the corrected code.
Algorithms whose code path did not change were not re-run; spot checks reproduced stored
results exactly.

- **No-bio control.** A clip on the weight phi3 made the no-bio attractor weights sum to
  0.9 rather than 1, which shrank particles towards the origin. Conclusions about the
  bio-leader in earlier drafts relied on this control and are withdrawn.
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
CEC2017 control result existed. Because of the corrections above, some experiments were
re-run after the rule was written. `docs/DEVIATIONS.md` lists every departure.

## What is new relative to the preprint

To complete after the final results.

## How previous reviewer concerns are addressed

To complete, with section numbers: novelty, statistics, fixed vs adaptive phi3, reference
verification, baselines, scalability.
