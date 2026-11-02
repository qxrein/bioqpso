# DEVIATIONS.md: where the study departs from docs/DECISION_RULE.md

`docs/DECISION_RULE.md` is locked and is left unedited. This file records what differs.

1. **Commit date.** The file ends with "committed before any E1/E2/E3 experiments were
   run". Its file timestamp (2026-10-05 16:28) precedes the first E1 log (16:53), but
   git shows the first commit containing it is `efdf818` of 2026-10-14. Git therefore
   does not prove it preceded the results. The paper says "decision rule specified before
   the analysis" and states the timestamp and the commit date (Amendment 1, A1.2).
2. **Threshold.** The rule says credited "overall" on **more than 5** of 10 functions.
   Earlier manuscript text said "at least 5". The paper now uses the locked wording.
   For N=28 the threshold is at least 15 of 28 (Amendment 1, A1.1), fixed before any
   CEC2017 control result existed.
3. **Re-runs after the lock.** Bugs found later forced re-runs of the no-bio controls
   and of all baselines (`docs/FINAL_FIXES.md`). The rules themselves did not change.
   These fixes were made after seeing earlier results, and the paper says so.
4. **Relative improvement.** Computed on f - f*, as the raw form is meaningless for the
   negative values of Styblinski-Tang. f* = 0 for the other functions; CEC2017 keeps the
   raw value, as locked in section 6.
5. **Added after the lock:** L-SHADE, E1b, E2c, E3s, E6, E7b and the medians table. The
   rules are applied to them in the same way, but they were not part of the rule as first written.
6. **Best-fixed baseline.** The locked text lists six fixed settings. phi3 = 0 is the
   no-bio control, so the sweep is {0.1,...,0.5} for E2. E2c uses {0.1, 0.3, 0.5}.
7. **Seeds.** `config/seeds.yaml` says the seeds were drawn from a SeedSequence, but the
   values look hand-chosen (e.g. 1234567891, 9012345679). They were fixed before the
   runs. The paper calls them "fixed in advance" and does not claim they were generated.
8. **E8.** Features are now standardised on the outer-train split and inner CV is
   5-fold, as planned. Madelon uses 630 FEs per run (1,530 elsewhere) for compute reasons.
9. **Rewritten hashes.** History was rewritten on 2026-11-02 only to remove commit-message
   trailers; dates and file contents are unchanged. `791cf44` (cited in Amendment 1, A1.3)
   is now `4e8e778`, `e090189` is `128caf1`, and the amendment `ca64d92` is `fff04fc`.
