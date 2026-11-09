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
10. **E9 re-run.** The first pass ran algorithms in sequence straight after hours of parallel
    load, and variants of equal cost differed by up to 1.77 times on one function. On 2026-11-03
    E9 was re-run with algorithms interleaved in shuffled order within each repetition and one
    discarded warm-up pass. E3s was paused with SIGSTOP from 00:56 to 01:00 IST for this; only
    `wall_time_s` of the runs in progress is affected, and no analysis uses that field. The
    first pass is kept as `results/e9_runtime/runtime_results_sequential_order.json`.
11. **Bolding in the feature-selection table** now includes the baselines, so that a selector is
    not shown as best when all features or MI top-k are better.
12. **CEC2017 instances.** On 2026-11-06 opfunu 1.0.4 was compared with the official C code
    (`docs/CEC2017_CHECK.md` section 7). It matches on F1 only; F3, F9 and F11 use the official
    formula with another function's data, and the other functions differ in formula. No experiment
    was re-run. The paper now states that its CEC2017 results refer to the opfunu suite, reports the
    D=50 analysis on all 28 functions and on the four with the official formula, and keeps the
    L-SHADE check on F1 only.
13. **Main text and supplement.** To fit about 25-30 pages, the sensitivity analysis (E6), E7b,
    per-function CEC2017 tables, all standard deviations and the former appendix tables moved to
    `supplementary.tex`; the main text keeps a summary of E6 and E7b.
14. **Official CEC2017 code not redistributed.** Amendment 2 says the official code is vendored in
    `bioqpso/benchmarks/cec17_official/`. For licence reasons it is instead installed by
    `scripts/fetch_cec2017_official.sh` from the official repository at a fixed commit, with SHA-256
    checks of the archive, the source and every data file, and the three documented patches. The
    installed files are byte-identical to those used for the runs (checked on 2026-11-07). The
    official prototype and globals that `cec17_capi.cpp` needs now come from a header the script
    extracts from the official `main.cpp`. The library rebuilt this way on 2026-11-07 00:34, during
    E3, gives bit-identical values to the one built before (58 function-dimension pairs, 20 points
    each, separate processes), so runs before and after the rebuild are comparable.
15. **E6 on the official suite.** After the official chain, the CEC part of E6 (official F1, F4,
    F11, F21; same settings, seeds and budget) is re-run on the official code
    (`scripts/run_e6_official.py`). The opfunu E6 stays in the supplement.
