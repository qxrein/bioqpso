# DECISION_RULE.md — Pre-registered Decision Rules

**Written:** 2026-10-05 (before any E1/E2/E3 results are seen)  
**Branch:** merged-unified  
**Status:** LOCKED — do not modify after first experiment run is started.

---

## 1. Purpose

This document pre-registers all inferential decision rules before any experiment
results are observed.  It prevents post-hoc cherry-picking of statistical tests,
thresholds, or families of comparison.  Results will be reported exactly as the
rules state, regardless of outcome.

---

## 2. Statistical framework

### 2.1 Per-function pairwise tests (Wilcoxon signed-rank)

- **Test:** Two-sided Wilcoxon signed-rank test, paired by run index (same seed
  → same initial population for all algorithms on the same run).
- **Alternative:** One-sided ("less than QPSO") for the claim that an algorithm
  *outperforms* QPSO; two-sided for general pairwise comparison.
- **Family:** All comparisons within one function (k−1 algorithms vs. QPSO,
  where k = number of algorithms).
- **Correction:** Holm–Bonferroni within each function family.
- **α:** 0.05 (Holm-corrected p < 0.05 required for significance).
- **Effect size:** Rank-biserial correlation r; bootstrap 95% CI with 10,000
  resamples.  Differences with |r| < 0.1 are flagged "negligible".
- **Practical significance threshold:** relative improvement
  (f_QPSO − f_alg)/|f_QPSO| < 1% is labelled "negligible" regardless of p-value.

### 2.2 Across-function global ranking (Friedman + Nemenyi/Holm)

- **Unit of analysis:** Functions (N = 10 for classical, N = 28 for CEC2017).
  Runs are NOT blocks for cross-function inference.
- **Statistic:** Mean rank of per-function mean fitness.
- **Test:** Friedman test over functions.
- **Post-hoc:** Nemenyi CD diagram with
  CD = q_α · √(k(k+1)/(6N)), where q_0.05 is the studentised range critical
  value, k = number of algorithms, N = number of functions.
- **Also:** Holm-corrected pairwise Wilcoxon across the k algorithms using
  per-function mean as the observation (Demšar 2006 recommended alternative).

---

## 3. E2 decision rule: does scheduling beat best-fixed φ3?

### 3.1 Baseline for comparison

For each (function, algorithm variant) cell, the **best-fixed-φ3 baseline** is
defined as:

> The best mean fitness achieved by any of the six fixed-φ3 variants
> {ant-fixed_p1, ant-fixed_p2, ant-fixed_p3, ant-fixed_p4, ant-fixed_p5,
>  bee-fixed_p1, …} **per leader type**, chosen post-hoc (conservative).

This is deliberately conservative: it allows the best-tuned fixed value to
compete against each schedule.

### 3.2 Credit rule for a schedule variant

A scheduled variant **S** (e.g. ant-cosine) is **credited on function F** if and
only if:

1. Its one-sided Wilcoxon p (vs. best-fixed for the same leader) < 0.05 after
   Holm correction over the 10 classical functions.
2. Its relative improvement over best-fixed > 1%.

A variant is credited "on a majority of functions" if it meets both conditions
on > 5 of the 10 classical functions.

### 3.3 Credit rule for the bio-leader

The bio-leader (Ant or Bee, at fixed φ3 = 0.3) is **credited on function F** if:

1. ant-fixed_p3 (or bee-fixed_p3) outperforms nobio (ant-nobio or bee-nobio)
   with one-sided Wilcoxon p < 0.05 (Holm-corrected) and relative improvement > 1%.

The bio-leader is credited "overall" if this holds on > 5 of 10 functions.

---

## 4. Manuscript framing rules (locked)

The abstract, Introduction contributions, and Conclusion will be written as
follows depending on the E2 outcome:

| E2 outcome | Framing |
|---|---|
| **(a)** scheduling beats best-fixed on majority (>5/10) of functions | Main claim: adaptive φ3 scheduling improves over the best tuned fixed value |
| **(b)** scheduling does NOT beat best-fixed on majority, but matches at lower tuning effort | Reframe: scheduling removes the need to tune φ3 at comparable or slightly lower accuracy |
| **(c)** bio-leader ≤ nobio on majority | Reframe as an honest negative/ablation: ring-neighbourhood leaders in QPSO do not consistently improve performance; paper becomes an ablation study |

The framing is selected after seeing E2 results; the selection criterion is
mechanical (count of credited functions), not subjective.

---

## 5. E1 decision rule: main classical table

No algorithm is "best on function F" in the manuscript unless its Holm-corrected
one-sided Wilcoxon p vs. QPSO is < 0.05 AND relative improvement > 1%.

Bold formatting in tables: best mean value (ties handled by relative difference
< 0.1%).

---

## 6. E3 CEC2017 decision rule

Same as §3–5 but applied to 28 CEC2017 functions at D=30.  Mean ranks computed
over N=28 functions (not N=10).  CD recomputed with this N.

---

## 7. Schwefel special treatment (E7)

Given the known boundary-optimum issue with Schwefel (optimum near 420.97 while
bounds are [−500, 500]):

- Schwefel is included in all tables.
- Any claim about which mechanism explains Schwefel results (boundary bias,
  boundary handler, schedule effect) requires support from E7.
- If E7 does not support a claimed mechanism, that mechanism claim is deleted.

---

## 8. Feature selection (E8)

Primary metric: mean accuracy on held-out outer test folds (nested CV, 5-outer ×
6-repeat = 30 folds).  Secondary: number of selected features.

An algorithm is credited on a dataset if:
1. Its paired Wilcoxon p (vs. QPSO, by outer fold) < 0.05 after Holm correction.
2. Accuracy improvement > 0.5 percentage points.

---

## 9. What does NOT change the rules

- A surprising result (e.g., all algorithms equivalent on Schwefel, or scheduling
  is universally worse than fixed) does not change these rules.
- Negative results are reported as-is.
- No functions, seeds, or runs are excluded post-hoc.

---

*This file was committed before any E1/E2/E3 experiments were run.*
*SHA of the first commit containing this file: see git log.*

---

## Amendment 1 — 2026-11-02 19:34 +0530

Appended by the author. The text above is unchanged.

**A1.1 CEC2017 threshold.** On the CEC2017 experiments (E2c, E3, E3s), the bio-leader or a
schedule is credited overall only if it is credited on **at least 15 of 28 functions** (a
majority). The classical suite stays at the locked rule: **more than 5 of 10**.

**A1.2 Wording.** This study is described as using a "decision rule specified before the
analysis", not as "pre-registered". The term "pre-registered" in the original text above is
kept only as a historical record. The dated facts are:

- file timestamp of this document before this amendment: 2026-10-05 16:28:44 +0530;
- first E1 log written: 2026-10-05 16:53:11 +0530;
- first git commit containing this document: efdf818, 2026-10-14 00:36:40 +0530.

The first commit postdates the first experiments. Only the file timestamp, which is not
tamper-evident, supports the claim that the rules came first.

**A1.3 State of knowledge at the time of this amendment.**

- No E2c result exists (no output file has been written) and no E3s run has started.
- The corrected E3 re-run is in progress and has not been read.
- Results from before the fixes of commit 791cf44 had been read, including the earlier E3
  light-budget ranks, which are superseded.
- The corrected E1 and E2 classical results had already been read, so A1.1 does not
  affect the classical rule, which remains as locked.

**A1.4 Freeze.** Baseline parameters are frozen from the commit that records them in
`docs/BASELINE_CHECK.md`. No algorithm's settings change after that commit.
