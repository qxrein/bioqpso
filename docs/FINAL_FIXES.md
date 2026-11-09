# FINAL_FIXES.md: defects addressed in the final pass (started 2026-11-02)

Status: **done** unless marked. Items marked *waiting* need results from a running experiment.

## Code defects found on the way

| Defect | Fix | Consequence |
|---|---|---|
| No-bio weights summed to 0.9, because a clip acted on phi3 in `redistribute_phi` | `_clip_unit` on [0,1] in `bioqpso/schedules.py`; regression tests | All earlier no-bio results are invalid and archived in `results/_superseded/`. Earlier claims about the bio-leader are withdrawn |
| GWO, WOA, ACO_R, ABC, PSO deviations | See `docs/BASELINE_CHECK.md` | Baselines re-run; settings frozen by Amendment 1 |

## Phase 1 list

| Item | Defect | How it was fixed | Status |
|---|---|---|---|
| 1.1 | E1 winners paragraph typed by hand | Generated: `\EoneWinFns*` macros | done |
| 1.2 | Bio-leader credit unclear | Per-leader test, `generated/bioleader_credit.tex`; five functions reported explicitly; decomposition stated, with `generated/decomposition.tex` added | done |
| 1.3 | Bolding inconsistent | Best per row at printed precision, with ties, in `value_table` | done |
| 1.4 | Number formats mixed | 3 s.f., scientific if \|x\| < 1e-2 or >= 1e4, true minus, one `\tabsetup` (scriptsize, tabcolsep 3pt). Tables of 13 algorithms are two panels instead of being scaled | done |
| 1.4b | Feature-selection bolding | Bold over all rows, baselines included; stated in the caption | done |
| 1.5 | CEC order and layout | Numeric order, two halves of 14, also for std | done |
| 1.6 | No effect sizes | `effect_sizes.tex`, `effect_sizes_shifted.tex`: rank-biserial r with bootstrap 95% CI | done |
| 1.7 | No medians | `classical_median.tex`, `classical_iqr.tex` in the appendix; mentioned in Section 4.2 | done |
| 1.8 | Heat map | Per-function ranks, no-bio column, both leaders | done |
| 1.9 | Scalability figure | Separate "= 0" band, `generated/scalability.tex`, paragraph in Section 4.7, 3,780 jobs confirmed | done |
| 1.10 | Convergence figure | Medians, 8 algorithms, shared legend, text in Section 4.10 | done |
| 1.11 | Feature selection | Real data (OpenML), SVM columns, 12 algorithms, results paragraph states where all features and MI top-k win | done |
| 1.12 | Runtime | E9 re-run on the fixed code with interleaved order (`docs/DEVIATIONS.md` item 10); GWO profile: gap is implementation only | done |
| 1.13 | Stale items | Author actions list, "earlier draft" text and Levy TODO removed | done |
| 1.14 | Floats | `[!htbp]` and `\FloatBarrier` | done |
| 1.15 | Baseline sanity | `docs/BASELINE_CHECK.md`; L-SHADE checked on E3s against published SCSS-L-SHADE values (Section 4.5) | done |

## Also corrected

- Table 1: Levy formula and the Rosenbrock and Ackley bounds now match the code.
- CEC2017 numbering in the paper and `docs/CEC2017_CHECK.md`: opfunu F_k is official F_{k+1} for k >= 2.
- "Pre-registered" removed; replaced by the wording and dates of Amendment 1.
- `Makefile`, `run_all.sh` and `README.md` referred to removed scripts and old results.
- `docs/SCHWEFEL.md` drew conclusions from the faulty control; rewritten from E7 and E7b.
- `references.bib`: uncited entries removed; the preprint's title and DOI corrected against CrossRef.
- E6: `Delta` is structurally inert (argmax on pheromone is scale-invariant); Friedman returned NaN and is now treated as p = 1, and figure ranks no longer split identical results.

## Added 2026-11-06

- opfunu checked against the official CEC2017 C code (`scripts/check_cec2017_official.py`,
  `docs/CEC2017_CHECK.md` section 7): identical on F1 only. Paper, abstract, threats and cover
  letter updated; D=50 analysis reported on all 28 and on the four official-formula functions.
- The D=50 ant-variant result is called weak evidence everywhere; it is not in the highlights.
- Main text 25 pages; supplement (`supplementary.tex`, same template) 29 pages.
- Declarations: not under review elsewhere.
- `scripts/build_upload.py` writes the single upload .tex.
- No file names, script names or repository paths in the paper or supplement prose; the upload build strips source comments.

## Added 2026-11-09: official CEC2017 code

- E3, E2c, E3s (D=30, 50) and the CEC part of E6 re-run on the official CEC2017 C code
  (Amendment 2; `scripts/run_official_cec.py`, `scripts/run_e6_official.py`). The paper's CEC2017
  sections, abstract, discussion, conclusion and cover letter now come from these runs only; the
  opfunu runs are in Supplementary Section S3 as a documented discrepancy.
- L-SHADE checked against jSO and LSHADE-SPACMA competition results (`docs/BASELINE_CHECK.md`,
  Supplementary Table S-lshade); SCSS-L-SHADE comparison and its reference removed.
- `Awad2017` corrected and verified against the official technical report.
