# TODO_INVENTORY.md — Phase 1 Inventory of manuscript.tex

Generated: 2026-10-07 (branch merged-unified)  
Source file: manuscript.tex (718 lines; this is manuscript_v2.tex per the phase plan)

---

## Part A — `\todo{}` items (15 total)

| # | Line | Text (abbreviated) | Status | Resolving file / action |
|---|------|-------------------|--------|------------------------|
| T1 | 147 | Add 3–5 adaptive/neighbourhood-QPSO papers; verify DOIs | **needs author** | Author provides papers; verified DOIs go into references.bib |
| T2 | 162 | Add at least one CEC-competitive DE variant (L-SHADE) | **needs experiment** | Implement L-SHADE in bioqpso/algorithms/; run E1+E3; re-run scripts/make_tables.py |
| T3 | 367 | Replace paragraph numbers by macros; re-check per-function statements | **needs verification** | numbers.tex exists; paragraph must use macros and be checked vs results/analysis/e1_classical/stats.csv |
| T4 | 376 | Add shifted/rotated classical functions to quantify GWO origin bias | **needs experiment** | New sub-experiment E1b; new config; update classical_mean.tex and observations paragraph |
| T5 | 407 | Confirm opfunu function ids and use those names in the text | **needs verification** | docs/CEC2017_CHECK.md has this; copy names into CEC2017 section |
| T6 | 452 | Optionally re-run fixed φ3=0 with clip removed | **needs experiment** | Fix FixedSchedule(0.0) clip bug; rerun only that cell in E2; update controls_fixed_sweep.tex |
| T7 | 475 | Verify which 4 functions are bio-leader-credited; resolve Levy contradiction | **needs verification** | results/analysis/e2_controls/ — compute pairwise (fixed-p03 vs nobio) per function |
| T8 | 496 | Replace Schwefel prose numbers by generated table from E7; add shifted-optimum run | **needs experiment + table** | Generate e7_schwefel.tex from results/e7_schwefel/; run shifted-optimum variant |
| T9 | 505 | Complete remaining D=100 E4 jobs (~950); regenerate scalability figure + table | **needs experiment** | E4 is actually complete (3780/3780 per metadata.json); regenerate scalability.pdf and add mean±std table |
| T10 | 535 | Run E8 and generate fs.tex | **partially done** | results/e8_feature_selection/e8_results.json EXISTS; fs.tex EXISTS but has comment header — fix and verify content matches E8 spec (inner 5-fold, not 3-fold) |
| T11 | 553 | Profile PerParticleAda; report per-1000-FE with IQR | **needs experiment** | Update E9 script; check if +88% is an implementation bug |
| T12 | 563 | Generate convergence multi-panel figure from unified results (classical + CEC2017) | **needs figure** | scripts/make_figures.py already generates convergence_classical.pdf — check; add CEC2017 panel |
| T13 | 571 | Re-run sensitivity as E6 with unified framework (30 runs, full budget, Schwefel) | **needs experiment** | scripts/run_e6_sensitivity.py (not yet written) |
| T14 | 682 | Confirm not under review; disclose preprint | **needs author** | Author action |
| T15 | 702 | Add CEC2017 standard-deviation table | **needs table** | Generate from results/e3_cec2017_d30 with make_tables.py |

---

## Part B — `\tbr` usages

None — no `\tbr` used in the document body (definition only at line 29).

---

## Part C — `\geninput{}` files (10 required)

| File | Exists? | Starts with `\begin{tabular}`? | Issues |
|------|---------|-------------------------------|--------|
| classical_mean.tex | ✅ | ✅ | OK |
| wilcoxon.tex | ✅ | ✅ | OK |
| cec2017_ranks.tex | ✅ | needs check | check below |
| cec2017_mean_a.tex | ✅ | needs check | check below |
| cec2017_mean_b.tex | ✅ | needs check | check below |
| controls_fixed_sweep.tex | ❌ **MISSING** | — | Must generate from results/e2_controls/ |
| fs.tex | ✅ but defective | ❌ starts with `%` comments | Fix: strip comments or regenerate |
| runtime.tex | ✅ | ✅ | OK |
| classical_std.tex | ✅ | needs check | check below |
| numbers.tex | ✅ (loaded via \InputIfFileExists) | N/A (macros only) | OK; needs CEC2017 and E8 macros added |

---

## Part D — `\includegraphics` files (3 required, must be .pdf)

| File | Exists as PDF? | Exists as PNG? | Status |
|------|---------------|----------------|--------|
| phi3_heatmap.pdf | ✅ | ✅ | OK |
| scalability.pdf | ✅ | ✅ | OK — but E4 was complete; update caption (remove "partial") |
| convergence_classical.pdf | ✅ | ✅ | OK |

---

## Part E — `\cite` keys (cross-reference with references.bib)

### In manuscript but MISSING from references.bib (10 keys → must be added):
- `Awad2017` (CEC2017 technical report; required by the plan)
- `Harris2020` (NumPy; required)
- `Holm1979` (Holm correction; required)
- `Hussein2025` — note: bib has `Hussein2025SRA`; key mismatch
- `Ibrahim2026` — note: bib has `Ibrahim2026SOA`; key mismatch
- `Kennedy1995` (PSO original; required)
- `Pedregosa2011` (scikit-learn; required)
- `Shi1998` (PSO inertia weight; required)
- `Socha2008` (ACO_R; required)
- `Virtanen2020` (SciPy; required)

### In references.bib but UNCITED (27 entries — review/clean):
Alzubi2022ClusterComp, Alzubi2023MNA, Alzubi2024NEM, Alzubi2025AEJ, Das2009,
Dehghani2024Bighorn, Dehghani2024CapeLynx, Dehghani2024Stoat, Eberhart1995,
Elgamal2020LSGWO, Elmasry2024QISSA, Hussein2025SRA (key mismatch), Ibrahim2026SOA
(key mismatch), Li2014, Lu2008, Manav2025, Movassagh2023JAIHC, Patel2025,
Qaraad2021ENSVM, Reviewer2026BreastMiRNA, Reviewer2026NSCLC, Reviewer2026Sjogren,
Reviewer2026Wilms, Sun2004QPSO, Wolpert1997, Zhang2023, wiley2023improvedqpso

**Action**: Add the 10 missing keys. Fix the 2 key mismatches (Hussein2025SRA→Hussein2025, Ibrahim2026SOA→Ibrahim2026). Remove uncited entries after author confirms. Phase 4 will verify each entry against DOI.

---

## Part F — Prose sentences with numbers (to be replaced by macros)

| Line | Sentence fragment | Number(s) | Source file |
|------|-------------------|-----------|-------------|
| 73–76 (abstract) | "GWO rank 3.50 … QPSO 4.50 … rank 5.20 … χ²=43.09 … p=1.0×10⁻⁵ … CD=5.27" | multiple | results/analysis/e1_classical/summary.json |
| 77–78 | "3.11–3.79 … QPSO 4.14 … CD=3.15" | CEC2017 ranks | results/analysis/e3_cec2017_d30/summary.json |
| 362–366 | "GWO rank 3.50, QPSO 4.50, BioQPSO ranks 5.20, 7.40, 8.40" | E1 ranks | results/analysis/e1_classical/ranks.csv |
| 362 | "QPSO Levy mean 3.3×10⁻²" | E1 stats | results/analysis/e1_classical/stats.csv |
| 403–408 | "χ²=218.71, p=9.6×10⁻⁴¹, CD=3.15, k=12, N=28, ranks 3.11/3.79/4.14/9.96" | E3 | results/analysis/e3_cec2017_d30/summary.json |
| 452 | "0/10 functions" | E2 decision rule | results/analysis/e2_controls/ |
| 470–471 | "4 of 10 functions" (bio-leader) | E2 | results/analysis/e2_controls/ |
| 490–492 | Schwefel numbers: 4277→1797, 8587→6062, 8050→7669 | E7 | results/e7_schwefel/ |
| 545 | "28\%" (AntBioQPSO overhead) | E9 | results/e9_runtime/runtime_results.json |
| 548 | "+88\%", "+121\%", "+11\%" | E9 | results/e9_runtime/runtime_results.json |

All these numbers must be replaced by macros from numbers.tex, which must in turn be generated from the result files.

---

## Part G — Summary by phase-3 category

### A. Needs experiments
- T2: L-SHADE baseline
- T4: E1b shifted classical functions
- T6: E2 φ3=0 clip fix and rerun
- T8: E7 shifted-optimum Schwefel
- T9: E4 — actually complete; just needs regeneration
- T10: E8 — done with 3-fold inner; spec says 5-fold; decide which to keep
- T11: E9 PerParticleAda profiling
- T12: convergence figure (CEC2017 panel missing)
- T13: E6 sensitivity full-budget

### B. Needs verification
- T3: macro replacement + per-function checks
- T5: opfunu id confirmation
- T7: bio-leader credits (Levy contradiction)
- T15: CEC2017 std table

### C. Needs author
- T1: references (3–5 papers)
- T14: preprint disclosure

### Critical blockers for compilation
1. `controls_fixed_sweep.tex` **missing** — manuscript will show "missing generated table" box
2. `fs.tex` has comment header — `\geninput` will print comment text before the table
3. 10 cite keys missing from bib — LaTeX will warn and produce `?` references
4. 2 bib key mismatches (Hussein2025SRA, Ibrahim2026SOA)

---

## Part H — E4 status clarification

`results/metadata.json` says "experiments_complete: E4" and 3780 files exist.  
The manuscript at line 505 still says "incomplete" and "partial" in the caption.  
**Action**: verify file count, then update the caption and remove the \todo.
