# Cover Letter Notes

## What is new vs. the preprint (Manav2026BioQPSO)

The preprint presented fixed-weight BioQPSO (AntBioQPSO and BeeBioQPSO) with
an initial set of experiments. This submission extends it as follows:

### New content
1. **Unified run** — all 12 algorithms (including GWO and WOA in own NumPy
   implementations) run under a single seed list fixed in advance with a shared
   FE budget of 30,030, replacing two inconsistent earlier batches.
2. **Ablation with a decision rule specified before the analysis (E2)** — full factorial: both leaders ×
   fixed φ3 ∈ {0.1,…,0.5} × all schedules × nobio control × random-φ3 control
   × QPSO, all on 10 classical functions, 30 runs.
3. **Schwefel diagnosis (E7)** — boundary mode × algorithm study that identifies
   the pheromone leader, not the boundary handler, as the primary cause.
4. **Honest negative findings** — the decision rule specified before the analysis was applied
   mechanically: no schedule beats the best fixed φ3 on a majority of functions
   (0/10 credited); the bio-leader is credited on only 4/10 functions. The
   abstract and conclusions are rewritten to reflect outcome (b).
5. **CEC2017 (pending E3)** — 28 functions (F5 excluded with documented reason),
   all 12 algorithms. To be inserted when E3 completes.
6. **Scalability (pending E4)** — D ∈ {30,50,100}, 30 runs. To be inserted.
7. **Reproducibility package** — seeds fixed in advance, git branch
   `merged-unified`, `docs/AUDIT.md`, `docs/REPRO.md`, `docs/CEC2017_CHECK.md`,
   `docs/SCHWEFEL.md`, `docs/DECISION_RULE.md`.

### Changes in response to reviewers
*(Fill in once reviewer comments are available.)*

| Reviewer | Comment | Response | Location in paper |
|---|---|---|---|
| R4 | Does scheduling beat fixed φ3? | E2 under the decision rule; result: no (0/10). Manuscript reframed as outcome (b). | Abstract, §4.2, §4 ablation |
| — | F5 always 500.00 | Documented as opfunu 1.0.4 defect (flat landscape); F5 excluded. | §4.3, docs/CEC2017_CHECK.md |
| — | Batch discrepancy Levy | Root cause: single divergent run, no shared seeds. Fixed. | docs/REPRO.md, §4.1 |
| — | GWO/WOA via mealpy | Reimplemented in own NumPy. | §4.1, bioqpso/algorithms/gwo.py, woa.py |

## Removed or weakened claims

| Claim removed/weakened | Reason | Evidence |
|---|---|---|
| "best on F2, F3, F4, F6, F7, F8" (CEC2017) | E3 not complete; old Run A data replaced | — |
| "clear optimum for W ∈ [15,25]" (Feedback) | Not supported by sensitivity data | §5, Table tab:fb |
| "mild optimum at φ3=0.2" | E2 shows flat landscape; std > mean differences | E2 analysis |
| "φ3 increase consistently worsens Schwefel" | E7 shows boundary mode change helps more | docs/SCHWEFEL.md |
| "×-faster-than-GWO/WOA" | GWO/WOA now in same code path; E9 pending | §4.5 |
| Styblinski-Tang Wilcoxon p < 0.05 | E1 unified run: p not significant after Holm | §4.4 |

## References removed (unverified / not cited in unified manuscript)
*(To be completed by author after checking all bibitem entries.)*

Add Shi & Eberhart 1998, Socha & Dorigo 2008, Demšar 2006, Holm 1979,
Awad et al. 2016 (CEC2017), if not already present. Remove any entry
that cannot be verified against a publisher/DOI page.

## Remaining open items (author action required)
1. Verify every `\bibitem` entry in `references.bib` against publisher/DOI.
2. Add 3–5 additional adaptive-QPSO papers that were actually read.
3. Insert E3/E4/E9 results when experiments complete.
4. Run E8 (nested CV feature selection, high-dim dataset).
5. Tag git release `v1.0-merged`.
6. Confirm no part is under review elsewhere; disclose preprint in cover letter.
