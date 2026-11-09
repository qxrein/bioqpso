# BioQPSO — Reproduction Package

Code and results for "Decomposing the Attractor of Quantum-Behaved PSO: What Random Weights,
Bio-Inspired Leaders and Weight Scheduling Each Contribute". Manuscript: `manuscript.tex`;
supplement: `supplementary.tex` (compile it first; `make paper`). `make upload DOI=...` builds the
single submission .tex in `upload/`.

## One-command reproduction

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
sh scripts/fetch_cec2017_official.sh   # official CEC2017 code: downloaded, verified, patched

make all   # tests, experiments, tables, figures, number check, paper
```

The full run takes several days on a 10-core machine, mostly `make e3s` (about 55 hours).
Each run is one JSON file, so an interrupted experiment resumes where it stopped. To rebuild only
the paper from existing results, use `make tables figures check paper`.

## Experiments

| Target | Experiment | Output in `results/` |
|---|---|---|
| `e1` | Classical functions | `e1_classical` |
| `e1b` | Shifted classical functions | `e1b_shifted` |
| `e2` | Controls on classical functions | `e2_controls` |
| `official` | CEC2017 on the official code: E3, E2c, E3s (D=30, 50) | `e3_official_d30`, `e2c_official`, `e3s_official_d30`, `e3s_official_d50` |
| `e2c`, `e3`, `e3s` | The same on opfunu 1.0.4 (supplement only; see `docs/CEC2017_CHECK.md`) | `e2c_controls_cec`, `e3_cec2017_d30`, `e3s_cec2017_d*` |
| `e4` | Scalability | `e4_scalability` |
| `e6` | Sensitivity analysis | `e6_sensitivity` |
| `e7`, `e7b` | Schwefel: boundary handling, optimum location | `e7_schwefel`, `e7b_schwefel_shift` |
| `e8` | Feature selection | `e8_feature_selection` |
| `e9` | Runtime, on an idle machine | `e9_runtime` |

## Official CEC2017 code

The official CEC2017 C code (https://github.com/P-N-Suganthan/CEC2017-BoundContrained) is not
redistributed here. `scripts/fetch_cec2017_official.sh` downloads it at a fixed commit, checks
SHA-256 checksums of the archive and of every file used, and applies three small portability
patches (see `bioqpso/benchmarks/cec17_official/README.md`). It needs `curl`, `bsdtar` or `unar`, and
`python3`; a C++ compiler builds the library on first use.

## Environment

```
Python 3.11.14
numpy    1.26.0
scipy    1.16.3
sklearn  1.8.0
opfunu   1.0.4
pytest   9.1.1
```

See `requirements.txt` for pinned versions. Seeds are fixed in advance in `config/seeds.yaml`
(see `docs/DEVIATIONS.md`).

## Provenance of results

- Every number in the paper comes from `results/` through `scripts/make_tables.py`, which writes
  `paper/generated/`. `scripts/check_numbers.py` checks the prose for hand-typed numbers.
- Results from before the fixes of 2026-11-02 are archived in `results/_superseded/`, and older
  batches in `results_legacy/`. Neither is used in the paper.

## Documentation

| File | Content |
|---|---|
| `docs/DECISION_RULE.md` | Decision rule specified before the analysis, with a dated amendment |
| `docs/DEVIATIONS.md` | Every departure from that rule |
| `docs/BASELINE_CHECK.md` | Frozen settings, sources and checks against the original papers |
| `docs/FINAL_FIXES.md` | Defects fixed in the final pass |
| `docs/CEC2017_CHECK.md` | CEC2017 numbering, the opfunu defects and the official code |
| `docs/SCHWEFEL.md` | Schwefel diagnosis |
| `docs/AUDIT.md`, `docs/REPRO.md` | Earlier code audit and batch discrepancy analysis |
