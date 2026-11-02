# BioQPSO — Unified Reproduction Package

Branch: `merged-unified` | Manuscript: `manuscript.tex`

## One-command reproduction

```bash
# Install dependencies
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run all experiments, analyse, generate figures/tables, compile paper
make all
```

Expected wall-clock time on a 10-core machine:
| Target | Time |
|--------|------|
| `make test` | < 1 min |
| `make e1` | ~15 min (8 workers) |
| `make e2` | ~40 min (8 workers) |
| `make e3` | ~96 min (4 workers) |
| `make e4` | ~156 min (4 workers, D=100) |
| `make e7` | ~6 min (8 workers) |
| `make e8` | ~38 min (6 workers) |
| `make e9` | ~15 min (sequential) |
| `make analyse` | < 1 min |
| `make figures` | < 1 min |
| `make tables` | < 1 min |
| `make paper` | < 1 min |

## Environment

```
Python 3.11.14
numpy    1.26.0
scipy    1.16.3
sklearn  1.8.0
opfunu   1.0.4
joblib   (auto-installed)
pyyaml   (auto-installed)
pytest   9.1.1
```

See `requirements.txt` for pinned versions.

## Seed list

Seeds fixed in advance: `config/seeds.yaml` (see `docs/DEVIATIONS.md`).
Do not change after experiments start.

## Provenance of results

- Legacy results (Batch A, Batch B) are preserved in `results_legacy/` for
  reference. They are NOT used in any table in the manuscript.
- All numbers in `manuscript.tex` come from files under `results/` produced
  by scripts in this repo on branch `merged-unified`.

## Documentation

| File | Content |
|------|---------|
| `docs/AUDIT.md` | Phase 1 code audit |
| `docs/REPRO.md` | Batch discrepancy root-cause analysis |
| `docs/CEC2017_CHECK.md` | CEC2017 validity check (F5 defect) |
| `docs/DECISION_RULE.md` | Decision rule specified before the analysis, with dated amendment |
| `docs/SCHWEFEL.md` | Schwefel diagnosis from E7 |
| `paper/cover_letter_notes.md` | Changes vs preprint, reviewer responses |

## Key findings

1. **E1 (classical)**: GWO rank 3.50; QPSO rank 4.50; AntBioQPSO/FeedbackAda
   rank 5.20. Friedman χ²=43.09, p=1.0×10⁻⁵; Nemenyi CD=5.27 (k=12, N=10).
   No BioQPSO variant separated from QPSO.

2. **E2 (controls)**: No schedule beats best-fixed φ3 on any function (0/10
   credited, decision-rule threshold 5/10). Framing outcome (b): scheduling
   removes tuning burden at comparable accuracy.

3. **E7 (Schwefel)**: Reflect boundary improves QPSO by 58% but AntBio variants
   by <5%. Primary mechanism is fixed-weight attractor structure, not boundary
   handling.

## Git

```bash
git log --oneline -5   # See recent commits on merged-unified
```

Tag `v1.0-merged` will be added once E3/E4/E8/E9 are complete.
