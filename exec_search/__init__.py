"""Executive Search module — dossier-depth evaluation for high-touch low-volume executive recruiting.

See `plans/executive-search-module-spec.md` for the full spec
(or `~/.cursor/plans/executive-search-module-spec_8a3af78b.plan.md`
during plan-iteration). Slice 1 ships the foundation only (schema,
launcher entry, confidentiality scaffolding, stub orchestrator). The
real pipeline arrives in Slices 2-10:

- Slice 2: dossier-depth evaluation pipeline (extends LinkedIn full-eval branch)
- Slice 3: off-LinkedIn signal interface + Perplexity baseline
  (`exec_search/signals/`)
- Slice 4: News API integration
- Slice 5: Crunchbase + PitchBook + per-search dossier-spend
  circuit-breaker (`exec_search/budget.py`)
- Slice 6: confidentiality enforcement (wires `shared/confidentiality.py`)
- Slice 7: Cloris-native shortlist destination + dossier rendering
- Slice 8: brief polish + reflection exec-register
- Slice 9: pre-launch investigation (new
  `market_intelligence/pre_launch.py`)
- Slice 10: prior-search exclusion + end-to-end demo
"""
