"""Researcher module — discovers and evaluates ML researchers.

See `plans/researcher-module-spec.md` for the full spec. Slice 1 ships
the foundation only (schema, launcher entry, stub orchestrator). The
real pipeline arrives in Slices 2-6:

- Slice 2: source clients (`researcher/sources/`)
- Slice 3: brief → query generator (`researcher/strategy.py`)
- Slice 4: acquisition + identity disambiguation
- Slice 5: evaluation pipeline + layered floor gates
- Slice 6: orchestrator + runtime state bridge
"""
