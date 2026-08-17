"""Experimental harnesses. NOT production tooling.

Modules under ``tools.experiments`` are offline analytical / comparison
harnesses. They never write to runtime state, never modify production
prompts or briefs, and never change operator-visible behavior.

Each module here MUST be guarded by an explicit ``--experiment``
confirmation flag at the CLI surface so it cannot be invoked by accident
or scripted into a production pipeline.
"""
