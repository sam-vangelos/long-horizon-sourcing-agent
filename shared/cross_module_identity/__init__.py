"""Cross-module identity adapters — Slice B.3 + C.8 + follow-ups.

Each per-pair adapter encodes the resolution semantics for one
cross-module identity bridge. Adapters are pure functions over the
``shared.identity_resolution_service._Candidate`` shape; the resolver
service composes them as additional passes after the existing
``_group_by_handle`` + ``_group_by_name_with_corroboration`` passes.

Adapters today (Multi-Agent Production Plan, Phase B / C):

- :mod:`researcher_to_linkedin` (Slice B.3) — ORCID-anchored
  high-confidence + name+affiliation-anchored medium-confidence.
- :mod:`designer_to_linkedin` (Slice C.8) — portfolio-URL-match
  high-confidence + name+company medium-confidence. Shipped.

Future adapters (per ``docs/cloris-cross-module-identity-resolution-spec.md``):

- ``researcher_to_designer`` — when both modules surface candidates
  for the same brief, dedup by ORCID / portfolio-URL.
- ``exec_search_to_linkedin`` — exec_search reuses the LinkedIn
  full-eval pipeline, so its candidates already carry LinkedIn
  identity; this adapter is mostly a no-op until an exec_search
  candidate originates outside LinkedIn (off-source signal feed).
"""
