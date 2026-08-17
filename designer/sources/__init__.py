"""Designer module per-source adapters.

Each source has its own adapter file (Behance v2 in :mod:`designer.sources.behance`,
Google CSE in :mod:`designer.sources.google_cse`, Dribbble v2 in
:mod:`designer.sources.dribbble`). The shape mirrors
:mod:`researcher.sources` (per the integration contract §1) so a
multi-source designer brief can fan out work-units across sources
without contortion.

Slice 2 ships :mod:`designer.sources.behance`. Slice 3 adds
:mod:`designer.sources.google_cse`. Slice 10 (optional) adds
:mod:`designer.sources.dribbble`.
"""
