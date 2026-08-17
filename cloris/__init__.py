"""Cloris desktop shell package.

Cloris is the standalone sourcing product for this repo. This package owns the
local app process (FastAPI + pywebview) that wraps the existing sourcing
runtime. See ``docs/cloris-ui-spec.md`` and ``docs/cloris-control-plane-spec.md``
for the product contract, and ``plans/cloris-shell-v0.md`` for the slicing.

Scope at this version: **v0 / Slice 1** — package skeleton, CLI entrypoint, and
app boot path only. There is no worker control, no status aggregation, no
pause/resume, no GitHub launch, and no semantic surfaces. Later slices add
those layers; this slice only proves the shell seam.
"""

__all__ = ["__version__"]

__version__ = "0.0.1"
