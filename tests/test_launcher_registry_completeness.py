"""Tests for the multi-agent-execution Phase 1 Slice 1.0 pioneer PR.

Pins the ``LauncherEntry`` shape declared upfront so that follow-on
slices 1.1–1.8 can populate fields per source without re-extending
the dataclass. Each populating slice MUST be able to find its field
already declared (with the right type-default) on every registered
source — otherwise the parallel execution payoff of slicing 1.0 out
as a single-window pioneer disappears.

What this test pins:

- Every registered source in ``LAUNCHERS`` has every required field
  declared on its ``LauncherEntry``. Defaults are documented per-field
  on the dataclass; the per-slice ratchets below pin which sources
  have populated their slice's fields with concrete values.
- The four pre-1.0 fields (``state_key_fn`` / ``state_dir_fn`` /
  ``orchestrator_argv_fn`` / ``save_destination_blocker_fn``) remain
  populated on every entry — the dataclass extension is purely
  additive, no existing field was renamed or dropped.
- Slice 1.0 itself populates none of the new fields. Each populating
  slice (1.1+) flips one of them; the per-slice ratchet here moves
  in lockstep so the test catches "I forgot to populate this source"
  regressions.

Per-slice population ratchet status:

- Slice 1.1 (``readiness_probe_fn``): populated on ``linkedin`` +
  ``github``; researcher / designer / exec_search land in Phase 2.2.
- Slice 1.2 (``in_process_dispatch_fn``): populated on every source.
- Slice 1.5 (``progress_kind``): populated on linkedin / github /
  researcher / designer with the canonical work-unit-kind constants;
  ``exec_search`` keeps the empty-string sentinel (no
  work-unit-aggregation channel today).
- Slice 1.6 (``form_strategy_fn``): populated on linkedin / github /
  researcher / designer with the per-module ``form_strategy_for_registry``
  adapter. ``exec_search`` keeps ``None`` until its strategy formation
  ships (out of scope for the multi-agent-execution Phase 1 cleanup).
- Slice 1.7 (``summarize_run_fn``): populated on every source via the
  per-source ``summarize_<source>_run`` helpers in
  ``shared/runtime_state/read_models.py``. The chief-of-staff agent
  (Phase 2.4 / 2.5) reads every source's latest-run snapshot uniformly
  through this field — unlike readiness (1.1) or strategy (1.6), there
  is no "source that simply doesn't have this yet" state, because every
  source writes runtime state into ``runtime_state.sqlite3`` the same
  way; the registry just exposes a read-only view of that for
  cross-source synthesis.
- Slice 1.4 (snapshot helpers): closes the five LinkedIn-special-case
  branches at ``market_intelligence/run_snapshots.py:287, 315, 409,
  461, 535``. Four callable fields and one bool flag populate
  per-source. Population matches today's call sites
  (``finalize_run_snapshot`` is invoked only with
  ``source ∈ {"linkedin", "github"}`` from
  ``linkedin/run_report.py`` and ``github/orchestrator.py``):
    - ``brief_id_for_snapshot_fn`` — linkedin only (the
      ``linkedin_project_id`` fallback chain is LinkedIn-specific;
      every other source falls through to the generic
      ``brief.id || role_title || stem`` shape supplied by the
      legacy else-branch).
    - ``progress_projection_fn`` — linkedin + github (the two
      sources that have per-source progress / stage / search-memory
      projection writers in
      ``shared/runtime_state/projections.py``).
    - ``snapshot_research_packet_fn`` — linkedin only (parallel to
      Slice 1.3's reflection-time builder, on a distinct registry
      slot so the two seams can diverge).
    - ``snapshot_state_key_fn`` — linkedin + github (the two
      sources whose snapshot output_dir namespace lives under
      ``output/runs/<source>/<state_key>/``).
    - ``reconstruct_report_analysis`` (bool) — ``True`` on linkedin
      only; the runtime ``not (run_dir / "run-report.json").exists()``
      check stays in the snapshot module.

Companion to :mod:`tests.test_cloris_launchers` (which exercises the
4-callable spawn contract end-to-end). This file exclusively pins
the field-shape invariants the pioneer slice introduces.
"""

from __future__ import annotations

import dataclasses

from cloris.launchers import LAUNCHERS, LauncherEntry, known_sources
from shared.runtime_state.store import (
    DESIGNER_BEHANCE_QUERY_KIND,
    EXEC_SEARCH_QUERY_KIND,
    GITHUB_QUERY_KIND,
    LINKEDIN_STRING_KIND,
    RESEARCHER_AUTHOR_QUERY_KIND,
)

# ---------------------------------------------------------------------------
# Field-shape invariants. The two lists below are the load-bearing
# contract Slice 1.0 declares. If a downstream slice adds a field, it
# updates these lists in the same PR — the test becomes the rolling
# ratchet that catches "I forgot to extend the dataclass" regressions.
# ---------------------------------------------------------------------------

_PRE_PIONEER_FIELDS: tuple[str, ...] = (
    "state_key_fn",
    "state_dir_fn",
    "orchestrator_argv_fn",
    "save_destination_blocker_fn",
)

# Pioneer callable fields that have NOT yet been populated by their
# follow-on slice. As 1.1–1.8 land, fields move from this tuple either
# to ``_POPULATED_PIONEER_CALLABLE_FIELDS`` (every source supplies a
# callable) or ``_PARTIALLY_POPULATED_PIONEER_CALLABLE_FIELDS`` (only
# some sources do; the rest stay ``None`` and the dispatch site keeps
# its None-fallback). Either way the test becomes the rolling ratchet
# that catches "I forgot to populate the right sources".
#
# Empty today: P10 deleted the unpopulated ``judge_candidate_fn`` slot
# (Slice 1.8, deferred) as dead theater rather than let it sit forever
# — a real broker arc adds its own slot when the demand materializes.
# Kept as a structural slot (mirrors ``_PIONEER_SCALAR_FIELDS`` below)
# so a future pioneer callable field reuses the same ratchet shape.
_UNPOPULATED_PIONEER_CALLABLE_FIELDS: tuple[str, ...] = ()

# Pioneer callable fields that HAVE been populated on every registered
# source. Every source must supply a non-None callable for each of
# these; any None would be a regression (the dispatch site removed its
# legacy if/elif ladder, so a None here would crash at runtime instead
# of falling through).
#
# Slice 1.2 populated ``in_process_dispatch_fn`` on all five sources
# (and closed the silent regression where ``exec_search`` was missing
# from the legacy ladder).
#
# Slice 1.7 populated ``summarize_run_fn`` on all five sources via
# per-source ``summarize_<source>_run`` helpers in
# ``shared/runtime_state/read_models.py``. Unlike the partial-
# population fields (research-packet / form-strategy), every source
# has runtime-state SQLite to summarize, so the field is fully
# populated from day one. Consumer callsites land in Phase 2.4
# (synthesis maturity) and Phase 2.5 (dispatch heuristic).
#
# Phase 2.2 (multi-agent-execution) populated ``readiness_probe_fn``
# on the remaining three sources (researcher / designer / exec_search)
# via per-module ``probe_<source>_readiness`` functions. Every
# registered source now supplies a non-None callable; the partial-
# population entry for ``readiness_probe_fn`` was retired in the same
# PR. The dispatch site at ``cloris/api.py:_readiness_blockers``
# preserves the ``probe_fn is None`` short-circuit defensively for
# the unknown-source case (``LAUNCHERS.get(source)`` returns ``None``)
# but no registered source exercises it any more.
_POPULATED_PIONEER_CALLABLE_FIELDS: tuple[str, ...] = (
    "in_process_dispatch_fn",
    "summarize_run_fn",
    "readiness_probe_fn",
)

# Pioneer callable fields that have been populated on a SUBSET of
# sources. Maps field name → tuple of sources expected to register a
# non-None callable. Sources NOT listed here keep the dataclass default
# (``None``); the dispatch site preserves the None-fallback so the
# legacy "no probe ⇒ no blockers" semantics survive.
#
# Slice 1.1 populated ``readiness_probe_fn`` on ``linkedin`` and
# ``github`` only. Researcher / designer / exec_search land in Phase
# 2.2 ("per-module readiness probe registration"); until then the
# aggregator at ``cloris/api.py:_readiness_blockers`` falls through to
# ``report = None`` for those sources.
#
# Slice 1.3 populated ``research_packet_builder_fn`` on ``linkedin``
# and ``github`` only — those are the two sources with
# reflection-time research packets today. Researcher / designer /
# exec_search keep ``None``; the dispatch site at
# ``market_intelligence/engine.py:_load_evidence_batch`` falls
# through unchanged for them. OSS Maintainers Slice 9 (Phase 3
# cleanup per spec §16) is the post-trial work that unifies the two
# underlying builders into a shared abstraction; out of scope here.
#
# Slice 1.6 populated ``form_strategy_fn`` on the four sources whose
# strategy formation has shipped (linkedin / github / researcher /
# designer) via per-module ``form_strategy_for_registry`` adapters.
# ``exec_search`` keeps ``None``; its strategy stage is not on the
# Phase 1 cleanup path. The chief-of-staff dispatch backend
# (Phase 2.5) will dispatch via
# ``LAUNCHERS[source].form_strategy_fn(brief, prior_run_data)`` for
# the four populated modules and skip exec_search until its
# orchestrator-level strategy stage lands.
#
# Slice 1.4 populated the four snapshot-helper callables on the
# sources whose ``finalize_run_snapshot`` call paths exist today
# (linkedin and github — see ``linkedin/run_report.py:236`` and
# ``github/orchestrator.py:1370``):
#
# - ``brief_id_for_snapshot_fn`` — linkedin only. The LinkedIn
#   ``linkedin_project_id``-or-id-or-stem fallback chain doesn't
#   apply to other sources; every other source falls through to the
#   generic ``brief.id || role_title || stem`` shape that the legacy
#   else-branch at ``run_snapshots.py:289`` supplied.
# - ``progress_projection_fn`` — linkedin + github. LinkedIn writes
#   3 projections (progress / stage / search-memory); GitHub writes 2
#   (progress / stage). Researcher / designer / exec_search keep
#   ``None`` and the dispatch site no-ops for them.
# - ``snapshot_research_packet_fn`` — linkedin only. Parallel to
#   Slice 1.3's reflection-time ``research_packet_builder_fn``;
#   distinct slot so the two seams can diverge.
# - ``snapshot_state_key_fn`` — linkedin + github. LinkedIn wraps
#   ``shared.output_paths.derive_brief_id`` (with its
#   ``source_config.linkedin.project_id`` fallback chain); GitHub
#   wraps ``shared.output_paths.github_state_key``. The dispatch
#   site at ``run_snapshots.py:_brief_namespace_key`` keeps
#   ``github_state_key`` as a None-fallback to preserve pre-slice
#   behavior for unrecognized sources.
_PARTIALLY_POPULATED_PIONEER_CALLABLE_FIELDS: dict[str, tuple[str, ...]] = {
    # A.6 (Multi-Agent Production Plan) widened research_packet_builder_fn
    # registration to every source (shims today; F.2b replaces shim
    # bodies with per-module packet content). Field is now uniformly
    # populated, but kept in the partial-population ratchet so the
    # source-set assertion stays explicit; if a future module spec
    # adds a new source, the ratchet catches missing registration.
    "research_packet_builder_fn": (
        "linkedin", "github", "researcher", "designer", "exec_search",
    ),
    "form_strategy_fn": ("linkedin", "github", "researcher", "designer", "exec_search"),
    "brief_id_for_snapshot_fn": ("linkedin",),
    "progress_projection_fn": ("linkedin", "github"),
    "snapshot_research_packet_fn": ("linkedin",),
    # A.7 widened snapshot_state_key_fn registration to every source
    # (each module wraps its own ``shared.output_paths`` state-key
    # helper); closes the dormant github_state_key fall-through at
    # ``market_intelligence/run_snapshots.py:_brief_namespace_key``.
    "snapshot_state_key_fn": (
        "linkedin", "github", "researcher", "designer", "exec_search",
    ),
}

_PIONEER_CALLABLE_FIELDS: tuple[str, ...] = (
    _UNPOPULATED_PIONEER_CALLABLE_FIELDS
    + _POPULATED_PIONEER_CALLABLE_FIELDS
    + tuple(_PARTIALLY_POPULATED_PIONEER_CALLABLE_FIELDS.keys())
)

# Scalar fields that still default to their typed-zero on every
# source. ``progress_kind`` (Slice 1.5) and
# ``reconstruct_report_analysis`` (Slice 1.4) have both lifted out of
# this tuple now that their populating slices have shipped. Their
# per-source expected-value ratchets live at
# :func:`test_progress_kind_registered_per_source_post_slice_1_5` and
# :func:`test_reconstruct_report_analysis_registered_per_source_post_slice_1_4`
# respectively. The tuple is currently empty — kept as a structural
# slot so a future scalar field added to ``LauncherEntry`` re-uses
# the same ratchet shape rather than inventing a new pattern.
_PIONEER_SCALAR_FIELDS: tuple[tuple[str, object], ...] = ()

# Slice 1.5 (multi-agent-execution Phase 1) ratchet: per-source
# ``progress_kind`` registration. Mirrors the canonical work-unit-kind
# constants in ``shared/runtime_state/store.py:37-41``. ``exec_search``
# has no work-unit-aggregation channel today and keeps the empty-string
# sentinel — the dispatch sites at ``cloris/control_plane.py`` guard
# with ``if progress_kind:`` so an empty string skips the
# ``read_models.work_unit_progress`` read for its state dirs.
_PROGRESS_KIND_BY_SOURCE: dict[str, str] = {
    "linkedin": LINKEDIN_STRING_KIND,
    "github": GITHUB_QUERY_KIND,
    "researcher": RESEARCHER_AUTHOR_QUERY_KIND,
    "designer": DESIGNER_BEHANCE_QUERY_KIND,
    # A.8 (Multi-Agent Production Plan): exec_search now declares its
    # own work-unit kind so the control-plane progress aggregation
    # can distinguish dossier work from LinkedIn's plain-eval work
    # even though both run inside the LinkedIn orchestrator process.
    # Per Phase D.1 the orchestrator emits work_units with this kind
    # for each candidate's dossier-eval cycle.
    "exec_search": EXEC_SEARCH_QUERY_KIND,
}

# Slice 1.4 (multi-agent-execution Phase 1) ratchet: per-source
# ``reconstruct_report_analysis`` flag. Closes the legacy
# ``source == "linkedin"`` predicate at the prior
# ``run_snapshots.py:535`` call-site by lifting the boolean onto the
# launcher entry. Only LinkedIn opts in to the
# reconstruct-on-missing path today (reflection runs trigger it when
# ``run-report.json`` is absent); the disk-presence check stays in
# the snapshot module since it's a runtime check, not a per-source
# contract.
_RECONSTRUCT_REPORT_ANALYSIS_BY_SOURCE: dict[str, bool] = {
    "linkedin": True,
    "github": False,
    "researcher": False,
    "designer": False,
    "exec_search": False,
}


def test_launcher_entry_declares_every_pioneer_field() -> None:
    """The dataclass declares every Slice 1.0 field, names exact-match.

    Catches the regression where a populating slice tries to set a
    field that was never declared (e.g., a typo in the field name on
    either side).
    """

    declared = {f.name for f in dataclasses.fields(LauncherEntry)}
    expected = (
        set(_PRE_PIONEER_FIELDS)
        | set(_PIONEER_CALLABLE_FIELDS)
        | {name for name, _ in _PIONEER_SCALAR_FIELDS}
    )
    missing = expected - declared
    assert not missing, (
        f"LauncherEntry is missing pioneer-slice fields: {sorted(missing)}. "
        "Declare them in cloris/launchers/__init__.py before populating."
    )


def test_pre_pioneer_fields_remain_on_every_entry() -> None:
    """Slice 1.0 is purely additive — no pre-1.0 field renamed or removed.

    The four-callable spawn contract pre-dates Slice 1.0 and every
    callsite (``cloris.worker``, ``cloris.api`` launch endpoints, the
    F2 readiness aggregator) reads them by name. Renaming would break
    the spawn path silently.
    """

    for source, entry in LAUNCHERS.items():
        for field in _PRE_PIONEER_FIELDS:
            assert hasattr(entry, field), (
                f"{source!r} entry missing pre-pioneer field {field!r}"
            )
            value = getattr(entry, field)
            assert callable(value), (
                f"{source!r}.{field} should be callable, got {type(value).__name__}"
            )


def test_every_registered_source_carries_every_pioneer_callable_field() -> None:
    """Every source has every Slice 1.0 callable field declared.

    Pioneer slice contract: declarations land here, populations land in
    1.1–1.8. ``None`` is the legitimate not-yet-populated value; what
    we forbid is "the attribute doesn't exist" (which would make
    ``LAUNCHERS[source].readiness_probe_fn`` raise ``AttributeError``
    on the dispatch site instead of returning ``None``).
    """

    for source in known_sources():
        entry = LAUNCHERS[source]
        for field in _PIONEER_CALLABLE_FIELDS:
            assert hasattr(entry, field), (
                f"{source!r} entry missing pioneer field {field!r}; "
                "Slice 1.0 should have declared it on the dataclass."
            )


def test_unpopulated_pioneer_callable_fields_default_to_none() -> None:
    """Pioneer callable fields not yet populated by their slice stay ``None``.

    Slices 1.1–1.8 populate one field per slice, in sequence. Until a
    slice ships, its field is the legitimate not-yet-populated ``None``
    — the dispatch site preserves the legacy if/elif behavior by
    checking ``if (fn := LAUNCHERS[source].field) is not None`` and
    falling through when ``None``.

    The populated-vs-unpopulated split lives in the two module-level
    tuples above; each populating slice moves its field from
    ``_UNPOPULATED_PIONEER_CALLABLE_FIELDS`` to
    ``_POPULATED_PIONEER_CALLABLE_FIELDS`` in the same PR.
    """

    for source in known_sources():
        entry = LAUNCHERS[source]
        for field in _UNPOPULATED_PIONEER_CALLABLE_FIELDS:
            value = getattr(entry, field)
            assert value is None, (
                f"{source!r}.{field} should still be None; got {value!r}. "
                "Was this populated outside the tracked sequence (1.1–1.8), "
                f"or is the populating slice's PR missing the move from "
                "_UNPOPULATED_PIONEER_CALLABLE_FIELDS to "
                "_POPULATED_PIONEER_CALLABLE_FIELDS in this test file?"
            )


def test_populated_pioneer_callable_fields_supply_a_callable_on_every_source() -> None:
    """Every populated pioneer field is non-None and callable, on every source.

    Once a slice populates a field for ALL sources, the dispatch site
    removes its legacy if/elif fallback — so a ``None`` value would
    crash at runtime instead of falling through. This test catches the
    regression where a new source (e.g., a sixth module) lands without
    populating already-populated fields.

    Slice 1.2: ``in_process_dispatch_fn`` lands on all five sources
    (linkedin, github, researcher, designer, exec_search). exec_search
    in particular closes the silent regression at the legacy frozen-app
    ladder which only supported the first four.

    Slice 1.7: ``summarize_run_fn`` lands on all five sources. Every
    source writes canonical runtime state to its per-state-dir
    ``runtime_state.sqlite3``, so the per-source helper exists
    uniformly; the chief-of-staff agent (Phase 2.4 / 2.5) reads across
    sources without branching on "does this module even have a
    summary?". A ``None`` here would mean a new source landed without
    registering its read helper and dispatch would KeyError at the
    first cross-source read.
    """

    for source in known_sources():
        entry = LAUNCHERS[source]
        for field in _POPULATED_PIONEER_CALLABLE_FIELDS:
            value = getattr(entry, field)
            assert value is not None, (
                f"{source!r}.{field} must be populated; got None. "
                "The dispatch site no longer falls through to a legacy "
                "branch — every registered source must supply this callable."
            )
            assert callable(value), (
                f"{source!r}.{field} must be callable; "
                f"got {type(value).__name__}."
            )


def test_partially_populated_pioneer_callable_fields_match_their_source_set() -> None:
    """Partially-populated fields are callable on the listed sources, None elsewhere.

    Some pioneer fields populate only a subset of sources by design.
    The dispatch site keeps its None-fallback so non-listed sources
    preserve the legacy behavior (e.g., readiness aggregator returns
    no blockers when the source has no probe). Two regressions to
    catch:

    1. A listed source forgot to register the callable (would silently
       lose source-level readiness instead of reporting blockers).
    2. A non-listed source registered something anyway (would either
       short-circuit Phase 2.2's planned probe wiring or surface
       half-baked readiness output).
    """

    all_sources = set(known_sources())
    for field, expected_sources in (
        _PARTIALLY_POPULATED_PIONEER_CALLABLE_FIELDS.items()
    ):
        unexpected = set(expected_sources) - all_sources
        assert not unexpected, (
            f"_PARTIALLY_POPULATED_PIONEER_CALLABLE_FIELDS lists "
            f"{sorted(unexpected)} for {field!r}, but those aren't "
            "registered sources. Update the map when sources change."
        )

        for source in known_sources():
            value = getattr(LAUNCHERS[source], field)
            if source in expected_sources:
                assert value is not None, (
                    f"{source!r}.{field} must be populated per the "
                    f"partial-population contract; got None."
                )
                assert callable(value), (
                    f"{source!r}.{field} must be callable; "
                    f"got {type(value).__name__}."
                )
            else:
                assert value is None, (
                    f"{source!r}.{field} should still be None until its "
                    "populating slice ships (e.g., readiness_probe_fn "
                    "for researcher/designer/exec_search lands in "
                    "Phase 2.2). Move the source into the partial map "
                    "in the same PR that registers the callable."
                )


def test_pioneer_scalar_fields_default_to_typed_zero() -> None:
    """Non-callable scalar fields not yet populated by their slice
    default to ``False`` / ``""`` per the plan.

    ``progress_kind`` (Slice 1.5) and ``reconstruct_report_analysis``
    (Slice 1.4) have both lifted out of this tuple now that their
    populating slices have shipped. Their per-source expected-value
    ratchets live at
    :func:`test_progress_kind_registered_per_source_post_slice_1_5`
    and :func:`test_reconstruct_report_analysis_registered_per_source_post_slice_1_4`
    respectively.

    The tuple is currently empty — kept as a structural slot so a
    future scalar field added to ``LauncherEntry`` re-uses the same
    ratchet shape rather than inventing a new pattern.
    """

    for source in known_sources():
        entry = LAUNCHERS[source]
        for field, expected in _PIONEER_SCALAR_FIELDS:
            value = getattr(entry, field)
            assert value == expected, (
                f"{source!r}.{field} should default to {expected!r}; "
                f"got {value!r}."
            )
            assert type(value) is type(expected), (
                f"{source!r}.{field} should be {type(expected).__name__}; "
                f"got {type(value).__name__}."
            )


def test_reconstruct_report_analysis_registered_per_source_post_slice_1_4() -> None:
    """Slice 1.4 ratchet: every registered source declares the right
    ``reconstruct_report_analysis`` flag on its launcher entry.

    The dispatch site at
    ``market_intelligence/run_snapshots.py:_should_reconstruct_report_analysis``
    reads ``LAUNCHERS[source].reconstruct_report_analysis`` directly —
    Slice 1.4 removed the legacy ``source == "linkedin"`` predicate at
    the prior :535 call-site. Only LinkedIn opts in to the
    reconstruct-on-missing path today (reflection runs trigger it
    when ``run-report.json`` is absent); flipping any other source's
    flag to ``True`` without a corresponding underlying builder would
    silently no-op (the builder field
    ``snapshot_research_packet_fn`` is also LinkedIn-only) — this
    test catches the registration mismatch.
    """

    for source in known_sources():
        assert source in _RECONSTRUCT_REPORT_ANALYSIS_BY_SOURCE, (
            f"{source!r} is registered in LAUNCHERS but has no expected "
            f"reconstruct_report_analysis value in this test's ratchet. "
            "New sources must declare a value here even if it's the "
            "default ``False`` — otherwise we lose the regression-catch "
            "shape that pins LinkedIn as the only source opting in."
        )
        expected = _RECONSTRUCT_REPORT_ANALYSIS_BY_SOURCE[source]
        actual = LAUNCHERS[source].reconstruct_report_analysis
        assert actual is expected, (
            f"{source!r}.reconstruct_report_analysis should be "
            f"{expected!r}; got {actual!r}. Only LinkedIn opts in "
            "today; if a new source needs reconstruct-on-missing, its "
            "module-spec slice must also register a "
            "snapshot_research_packet_fn or the flag is a no-op."
        )
        assert isinstance(actual, bool), (
            f"{source!r}.reconstruct_report_analysis should be bool; "
            f"got {type(actual).__name__}."
        )


def test_progress_kind_registered_per_source_post_slice_1_5() -> None:
    """Slice 1.5 ratchet: every registered source declares the right
    ``progress_kind`` value on its launcher entry.

    The control_plane status aggregator and run-report builder read
    ``LAUNCHERS[source].progress_kind`` directly — Slice 1.5 removed
    the legacy ``_progress_kind_for_source`` ladder at
    ``cloris/control_plane.py:170-187`` along with the duplicate
    ``_SOURCES`` literal at ``cloris/control_plane.py:198``. The
    registry is the single source-of-truth.

    Coupling values to the canonical ``shared/runtime_state/store.py``
    constants (rather than re-typing string literals here) catches the
    "I renamed the kind constant but forgot the registry" regression
    at test time instead of letting work-unit progress silently land
    as 0/0 in the recruiter's homescreen card.
    """

    for source in known_sources():
        assert source in _PROGRESS_KIND_BY_SOURCE, (
            f"{source!r} is registered in LAUNCHERS but has no expected "
            f"progress_kind in this test's ratchet. New sources must "
            "either declare a non-empty work-unit kind in "
            "shared/runtime_state/store.py and register it on the "
            "LauncherEntry, or document the empty-string sentinel "
            "(no work-unit-aggregation channel) in this ratchet."
        )
        expected = _PROGRESS_KIND_BY_SOURCE[source]
        actual = LAUNCHERS[source].progress_kind
        assert actual == expected, (
            f"{source!r}.progress_kind should be {expected!r} "
            f"(from shared/runtime_state/store.py); got {actual!r}."
        )
        assert isinstance(actual, str), (
            f"{source!r}.progress_kind should be str; "
            f"got {type(actual).__name__}."
        )


def test_launcher_entry_remains_frozen_dataclass() -> None:
    """Frozen-ness is part of the contract.

    The registry is module-scope and immutable post-import (per the
    module docstring's contract notes). Frozen dataclass enforces this
    at the entry granularity — slice-time errors become declaration-time
    errors instead of runtime mutation bugs.
    """

    params = LauncherEntry.__dataclass_params__  # type: ignore[attr-defined]
    assert params.frozen is True, (
        "LauncherEntry must remain frozen=True; the registry's "
        "module-scope-immutable invariant depends on it."
    )


def test_pioneer_did_not_drop_existing_sources() -> None:
    """The five sources registered pre-1.0 stay registered.

    Pioneer slice is additive on the dataclass; it doesn't touch the
    LAUNCHERS dict entries themselves. Catches the regression where a
    rebase against a parallel module-spec window accidentally drops a
    source.
    """

    for source in ("linkedin", "github", "researcher", "designer", "exec_search"):
        assert source in LAUNCHERS, (
            f"{source!r} disappeared from LAUNCHERS. Slice 1.0 must "
            "not drop pre-existing entries."
        )
