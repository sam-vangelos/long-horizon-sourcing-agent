"""Per-branch regression tests for ``market_intelligence/run_snapshots.py``.

Multi-agent-execution Phase 1 Slice 1.4 replaces five LinkedIn-special-
case branches in ``market_intelligence/run_snapshots.py`` with registry
dispatch via :data:`cloris.launchers.LAUNCHERS`. Each branch closes
independently; this file pins the per-branch dispatch invariant and
the adapter-contract invariant for each registered helper.

Closed branches and their pre-/post-slice shapes:

- ``run_snapshots.py:287`` (``_runtime_brief_id``) — pre-slice was
  ``if source == "linkedin": return str(raw.get("linkedin_project_id")
  or brief.linkedin_project_id or brief.id or brief_path.stem)`` /
  ``return str(brief.id or brief.role_title or brief_path.stem)``.
  Post-slice the LinkedIn branch dispatches to
  ``LAUNCHERS["linkedin"].brief_id_for_snapshot_fn(brief, raw,
  brief_path)``; every other source falls through to the generic
  shape.

- ``run_snapshots.py:315`` (``_rebuild_run_scoped_projections``) —
  pre-slice was a 3-call LinkedIn branch (``write_linkedin_progress_projection``
  + ``write_linkedin_stage_projections`` +
  ``write_linkedin_search_memory_projection``) and a 2-call else
  branch (``write_github_progress_projection`` +
  ``write_github_stage_projections``). Post-slice each source's
  registered ``progress_projection_fn`` owns the projection orchestration;
  unrecognized sources no-op (pre-slice's else-branch was a dormant
  bug for non-{linkedin,github} sources, never hit in production).

- ``run_snapshots.py:409`` (``_research_batch_from_run_dir``) —
  pre-slice was ``if source == "linkedin": batch =
  maybe_build_and_persist_research_packet(batch,
  reconstruct_report_analysis=...)``. Post-slice dispatches via
  ``LAUNCHERS[source].snapshot_research_packet_fn``; LinkedIn is the
  only registered source. Distinct registry slot from Slice 1.3's
  reflection-time ``research_packet_builder_fn`` so the two seams can
  diverge if their kwarg shapes ever do.

- ``run_snapshots.py:461`` (``_brief_namespace_key``) — pre-slice was
  ``if source == "linkedin": return derive_brief_id(...)`` /
  ``return github_state_key(...)``. Post-slice dispatches via
  ``LAUNCHERS[source].snapshot_state_key_fn``; LinkedIn and GitHub
  register their respective state-key wrappers. The dispatch site
  preserves ``github_state_key`` as the None-fallback for unrecognized
  sources (preserves pre-slice behavior — those sources don't reach
  the dispatch site in production today, but the fallback honors the
  pre-slice contract verbatim).

- ``run_snapshots.py:535`` (``finalize_run_snapshot``) — pre-slice was
  ``reconstruct_report_analysis=source == "linkedin" and not (run_dir
  / "run-report.json").exists()``. Post-slice the per-source bool
  flag lifts onto ``LAUNCHERS[source].reconstruct_report_analysis``;
  the disk-presence check stays in the snapshot module (it's a
  runtime check, not a per-source contract). Behavior preserved
  bit-for-bit: only LinkedIn flips the flag to ``True``.

Companion to:

- :mod:`tests.test_launcher_registry_completeness` — pins the
  ``LauncherEntry`` shape and which sources populate each Slice 1.4
  field. The ``_PARTIALLY_POPULATED_PIONEER_CALLABLE_FIELDS`` and
  ``_RECONSTRUCT_REPORT_ANALYSIS_BY_SOURCE`` ratchets there move in
  lockstep with Slice 1.4.
- :mod:`tests.test_market_intelligence` — covers
  ``finalize_run_snapshot`` / ``import_legacy_run_snapshot`` end-to-
  end with realistic on-disk fixtures. This file is the narrow
  per-branch regression that catches dispatch-site mistakes without
  standing up the heavy fixture pipeline.
- :mod:`tests.test_market_intelligence_engine` — the Slice 1.3
  parallel for the reflection-time research-packet path. The
  snapshot-time path here mirrors that file's dispatch pattern.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import market_intelligence.run_snapshots as run_snapshots_mod
import shared.output_paths as output_paths_mod
import shared.runtime_state.projections as projections_mod
from cloris.launchers import LAUNCHERS, known_sources
import cloris.launchers as launchers_mod
from market_intelligence.schema import MarketEvidenceBatch


# ---------------------------------------------------------------------------
# Branch :287 — _runtime_brief_id dispatch + LinkedIn adapter contract.
# ---------------------------------------------------------------------------


def _stub_brief(
    *,
    id: str = "brief-id",
    role_title: str = "Role Title",
    linkedin_project_id: str = "",
) -> Any:
    """Minimal duck-typed stand-in for ``shared.brief_loader.Brief``.

    The five snapshot helpers read ``brief.id`` / ``brief.role_title``
    / ``brief.linkedin_project_id``; using ``SimpleNamespace`` keeps
    the test independent of the real Brief dataclass evolution.
    """

    return SimpleNamespace(
        id=id,
        role_title=role_title,
        linkedin_project_id=linkedin_project_id,
    )


def test_runtime_brief_id_linkedin_uses_registered_adapter(tmp_path: Path) -> None:
    """LinkedIn dispatches to its registered adapter, returning the
    ``linkedin_project_id``-first fallback chain.

    Pins the pre-slice behavior at the prior :287 ladder verbatim:
    ``raw["linkedin_project_id"] || brief.linkedin_project_id ||
    brief.id || stem``.
    """

    brief_path = tmp_path / "head-ai-lab.json"
    brief_path.touch()
    brief = _stub_brief(id="head-ai-lab", linkedin_project_id="0000")
    raw = {"linkedin_project_id": "3000000006"}

    result = run_snapshots_mod._runtime_brief_id(
        source="linkedin", brief=brief, raw=raw, brief_path=brief_path
    )

    assert result == "3000000006", (
        "LinkedIn brief id must come from raw['linkedin_project_id'] "
        "first (matches the F2 source_config fallback at "
        "derive_brief_id, even though _runtime_brief_id reads the "
        "flat field directly — pre-slice ladder used the same "
        "precedence)."
    )


def test_runtime_brief_id_linkedin_falls_back_through_chain(
    tmp_path: Path,
) -> None:
    """Every step in the LinkedIn fallback chain produces the right id.

    Walks the pre-slice ladder's ``or``-chain end to end; each step
    must yield the next value when its predecessor is empty.
    """

    brief_path = tmp_path / "fde-stem.json"
    brief_path.touch()

    # Step 2: brief.linkedin_project_id when raw is empty.
    brief = _stub_brief(id="fde", linkedin_project_id="proj-from-brief")
    assert (
        run_snapshots_mod._runtime_brief_id(
            source="linkedin",
            brief=brief,
            raw={},
            brief_path=brief_path,
        )
        == "proj-from-brief"
    )

    # Step 3: brief.id when neither raw nor brief carries a project_id.
    brief = _stub_brief(id="fde", linkedin_project_id="")
    assert (
        run_snapshots_mod._runtime_brief_id(
            source="linkedin",
            brief=brief,
            raw={},
            brief_path=brief_path,
        )
        == "fde"
    )

    # Step 4: brief_path stem when nothing else is set.
    brief = _stub_brief(id="", role_title="", linkedin_project_id="")
    assert (
        run_snapshots_mod._runtime_brief_id(
            source="linkedin",
            brief=brief,
            raw={},
            brief_path=brief_path,
        )
        == "fde-stem"
    )


@pytest.mark.parametrize("source", ["github", "researcher", "designer", "exec_search"])
def test_runtime_brief_id_non_linkedin_falls_through_to_generic_shape(
    source: str, tmp_path: Path
) -> None:
    """Non-LinkedIn sources fall through to ``brief.id || role_title || stem``.

    Pre-slice the else-branch supplied this generic shape to every
    non-LinkedIn source. Post-slice, those sources keep
    ``brief_id_for_snapshot_fn = None`` so the dispatch site falls
    through to the same generic shape — bit-for-bit preserved.

    Critically, the ``raw["linkedin_project_id"]`` is ignored here:
    the LinkedIn-specific fallback chain MUST NOT leak into other
    sources.
    """

    brief_path = tmp_path / "stem.json"
    brief_path.touch()
    brief = _stub_brief(id="github-brief-id", role_title="Role")
    raw = {"linkedin_project_id": "should-be-ignored"}

    result = run_snapshots_mod._runtime_brief_id(
        source=source, brief=brief, raw=raw, brief_path=brief_path
    )

    assert result == "github-brief-id"
    assert "should-be-ignored" not in result


def test_linkedin_brief_id_for_snapshot_adapter_is_pure_pass_through(
    tmp_path: Path,
) -> None:
    """The registered adapter is a thin computation, no side effects."""

    brief_path = tmp_path / "x.json"
    brief = _stub_brief(linkedin_project_id="proj-id")

    result = launchers_mod._linkedin_brief_id_for_snapshot(
        brief, {}, brief_path
    )

    assert result == "proj-id"


def test_brief_id_for_snapshot_fn_registered_only_for_linkedin() -> None:
    """Pins which sources populate the field — companion to the registry-completeness ratchet.

    The non-LinkedIn else-branch at the prior :289 supplied a generic
    fallback to every other source; registering an adapter for them
    here would short-circuit that path. Today only LinkedIn has the
    LinkedIn-special ``project_id`` chain.
    """

    populated = {
        source
        for source, entry in LAUNCHERS.items()
        if entry.brief_id_for_snapshot_fn is not None
    }

    assert populated == {"linkedin"}, (
        "brief_id_for_snapshot_fn should be populated only for "
        "linkedin at Slice 1.4. If you're adding a new source's "
        "snapshot brief-id, also extend the partial-population "
        "ratchet in tests/test_launcher_registry_completeness.py."
    )


# ---------------------------------------------------------------------------
# Branch :315 — _rebuild_run_scoped_projections dispatch + adapter contract.
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_projection_writers(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, list[tuple]]:
    """Patch all five projection writers to record their invocations.

    The Slice 1.4 adapters
    (``_linkedin_progress_projection`` /
    ``_github_progress_projection``) lazy-import the writers from
    :mod:`shared.runtime_state.projections`, so we patch the symbols
    at their definition module — same pattern as the Slice 1.3
    research-packet adapter test fixture.
    """

    calls: dict[str, list[tuple]] = {
        "linkedin_progress": [],
        "linkedin_stage": [],
        "linkedin_search_memory": [],
        "github_progress": [],
        "github_stage": [],
    }

    def _record(name: str):
        def fake(*args: Any, **kwargs: Any) -> None:
            calls[name].append((args, kwargs))

        return fake

    monkeypatch.setattr(
        projections_mod,
        "write_linkedin_progress_projection",
        _record("linkedin_progress"),
    )
    monkeypatch.setattr(
        projections_mod,
        "write_linkedin_stage_projections",
        _record("linkedin_stage"),
    )
    monkeypatch.setattr(
        projections_mod,
        "write_linkedin_search_memory_projection",
        _record("linkedin_search_memory"),
    )
    monkeypatch.setattr(
        projections_mod,
        "write_github_progress_projection",
        _record("github_progress"),
    )
    monkeypatch.setattr(
        projections_mod,
        "write_github_stage_projections",
        _record("github_stage"),
    )
    return calls


def _make_runtime_state_dir(state_dir: Path) -> None:
    """Create a minimal SQLite the snapshot module's existence check passes.

    ``_rebuild_run_scoped_projections`` short-circuits on a missing
    ``runtime_state.sqlite3`` — for dispatch-routing tests we just
    need the file to exist. The fake projection writers don't
    actually touch the DB.
    """

    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "runtime_state.sqlite3").touch()


def test_progress_projection_dispatch_routes_linkedin_to_three_writers(
    tmp_path: Path,
    fake_projection_writers: dict[str, list[tuple]],
) -> None:
    """LinkedIn invokes ``write_linkedin_*`` x3, no GitHub writers fire.

    Same call shape as the legacy linkedin branch at the prior :316-
    :330 (progress + stage + search-memory).
    """

    state_dir = tmp_path / "state"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _make_runtime_state_dir(state_dir)

    run_snapshots_mod._rebuild_run_scoped_projections(
        source="linkedin",
        run_dir=run_dir,
        source_dir=state_dir,
        brief_id="3000000006",
        run_id=42,
    )

    assert len(fake_projection_writers["linkedin_progress"]) == 1
    assert len(fake_projection_writers["linkedin_stage"]) == 1
    assert len(fake_projection_writers["linkedin_search_memory"]) == 1
    assert fake_projection_writers["github_progress"] == []
    assert fake_projection_writers["github_stage"] == []

    # Search-memory path is brief-id-scoped — pre-slice this was the
    # f"search_memory-{brief_id}.json" filename inside the run_dir.
    _, search_memory_kwargs = fake_projection_writers["linkedin_search_memory"][0]
    assert search_memory_kwargs["brief_id"] == "3000000006"
    assert search_memory_kwargs["path"] == run_dir / "search_memory-3000000006.json"


def test_progress_projection_dispatch_routes_github_to_two_writers(
    tmp_path: Path,
    fake_projection_writers: dict[str, list[tuple]],
) -> None:
    """GitHub invokes ``write_github_*`` x2, no LinkedIn writers fire.

    Same call shape as the legacy ``return``-less else-branch at
    the prior :332-:339 (progress + stage; no search-memory — GitHub
    has no per-brief search-memory projection in
    ``shared/runtime_state/projections.py``).
    """

    state_dir = tmp_path / "state"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _make_runtime_state_dir(state_dir)

    run_snapshots_mod._rebuild_run_scoped_projections(
        source="github",
        run_dir=run_dir,
        source_dir=state_dir,
        brief_id="github-brief",
        run_id=7,
    )

    assert len(fake_projection_writers["github_progress"]) == 1
    assert len(fake_projection_writers["github_stage"]) == 1
    assert fake_projection_writers["linkedin_progress"] == []
    assert fake_projection_writers["linkedin_stage"] == []
    assert fake_projection_writers["linkedin_search_memory"] == []


@pytest.mark.parametrize("source", ["researcher", "designer", "exec_search"])
def test_progress_projection_dispatch_no_op_for_unregistered_sources(
    source: str,
    tmp_path: Path,
    fake_projection_writers: dict[str, list[tuple]],
) -> None:
    """Sources without a registered fn no-op silently.

    Pre-slice the else-branch wrote GitHub-style projections for
    these sources — a dormant bug never hit in production
    (researcher/designer/exec_search don't invoke
    ``finalize_run_snapshot`` yet). Post-slice they no-op cleanly.
    Per the slice plan this is registry cleanup, not a snapshot-
    semantics change (the pre-slice path was unreachable).
    """

    state_dir = tmp_path / "state"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _make_runtime_state_dir(state_dir)

    run_snapshots_mod._rebuild_run_scoped_projections(
        source=source,
        run_dir=run_dir,
        source_dir=state_dir,
        brief_id="brief-id",
        run_id=1,
    )

    for writer in fake_projection_writers.values():
        assert writer == []


def test_progress_projection_dispatch_short_circuits_when_run_id_is_none(
    tmp_path: Path,
    fake_projection_writers: dict[str, list[tuple]],
) -> None:
    """Snapshot module's pre-existing ``run_id is None`` guard still applies.

    Pre-slice the linkedin branch was preceded by ``if run_id is
    None: return``; post-slice the guard stays in the snapshot module
    so the dispatch never fires for runs with no canonical id (e.g.,
    a state-dir without a runtime SQLite row).
    """

    state_dir = tmp_path / "state"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _make_runtime_state_dir(state_dir)

    run_snapshots_mod._rebuild_run_scoped_projections(
        source="linkedin",
        run_dir=run_dir,
        source_dir=state_dir,
        brief_id="brief",
        run_id=None,
    )

    for writer in fake_projection_writers.values():
        assert writer == []


def test_progress_projection_dispatch_short_circuits_when_sqlite_missing(
    tmp_path: Path,
    fake_projection_writers: dict[str, list[tuple]],
) -> None:
    """``runtime_state.sqlite3`` absence still short-circuits.

    Pre-slice the SQLite existence check ran before the linkedin
    branch; same here post-slice. State-dirs without a canonical
    runtime DB skip projection rebuilds entirely.
    """

    state_dir = tmp_path / "state"
    state_dir.mkdir()
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    run_snapshots_mod._rebuild_run_scoped_projections(
        source="linkedin",
        run_dir=run_dir,
        source_dir=state_dir,
        brief_id="brief",
        run_id=1,
    )

    for writer in fake_projection_writers.values():
        assert writer == []


def test_linkedin_progress_projection_adapter_swallows_legacy_payload_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Adapter preserves the pre-slice ``_try_projection`` legacy guard.

    Pre-slice ``_try_projection`` swallowed ``(TypeError, ValueError,
    KeyError)`` so legacy-shaped runtime payloads didn't fail the
    whole snapshot import. The adapter must preserve this exact
    catch list — anything narrower would re-introduce the legacy-
    payload bug.
    """

    def raises_value_error(*args: Any, **kwargs: Any) -> None:
        raise ValueError("legacy payload")

    monkeypatch.setattr(
        projections_mod,
        "write_linkedin_progress_projection",
        raises_value_error,
    )
    monkeypatch.setattr(
        projections_mod, "write_linkedin_stage_projections", raises_value_error
    )
    monkeypatch.setattr(
        projections_mod,
        "write_linkedin_search_memory_projection",
        raises_value_error,
    )

    run_dir = tmp_path / "run"
    run_dir.mkdir()

    # Should not raise — every underlying call is wrapped in the guard.
    launchers_mod._linkedin_progress_projection(
        store=None,  # the fakes ignore it
        brief_id="b",
        run_id=1,
        run_dir=run_dir,
    )


def test_progress_projection_fn_registered_for_linkedin_and_github_only() -> None:
    """Pins which sources populate the field — companion to the registry-completeness ratchet."""

    populated = {
        source
        for source, entry in LAUNCHERS.items()
        if entry.progress_projection_fn is not None
    }

    assert populated == {"linkedin", "github"}, (
        "progress_projection_fn should be populated only for linkedin "
        "and github at Slice 1.4. Other sources keep None and the "
        "dispatch site no-ops for them."
    )


# ---------------------------------------------------------------------------
# Branch :409 — _research_batch_from_run_dir dispatch + adapter contract.
# ---------------------------------------------------------------------------


def _make_batch(source: str) -> MarketEvidenceBatch:
    """Minimal batch for snapshot research-packet dispatch routing.

    Mirrors :func:`tests.test_market_intelligence_engine._make_batch`
    so the parallel-seam discipline shows in both files.
    """

    return MarketEvidenceBatch(
        run_ref=f"{source}:fixture",
        source=source,
        output_dir=f"/tmp/{source}-fixture",
        brief_version="2.0",
        generated_at="2026-05-04T00:00:00Z",
    )


@pytest.fixture
def fake_research_packet_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[MarketEvidenceBatch, bool]]:
    """Patch :func:`maybe_build_and_persist_research_packet`.

    The Slice 1.4 adapter ``_linkedin_snapshot_research_packet``
    lazy-imports the underlying builder from
    :mod:`market_intelligence.research_context`, so we patch the
    symbol at its definition module. Same lazy-import-compatibility
    pattern as the Slice 1.3 fixture.
    """

    calls: list[tuple[MarketEvidenceBatch, bool]] = []

    def fake_builder(
        batch: MarketEvidenceBatch,
        *,
        reconstruct_report_analysis: bool,
    ) -> MarketEvidenceBatch:
        calls.append((batch, reconstruct_report_analysis))
        return batch

    import market_intelligence.research_context as research_context_mod

    monkeypatch.setattr(
        research_context_mod,
        "maybe_build_and_persist_research_packet",
        fake_builder,
    )
    return calls


def _dispatch_snapshot_research_packet(
    batch: MarketEvidenceBatch,
    *,
    reconstruct_report_analysis: bool,
) -> MarketEvidenceBatch:
    """Inline copy of the post-slice dispatch site at run_snapshots.py:412-422.

    Replicated here so the test exercises dispatch routing without
    standing up the heavy ``_research_batch_from_run_dir`` fixture
    (run-dir, run-report, final_judgments JSONL, search_memory glob).
    If the dispatch shape changes, this helper must change with it.
    """

    launcher = LAUNCHERS.get(batch.source)
    if launcher is not None and (
        builder := launcher.snapshot_research_packet_fn
    ) is not None:
        batch = builder(
            batch,
            reconstruct_report_analysis=reconstruct_report_analysis,
        )
    return batch


def test_snapshot_research_packet_dispatch_routes_linkedin(
    fake_research_packet_builder: list[tuple[MarketEvidenceBatch, bool]],
) -> None:
    """LinkedIn batch dispatches to the underlying builder; kwarg forwarded."""

    batch = _make_batch("linkedin")

    result = _dispatch_snapshot_research_packet(
        batch, reconstruct_report_analysis=True
    )

    assert result is batch
    assert len(fake_research_packet_builder) == 1
    forwarded_batch, forwarded_kwarg = fake_research_packet_builder[0]
    assert forwarded_batch is batch
    assert forwarded_kwarg is True, (
        "LinkedIn snapshot adapter must forward "
        "reconstruct_report_analysis verbatim — it gates the "
        "report-analysis reconstruction path the LinkedIn-only "
        "snapshot builder owns."
    )


def test_snapshot_research_packet_dispatch_forwards_kwarg_false(
    fake_research_packet_builder: list[tuple[MarketEvidenceBatch, bool]],
) -> None:
    """``reconstruct_report_analysis=False`` forwards as-is, not coerced."""

    batch = _make_batch("linkedin")

    _dispatch_snapshot_research_packet(batch, reconstruct_report_analysis=False)

    assert len(fake_research_packet_builder) == 1
    _, forwarded_kwarg = fake_research_packet_builder[0]
    assert forwarded_kwarg is False


@pytest.mark.parametrize(
    "source", ["github", "researcher", "designer", "exec_search"]
)
def test_snapshot_research_packet_dispatch_no_op_for_non_linkedin_sources(
    source: str,
    fake_research_packet_builder: list[tuple[MarketEvidenceBatch, bool]],
) -> None:
    """Sources without a registered fn fall through unchanged.

    Pre-slice the if/else fell through and returned the batch
    unchanged for every non-LinkedIn source. Post-slice
    ``snapshot_research_packet_fn`` is ``None`` for every other
    source so the dispatch site preserves the fall-through.
    """

    batch = _make_batch(source)

    result = _dispatch_snapshot_research_packet(
        batch, reconstruct_report_analysis=True
    )

    assert result is batch
    assert fake_research_packet_builder == []


def test_snapshot_research_packet_fn_registered_only_for_linkedin() -> None:
    """Pins which sources populate the field.

    Slice 1.3's reflection-time ``research_packet_builder_fn``
    registered LinkedIn AND GitHub (both sources have reflection-
    time research-packet contracts). Slice 1.4's snapshot-time
    counterpart registers only LinkedIn (GitHub doesn't build a
    snapshot-time research packet — see the pre-slice :409 branch
    which had no ``elif source == "github"`` arm). Catches the
    regression where a future module spec accidentally registers
    a snapshot builder.
    """

    populated = {
        source
        for source, entry in LAUNCHERS.items()
        if entry.snapshot_research_packet_fn is not None
    }

    assert populated == {"linkedin"}, (
        "snapshot_research_packet_fn should be populated only for "
        "linkedin at Slice 1.4. If you're adding a new source's "
        "snapshot-time research-packet builder, also extend the "
        "partial-population ratchet in "
        "tests/test_launcher_registry_completeness.py."
    )


# ---------------------------------------------------------------------------
# Branch :461 — _brief_namespace_key dispatch + adapter contracts.
# ---------------------------------------------------------------------------


def test_brief_namespace_key_linkedin_uses_derive_brief_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """LinkedIn dispatches to its registered adapter (wraps ``derive_brief_id``).

    Pre-slice the linkedin branch called
    ``derive_brief_id(brief_path=..., brief=..., raw=...)``. Post-
    slice the same call goes through the registered adapter; we
    monkeypatch the underlying function and verify the adapter
    forwards all three kwargs verbatim.
    """

    captured: dict[str, Any] = {}

    def fake_derive_brief_id(*, brief_path: Any, brief: Any, raw: Any) -> str:
        captured["brief_path"] = brief_path
        captured["brief"] = brief
        captured["raw"] = raw
        return "linkedin-state-key"

    monkeypatch.setattr(
        output_paths_mod, "derive_brief_id", fake_derive_brief_id
    )

    brief = _stub_brief()
    raw = {"linkedin_project_id": "p"}
    brief_path = tmp_path / "brief.json"

    result = run_snapshots_mod._brief_namespace_key(
        source="linkedin", brief_path=brief_path, brief=brief, raw=raw
    )

    assert result == "linkedin-state-key"
    assert captured == {"brief_path": brief_path, "brief": brief, "raw": raw}


def test_brief_namespace_key_github_uses_github_state_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GitHub dispatches to its registered adapter (wraps ``github_state_key``).

    Pre-slice GitHub fell through the else-branch which directly
    called ``github_state_key(brief_path=..., brief=...)`` — note no
    ``raw`` kwarg (GitHub doesn't read the raw brief dict). The
    post-slice adapter accepts ``raw`` for signature uniformity and
    discards it; this test pins the discard so the underlying
    ``github_state_key`` doesn't receive an unexpected kwarg.
    """

    captured: dict[str, Any] = {}

    def fake_github_state_key(*, brief_path: Any, brief: Any) -> str:
        captured["brief_path"] = brief_path
        captured["brief"] = brief
        return "github-state-key"

    monkeypatch.setattr(
        output_paths_mod, "github_state_key", fake_github_state_key
    )

    brief = _stub_brief()
    raw = {"unused": "by-github"}
    brief_path = tmp_path / "brief.json"

    result = run_snapshots_mod._brief_namespace_key(
        source="github", brief_path=brief_path, brief=brief, raw=raw
    )

    assert result == "github-state-key"
    assert captured == {"brief_path": brief_path, "brief": brief}, (
        "GitHub adapter must discard the LinkedIn-specific raw kwarg — "
        "passing it through would TypeError on the underlying "
        "github_state_key signature."
    )


@pytest.mark.parametrize(
    "source,expected_key_fn",
    [
        ("researcher", "researcher_state_key"),
        ("designer", "designer_state_key"),
        ("exec_search", "exec_search_state_key"),
    ],
)
def test_brief_namespace_key_dispatches_per_source_for_non_linkedin_github(
    source: str,
    expected_key_fn: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Slice A.7 wired per-source snapshot state-key adapters.

    Pre-A.7 the dispatch fell through to ``github_state_key`` for
    researcher / designer / exec_search — a wrong-source fallback
    that would corrupt the snapshot namespace once those modules
    invoked ``finalize_run_snapshot``. A.7 closes this dormant
    fall-through: each module now registers
    :attr:`cloris.launchers.LauncherEntry.snapshot_state_key_fn`
    wrapping the appropriate ``shared.output_paths`` helper.

    This test pins the per-source dispatch by monkey-patching the
    expected state-key helper at the source-modules level and
    asserting the returned namespace key matches.
    """

    captured: dict[str, Any] = {}

    def fake_state_key(*, brief_path: Any, brief: Any) -> str:
        captured["called_with"] = (brief_path, brief)
        return f"fake-{source}-state-key"

    import shared.output_paths as output_paths_mod

    monkeypatch.setattr(output_paths_mod, expected_key_fn, fake_state_key)

    brief = _stub_brief()
    brief_path = tmp_path / "brief.json"

    result = run_snapshots_mod._brief_namespace_key(
        source=source, brief_path=brief_path, brief=brief, raw={}
    )

    assert result == f"fake-{source}-state-key", (
        f"{source} should dispatch through {expected_key_fn} after Slice A.7, "
        "not the legacy github_state_key fallback."
    )
    assert captured["called_with"] == (brief_path, brief)
    assert captured["called_with"] == (brief_path, brief)


def test_snapshot_state_key_fn_registered_for_every_known_source() -> None:
    """Slice A.7 (Multi-Agent Production Plan) widened registration to all sources.

    Pre-A.7 the field was populated only for linkedin / github;
    researcher / designer / exec_search registered ``None`` and the
    dispatch site at :func:`market_intelligence.run_snapshots._brief_namespace_key`
    fell through to ``github_state_key`` — a wrong-source fallback
    that would corrupt the snapshot namespace once those modules hit
    ``finalize_run_snapshot``. A.7 closes the dormant fall-through;
    every source now registers a per-module snapshot adapter that
    wraps the appropriate ``shared.output_paths`` state-key helper.
    """

    populated = {
        source
        for source, entry in LAUNCHERS.items()
        if entry.snapshot_state_key_fn is not None
    }

    assert populated == {
        "linkedin",
        "github",
        "researcher",
        "designer",
        "exec_search",
    }, (
        "snapshot_state_key_fn should be populated for every "
        "registered source post-Slice A.7. The github_state_key "
        "fallback at run_snapshots._brief_namespace_key remains as a "
        "defensive backstop for unknown sources but should not be hit "
        "for any known module."
    )


# ---------------------------------------------------------------------------
# Branch :535 — _should_reconstruct_report_analysis dispatch + scalar contract.
# ---------------------------------------------------------------------------


def test_should_reconstruct_report_analysis_linkedin_run_report_missing(
    tmp_path: Path,
) -> None:
    """LinkedIn + missing run-report.json → True.

    Pre-slice predicate at the prior :535:
    ``source == "linkedin" and not (run_dir / "run-report.json").exists()``.
    LinkedIn's flag is ``True``; the disk-presence check stays in
    the snapshot module.
    """

    run_dir = tmp_path / "run"
    run_dir.mkdir()

    assert (
        run_snapshots_mod._should_reconstruct_report_analysis(
            source="linkedin", run_dir=run_dir
        )
        is True
    )


def test_should_reconstruct_report_analysis_linkedin_run_report_present(
    tmp_path: Path,
) -> None:
    """LinkedIn + present run-report.json → False.

    Disk-presence check still gates reconstruction even with the
    flag flipped on. This is the "don't reconstruct what's already
    there" semantic that the pre-slice predicate's right-hand
    conjunct supplied.
    """

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run-report.json").write_text("{}")

    assert (
        run_snapshots_mod._should_reconstruct_report_analysis(
            source="linkedin", run_dir=run_dir
        )
        is False
    )


@pytest.mark.parametrize("source", ["github", "researcher", "designer", "exec_search"])
def test_should_reconstruct_report_analysis_non_linkedin_always_false(
    source: str, tmp_path: Path
) -> None:
    """Every non-LinkedIn source returns False regardless of run-report presence.

    Pre-slice predicate's left-hand conjunct (``source ==
    "linkedin"``) was the LinkedIn-special gate; post-slice the
    registry flag is ``False`` for every other source so
    reconstruction never fires regardless of disk state. Bit-for-
    bit pre-slice behavior preserved.
    """

    run_dir_missing = tmp_path / "missing"
    run_dir_missing.mkdir()
    run_dir_present = tmp_path / "present"
    run_dir_present.mkdir()
    (run_dir_present / "run-report.json").write_text("{}")

    assert (
        run_snapshots_mod._should_reconstruct_report_analysis(
            source=source, run_dir=run_dir_missing
        )
        is False
    )
    assert (
        run_snapshots_mod._should_reconstruct_report_analysis(
            source=source, run_dir=run_dir_present
        )
        is False
    )


def test_should_reconstruct_report_analysis_unknown_source_returns_false(
    tmp_path: Path,
) -> None:
    """Unknown source falls through to ``False`` — preserves legacy ladder semantics.

    Pre-slice the predicate ``source == "linkedin"`` evaluated to
    False for any non-LinkedIn string, including unknown ones.
    Post-slice ``LAUNCHERS.get(unknown_source)`` returns ``None``
    and the dispatch returns ``False`` — same outcome.
    """

    run_dir = tmp_path / "run"
    run_dir.mkdir()

    assert (
        run_snapshots_mod._should_reconstruct_report_analysis(
            source="not-a-real-source", run_dir=run_dir
        )
        is False
    )


def test_reconstruct_report_analysis_flag_set_only_for_linkedin() -> None:
    """Pins the per-source flag values — companion to the registry-completeness ratchet."""

    flagged = {
        source
        for source, entry in LAUNCHERS.items()
        if entry.reconstruct_report_analysis is True
    }

    assert flagged == {"linkedin"}, (
        "reconstruct_report_analysis should be True only on linkedin "
        "at Slice 1.4. If a new source needs reconstruct-on-missing, "
        "its module-spec slice must also register a "
        "snapshot_research_packet_fn — flipping the flag without the "
        "builder is a no-op (and a bug if it shadows another source's "
        "intent)."
    )


# ---------------------------------------------------------------------------
# Slice W1.7 — finalized LinkedIn snapshots carry shadow experiment artifacts.
# ---------------------------------------------------------------------------


def _write_shadow_snapshot_brief(brief_path: Path, *, project_id: str) -> None:
    brief_path.parent.mkdir(parents=True, exist_ok=True)
    brief_path.write_text(
        json.dumps(
            {
                "name": "shadow-artifact-brief",
                "role_title": "Shadow Artifact Role",
                "linkedin_project": "Shadow Artifact Project",
                "linkedin_project_id": project_id,
                "minimum_bar": "Evidence that the candidate built relevant systems.",
                "archetypes": [],
            },
            indent=2,
        )
    )


def _finalize_shadow_snapshot_fixture(
    tmp_path: Path, *, include_shadow_artifacts: bool
) -> tuple[Path, dict]:
    fixture_root = tmp_path / (
        "with-shadow" if include_shadow_artifacts else "without-shadow"
    )
    project_id = "shadow-with" if include_shadow_artifacts else "shadow-without"
    brief_path = fixture_root / "brief.json"
    _write_shadow_snapshot_brief(brief_path, project_id=project_id)

    state_dir = fixture_root / "output" / "state" / "linkedin" / project_id
    state_dir.mkdir(parents=True)
    if include_shadow_artifacts:
        shadow_strategy_dir = state_dir / "shadow_strategy"
        shadow_strategy_dir.mkdir()
        (shadow_strategy_dir / "one.json").write_text(
            json.dumps({"strategy": "one"}, indent=2)
        )
        (state_dir / "shadow_judgments.jsonl").write_text(
            json.dumps({"candidate": "Ada Shadow", "decision": "SAVE"}) + "\n"
        )

    run_dir = run_snapshots_mod.finalize_run_snapshot(
        source="linkedin",
        brief_path=brief_path,
        state_dir=state_dir,
    )
    manifest = json.loads((run_dir / "run-manifest.json").read_text())
    return run_dir, manifest


def test_finalize_run_snapshot_includes_linkedin_shadow_artifacts(
    tmp_path: Path,
) -> None:
    run_dir, manifest = _finalize_shadow_snapshot_fixture(
        tmp_path,
        include_shadow_artifacts=True,
    )

    assert (run_dir / "shadow_strategy" / "one.json").exists()
    assert (run_dir / "shadow_judgments.jsonl").exists()
    assert "shadow_strategy/one.json" in manifest["artifacts_present"]
    assert "shadow_judgments.jsonl" in manifest["artifacts_present"]


def test_finalize_run_snapshot_without_linkedin_shadow_artifacts_has_no_phantoms(
    tmp_path: Path,
) -> None:
    run_dir, manifest = _finalize_shadow_snapshot_fixture(
        tmp_path,
        include_shadow_artifacts=False,
    )
    shadow_entries = {"shadow_strategy/one.json", "shadow_judgments.jsonl"}

    assert shadow_entries.isdisjoint(manifest["artifacts_present"])
    assert not (run_dir / "shadow_strategy").exists()
    assert not (run_dir / "shadow_judgments.jsonl").exists()


def test_archive_stale_outputs_rotates_shadow_artifacts(tmp_path: Path) -> None:
    from linkedin.orchestrator import Pipeline

    state_dir = tmp_path / "output" / "state" / "linkedin" / "shadow-brief"
    state_dir.mkdir(parents=True)
    (state_dir / "shadow_judgments.jsonl").write_text(
        json.dumps({"candidate": "Ada Shadow", "decision": "SAVE"}) + "\n"
    )
    shadow_strategy_dir = state_dir / "shadow_strategy"
    shadow_strategy_dir.mkdir()
    (shadow_strategy_dir / "one.json").write_text(
        json.dumps({"strategy": "one"}, indent=2)
    )
    pipeline = SimpleNamespace(
        output_dir=state_dir,
        _brief_id="shadow-brief",
        snippets_path=state_dir / "snippets.jsonl",
        facial_path=state_dir / "facial_judgments.jsonl",
        profiles_path=state_dir / "profile_summaries.jsonl",
        final_path=state_dir / "final_judgments.jsonl",
    )

    Pipeline._archive_stale_outputs(pipeline)

    assert not (state_dir / "shadow_judgments.jsonl").exists()
    assert not (state_dir / "shadow_strategy").exists()
    archive_root = (
        tmp_path / "output" / "archive" / "linkedin" / "shadow_brief" / "state-resets"
    )
    archive_roots = list(archive_root.iterdir())
    assert len(archive_roots) == 1
    archive_dir = archive_roots[0]
    assert (archive_dir / "shadow_judgments.jsonl").exists()
    assert (archive_dir / "shadow_strategy" / "one.json").exists()


# ---------------------------------------------------------------------------
# Drift guards — pin the production dispatch shape.
# ---------------------------------------------------------------------------


def test_run_snapshots_module_no_longer_imports_lifted_helpers() -> None:
    """The snapshot module's top-level imports drop the lifted symbols.

    Pre-slice ``run_snapshots.py`` imported
    ``maybe_build_and_persist_research_packet``, ``derive_brief_id``,
    ``write_linkedin_progress_projection``,
    ``write_linkedin_search_memory_projection``,
    ``write_linkedin_stage_projections``, and
    ``write_github_progress_projection`` /
    ``write_github_stage_projections`` at module scope. Post-slice
    every one of those lookups happens inside the registered
    adapters at :mod:`cloris.launchers`. Catches the regression
    where a rebase reintroduces the legacy ladder alongside the
    registry dispatch (double-fire would write projections twice
    per LinkedIn run).
    """

    lifted_symbols = (
        "maybe_build_and_persist_research_packet",
        "derive_brief_id",
        "write_linkedin_progress_projection",
        "write_linkedin_search_memory_projection",
        "write_linkedin_stage_projections",
        "write_github_progress_projection",
        "write_github_stage_projections",
    )

    for symbol in lifted_symbols:
        assert not hasattr(run_snapshots_mod, symbol), (
            f"market_intelligence/run_snapshots.py still imports "
            f"{symbol!r} at module scope. Slice 1.4 lifted it into "
            f"the registered adapter at cloris/launchers/__init__.py — "
            "drop the import."
        )


def test_run_snapshots_module_has_no_runtime_source_equality_branches() -> None:
    """No ``if source == "linkedin"`` runtime branches remain.

    Slice 1.4 closes five such branches at the prior :287, :315,
    :409, :461, :535. Docstrings and comments may reference the
    pre-slice shape historically; this test scans the module's
    actual AST to catch a runtime branch sneaking back in.
    """

    import ast
    import inspect

    source_text = inspect.getsource(run_snapshots_mod)
    tree = ast.parse(source_text)

    runtime_source_compares: list[tuple[int, str]] = []

    class SourceCompareFinder(ast.NodeVisitor):
        def visit_Compare(self, node: ast.Compare) -> None:
            # Look for ``<something>.source == "linkedin"`` or
            # ``source == "linkedin"`` style runtime predicates.
            if not (
                isinstance(node.left, ast.Name) and node.left.id == "source"
            ):
                self.generic_visit(node)
                return
            for op, comparator in zip(node.ops, node.comparators):
                if not isinstance(op, ast.Eq):
                    continue
                if (
                    isinstance(comparator, ast.Constant)
                    and isinstance(comparator.value, str)
                    and comparator.value in ("linkedin", "github")
                ):
                    runtime_source_compares.append(
                        (node.lineno, comparator.value)
                    )
            self.generic_visit(node)

    SourceCompareFinder().visit(tree)

    assert runtime_source_compares == [], (
        "Slice 1.4 closed every ``source == 'linkedin'`` / "
        "``source == 'github'`` runtime branch in run_snapshots.py. "
        "Found a recurrence at: "
        f"{runtime_source_compares}. Route via "
        "``LAUNCHERS[source].<helper>`` instead."
    )


def test_known_sources_unchanged_by_slice_1_4() -> None:
    """The set of registered sources is structurally untouched.

    Slice 1.4 only adds Per-source helpers; it must not drop a
    source nor add one. Catches the regression where a rebase
    against a parallel module-spec window accidentally drops or
    duplicates an entry.
    """

    assert set(known_sources()) == {
        "linkedin",
        "github",
        "researcher",
        "designer",
        "exec_search",
    }
