"""Engine-level regression tests for ``market_intelligence/engine.py``.

Multi-agent-execution Phase 1 Slice 1.3 replaces the legacy if/elif
source ladder inside :func:`market_intelligence.engine._load_evidence_batch`
with registry dispatch via
:attr:`cloris.launchers.LAUNCHERS`'s ``research_packet_builder_fn``.

Pre-slice (engine.py:1621-1636 in slice 1.0)::

    if batch.source == "linkedin":
        batch = maybe_build_and_persist_research_packet(
            batch, reconstruct_report_analysis=reconstruct_report_analysis,
        )
    elif batch.source == "github":
        from market_intelligence.github_reflection import (
            maybe_build_and_persist_github_research_packet,
        )
        batch = maybe_build_and_persist_github_research_packet(batch)
    return batch

Post-slice::

    launcher = LAUNCHERS.get(batch.source)
    if launcher is not None and (
        builder := launcher.research_packet_builder_fn
    ) is not None:
        batch = builder(
            batch,
            reconstruct_report_analysis=reconstruct_report_analysis,
        )
    return batch

Behavior is byte-equivalent: LinkedIn batches still call
``maybe_build_and_persist_research_packet`` with the
``reconstruct_report_analysis`` kwarg; GitHub batches still call
``maybe_build_and_persist_github_research_packet`` without it; every
other source still no-ops. This file pins that invariant directly
against the registered adapter callables and against the dispatch
shape on a multi-source evidence batch.

Companion to:

- :mod:`tests.test_launcher_registry_completeness` — pins the
  ``LauncherEntry`` shape and which sources populate
  ``research_packet_builder_fn`` (linkedin, github only).
- :mod:`tests.test_github_reflection` — pins the GitHub builder's
  underlying source-defensive behavior (no-op outside github source).
- :mod:`tests.test_market_intelligence` — covers
  ``_load_evidence_batch`` end-to-end with realistic on-disk fixtures.
"""

from __future__ import annotations

import pytest

import market_intelligence.engine as engine_mod
import market_intelligence.github_reflection as github_reflection_mod
import market_intelligence.research_context as research_context_mod
from cloris.launchers import LAUNCHERS, known_sources
from market_intelligence.schema import MarketEvidenceBatch


def _make_batch(source: str) -> MarketEvidenceBatch:
    """Build a minimal :class:`MarketEvidenceBatch` for dispatch routing.

    Real batches in production carry report / runtime_summary /
    final_judgments / metrics_summary; for dispatch-routing tests
    those don't matter — only ``source`` does, since the registry
    dispatches on it. The underlying builders are mocked, so they
    never inspect the rest of the batch shape.
    """

    return MarketEvidenceBatch(
        run_ref=f"{source}:fixture",
        source=source,
        output_dir=f"/tmp/{source}-fixture",
        brief_version="2.0",
        generated_at="2026-05-04T00:00:00Z",
    )


def _dispatch_via_registry(
    batch: MarketEvidenceBatch,
    *,
    reconstruct_report_analysis: bool,
) -> MarketEvidenceBatch:
    """Replicate the post-slice dispatch shape inline from ``engine.py``.

    This is the exact 5-line invariant ``_load_evidence_batch`` uses
    post-slice. Replicated here so the test exercises dispatch routing
    without standing up the heavy ``_load_evidence_batch`` fixture
    (run manifest, runtime SQLite, final_judgments JSONL, etc., which
    :mod:`tests.test_market_intelligence` already covers end-to-end).
    If the dispatch shape changes, this helper must change with it.
    """

    launcher = LAUNCHERS.get(batch.source)
    if launcher is not None and (
        builder := launcher.research_packet_builder_fn
    ) is not None:
        batch = builder(
            batch,
            reconstruct_report_analysis=reconstruct_report_analysis,
        )
    return batch


@pytest.fixture
def captured_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, list[tuple[MarketEvidenceBatch, ...]]]:
    """Patch both underlying builders to record their invocations.

    Slice 1.3's adapter wrappers
    (:func:`cloris.launchers._linkedin_research_packet_builder` /
    :func:`cloris.launchers._github_research_packet_builder`) lazy-
    import the underlying builders at call time, mirroring the
    Slice 1.1 readiness-probe lazy-import pattern at
    :func:`cloris.launchers._linkedin_readiness_probe`. The
    monkeypatch replaces the symbols at their definition modules so
    the adapters pick up the fakes when invoked.
    """

    calls: dict[str, list[tuple[MarketEvidenceBatch, ...]]] = {
        "linkedin": [],
        "github": [],
    }

    def fake_linkedin_builder(
        batch: MarketEvidenceBatch,
        *,
        reconstruct_report_analysis: bool,
    ) -> MarketEvidenceBatch:
        calls["linkedin"].append((batch, reconstruct_report_analysis))
        return batch

    def fake_github_builder(
        batch: MarketEvidenceBatch,
    ) -> MarketEvidenceBatch:
        calls["github"].append((batch,))
        return batch

    monkeypatch.setattr(
        research_context_mod,
        "maybe_build_and_persist_research_packet",
        fake_linkedin_builder,
    )
    monkeypatch.setattr(
        github_reflection_mod,
        "maybe_build_and_persist_github_research_packet",
        fake_github_builder,
    )
    return calls


# ---------------------------------------------------------------------------
# Dispatch routing — pin the per-source builder calls.
# ---------------------------------------------------------------------------


def test_research_packet_dispatch_routes_linkedin_to_linkedin_builder(
    captured_calls: dict[str, list[tuple]],
) -> None:
    """LinkedIn batch dispatches to the LinkedIn underlying builder, kwarg forwarded.

    Same call shape as the legacy
    ``if batch.source == "linkedin": maybe_build_and_persist_research_packet(batch, reconstruct_report_analysis=...)``.
    """

    batch = _make_batch("linkedin")

    result = _dispatch_via_registry(batch, reconstruct_report_analysis=True)

    assert result is batch
    assert len(captured_calls["linkedin"]) == 1
    forwarded_batch, forwarded_kwarg = captured_calls["linkedin"][0]
    assert forwarded_batch is batch
    assert forwarded_kwarg is True, (
        "LinkedIn adapter must forward reconstruct_report_analysis "
        "verbatim — it gates the report-analysis reconstruction path "
        "at run_snapshots.py:535 (LinkedIn-only)."
    )
    assert captured_calls["github"] == []


def test_research_packet_dispatch_routes_github_to_github_builder(
    captured_calls: dict[str, list[tuple]],
) -> None:
    """GitHub batch dispatches to the GitHub underlying builder, kwarg discarded.

    Same call shape as the legacy
    ``elif batch.source == "github": maybe_build_and_persist_github_research_packet(batch)``.
    The LinkedIn-specific ``reconstruct_report_analysis`` kwarg is
    accepted by the adapter and discarded — the underlying GitHub
    builder takes only ``(batch,)``.
    """

    batch = _make_batch("github")

    result = _dispatch_via_registry(batch, reconstruct_report_analysis=True)

    assert result is batch
    assert len(captured_calls["github"]) == 1
    (forwarded_batch,) = captured_calls["github"][0]
    assert forwarded_batch is batch
    assert captured_calls["linkedin"] == []


def test_research_packet_dispatch_kwarg_false_still_forwards_to_linkedin(
    captured_calls: dict[str, list[tuple]],
) -> None:
    """``reconstruct_report_analysis=False`` forwards as-is, not coerced."""

    batch = _make_batch("linkedin")

    _dispatch_via_registry(batch, reconstruct_report_analysis=False)

    assert len(captured_calls["linkedin"]) == 1
    _, forwarded_kwarg = captured_calls["linkedin"][0]
    assert forwarded_kwarg is False


@pytest.mark.parametrize("source", ["researcher", "designer", "exec_search"])
def test_research_packet_dispatch_no_op_for_sources_without_a_registered_builder(
    source: str,
    captured_calls: dict[str, list[tuple]],
) -> None:
    """Sources without a registered builder fall through unchanged.

    Pre-slice: the if/elif fell through and returned the batch
    unchanged. Post-slice: ``research_packet_builder_fn`` is ``None``
    for these sources so the dispatch site preserves the fall-through.
    Researcher / Designer / exec_search will gain their own builders
    only when their reflection paths grow research-packet semantics
    (out of scope for Slice 1.3).
    """

    batch = _make_batch(source)

    result = _dispatch_via_registry(batch, reconstruct_report_analysis=True)

    assert result is batch
    assert captured_calls["linkedin"] == []
    assert captured_calls["github"] == []


def test_research_packet_dispatch_no_op_on_unknown_source(
    captured_calls: dict[str, list[tuple]],
) -> None:
    """Unknown source falls through unchanged — preserves legacy ladder semantics.

    Pre-slice: the if/elif fell through silently. Post-slice:
    ``LAUNCHERS.get(batch.source)`` returns ``None`` for unregistered
    sources and the dispatch is skipped. Today this can't happen in
    production (every source emerges from
    ``_load_evidence_batch``'s source-detection logic which always
    picks linkedin or github), but the contract has to survive a
    future source landing without a research-packet builder yet.
    """

    batch = _make_batch("not-a-real-source")

    result = _dispatch_via_registry(batch, reconstruct_report_analysis=True)

    assert result is batch
    assert captured_calls["linkedin"] == []
    assert captured_calls["github"] == []


def test_research_packet_dispatch_routes_multi_source_batch_correctly(
    captured_calls: dict[str, list[tuple]],
) -> None:
    """Walking every registered source produces the same builder calls as the if/elif.

    This is the per-plan ("fixture-based regression: registry dispatch
    produces the same builder calls as the if/elif on a multi-source
    evidence batch") invariant. Mirrors the production scenario where
    ``_load_evidence_batch`` is invoked once per evidence source the
    brief touched (LinkedIn run + GitHub run + future modules). The
    registry-driven dispatch must produce exactly:

    - one ``maybe_build_and_persist_research_packet`` call (for
      linkedin), with ``reconstruct_report_analysis`` forwarded;
    - one ``maybe_build_and_persist_github_research_packet`` call
      (for github);
    - zero calls for researcher / designer / exec_search.
    """

    sources = list(known_sources())
    batches_by_source = {source: _make_batch(source) for source in sources}

    for source in sources:
        _dispatch_via_registry(
            batches_by_source[source],
            reconstruct_report_analysis=True,
        )

    assert len(captured_calls["linkedin"]) == 1, (
        "linkedin builder should fire exactly once for the linkedin batch."
    )
    assert captured_calls["linkedin"][0][0] is batches_by_source["linkedin"]
    assert captured_calls["linkedin"][0][1] is True

    assert len(captured_calls["github"]) == 1, (
        "github builder should fire exactly once for the github batch."
    )
    assert captured_calls["github"][0][0] is batches_by_source["github"]


# ---------------------------------------------------------------------------
# Direct adapter contract — pin the kwarg-forwarding / kwarg-discarding shape.
# ---------------------------------------------------------------------------


def test_linkedin_research_packet_adapter_forwards_kwarg_to_underlying_builder(
    captured_calls: dict[str, list[tuple]],
) -> None:
    """The LinkedIn adapter is a pure pass-through to the underlying builder.

    Slice 1.3's
    :func:`cloris.launchers._linkedin_research_packet_builder` lazy-
    imports :func:`market_intelligence.research_context.maybe_build_and_persist_research_packet`
    and forwards both positional and keyword arguments verbatim.
    """

    batch = _make_batch("linkedin")
    adapter = LAUNCHERS["linkedin"].research_packet_builder_fn
    assert adapter is not None

    adapter(batch, reconstruct_report_analysis=True)

    assert captured_calls["linkedin"] == [(batch, True)]
    assert captured_calls["github"] == []


def test_github_research_packet_adapter_discards_linkedin_specific_kwarg(
    captured_calls: dict[str, list[tuple]],
) -> None:
    """The GitHub adapter accepts and discards ``reconstruct_report_analysis``.

    The underlying
    :func:`market_intelligence.github_reflection.maybe_build_and_persist_github_research_packet`
    takes only ``(batch,)``. The adapter accepts the LinkedIn-specific
    kwarg so the dispatch site can call every source's adapter with a
    uniform signature — but it must not forward the kwarg, or the
    underlying builder would raise ``TypeError`` on the unknown kwarg.
    """

    batch = _make_batch("github")
    adapter = LAUNCHERS["github"].research_packet_builder_fn
    assert adapter is not None

    adapter(batch, reconstruct_report_analysis=True)

    assert captured_calls["github"] == [(batch,)]
    assert captured_calls["linkedin"] == []


def test_research_packet_builder_registered_for_every_known_source() -> None:
    """Slice A.6 (Multi-Agent Production Plan) widened registration to all sources.

    Pre-A.6 the field was populated only for linkedin / github; researcher
    / designer / exec_search registered ``None`` and the dispatch site
    at :func:`market_intelligence.engine._load_evidence_batch` fell
    through to a no-op pass-through. A.6 installs per-module shim
    builders (currently pass-through) on the three formerly-None
    sources so the registry's per-source slot is uniformly populated;
    this lets F.2b ship per-module reflection packet content by
    extending the shim bodies in place rather than touching the
    LauncherEntry rows.

    The shims are pass-through today (byte-equivalent to the prior
    None-fallback at the engine dispatch site), so this widening
    introduces zero behavior change for existing multi-source
    reflection paths. F.2b will replace each shim body with the
    real per-module packet shape (researcher publication-record
    rollups, designer rubric-score distribution, exec_search
    per-signal coverage rate).
    """

    populated = {
        source
        for source, entry in LAUNCHERS.items()
        if entry.research_packet_builder_fn is not None
    }

    assert populated == {
        "linkedin",
        "github",
        "researcher",
        "designer",
        "exec_search",
    }, (
        "research_packet_builder_fn should be populated for every "
        "registered source post-Slice A.6. If you're adding a new "
        "source, register a shim (pass-through is fine until F.2b "
        "ships its packet content)."
    )


# ---------------------------------------------------------------------------
# Drift guard — pin the production dispatch shape against accidental edits.
# ---------------------------------------------------------------------------


def test_engine_module_no_longer_imports_underlying_research_packet_builders() -> None:
    """The engine module's imports drop the underlying builder symbols post-slice.

    Pre-slice: ``engine.py`` imported
    ``maybe_build_and_persist_research_packet`` at module scope and
    lazy-imported ``maybe_build_and_persist_github_research_packet``
    inside the elif. Post-slice: both lookups happen inside the
    registered adapters
    (``cloris.launchers._linkedin_research_packet_builder`` /
    ``_github_research_packet_builder``); engine.py shouldn't
    reference either symbol directly. Catches the regression where a
    rebase reintroduces the legacy ladder alongside the registry
    dispatch (double-fire would persist two packets per LinkedIn run).
    """

    assert not hasattr(engine_mod, "maybe_build_and_persist_research_packet"), (
        "engine.py imported maybe_build_and_persist_research_packet "
        "directly post-slice. The Slice 1.3 dispatch site routes via "
        "LAUNCHERS — drop the import."
    )
    assert not hasattr(engine_mod, "maybe_build_and_persist_github_research_packet"), (
        "engine.py imported maybe_build_and_persist_github_research_packet "
        "directly post-slice. The Slice 1.3 dispatch site routes via "
        "LAUNCHERS — drop the (lazy) import."
    )
