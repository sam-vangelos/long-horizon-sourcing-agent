"""Tests for multi-agent-execution Phase 1 Slice 1.1 + Phase 2.2 —
registry-driven launch-readiness dispatch.

Pre-1.1, ``cloris/api.py:_readiness_blockers`` branched on ``source`` to
call :func:`linkedin.health.probe_linkedin_readiness` /
:func:`github.health.probe_github_readiness` / fall through to
``report = None``. Slice 1.1 routed linkedin / github through the
registry (``LAUNCHERS[source].readiness_probe_fn``) with a None-
fallback for sources that hadn't shipped a probe yet. Phase 2.2
shipped the remaining three probes — :mod:`researcher.health`,
:mod:`designer.health`, :mod:`exec_search.health` — and registered
them on their launcher entries. The dispatch site retains the None-
fallback defensively for the unknown-source path
(``LAUNCHERS.get(source)`` returns ``None``).

What this test pins:

- The ``readiness_probe_fn`` field shape on ``LAUNCHERS`` entries:
  populated for every registered source (linkedin / github post-1.1;
  researcher / designer / exec_search post-2.2).
- Registry dispatch matches the prior if-elif behavior on a fixture
  brief: stubbed probes flow through to the aggregator's blockers, and
  the per-source probes do not cross-call each other.
- Late-binding via inline imports inside the registered wrappers — the
  existing endpoint tests at ``tests/test_launch_readiness_endpoint.py``
  rely on ``monkeypatch.setattr(linkedin.health, "probe_linkedin_readiness",
  ...)`` to stub the probe, which only works if the registry resolves
  the symbol at call time (capturing the function object at import
  time would silently bypass the fixture).
"""

from __future__ import annotations

import pytest

import designer.health as designer_health
import exec_search.health as exec_search_health
import github.health as github_health
import linkedin.health as linkedin_health
import researcher.health as researcher_health
from cloris.api import _readiness_blockers
from cloris.launchers import LAUNCHERS, known_sources


_LINKEDIN_BLOCKER = linkedin_health.ReadinessBlocker(
    kind="net",
    message="Cloris can't reach Chrome over CDP.",
    remediation="Run ./launch-chrome.sh --force, open linkedin.com/talent.",
)

_GITHUB_BLOCKER = github_health.ReadinessBlocker(
    kind="config",
    message="No GitHub token configured.",
    remediation="Add GITHUB_TOKEN to your .env file.",
)


def _stub_linkedin_blocked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        linkedin_health,
        "probe_linkedin_readiness",
        lambda **_kwargs: linkedin_health.ReadinessReport(
            ready=False, blockers=(_LINKEDIN_BLOCKER,)
        ),
    )


def _stub_linkedin_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        linkedin_health,
        "probe_linkedin_readiness",
        lambda **_kwargs: linkedin_health.ReadinessReport(
            ready=True, blockers=()
        ),
    )


def _stub_github_blocked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        github_health,
        "probe_github_readiness",
        lambda **_kwargs: github_health.ReadinessReport(
            ready=False, blockers=(_GITHUB_BLOCKER,)
        ),
    )


def _stub_github_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        github_health,
        "probe_github_readiness",
        lambda **_kwargs: github_health.ReadinessReport(
            ready=True, blockers=()
        ),
    )


# ---------------------------------------------------------------------------
# Field-shape contract
# ---------------------------------------------------------------------------


def test_every_source_registers_readiness_probe_fn() -> None:
    """Slice 1.1 + Phase 2.2 populate readiness on every source.

    Slice 1.1 shipped linkedin + github; Phase 2.2 shipped researcher
    + designer + exec_search via :mod:`researcher.health` /
    :mod:`designer.health` / :mod:`exec_search.health`. Every
    registered source must supply a non-None callable so the
    dispatch aggregator surfaces real blockers uniformly across the
    five modules.
    """

    for source in known_sources():
        entry = LAUNCHERS[source]
        assert entry.readiness_probe_fn is not None, (
            f"{source!r}.readiness_probe_fn must be populated; got None. "
            "Phase 2.2 wired researcher / designer / exec_search; if a "
            "new source was added without a probe, register one in the "
            "same PR or document why none is required."
        )
        assert callable(entry.readiness_probe_fn), (
            f"{source!r}.readiness_probe_fn must be callable; "
            f"got {type(entry.readiness_probe_fn).__name__}."
        )


def test_readiness_probe_fn_present_on_every_registered_source() -> None:
    """Field declaration covers every source, populated or not.

    Slice 1.0 declared ``readiness_probe_fn`` on ``LauncherEntry``; if
    ``hasattr`` ever fails, the dispatch site at
    ``_readiness_blockers`` will raise ``AttributeError`` instead of
    falling through to ``report = None``.
    """

    for source in known_sources():
        entry = LAUNCHERS[source]
        assert hasattr(entry, "readiness_probe_fn"), (
            f"{source!r} entry missing readiness_probe_fn declaration; "
            "Slice 1.0 should have declared it on the dataclass."
        )


# ---------------------------------------------------------------------------
# Dispatch behavior — matches prior if-elif ladder
# ---------------------------------------------------------------------------


def test_dispatch_routes_linkedin_through_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LinkedIn aggregator surfaces the registered probe's blockers.

    Mirrors the prior ``if source == "linkedin": probe_linkedin_readiness()``
    branch byte-for-byte. The ``brief_id`` is opaque here — Layer 2
    (per-brief save destination) only fires on a resolvable brief, and
    ``"nonexistent-brief-id"`` will not resolve.
    """

    _stub_linkedin_blocked(monkeypatch)
    _stub_github_ready(monkeypatch)

    blockers = _readiness_blockers("linkedin", "nonexistent-brief-id")

    assert _LINKEDIN_BLOCKER in blockers, (
        "Registry dispatch lost the LinkedIn probe's blocker."
    )


def test_dispatch_routes_github_through_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GitHub aggregator surfaces the registered probe's blockers.

    Mirrors the prior ``elif source == "github": probe_github_readiness()``
    branch byte-for-byte.
    """

    _stub_linkedin_ready(monkeypatch)
    _stub_github_blocked(monkeypatch)

    blockers = _readiness_blockers("github", "nonexistent-brief-id")

    assert _GITHUB_BLOCKER in blockers, (
        "Registry dispatch lost the GitHub probe's blocker."
    )


def test_dispatch_does_not_cross_call_probes_across_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Per-source dispatch never picks up another source's blockers.

    Stub the linkedin / github probes to fail so a regression that
    accidentally routed through them would surface as unexpected
    blockers on the other sources. The researcher / designer /
    exec_search probes (Phase 2.2) report their own blockers — those
    come from :mod:`researcher.health` etc. and are tested in their
    own per-module test files. The contract pinned here is exclusively
    "no cross-call".
    """

    _stub_linkedin_blocked(monkeypatch)
    _stub_github_blocked(monkeypatch)

    for source in ("researcher", "designer", "exec_search"):
        blockers = _readiness_blockers(source, "nonexistent-brief-id")
        assert _LINKEDIN_BLOCKER not in blockers, (
            f"{source!r} aggregator picked up a LinkedIn blocker — "
            "registry dispatch must not cross-call probes."
        )
        assert _GITHUB_BLOCKER not in blockers, (
            f"{source!r} aggregator picked up a GitHub blocker — "
            "registry dispatch must not cross-call probes."
        )


def test_dispatch_handles_unknown_source_with_none_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unknown sources fall through cleanly (defensive — caller already gates).

    The pre-1.1 ladder's ``else: report = None`` branch covered both
    "registered source without a probe" and "unknown source". The
    registry dispatch uses ``LAUNCHERS.get(source)`` so an unknown
    source still surfaces as ``probe_fn = None`` rather than raising
    ``KeyError``. The launch handler at ``_launch_for_source_impl``
    already raises ``UnknownSourceError`` before reaching this
    aggregator, but the safety net stays in place.
    """

    _stub_linkedin_blocked(monkeypatch)
    _stub_github_blocked(monkeypatch)

    blockers = _readiness_blockers(
        "nonexistent_source_for_test_only", "nonexistent-brief-id"
    )

    assert blockers == [], (
        "Unknown source should produce zero layer-1 blockers via the "
        "None-fallback (caller is responsible for unknown_source 422s)."
    )


# ---------------------------------------------------------------------------
# Late binding — monkeypatch fixtures still work via inline imports
# ---------------------------------------------------------------------------


def test_registered_probes_are_late_bound_for_monkeypatch_compat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The wrapper functions resolve the underlying probe at call time.

    ``tests/test_launch_readiness_endpoint.py`` and
    ``tests/test_save_destination_config.py`` stub the probes with
    ``monkeypatch.setattr(linkedin.health, "probe_linkedin_readiness", ...)``
    AFTER the registry has been constructed. If the registered callable
    captured the original function object at module-import time (e.g.,
    ``readiness_probe_fn=probe_linkedin_readiness``), the monkeypatch
    would have no effect — the registry would still call the real
    probe, and those tests would fail in subtle ways.

    This test pins the wrapper-with-inline-import contract: the
    callable must observe the monkeypatched value when invoked.
    """

    sentinel_blockers: tuple[linkedin_health.ReadinessBlocker, ...] = (
        linkedin_health.ReadinessBlocker(
            kind="auth",
            message="sentinel-linkedin",
            remediation="sentinel-remediation",
        ),
    )
    monkeypatch.setattr(
        linkedin_health,
        "probe_linkedin_readiness",
        lambda **_kwargs: linkedin_health.ReadinessReport(
            ready=False, blockers=sentinel_blockers
        ),
    )

    fn = LAUNCHERS["linkedin"].readiness_probe_fn
    assert fn is not None
    report = fn()
    assert report.blockers == sentinel_blockers, (
        "Registered linkedin probe didn't observe the monkeypatch — "
        "registry must use late-binding (inline import inside the "
        "wrapper), not capture the function reference at import time."
    )

    sentinel_github: tuple[github_health.ReadinessBlocker, ...] = (
        github_health.ReadinessBlocker(
            kind="config",
            message="sentinel-github",
            remediation="sentinel-remediation",
        ),
    )
    monkeypatch.setattr(
        github_health,
        "probe_github_readiness",
        lambda **_kwargs: github_health.ReadinessReport(
            ready=False, blockers=sentinel_github
        ),
    )

    fn = LAUNCHERS["github"].readiness_probe_fn
    assert fn is not None
    report = fn()
    assert report.blockers == sentinel_github, (
        "Registered github probe didn't observe the monkeypatch — "
        "registry must use late-binding (inline import inside the "
        "wrapper), not capture the function reference at import time."
    )


def test_phase_2_2_probes_are_late_bound_for_monkeypatch_compat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Researcher / designer / exec_search probes use late-binding too.

    Mirrors the contract pinned for linkedin / github in
    :func:`test_registered_probes_are_late_bound_for_monkeypatch_compat`.
    Phase 2.2 added per-module ``probe_<source>_readiness`` functions
    and registered each via an inline-import shim
    (``_researcher_readiness_probe`` / ``_designer_readiness_probe``
    / ``_exec_search_readiness_probe``); a future regression that
    captured the function reference at import time would silently
    bypass any test fixture or hot-reload patching the underlying
    module.
    """

    sentinel_researcher: tuple[researcher_health.ReadinessBlocker, ...] = (
        researcher_health.ReadinessBlocker(
            kind="config",
            message="sentinel-researcher",
            remediation="sentinel-remediation",
        ),
    )
    monkeypatch.setattr(
        researcher_health,
        "probe_researcher_readiness",
        lambda **_kwargs: researcher_health.ReadinessReport(
            ready=False, blockers=sentinel_researcher
        ),
    )
    fn = LAUNCHERS["researcher"].readiness_probe_fn
    assert fn is not None
    assert fn().blockers == sentinel_researcher, (
        "Registered researcher probe didn't observe the monkeypatch — "
        "registry must use late-binding (inline import inside the "
        "wrapper), not capture the function reference at import time."
    )

    sentinel_designer: tuple[designer_health.ReadinessBlocker, ...] = (
        designer_health.ReadinessBlocker(
            kind="config",
            message="sentinel-designer",
            remediation="sentinel-remediation",
        ),
    )
    monkeypatch.setattr(
        designer_health,
        "probe_designer_readiness",
        lambda **_kwargs: designer_health.ReadinessReport(
            ready=False, blockers=sentinel_designer
        ),
    )
    fn = LAUNCHERS["designer"].readiness_probe_fn
    assert fn is not None
    assert fn().blockers == sentinel_designer, (
        "Registered designer probe didn't observe the monkeypatch — "
        "registry must use late-binding (inline import inside the "
        "wrapper), not capture the function reference at import time."
    )

    sentinel_exec_search: tuple[exec_search_health.ReadinessBlocker, ...] = (
        exec_search_health.ReadinessBlocker(
            kind="config",
            message="sentinel-exec-search",
            remediation="sentinel-remediation",
        ),
    )
    monkeypatch.setattr(
        exec_search_health,
        "probe_exec_search_readiness",
        lambda **_kwargs: exec_search_health.ReadinessReport(
            ready=False, blockers=sentinel_exec_search
        ),
    )
    fn = LAUNCHERS["exec_search"].readiness_probe_fn
    assert fn is not None
    assert fn().blockers == sentinel_exec_search, (
        "Registered exec_search probe didn't observe the monkeypatch — "
        "registry must use late-binding (inline import inside the "
        "wrapper), not capture the function reference at import time."
    )
