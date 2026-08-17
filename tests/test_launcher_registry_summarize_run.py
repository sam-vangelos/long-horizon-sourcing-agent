"""Tests for the multi-agent-execution Phase 1 Slice 1.7 wiring.

Pins the per-source ``summarize_run_fn`` registrations on
``LAUNCHERS`` and the read-only invariant the chief-of-staff agent
(Phase 2.4 synthesis extensions, Phase 2.5 dispatch heuristic) depends
on when it reads run summaries across sources uniformly.

What this file pins (companion to ``tests/test_launcher_registry_completeness.py``,
which pins the ratchet that ``summarize_run_fn`` is populated on every
registered source):

- **Per-source dispatch**. Each ``LAUNCHERS[source].summarize_run_fn``
  routes to the matching ``shared.runtime_state.read_models.summarize_<source>_run``
  helper. Catches the regression where a new source registers the
  wrong helper or the registry is wired to the wrong adapter (e.g.,
  designer accidentally registered with ``summarize_researcher_run``).
- **Behavior on a populated state dir**. Each helper returns the
  latest ``runs`` row as a :class:`RunSummary`, byte-equivalent to a
  direct ``latest_run_summary`` call. The five per-source helpers are
  intentionally thin wrappers today — Slice 1.7's promise is that they
  share the same read shape (per-source elaboration is forward-compat
  surface for Phase 2.4/2.5 per-source summary growth).
- **Behavior on a missing state dir**. Each helper returns the
  default ``RunSummary()`` (every field ``None`` / ``False``) instead
  of raising or creating the SQLite file as a side effect. The
  default-return semantics are load-bearing — callers (Phase 2.4 / 2.5)
  use ``summary.id is None`` to disambiguate "no run yet" from "run
  exists", and an exception here would force per-source try/except in
  every cross-source synthesis call site.
- **Read-only invariant — file-level**. Calling each helper against a
  populated SQLite leaves the file's bytes and mtime unchanged. This
  catches any future change to the underlying helper that swaps
  ``_open_readonly`` for a writable mode (would silently work today
  because the schema is already there, but breaks the invariant the
  next time a writer runs DDL on a stale schema).
- **Read-only invariant — connection-URI**. Monkey-patch
  ``sqlite3.connect`` to capture the URI/mode each helper opens with;
  assert every connection carries ``mode=ro`` and ``uri=True``. This
  mirrors the structural invariant the existing layering test at
  ``tests/test_read_models.py:45`` enforces (no
  ``RuntimeStateStore`` import) — same hostile-writer-on-read-path
  class of bug, caught at the connection-time seam instead of the
  import-time seam.

The ratchet that every source is populated lives in
:mod:`tests.test_launcher_registry_completeness` (specifically
``_POPULATED_PIONEER_CALLABLE_FIELDS``); this file's tests assume that
ratchet has already passed and exercise the per-source dispatch +
invariant contracts.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Callable

import pytest

from cloris.launchers import LAUNCHERS, known_sources
from shared.runtime_state import read_models
from shared.runtime_state.read_models import (
    RunSummary,
    latest_run_summary,
    summarize_designer_run,
    summarize_exec_search_run,
    summarize_github_run,
    summarize_linkedin_run,
    summarize_researcher_run,
)
from shared.runtime_state.store import RuntimeStateStore


# Per-source expected helper identity. Pinning by ``is`` (not name) so
# the test catches a wiring regression where a same-named alias gets
# registered (e.g., re-export via a stub module).
_EXPECTED_SUMMARIZER_BY_SOURCE: dict[str, Callable[[Path], RunSummary]] = {
    "linkedin": summarize_linkedin_run,
    "github": summarize_github_run,
    "researcher": summarize_researcher_run,
    "designer": summarize_designer_run,
    "exec_search": summarize_exec_search_run,
}


def test_every_source_dispatches_to_its_per_source_helper() -> None:
    """Each ``summarize_run_fn`` is the exact per-source helper.

    The chief-of-staff agent calls
    ``LAUNCHERS[source].summarize_run_fn(state_dir)`` for cross-source
    synthesis; the per-source anchor (vs. one shared helper registered
    five times) is the seam Phase 2.4/2.5 will grow source-specific
    summary shape into. Identity check (``is``) catches the wiring
    regression where a refactor "consolidates" the helpers into one
    shared callable and silently loses the per-source seam.
    """

    assert set(_EXPECTED_SUMMARIZER_BY_SOURCE) == set(known_sources()), (
        "Every registered source must have an expected summarizer "
        "in this test's map. New source? Add the per-source helper "
        "in shared/runtime_state/read_models.py and the expectation "
        "here in the same PR."
    )
    for source, expected in _EXPECTED_SUMMARIZER_BY_SOURCE.items():
        registered = LAUNCHERS[source].summarize_run_fn
        assert registered is expected, (
            f"{source!r}.summarize_run_fn should be {expected.__name__} "
            f"(identity match); got {getattr(registered, '__name__', registered)!r}."
        )


def _seed_run(db_path: Path, *, source: str) -> int:
    """Seed one ``runs`` row via the writer; return its id.

    Uses ``RuntimeStateStore`` directly because the read path under
    test must NOT instantiate the writer (that's the whole layering
    rule the read-models module enforces). Test-side seeding is the
    legitimate use of the writer — analogous to ``_seed_runs`` /
    ``_direct_insert_attempt`` in :mod:`tests.test_read_models`.
    """

    store = RuntimeStateStore(db_path)
    run_id = store.start_run(
        source=source,
        brief_id=f"brief-{source}-summarize-test",
        output_dir=str(db_path.parent),
        mode="fresh",
    )
    store.finish_run(run_id, "completed", stop_reason="normal")
    return run_id


@pytest.mark.parametrize("source", sorted(_EXPECTED_SUMMARIZER_BY_SOURCE))
def test_summarize_run_returns_default_on_missing_state_dir(
    source: str, tmp_path: Path
) -> None:
    """Helpers collapse to ``RunSummary()`` when the state dir has no DB.

    Two contracts in one. (1) The dispatch site (Phase 2.4 / 2.5)
    treats ``summary.id is None`` as "no run yet" and a missing state
    dir falls into that bucket; raising here would force per-source
    try/except in every cross-source call site. (2) The read path must
    NOT create the DB file as a side effect — same invariant the
    existing read-models layering test pins for
    :func:`latest_run_summary` at ``tests/test_read_models.py:73``.
    """

    summarizer = LAUNCHERS[source].summarize_run_fn
    assert summarizer is not None  # ratcheted by completeness test
    db_path = tmp_path / "runtime_state.sqlite3"
    assert not db_path.exists()

    summary = summarizer(tmp_path)

    assert summary == RunSummary(), (
        f"{source}: missing state dir should collapse to default "
        f"RunSummary(); got {summary!r}."
    )
    assert not db_path.exists(), (
        f"{source}: summarize_run_fn must not create runtime_state.sqlite3 "
        "as a side effect on a missing-DB state dir."
    )


@pytest.mark.parametrize("source", sorted(_EXPECTED_SUMMARIZER_BY_SOURCE))
def test_summarize_run_returns_latest_run_for_populated_state_dir(
    source: str, tmp_path: Path
) -> None:
    """Helpers return the latest ``runs`` row as a :class:`RunSummary`.

    Byte-equivalent to a direct :func:`latest_run_summary` call. The
    five per-source helpers are intentionally thin wrappers today —
    Slice 1.7's promise is the registry-side dispatch shape, not
    per-source semantics. Phase 2.4/2.5 may grow source-specific
    summary fields; the per-source anchor here is where that growth
    lands without forking the registry contract.
    """

    summarizer = LAUNCHERS[source].summarize_run_fn
    assert summarizer is not None
    db_path = tmp_path / "runtime_state.sqlite3"
    run_id = _seed_run(db_path, source=source)

    summary = summarizer(tmp_path)

    assert summary == latest_run_summary(db_path), (
        f"{source}: per-source helper must agree with latest_run_summary "
        "on the same DB; divergence here means the per-source helper "
        "is doing something other than the documented thin wrapper."
    )
    assert summary.id == run_id
    assert summary.status == "completed"
    assert summary.stop_reason == "normal"
    assert summary.mode == "fresh"


@pytest.mark.parametrize("source", sorted(_EXPECTED_SUMMARIZER_BY_SOURCE))
def test_summarize_run_does_not_mutate_the_sqlite_file(
    source: str, tmp_path: Path
) -> None:
    """Calling the helper leaves the SQLite file's bytes + mtime unchanged.

    File-level invariant for the no-DDL / no-INSERT contract. A future
    change that swaps ``_open_readonly`` for a writable connection
    would still pass the behavior tests above (the schema is already
    there) but would silently re-introduce the hazard the read path
    exists to prevent. Bytes + mtime is a coarse proxy that catches
    any write-mode open even if the underlying SQL is read-only —
    SQLite touches the WAL / shm files on a writable open.
    """

    summarizer = LAUNCHERS[source].summarize_run_fn
    assert summarizer is not None
    db_path = tmp_path / "runtime_state.sqlite3"
    _seed_run(db_path, source=source)

    pre_bytes = db_path.read_bytes()
    pre_mtime_ns = db_path.stat().st_mtime_ns

    for _ in range(3):
        summarizer(tmp_path)

    assert db_path.read_bytes() == pre_bytes, (
        f"{source}: summarize_run_fn changed the bytes of "
        "runtime_state.sqlite3. Read path must not run DDL / INSERT."
    )
    assert db_path.stat().st_mtime_ns == pre_mtime_ns, (
        f"{source}: summarize_run_fn touched the mtime of "
        "runtime_state.sqlite3. SQLite only updates mtime on writable "
        "opens — this strongly suggests a non-``mode=ro`` connection."
    )


@pytest.mark.parametrize("source", sorted(_EXPECTED_SUMMARIZER_BY_SOURCE))
def test_summarize_run_opens_sqlite_in_readonly_uri_mode(
    source: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every connection the helper opens carries ``mode=ro`` + ``uri=True``.

    Connection-time invariant — strongest of the read-only checks.
    Catches a regression where a future helper "optimizes" past
    :func:`_open_readonly` and opens the file directly (e.g., via
    ``sqlite3.connect(str(db_path))``), which would silently work for
    pure-SELECT code paths but break the invariant the chief-of-staff
    agent (Phase 2.4 / 2.5) depends on for cross-source reads against
    state dirs that other workers may be actively writing.

    Mirrors the structural invariant the existing layering test at
    ``tests/test_read_models.py:45`` enforces (no
    ``RuntimeStateStore`` import). Same hostile-writer-on-read-path
    bug class, caught at the connection-time seam.
    """

    summarizer = LAUNCHERS[source].summarize_run_fn
    assert summarizer is not None
    db_path = tmp_path / "runtime_state.sqlite3"
    _seed_run(db_path, source=source)

    seen_calls: list[tuple[tuple, dict]] = []
    real_connect = sqlite3.connect

    def recording_connect(*args, **kwargs):
        seen_calls.append((args, dict(kwargs)))
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(read_models.sqlite3, "connect", recording_connect)

    summary = summarizer(tmp_path)
    assert summary.id is not None  # confirms the helper actually read

    assert seen_calls, (
        f"{source}: summarize_run_fn opened no SQLite connection on a "
        "populated state dir — the read can't have come from canonical "
        "state. (If a per-source helper grows a non-SQLite read path, "
        "extend this test rather than skip it.)"
    )
    for args, kwargs in seen_calls:
        target = args[0] if args else kwargs.get("database", "")
        assert isinstance(target, str), (
            f"{source}: connection target should be a string URI; "
            f"got {type(target).__name__} ({target!r})."
        )
        assert "mode=ro" in target, (
            f"{source}: SQLite connection opened without ``mode=ro``. "
            f"URI was {target!r}. Read path must funnel through "
            "shared.runtime_state.read_models._open_readonly."
        )
        assert kwargs.get("uri") is True, (
            f"{source}: SQLite connection opened without ``uri=True``; "
            "the ``mode=ro`` flag is silently ignored without it. "
            f"kwargs were {kwargs!r}."
        )
