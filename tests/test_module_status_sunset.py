"""Tests for the P7.1 sunset marker on ``GET /api/status``'s ``modules`` field.

Reopen P7.1 (spec §8) set ``designer``/``exec_search`` to
``launchable=False, sunset=True`` in the launcher registry
(``cloris/launchers/__init__.py:LAUNCHERS``) and gates every launch attempt
with a 409 at the single spawn choke point. The registry also promised "a
sunset:true marker the frontend can render as 'Paused for now'" — these
tests pin that the marker actually flows from the registry into
:func:`cloris.control_plane._module_statuses`'s payload, and that a
registry ``launchable=False`` forces the payload ``launchable`` false
regardless of ``pipeline_state``/trial visibility.
"""

from __future__ import annotations

import dataclasses

from cloris.control_plane import _module_statuses
from cloris.launchers import LAUNCHERS
from shared import config


def test_sunset_modules_carry_sunset_true_and_unlaunchable(monkeypatch) -> None:
    monkeypatch.setattr(config, "CLORIS_TRIAL_MODE", False)
    statuses = {m.source: m for m in _module_statuses()}

    for source in ("designer", "exec_search"):
        assert LAUNCHERS[source].sunset is True, (
            f"test assumption broken: {source} is no longer sunset in the registry"
        )
        status = statuses[source]
        assert status.sunset is True, f"{source} should carry sunset=True on the wire"
        assert status.launchable is False, (
            f"{source} is administratively retired; launchable must be False"
        )


def test_non_sunset_modules_carry_sunset_false() -> None:
    statuses = {m.source: m for m in _module_statuses()}

    for source in ("linkedin", "github"):
        assert LAUNCHERS[source].sunset is False, (
            f"test assumption broken: {source} is unexpectedly sunset in the registry"
        )
        assert statuses[source].sunset is False, (
            f"{source} is not sunset; the payload must not claim otherwise"
        )


def test_linkedin_launchable_unaffected_by_sunset_gate(monkeypatch) -> None:
    monkeypatch.setattr(config, "CLORIS_TRIAL_MODE", False)
    statuses = {m.source: m for m in _module_statuses()}
    # LinkedIn is production + registry-launchable — the new
    # ``launcher.launchable`` AND-gate must not regress its existing
    # launchable=True contract.
    assert statuses["linkedin"].launchable is True


def test_registry_launchable_false_overrides_production_pipeline_state(monkeypatch) -> None:
    """Registry ``launchable=False`` must win even if pipeline_state is
    "production" — the payload must never promise a launch the single
    spawn choke point will refuse with a 409."""
    monkeypatch.setattr(config, "CLORIS_TRIAL_MODE", False)
    # ``LauncherEntry`` is a frozen dataclass — swap the whole registry
    # entry for a copy with ``pipeline_state`` bumped, via
    # ``monkeypatch.setitem`` so the dict mutation auto-reverts.
    promoted = dataclasses.replace(LAUNCHERS["designer"], pipeline_state="production")
    monkeypatch.setitem(LAUNCHERS, "designer", promoted)
    statuses = {m.source: m for m in _module_statuses()}
    assert statuses["designer"].pipeline_state == "production"
    assert statuses["designer"].launchable is False, (
        "administrative launchable=False must override pipeline_state=production"
    )
