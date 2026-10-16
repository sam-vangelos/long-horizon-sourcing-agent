"""Tests for :mod:`github.project_quality` (OSS Maintainers Slice 5).

Per spec §13: fixture-based scoring on representative inputs. Tests
mock the GitHub client + network_dependents fetch so the score
function is the unit under test. OSSF lookup is monkeypatched with
synthetic scores where a specific value matters (P6.7: the shipped
snapshot ships schema-only, with the fabricated rows deleted — tests
must not depend on production snapshot contents).
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from github import project_quality as pq
from github.client import GitHubClient


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from github import maintainer_signal_cache as mcache

    monkeypatch.setattr(mcache, "CACHE_ROOT", tmp_path / "cache")


def _iso(dt: datetime) -> str:
    return dt.isoformat().replace("+00:00", "Z")


def _make_client(
    *,
    repo: dict | None = None,
    contributors: list[dict] | None = None,
    releases: list[dict] | None = None,
    contributors_fail: bool = False,
    releases_fail: bool = False,
) -> GitHubClient:
    client = GitHubClient(token="dummy")
    client.get_repo = AsyncMock(return_value=repo)
    if contributors_fail:
        client.get_repo_contributors = AsyncMock(side_effect=RuntimeError("contributors down"))
    else:
        client.get_repo_contributors = AsyncMock(return_value=contributors or [])
    if releases_fail:
        client.list_repo_releases = AsyncMock(side_effect=RuntimeError("releases down"))
    else:
        client.list_repo_releases = AsyncMock(return_value=releases or [])
    return client


# ---------------------------------------------------------------------------
# _log_scale + _band_for unit tests
# ---------------------------------------------------------------------------


class TestLogScale:
    def test_zero_value_returns_zero(self) -> None:
        assert pq._log_scale(0, max_log=3.0) == 0.0

    def test_negative_returns_zero(self) -> None:
        assert pq._log_scale(-1, max_log=3.0) == 0.0

    def test_saturates_at_max_log(self) -> None:
        # 10^3 = 1000 → score 1.0
        assert pq._log_scale(1000, max_log=3.0) == 1.0
        # 10x more saturates at 1.0
        assert pq._log_scale(100_000, max_log=3.0) == 1.0

    def test_mid_range(self) -> None:
        # log10(100) / 3 = 2/3 ≈ 0.667
        assert pq._log_scale(100, max_log=3.0) == pytest.approx(0.667, abs=0.01)


class TestBandFor:
    def test_critical_band(self) -> None:
        assert pq._band_for(0.9) == "critical"
        assert pq._band_for(0.66) == "critical"

    def test_established_band(self) -> None:
        assert pq._band_for(0.5) == "established"
        assert pq._band_for(0.33) == "established"

    def test_niche_band(self) -> None:
        assert pq._band_for(0.1) == "niche"
        assert pq._band_for(0.0) == "niche"


# ---------------------------------------------------------------------------
# score_project — high-criticality known anchor
# ---------------------------------------------------------------------------


def test_high_criticality_anchor_scores_critical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """kubernetes/kubernetes has OSSF + dependents + many contributors."""

    now = datetime.now(timezone.utc)
    repo = {
        "created_at": _iso(now - timedelta(days=365 * 9)),  # 9 years old
        "pushed_at": _iso(now - timedelta(days=2)),  # recent push
    }
    contributors = [{"login": f"contrib_{i}"} for i in range(450)]
    # Releases on a regular cadence.
    releases = [
        {"tag_name": f"v{i}", "published_at": _iso(now - timedelta(days=30 * i))}
        for i in range(1, 11)
    ]

    client = _make_client(repo=repo, contributors=contributors, releases=releases)
    monkeypatch.setattr(
        "github.network_dependents.fetch_dependents_count",
        AsyncMock(return_value=120_000),
    )
    # P6.7: the shipped OSSF snapshot ships schema-only (no fabricated
    # rows) — synthesize the "known high-criticality anchor" score
    # this test is actually about, rather than depending on production
    # snapshot contents.
    monkeypatch.setattr(pq, "lookup_criticality_score", lambda owner, repo: 0.99432)

    result = asyncio.run(pq.score_project("kubernetes", "kubernetes", client))

    assert result.score >= 0.66
    assert result.criticality_band == "critical"
    assert "ossf_criticality_raw" in result.signals
    assert result.signals["ossf_criticality_raw"] > 0.9
    assert result.signals["downstream_dependents_raw"] == 120_000
    assert result.signals["contributor_diversity_raw"] == 450


def test_niche_unknown_project_scores_low(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unknown project with no signal returns an unknown band."""

    client = _make_client(
        repo=None,
        contributors=None,
        releases=None,
        contributors_fail=True,
        releases_fail=True,
    )
    monkeypatch.setattr(
        "github.network_dependents.fetch_dependents_count",
        AsyncMock(return_value=None),
    )

    result = asyncio.run(pq.score_project("nobody-org", "nobody-repo", client))

    assert result.score is None
    assert result.criticality_band == "unknown"
    assert result.criticality_band != "niche"
    # Several signals are unavailable.
    unavailable = result.signals.get("unavailable", [])
    assert "ossf_criticality" in unavailable
    assert "downstream_dependents" in unavailable
    assert "age_x_activity" in unavailable
    assert "downstream_dependents" not in result.signals
    assert "ossf_criticality" not in result.signals


def test_irregular_release_cadence_scores_low(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Burst-then-silence releases score lower than regular cadence."""

    now = datetime.now(timezone.utc)
    # 5 releases in one week, then a 2-year gap, then 1 more.
    bursty = [
        {"tag_name": f"v{i}", "published_at": _iso(now - timedelta(days=730 + i))}
        for i in range(5)
    ]
    bursty.append({"tag_name": "v6", "published_at": _iso(now - timedelta(days=10))})

    client = _make_client(
        repo={
            "created_at": _iso(now - timedelta(days=365 * 3)),
            "pushed_at": _iso(now - timedelta(days=10)),
        },
        contributors=[{"login": "alice"}],
        releases=bursty,
    )
    monkeypatch.setattr(
        "github.network_dependents.fetch_dependents_count",
        AsyncMock(return_value=None),
    )

    result = asyncio.run(pq.score_project("bursty-org", "bursty-repo", client))

    # Cadence signal exists but is low (high coefficient of variation).
    assert "release_cadence" in result.signals
    assert result.signals["release_cadence"] < 0.5


def test_age_x_activity_recent_push_scores_high(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime.now(timezone.utc)
    client = _make_client(
        repo={
            "created_at": _iso(now - timedelta(days=365 * 5)),  # 5 years
            "pushed_at": _iso(now - timedelta(days=1)),  # pushed yesterday
        },
        contributors=[{"login": "alice"}],
        releases=[],
    )
    monkeypatch.setattr(
        "github.network_dependents.fetch_dependents_count",
        AsyncMock(return_value=None),
    )

    result = asyncio.run(pq.score_project("active-org", "active-repo", client))

    # 5/10 age * 1.0 activity = 0.5
    assert result.signals["age_x_activity"] == pytest.approx(0.5, abs=0.05)


def test_age_x_activity_stale_repo_scores_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime.now(timezone.utc)
    client = _make_client(
        repo={
            "created_at": _iso(now - timedelta(days=365 * 8)),  # 8 years
            "pushed_at": _iso(now - timedelta(days=400)),  # 400 days ago
        },
        contributors=[],
        releases=[],
    )
    monkeypatch.setattr(
        "github.network_dependents.fetch_dependents_count",
        AsyncMock(return_value=None),
    )

    result = asyncio.run(pq.score_project("stale-org", "stale-repo", client))

    # Activity is zero; product is zero regardless of age.
    assert result.signals["age_x_activity"] == 0.0


def test_unavailable_signals_normalize_against_available_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A project with only OSSF + dependents data scores fairly.

    Spec contract: missing signals contribute zero; the composite
    normalizes against the SUM of available weights, not the total.
    Without this discipline a project with only 2/5 signals would
    cap at ~0.4 even with both signals at max.
    """

    client = _make_client(
        repo=None,
        contributors=None,
        releases=None,
        contributors_fail=True,
        releases_fail=True,
    )
    monkeypatch.setattr(
        "github.network_dependents.fetch_dependents_count",
        AsyncMock(return_value=10_000),
    )
    # P6.7: synthesize the OSSF score directly — the shipped snapshot
    # ships schema-only, so production data can't be relied on here.
    monkeypatch.setattr(pq, "lookup_criticality_score", lambda owner, repo: 0.97845)

    result = asyncio.run(pq.score_project("rust-lang", "rust", client))

    # Two signals available (OSSF + dependents); the composite
    # weighted average against THOSE two should land in the
    # critical band.
    assert result.criticality_band in ("established", "critical")
    assert "ossf_criticality" not in result.signals.get("unavailable", [])
    assert "downstream_dependents" not in result.signals.get("unavailable", [])


# ---------------------------------------------------------------------------
# P6.7 — OSSF absence degrades visibly, not silently
# ---------------------------------------------------------------------------


def test_ossf_absence_degrades_visibly(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A project absent from the OSSF snapshot (the ordinary case now
    that the shipped snapshot ships schema-only per P6.7) must degrade
    VISIBLY: a log line naming the owner/repo, plus a
    ``signals["ossf_criticality_note"]`` a report layer can surface —
    not silence. Exercises the real (now-empty) shipped snapshot, no
    monkeypatch of the lookup itself.
    """

    client = _make_client(
        repo=None,
        contributors=None,
        releases=None,
        contributors_fail=True,
        releases_fail=True,
    )
    monkeypatch.setattr(
        "github.network_dependents.fetch_dependents_count",
        AsyncMock(return_value=None),
    )

    with caplog.at_level("INFO", logger="github.project_quality"):
        result = asyncio.run(
            pq.score_project("some-unlisted-org", "some-unlisted-repo", client)
        )

    assert "ossf_criticality" in result.signals.get("unavailable", [])
    note = result.signals.get("ossf_criticality_note")
    assert note, "expected a report-visible note explaining the OSSF drop"
    assert "some-unlisted-org/some-unlisted-repo" in note

    matching_logs = [
        r
        for r in caplog.records
        if "ossf_criticality" in r.getMessage()
        or "OSSF" in r.getMessage()
    ]
    assert matching_logs, (
        "expected a log line naming the OSSF drop; degradation must not "
        "be silent"
    )


# ---------------------------------------------------------------------------
# W2-A3 — unknown ≠ poor; absent signals; resolver dependents
# ---------------------------------------------------------------------------


def test_unknown_is_not_scored_as_poor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All signals unavailable → unknown band, not niche."""

    client = _make_client(
        repo=None,
        contributors=None,
        releases=None,
        contributors_fail=True,
        releases_fail=True,
    )
    monkeypatch.setattr(
        "github.network_dependents.fetch_dependents_count",
        AsyncMock(return_value=None),
    )

    result = asyncio.run(pq.score_project("ghost-org", "ghost-repo", client))

    assert result.score is None
    assert result.criticality_band == "unknown"
    assert result.criticality_band != "niche"
    assert len(result.signals.get("unavailable", [])) == len(pq.SIGNAL_WEIGHTS)
    assert result.to_dict()["score"] is None
    assert result.to_dict()["criticality_band"] == "unknown"


def test_partial_availability_ignores_absent_signals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One strong present signal + four unavailable is not zero-diluted."""

    client = _make_client(
        repo=None,
        contributors=None,
        releases=None,
        contributors_fail=True,
        releases_fail=True,
    )
    monkeypatch.setattr(
        "github.network_dependents.fetch_dependents_count",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(pq, "lookup_criticality_score", lambda owner, repo: 0.95)

    result = asyncio.run(pq.score_project("solo-signal-org", "solo-signal-repo", client))

    assert result.score is not None
    assert result.score == pytest.approx(0.95, abs=0.001)
    assert result.criticality_band == "critical"
    assert "ossf_criticality" in result.signals
    assert "ossf_criticality" not in result.signals.get("unavailable", [])
    assert "downstream_dependents" not in result.signals
    assert "contributor_diversity" not in result.signals


def test_dependents_from_resolver_fixture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolver supplies dependents; failure falls through to network scrape."""

    client = _make_client(
        repo=None,
        contributors=None,
        releases=None,
        contributors_fail=True,
        releases_fail=True,
    )

    resolver = AsyncMock()
    resolver.resolve_repo_packages = AsyncMock(
        return_value=[
            {"registry": "npm", "name": "lodash", "downloads": 50_000_000},
            {"registry": "npm", "name": "lodash-es", "downloads": 1_000_000},
        ]
    )
    resolver.reverse_dependency_count = AsyncMock(return_value=12_345)

    result = asyncio.run(
        pq.score_project(
            "lodash",
            "lodash",
            client,
            ecosystems_resolver=resolver,
        )
    )

    assert result.signals["downstream_dependents_raw"] == 12_345
    assert result.signals["downstream_dependents_source"] == "resolver"
    assert "downstream_dependents" in result.signals
    assert "downstream_dependents" not in result.signals.get("unavailable", [])
    resolver.resolve_repo_packages.assert_awaited_once_with(
        "https://github.com/lodash/lodash"
    )
    resolver.reverse_dependency_count.assert_awaited_once_with("npm", "lodash")

    resolver_fail = AsyncMock()
    resolver_fail.resolve_repo_packages = AsyncMock(return_value=[])
    network_fetch = AsyncMock(return_value=7_500)
    monkeypatch.setattr(
        "github.network_dependents.fetch_dependents_count",
        network_fetch,
    )

    fallback = asyncio.run(
        pq.score_project(
            "fallback-org",
            "fallback-repo",
            client,
            ecosystems_resolver=resolver_fail,
        )
    )

    assert fallback.signals["downstream_dependents_raw"] == 7_500
    assert fallback.signals["downstream_dependents_source"] == "network"
    network_fetch.assert_awaited_once_with(
        "fallback-org", "fallback-repo", throttle_seconds=1.0
    )


def test_scrape_not_called_when_resolver_answers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolver-first ordering must short-circuit before the HTML scrape."""

    client = _make_client(
        repo=None,
        contributors=None,
        releases=None,
        contributors_fail=True,
        releases_fail=True,
    )

    resolver = AsyncMock()
    resolver.resolve_repo_packages = AsyncMock(
        return_value=[{"registry": "npm", "name": "pkg", "downloads": 1}]
    )
    resolver.reverse_dependency_count = AsyncMock(return_value=42)

    network_fetch = AsyncMock(return_value=99_999)
    monkeypatch.setattr(
        "github.network_dependents.fetch_dependents_count",
        network_fetch,
    )

    result = asyncio.run(
        pq.score_project(
            "resolver-org",
            "resolver-repo",
            client,
            ecosystems_resolver=resolver,
        )
    )

    assert result.signals["downstream_dependents_raw"] == 42
    assert result.signals["downstream_dependents_source"] == "resolver"
    network_fetch.assert_not_awaited()


def test_scrape_fallback_is_throttled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HTML scrape fallback must pass a nonzero throttle_seconds."""

    client = _make_client(
        repo=None,
        contributors=None,
        releases=None,
        contributors_fail=True,
        releases_fail=True,
    )

    network_fetch = AsyncMock(return_value=500)
    monkeypatch.setattr(
        "github.network_dependents.fetch_dependents_count",
        network_fetch,
    )

    asyncio.run(pq.score_project("throttle-org", "throttle-repo", client))

    network_fetch.assert_awaited_once_with(
        "throttle-org", "throttle-repo", throttle_seconds=1.0
    )
