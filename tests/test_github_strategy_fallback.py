"""P6.6 — fallback strategy shape fix.

Pins ``github.strategy._default_queries`` location precedence:
``permanent_filters["Location"]`` first, then ``brief.raw["geography"]``,
then no invented location. The pre-fix code iterated
``brief.permanent_filters`` (a dict) as if it were a list of filter
entries — which both misread the dict (a plain-string dict key like
"Location" satisfied ``isinstance(f, str)`` and got appended as the
"location" value itself) and defaulted to a fabricated "Brazil"
whenever the dict was empty.

Deliberately does NOT depend on the optional GitHub FDE brief fixture
under ``config/`` (unlike ``tests/test_github_strategy.py``, which
skips at module level when that fixture is absent) — a minimal
Brief-shaped stub is enough to exercise this helper.
"""

from __future__ import annotations

import pytest

from github.strategy import _default_queries


class _FakeBrief:
    """Minimal Brief-shaped stub — only the attributes `_default_queries`
    reads: `permanent_filters`, `raw`, `target_projects`, `role_title`."""

    def __init__(
        self,
        *,
        permanent_filters: dict | None = None,
        raw: dict | None = None,
        role_title: str = "Platform Engineer",
        target_projects: list[str] | None = None,
    ) -> None:
        self.permanent_filters = dict(permanent_filters or {})
        self.raw = dict(raw or {})
        self.role_title = role_title
        self.target_projects = list(target_projects or [])
        self.target_stacks: list[str] = []


def test_default_queries_honors_explicit_location_filter() -> None:
    """`permanent_filters["Location"]` scopes the role-title user search."""

    queries = _default_queries(
        _FakeBrief(permanent_filters={"Location": "Portugal"}),
    )
    user_search = [q for q in queries if q.channel == "user_search"]
    assert user_search, "expected fallback user_search queries"
    for q in user_search:
        assert "location:Portugal" in q.query
        assert "location:Location" not in q.query
        assert "Platform Engineer" in q.query or "Platform" in q.query


def test_default_queries_falls_back_to_brief_geography() -> None:
    """No `permanent_filters["Location"]` but `raw["geography"]` is set:
    geography comes from the brief, not an invented default."""

    queries = _default_queries(
        _FakeBrief(permanent_filters={}, raw={"geography": "Argentina"}),
    )
    user_search = [q for q in queries if q.channel == "user_search"]
    assert user_search
    for q in user_search:
        assert "location:Argentina" in q.query


def test_default_queries_geography_less_brief_emits_no_invented_country() -> None:
    """Neither an explicit Location filter nor a brief geography: emit
    location-less role-title queries. The pre-fix code invented "Brazil" here."""

    queries = _default_queries(
        _FakeBrief(permanent_filters={}, raw={}, role_title="Systems Engineer"),
    )
    user_search = [q for q in queries if q.channel == "user_search"]
    assert user_search
    for q in user_search:
        assert "Brazil" not in q.query
        assert "Brazil" not in q.name
        assert "location:" not in q.query
    assert any("Systems Engineer" in q.query for q in user_search)


def test_default_queries_prefers_permanent_filter_over_brief_geography() -> None:
    """When both are present, the explicit Location filter wins (it is
    the more specific/authoritative signal)."""

    queries = _default_queries(
        _FakeBrief(
            permanent_filters={"Location": "Portugal"},
            raw={"geography": "Argentina"},
        ),
    )
    user_search = [q for q in queries if q.channel == "user_search"]
    assert user_search
    for q in user_search:
        assert "location:Portugal" in q.query
        assert "Argentina" not in q.query


def test_default_queries_seeds_target_projects_without_vertical_defaults() -> None:
    queries = _default_queries(
        _FakeBrief(
            role_title="",
            target_projects=["kubernetes/kubernetes"],
        ),
    )
    channels = {q.channel for q in queries}
    assert channels == {"repo_mining"}
    assert all(q.target_repo == "kubernetes/kubernetes" for q in queries)


def test_default_queries_empty_when_no_brief_signals() -> None:
    from unittest.mock import patch

    from github.strategy import form_github_strategy

    with patch("github.strategy.opus_llm_cached", side_effect=RuntimeError("llm down")), \
         patch("github.strategy._build_strategy_system", return_value="system"), \
         patch("github.strategy._build_strategy_user", return_value="user"):
        with pytest.raises(
            RuntimeError,
            match="strategy formation failed and no fallback queries are derivable from the brief",
        ):
            form_github_strategy(_FakeBrief(role_title=""))
