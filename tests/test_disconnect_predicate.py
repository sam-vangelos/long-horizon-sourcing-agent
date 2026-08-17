"""Tests for shared browser-disconnect classification."""

import pytest

from linkedin.acquisition import (
    _BROWSER_DISCONNECT_PATTERNS,
    _is_browser_disconnect_error as acquisition_is_browser_disconnect_error,
)


_DISCONNECT_PATTERNS = (
    "page crashed",
    "target crashed",
    "target closed",
    "target page, context or browser has been closed",
    "has been closed",
    "connection closed",
    "session closed",
    "broken pipe",
    "browser has been closed",
    "page closed",
    "context closed",
    "page.createisolatedworld",
    "page.addscripttoevaluateonnewdocument",
    "cannot get world",
)


def orchestrator_is_browser_disconnect_error(error):
    from linkedin.orchestrator import _is_browser_disconnect_error

    return _is_browser_disconnect_error(error)


def test_disconnect_patterns_are_the_union():
    assert _BROWSER_DISCONNECT_PATTERNS == _DISCONNECT_PATTERNS


@pytest.mark.parametrize("pattern", _DISCONNECT_PATTERNS)
def test_every_disconnect_pattern_matches_both_entry_points(pattern):
    error = RuntimeError(f"Playwright failed: {pattern}")

    assert acquisition_is_browser_disconnect_error(error)
    assert orchestrator_is_browser_disconnect_error(error)


def test_unrelated_error_does_not_match_either_entry_point():
    error = RuntimeError("profile panel did not render")

    assert not acquisition_is_browser_disconnect_error(error)
    assert not orchestrator_is_browser_disconnect_error(error)


def test_render_failure_is_opt_in_for_acquisition():
    from linkedin.orchestrator import PageRenderFailedError

    error = PageRenderFailedError("results page had no reviewable card slots")

    assert not acquisition_is_browser_disconnect_error(error)
    assert acquisition_is_browser_disconnect_error(error, include_render_failures=True)
    assert orchestrator_is_browser_disconnect_error(error)
