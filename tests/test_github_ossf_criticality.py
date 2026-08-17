"""Tests for :mod:`github.ossf_criticality` (OSS Maintainers Slice 5).

Covers:

- Snapshot lookup hits known entries and is case-insensitive.
- Missing keys return ``None``.
- Malformed CSV rows are skipped without raising.
- Missing snapshot file returns an empty cache (fail-soft).
- P6.7: the shipped snapshot at ``data/ossf_criticality_snapshot.csv``
  ships with schema-only (header + ``repo,score`` column row, zero
  data rows) — the fabricated "illustrative ordering" scores that
  used to live there have been deleted (never invent data). Lookup
  mechanics themselves are exercised against synthetic tmp_path
  snapshots, not the shipped file, so these tests don't silently
  regress into re-depending on production data.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from github import ossf_criticality as oc


@pytest.fixture(autouse=True)
def _reset_cache() -> None:
    oc.reset_snapshot_cache()
    yield
    oc.reset_snapshot_cache()


def test_lookup_finds_known_high_criticality_anchor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Lookup mechanics against a populated (synthetic) snapshot — not
    the shipped file, which ships schema-only per P6.7."""

    snapshot = tmp_path / "snap.csv"
    snapshot.write_text("repo,score\nkubernetes/kubernetes,0.99432\n")
    monkeypatch.setattr(oc, "SNAPSHOT_PATH", snapshot)
    oc.reset_snapshot_cache()

    score = oc.lookup_criticality_score("kubernetes", "kubernetes")
    assert score is not None
    assert score > 0.9


def test_lookup_is_case_insensitive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = tmp_path / "snap.csv"
    snapshot.write_text("repo,score\nkubernetes/kubernetes,0.99432\n")
    monkeypatch.setattr(oc, "SNAPSHOT_PATH", snapshot)
    oc.reset_snapshot_cache()

    score = oc.lookup_criticality_score("KUBERNETES", "Kubernetes")
    assert score is not None


def test_lookup_returns_none_for_unknown_repo() -> None:
    assert oc.lookup_criticality_score("nonexistent-org", "nonexistent-repo") is None


def test_lookup_returns_none_for_empty_inputs() -> None:
    assert oc.lookup_criticality_score("", "kubernetes") is None
    assert oc.lookup_criticality_score("kubernetes", "") is None


def test_shipped_snapshot_ships_schema_only_no_fabricated_rows() -> None:
    """P6.7 regression lock: the fabricated "illustrative ordering"
    rows that used to ship in ``data/ossf_criticality_snapshot.csv``
    (kubernetes/kubernetes, rust-lang/rust, etc., with invented scores
    consumed as authoritative at weight 1.5) have been deleted. The
    shipped file must carry zero data rows — header + schema only —
    until an operator runs ``scripts/refresh_ossf_criticality.py``
    against the real upstream data (spec §14 operator gate). If this
    test starts failing because someone re-populated the file, that
    re-population must be real upstream data, not invented scores.
    """

    previously_fabricated_anchors = [
        ("kubernetes", "kubernetes"),
        ("rust-lang", "rust"),
        ("torvalds", "linux"),
        ("pytorch", "pytorch"),
        ("astral-sh", "uv"),
        ("facebook", "react"),
        ("vercel", "next.js"),
        ("etcd-io", "etcd"),
    ]
    for owner, repo in previously_fabricated_anchors:
        assert oc.lookup_criticality_score(owner, repo) is None, (
            f"{owner}/{repo} resolved to a score — the shipped snapshot "
            "should be empty (schema-only) per P6.7"
        )

    body = oc.SNAPSHOT_PATH.read_text(encoding="utf-8")
    data_lines = [
        line
        for line in body.splitlines()
        if line.strip()
        and not line.lstrip().startswith("#")
        and line.strip().lower() != "repo,score"
    ]
    assert data_lines == [], f"expected zero data rows, found: {data_lines}"


def test_missing_snapshot_falls_back_to_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing snapshot file should not crash; lookup returns None."""

    monkeypatch.setattr(oc, "SNAPSHOT_PATH", tmp_path / "missing.csv")
    oc.reset_snapshot_cache()

    assert oc.lookup_criticality_score("kubernetes", "kubernetes") is None


def test_malformed_rows_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Malformed rows in the snapshot are skipped, not crash-inducing."""

    snapshot = tmp_path / "snap.csv"
    snapshot.write_text(
        "# header line\n"
        "repo,score\n"
        "kubernetes/kubernetes,0.95\n"
        "garbage_no_score\n"  # no comma — skipped
        "rust-lang/rust,not-a-number\n"  # non-numeric score — skipped
        "etcd-io/etcd,0.88\n"
    )
    monkeypatch.setattr(oc, "SNAPSHOT_PATH", snapshot)
    oc.reset_snapshot_cache()

    assert oc.lookup_criticality_score("kubernetes", "kubernetes") == 0.95
    assert oc.lookup_criticality_score("etcd-io", "etcd") == 0.88
    assert oc.lookup_criticality_score("rust-lang", "rust") is None


def test_score_clamped_to_unit_interval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Defensive clamp: out-of-range scores get bounded to [0, 1]."""

    snapshot = tmp_path / "snap.csv"
    snapshot.write_text(
        "repo,score\n"
        "weird-org/weird-repo,1.5\n"
        "negative-org/negative-repo,-0.3\n"
    )
    monkeypatch.setattr(oc, "SNAPSHOT_PATH", snapshot)
    oc.reset_snapshot_cache()

    assert oc.lookup_criticality_score("weird-org", "weird-repo") == 1.0
    assert oc.lookup_criticality_score("negative-org", "negative-repo") == 0.0


# ---------------------------------------------------------------------------
# Move #20: staleness warning at load
# ---------------------------------------------------------------------------


def test_stale_snapshot_emits_warning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A snapshot whose ``# snapshot_date:`` header is older than the
    14-day threshold should log a warning at load time. The data still
    loads — the warning is for operator awareness, not a hard fail.
    """

    snapshot = tmp_path / "snap.csv"
    snapshot.write_text(
        "# snapshot_source: https://github.com/ossf/criticality_score\n"
        "# snapshot_date: 2024-01-01\n"
        "# refresh_cadence_target: weekly\n"
        "repo,score\n"
        "kubernetes/kubernetes,0.99\n"
    )
    monkeypatch.setattr(oc, "SNAPSHOT_PATH", snapshot)
    oc.reset_snapshot_cache()

    with caplog.at_level("WARNING", logger="github.ossf_criticality"):
        score = oc.lookup_criticality_score("kubernetes", "kubernetes")

    assert score == 0.99
    stale_records = [
        r for r in caplog.records
        if "snapshot is" in r.getMessage() and "days old" in r.getMessage()
    ]
    assert stale_records, "expected staleness warning, got: %r" % [
        r.getMessage() for r in caplog.records
    ]


def test_fresh_snapshot_does_not_emit_staleness_warning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A snapshot whose ``# snapshot_date:`` is within the threshold
    must not log a staleness warning."""

    from datetime import datetime, timezone

    today = datetime.now(timezone.utc).date().isoformat()
    snapshot = tmp_path / "snap.csv"
    snapshot.write_text(
        "# snapshot_source: https://github.com/ossf/criticality_score\n"
        f"# snapshot_date: {today}\n"
        "repo,score\n"
        "kubernetes/kubernetes,0.99\n"
    )
    monkeypatch.setattr(oc, "SNAPSHOT_PATH", snapshot)
    oc.reset_snapshot_cache()

    with caplog.at_level("WARNING", logger="github.ossf_criticality"):
        oc.lookup_criticality_score("kubernetes", "kubernetes")

    stale_records = [
        r for r in caplog.records
        if "days old" in r.getMessage()
    ]
    assert not stale_records


def test_snapshot_without_date_header_skips_staleness_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A snapshot missing the ``# snapshot_date:`` header is treated as
    "unknown age" — no warning, no crash."""

    snapshot = tmp_path / "snap.csv"
    snapshot.write_text(
        "repo,score\n"
        "kubernetes/kubernetes,0.99\n"
    )
    monkeypatch.setattr(oc, "SNAPSHOT_PATH", snapshot)
    oc.reset_snapshot_cache()

    with caplog.at_level("WARNING", logger="github.ossf_criticality"):
        score = oc.lookup_criticality_score("kubernetes", "kubernetes")

    assert score == 0.99
    stale_records = [
        r for r in caplog.records
        if "days old" in r.getMessage()
    ]
    assert not stale_records


# ---------------------------------------------------------------------------
# Move #20: refresh script parser smoke
# ---------------------------------------------------------------------------


def test_refresh_script_parses_upstream_csv_and_writes_snapshot(
    tmp_path: Path,
) -> None:
    """The refresh script should round-trip a synthetic upstream CSV
    into the local snapshot format with an updated metadata header.

    Uses ``--input-file`` to short-circuit the network so this stays
    in the offline default test suite.
    """

    from scripts.refresh_ossf_criticality import main

    upstream = tmp_path / "upstream.csv"
    upstream.write_text(
        "url,criticality_score,language\n"
        "https://github.com/kubernetes/kubernetes,0.987,Go\n"
        "github.com/rust-lang/rust,0.972,Rust\n"
        "garbage_no_slash,0.5,Other\n"
        "https://github.com/torvalds/linux,not-a-number,C\n"
    )
    output = tmp_path / "snap.csv"

    rc = main(
        [
            "--input-file",
            str(upstream),
            "--output-path",
            str(output),
        ]
    )
    assert rc == 0
    assert output.exists()

    body = output.read_text()
    assert "# snapshot_date:" in body
    assert "kubernetes/kubernetes" in body
    assert "rust-lang/rust" in body
    # Garbage rows are skipped silently.
    assert "garbage_no_slash" not in body
    assert "torvalds/linux" not in body  # non-numeric score skipped


def test_refresh_script_returns_one_when_upstream_is_unparseable(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An upstream payload with no recognized columns returns exit 1
    and leaves any existing snapshot untouched."""

    from scripts.refresh_ossf_criticality import main

    upstream = tmp_path / "upstream.csv"
    upstream.write_text("totally,unrelated,headers\nfoo,bar,baz\n")
    output = tmp_path / "snap.csv"

    rc = main(
        [
            "--input-file",
            str(upstream),
            "--output-path",
            str(output),
        ]
    )
    assert rc == 1
    assert not output.exists()
    err = capsys.readouterr().err
    assert "parsed zero rows" in err
