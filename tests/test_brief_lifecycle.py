from pathlib import Path

from shared.brief_lifecycle import classify_brief_path, discover_briefs, summarize_brief_lifecycle


def test_classify_brief_path_recognizes_lifecycle_markers():
    assert classify_brief_path(Path("config/brief-role.json")) == "active"
    assert classify_brief_path(Path("config/brief-role-draft.json")) == "draft"
    assert classify_brief_path(Path("config/brief-role.bak-20260414-120000.json")) == "draft"
    assert classify_brief_path(Path("config/archive/brief-role.json")) == "archived"
    assert classify_brief_path(Path("config/superseded/brief-role.json")) == "superseded"


def test_discover_briefs_filters_to_requested_lifecycle(tmp_path: Path):
    (tmp_path / "brief-active.json").write_text("{}")
    (tmp_path / "brief-draft-draft.json").write_text("{}")
    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()
    (archive_dir / "brief-old.json").write_text("{}")

    assert discover_briefs(tmp_path, include=("active",), recursive=True) == [
        tmp_path / "brief-active.json"
    ]
    assert discover_briefs(tmp_path, include=("draft",), recursive=True) == [
        tmp_path / "brief-draft-draft.json"
    ]


def test_summarize_brief_lifecycle_groups_paths(tmp_path: Path):
    (tmp_path / "brief-active.json").write_text("{}")
    (tmp_path / "brief-role-draft.json").write_text("{}")
    superseded_dir = tmp_path / "superseded"
    superseded_dir.mkdir()
    (superseded_dir / "brief-old.json").write_text("{}")

    summary = summarize_brief_lifecycle(tmp_path, recursive=True)

    assert summary["active"] == [tmp_path / "brief-active.json"]
    assert summary["draft"] == [tmp_path / "brief-role-draft.json"]
    assert summary["superseded"] == [superseded_dir / "brief-old.json"]
