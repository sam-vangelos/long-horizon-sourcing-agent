import json
import shutil
from pathlib import Path

import pytest

from shared.recruiter_brief_resolution import resolve_linkedin_brief_path_for_github_run

REPO_ROOT = Path(__file__).resolve().parents[1]
FDE_DIR = REPO_ROOT / "config/Forward-Deployed-Engineer-NYC"
GITHUB_BRIEF = FDE_DIR / "brief-forward-deployed-engineer-us-github-v1.json"
LINKEDIN_BRIEF = FDE_DIR / "brief-forward-deployed-engineer-us-v1.4.json"


@pytest.mark.skipif(not GITHUB_BRIEF.is_file(), reason="FDE GitHub brief fixture missing")
@pytest.mark.skipif(not LINKEDIN_BRIEF.is_file(), reason="FDE LinkedIn brief fixture missing")
def test_resolve_linkedin_brief_from_run_manifest_sibling(tmp_path):
    gh_copy = tmp_path / GITHUB_BRIEF.name
    li_copy = tmp_path / LINKEDIN_BRIEF.name
    shutil.copy(GITHUB_BRIEF, gh_copy)
    shutil.copy(LINKEDIN_BRIEF, li_copy)

    manifest = {"brief_path": str(gh_copy.resolve())}
    (tmp_path / "run-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    resolved = resolve_linkedin_brief_path_for_github_run(tmp_path)
    assert resolved.resolve() == li_copy.resolve()


@pytest.mark.skipif(not LINKEDIN_BRIEF.is_file(), reason="FDE LinkedIn brief fixture missing")
def test_resolve_linkedin_brief_explicit_override(tmp_path):
    resolved = resolve_linkedin_brief_path_for_github_run(
        tmp_path,
        explicit_linkedin_brief=LINKEDIN_BRIEF,
    )
    assert resolved.resolve() == LINKEDIN_BRIEF.resolve()


@pytest.mark.skipif(not GITHUB_BRIEF.is_file(), reason="FDE GitHub brief fixture missing")
def test_resolve_linkedin_brief_picks_highest_numeric_version(tmp_path):
    """Plan §6: with siblings v1 and v1.4, the resolver must return v1.4 (numeric
    version), not the lexicographically greatest."""
    v1_brief = REPO_ROOT / "config/Forward-Deployed-Engineer-NYC/brief-forward-deployed-engineer-us-v1.json"
    v14_brief = REPO_ROOT / "config/Forward-Deployed-Engineer-NYC/brief-forward-deployed-engineer-us-v1.4.json"
    if not v1_brief.is_file() or not v14_brief.is_file():
        pytest.skip("multi-version FDE brief fixtures missing")

    gh_copy = tmp_path / GITHUB_BRIEF.name
    v1_copy = tmp_path / v1_brief.name
    v14_copy = tmp_path / v14_brief.name
    shutil.copy(GITHUB_BRIEF, gh_copy)
    shutil.copy(v1_brief, v1_copy)
    shutil.copy(v14_brief, v14_copy)

    manifest = {"brief_path": str(gh_copy.resolve())}
    (tmp_path / "run-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    resolved = resolve_linkedin_brief_path_for_github_run(tmp_path)
    assert resolved.resolve() == v14_copy.resolve()


@pytest.mark.skipif(not GITHUB_BRIEF.is_file(), reason="FDE GitHub brief fixture missing")
def test_resolve_linkedin_brief_fails_closed_on_unparseable_sibling(tmp_path):
    """Plan §6: any sibling without a parseable -vN[.N]* suffix forces fail-closed."""
    v14_brief = REPO_ROOT / "config/Forward-Deployed-Engineer-NYC/brief-forward-deployed-engineer-us-v1.4.json"
    if not v14_brief.is_file():
        pytest.skip("FDE v1.4 brief fixture missing")

    gh_copy = tmp_path / GITHUB_BRIEF.name
    v14_copy = tmp_path / v14_brief.name
    bak_copy = tmp_path / "brief-forward-deployed-engineer-us-v1.4-draft.bak.json"
    shutil.copy(GITHUB_BRIEF, gh_copy)
    shutil.copy(v14_brief, v14_copy)
    shutil.copy(v14_brief, bak_copy)

    manifest = {"brief_path": str(gh_copy.resolve())}
    (tmp_path / "run-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="parseable -vN"):
        resolve_linkedin_brief_path_for_github_run(tmp_path)


@pytest.mark.skipif(not GITHUB_BRIEF.is_file(), reason="FDE GitHub brief fixture missing")
def test_resolve_linkedin_brief_fails_closed_when_two_siblings_tie_at_top_version(tmp_path):
    """Plan §6: when two siblings share the highest version, fail closed."""
    v14_brief = REPO_ROOT / "config/Forward-Deployed-Engineer-NYC/brief-forward-deployed-engineer-us-v1.4.json"
    if not v14_brief.is_file():
        pytest.skip("FDE v1.4 brief fixture missing")

    gh_copy = tmp_path / GITHUB_BRIEF.name
    a_copy = tmp_path / "brief-forward-deployed-engineer-us-alt-a-v1.4.json"
    b_copy = tmp_path / "brief-forward-deployed-engineer-us-alt-b-v1.4.json"
    shutil.copy(GITHUB_BRIEF, gh_copy)
    shutil.copy(v14_brief, a_copy)
    shutil.copy(v14_brief, b_copy)

    manifest = {"brief_path": str(gh_copy.resolve())}
    (tmp_path / "run-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="share the highest version"):
        resolve_linkedin_brief_path_for_github_run(tmp_path)


def test_parse_brief_version_handles_common_shapes():
    from shared.recruiter_brief_resolution import _parse_brief_version

    assert _parse_brief_version("brief-foo-v1") == (1,)
    assert _parse_brief_version("brief-foo-v1.4") == (1, 4)
    assert _parse_brief_version("brief-foo-v2.10.3") == (2, 10, 3)
    assert _parse_brief_version("brief-foo") is None
    assert _parse_brief_version("brief-foo-v1.4-draft") is None
    assert _parse_brief_version("brief-v1-foo") is None
