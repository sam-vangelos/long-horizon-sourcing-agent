"""``prior_search.ruled_out_urls_file``: a skip list that lives outside git.

A campaign's known-people list (1,555 LinkedIn URLs on 2026-10-14) is candidate
data and never enters the repo, so a seed names it by absolute path instead of
inlining it. The loader reads the file into ``prior_search.ruled_out_urls``,
which every consumer already honours: the LinkedIn ledger skips a matching
search card unopened, the Exa do-not-save gate drops a matching public profile
before judgment, and preflight carries the reference into the generated brief,
which re-reads the file each time it loads.

URLs are normalised to the forms each source matches on: a Recruiter talent
URL to ``https://www.linkedin.com/talent/profile/<id>`` (the form a search card
carries), a public URL to ``https://www.linkedin.com/in/<handle>``. A file that
is named but cannot be read, a relative path, or a line that is not a LinkedIn
profile URL fails the load: a campaign that believes it has a skip list must
never run without one.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from shared.brief_loader import load_brief

SEED = "config/spl-test/brief-spl-test-v1.json"
EXA = "tests/fixtures/field_engineering_resume/preflight_v2_brief.json"

LINES = [
    "https://www.linkedin.com/talent/profile/AEMAAAKnownOne",
    "https://linkedin.com/talent/profile/AEMAAAKnownTwo?trk=abc",
    "https://www.linkedin.com/in/known-person-one/",
    "https://linkedin.com/in/Known-Person-Two",
    "",
    "# a comment line",
    "https://www.linkedin.com/talent/profile/AEMAAAKnownOne",
]


def _skip_file(tmp_path: Path, lines=LINES) -> Path:
    path = tmp_path / "outside-git" / "known-people.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _seed_with(tmp_path: Path, source: str, prior_search: dict) -> Path:
    raw = json.loads(Path(source).read_text(encoding="utf-8"))
    raw["prior_search"] = prior_search
    seed = tmp_path / "seed.json"
    seed.write_text(json.dumps(raw), encoding="utf-8")
    return seed


def test_the_file_is_read_into_the_skip_list_normalised_and_deduplicated(tmp_path) -> None:
    skip = _skip_file(tmp_path)
    inline = "https://www.linkedin.com/talent/profile/AEMAAAInline"
    seed = _seed_with(
        tmp_path, SEED, {"ruled_out_urls": [inline], "ruled_out_urls_file": str(skip)}
    )

    urls = load_brief(str(seed)).prior_search.ruled_out_urls

    assert urls == [
        inline,
        "https://www.linkedin.com/talent/profile/AEMAAAKnownOne",
        "https://www.linkedin.com/talent/profile/AEMAAAKnownTwo",
        "https://www.linkedin.com/in/known-person-one",
        "https://www.linkedin.com/in/Known-Person-Two",
    ]


def test_a_v2_brief_reads_the_file_too(tmp_path) -> None:
    skip = _skip_file(tmp_path)
    seed = _seed_with(tmp_path, EXA, {"ruled_out_urls_file": str(skip)})

    brief = load_brief(str(seed))

    assert "https://www.linkedin.com/in/known-person-one" in brief.prior_search.ruled_out_urls
    assert brief._new_brief.prior_search.ruled_out_urls == brief.prior_search.ruled_out_urls


@pytest.mark.parametrize(
    "path_value, message",
    [
        ("relative/known-people.txt", "absolute"),
        ("/nonexistent-dir-for-cloris-test/known-people.txt", "cannot be read"),
    ],
)
def test_a_named_file_that_cannot_be_used_fails_the_load(tmp_path, path_value, message) -> None:
    from shared.brief_v2_schema import BriefSchemaError

    seed = _seed_with(tmp_path, SEED, {"ruled_out_urls_file": path_value})

    with pytest.raises(BriefSchemaError, match=message):
        load_brief(str(seed))


def test_a_line_that_is_not_a_linkedin_profile_fails_the_load(tmp_path) -> None:
    from shared.brief_v2_schema import BriefSchemaError

    skip = _skip_file(tmp_path, ["https://www.linkedin.com/in/ok", "https://example.com/someone"])
    seed = _seed_with(tmp_path, SEED, {"ruled_out_urls_file": str(skip)})

    with pytest.raises(BriefSchemaError, match="line 2"):
        load_brief(str(seed))


def test_a_talent_id_written_in_a_public_url_slot_is_read_as_a_talent_url(tmp_path) -> None:
    skip = _skip_file(tmp_path, ["https://www.linkedin.com/in/(talent AEMAAASlotted_id-1)"])
    seed = _seed_with(tmp_path, SEED, {"ruled_out_urls_file": str(skip)})

    urls = load_brief(str(seed)).prior_search.ruled_out_urls

    assert urls == ["https://www.linkedin.com/talent/profile/AEMAAASlotted_id-1"]


@pytest.mark.parametrize(
    "line",
    ["https://www.linkedin.com/in/(someone else)", "https://www.linkedin.com/in/two words", "linkedin.com/in/no-scheme"],
)
def test_a_half_url_line_fails_the_load_rather_than_half_parsing(tmp_path, line) -> None:
    from shared.brief_v2_schema import BriefSchemaError

    skip = _skip_file(tmp_path, [line])
    seed = _seed_with(tmp_path, SEED, {"ruled_out_urls_file": str(skip)})

    with pytest.raises(BriefSchemaError, match="line 1"):
        load_brief(str(seed))


def test_the_ledger_skips_a_card_from_the_file_unopened(tmp_path) -> None:
    from linkedin.candidate_ledger import CandidateLedger

    skip = _skip_file(tmp_path)
    brief = load_brief(str(_seed_with(tmp_path, SEED, {"ruled_out_urls_file": str(skip)})))
    ledger = object.__new__(CandidateLedger)
    ledger._seen_urls = set()

    ledger._merge_prior_search_exclusion(brief)

    assert ledger.is_terminal("https://www.linkedin.com/talent/profile/AEMAAAKnownTwo")
    assert not ledger.is_terminal("https://www.linkedin.com/talent/profile/AEMAAAStranger")


def test_preflight_carries_the_reference_and_the_generated_brief_rereads_it(tmp_path) -> None:
    from shared.preflight_v2 import preflight_overrides_from_brief

    skip = _skip_file(tmp_path)
    seed = load_brief(str(_seed_with(tmp_path, SEED, {"ruled_out_urls_file": str(skip)})))

    overrides = preflight_overrides_from_brief(seed, {}, geography="United States")

    assert overrides["prior_search"]["ruled_out_urls_file"] == str(skip)
    generated = _seed_with(tmp_path, EXA, overrides["prior_search"])
    assert load_brief(str(generated)).prior_search.ruled_out_urls == seed.prior_search.ruled_out_urls


def test_the_exa_gate_drops_a_public_profile_from_the_file(tmp_path) -> None:
    from exa.identity import DoNotSaveGate
    from exa.schemas import ExaCandidate

    skip = _skip_file(tmp_path)
    brief = load_brief(str(_seed_with(tmp_path, EXA, {"ruled_out_urls_file": str(skip)})))
    gate = DoNotSaveGate.from_brief(brief)

    def candidate(url: str) -> ExaCandidate:
        return ExaCandidate(identity_key=url, name="Someone New", linkedin_url=url)

    assert gate.blocks(candidate("https://www.linkedin.com/in/known-person-one"))
    assert gate.blocks(candidate("https://www.linkedin.com/in/known-person-two/"))
    assert gate.blocks(candidate("https://www.linkedin.com/in/a-stranger")) is None
    # The brief's own named entries still block on the name alone.
    assert gate.blocks(ExaCandidate(identity_key="x", name="Avery Holt"))


def test_without_prior_search_the_exa_gate_is_the_do_not_save_list_alone() -> None:
    from exa.identity import DoNotSaveGate

    gate = DoNotSaveGate.from_brief(SimpleNamespace(do_not_save=[], prior_search=None))

    assert not gate
