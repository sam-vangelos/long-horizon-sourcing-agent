"""A preflight seed's ``prior_search.ruled_out_urls`` reaches the dedup set.

Before this, only a full V2 brief hydrated ``prior_search``. A preflight seed
(the legacy shape, ``needs_preflight()`` True) loaded through
``normalize_brief``, which never read it, so the fresh run's history merge
(evaluation_loop ``_load_candidate_history``, which runs before preflight)
saw ``None``; and preflight never carried it into the generated brief, so
every resumed session reloaded a frozen brief without it. A skip list of
people already contacted outside Cloris was a silent no-op on both paths.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

from linkedin.candidate_ledger import CandidateLedger
from shared.brief_loader import _load_v2_brief, load_brief, normalize_brief
from shared.preflight_v2 import (
    overlay_seed_onto_resumed_brief,
    preflight_overrides_from_brief,
)

RULED_OUT = [
    "https://www.linkedin.com/talent/profile/AEMAAExampleProfile00000000000000000001",
    "https://www.linkedin.com/talent/profile/AEMAAExampleProfile00000000000000000002",
]
NOTES = "Agency roster plus InMails sent outside Cloris"
FROZEN_BRIEF = (
    Path(__file__).parent / "fixtures" / "field_engineering_resume" / "preflight_v2_brief.json"
)


def _seed(**extra) -> dict:
    seed = {
        "name": "prior-search-seed",
        "role_title": "Strategic Project Lead",
        "linkedin_project": "Test project",
        "linkedin_project_id": "123",
        "permanent_filters": {"Location": "United States"},
        "employer_blacklist": ["Babbage"],
        "alumni_blacklist": ["Babbage"],
        "instructions": ["THE TARGET: an operations leader who can read code."],
        "jd": "# The Role\n\nBabbage is hiring a Strategic Project Lead.",
    }
    seed.update(extra)
    return seed


def _with_prior_search() -> dict:
    return _seed(prior_search={"ruled_out_urls": list(RULED_OUT), "ruled_out_notes": NOTES})


def test_a_preflight_seed_hydrates_prior_search(tmp_path: Path) -> None:
    path = tmp_path / "seed.json"
    path.write_text(json.dumps(_with_prior_search()))

    brief = load_brief(path)

    assert brief.needs_preflight()
    assert brief.prior_search is not None
    assert brief.prior_search.ruled_out_urls == RULED_OUT
    assert brief.prior_search.ruled_out_notes == NOTES


def test_a_seed_without_prior_search_still_loads_none() -> None:
    assert normalize_brief(_seed()).prior_search is None


def test_the_fresh_history_merge_suppresses_the_seeds_urls() -> None:
    ledger = CandidateLedger()
    ledger._seen_urls = set()

    ledger._merge_prior_search_exclusion(normalize_brief(_with_prior_search()))

    assert all(ledger.is_terminal(url) for url in RULED_OUT)


def test_preflight_carries_prior_search_into_the_generated_brief() -> None:
    seed = normalize_brief(_with_prior_search())

    overrides = preflight_overrides_from_brief(seed, {}, geography="United States")

    assert overrides["prior_search"]["ruled_out_urls"] == RULED_OUT


def test_a_resumed_session_reloads_prior_search_from_the_seed() -> None:
    frozen = json.loads(FROZEN_BRIEF.read_text())
    assert "prior_search" not in frozen
    seed = normalize_brief(_with_prior_search())

    merged = overlay_seed_onto_resumed_brief(copy.deepcopy(frozen), seed, emit=lambda _line: None)
    reloaded = _load_v2_brief(merged)

    assert reloaded.prior_search.ruled_out_urls == RULED_OUT
    ledger = CandidateLedger()
    ledger._seen_urls = set()
    ledger._merge_prior_search_exclusion(reloaded)
    assert all(ledger.is_terminal(url) for url in RULED_OUT)


def test_a_malformed_prior_search_is_not_carried() -> None:
    seed = normalize_brief(_seed(prior_search={"ruled_out_urls": "not-a-list"}))

    overrides = preflight_overrides_from_brief(seed, {}, geography="United States")

    assert "prior_search" not in overrides
