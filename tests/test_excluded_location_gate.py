"""A brief's ``excluded_locations`` settles a card as FACIAL_NO with no model call.

LinkedIn Recruiter's Location filter can only include, so a search whose
territory is "the United States except four markets" keeps the filter on the
United States and needs the exclusion somewhere else. The full judge cannot
carry it: its profile text has no location line (shared/judger.py
``_profile_to_text``), and the facial judge never reads the seed's
instructions. On the dead SPL SWE run, 59% of profile opens and 76% of saves
were in markets another recruiter covers. The gate reads the card's location
before any judgment, the way the employer and alumni gates do.

An entry matches when its comma-separated parts appear, in order and whole,
in the card's location: "New York, New York" matches "New York, New York,
United States" and never "Buffalo, New York, United States"; "Texas" matches
any "<city>, Texas, United States". A location the list does not name, a bare
"United States", or no location at all is never excluded.
"""

from __future__ import annotations

import asyncio
import copy
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared.excluded_locations import excluded_location_match

EXCLUDED = [
    "San Francisco Bay Area",
    "San Francisco, California",
    "Palo Alto, California",
    "Greater Seattle Area",
    "Seattle, Washington",
    "Bellevue, Washington",
    "New York City Metropolitan Area",
    "New York, New York",
    "Brooklyn, New York",
    "Texas",
    "Dallas-Fort Worth Metroplex",
]


@pytest.mark.parametrize(
    "location, entry",
    [
        ("San Francisco Bay Area", "San Francisco Bay Area"),
        ("San Francisco, California, United States", "San Francisco, California"),
        ("Palo Alto, California, United States", "Palo Alto, California"),
        ("Greater Seattle Area", "Greater Seattle Area"),
        ("Bellevue, Washington, United States", "Bellevue, Washington"),
        ("New York, New York, United States", "New York, New York"),
        ("New York City Metropolitan Area", "New York City Metropolitan Area"),
        ("Brooklyn, New York, United States", "Brooklyn, New York"),
        ("Austin, Texas, United States", "Texas"),
        ("Texas, United States", "Texas"),
        ("Dallas-Fort Worth Metroplex", "Dallas-Fort Worth Metroplex"),
        ("  austin,  TEXAS , united states ", "Texas"),
    ],
)
def test_an_excluded_location_matches_its_entry(location: str, entry: str) -> None:
    assert excluded_location_match(location, EXCLUDED) == entry


@pytest.mark.parametrize(
    "location",
    [
        "Buffalo, New York, United States",
        "New York, United States",
        "Los Angeles, California, United States",
        "California, United States",
        "Spokane, Washington, United States",
        "Washington, United States",
        "Washington DC-Baltimore Area",
        "Denver, Colorado, United States",
        "Texas City Area",
        "United States",
        "",
        None,
    ],
)
def test_a_location_the_list_does_not_name_is_in_scope(location) -> None:
    assert excluded_location_match(location, EXCLUDED) is None


@pytest.mark.parametrize(
    "location, entry",
    [
        # The model card extractor keeps LinkedIn's industry suffix; five hub
        # cards in the historical snippets carry one.
        ("San Francisco Bay Area · Legal Services", "San Francisco Bay Area"),
        ("San Francisco Bay Area · Technology, Information and Internet", "San Francisco Bay Area"),
        ("New York City Metropolitan Area · Professional Services", "New York City Metropolitan Area"),
        ("Austin, Texas, United States · IT Services and IT Consulting", "Texas"),
    ],
)
def test_an_industry_suffix_does_not_hide_an_excluded_location(location: str, entry: str) -> None:
    assert excluded_location_match(location, EXCLUDED) == entry


def test_an_industry_suffix_on_an_in_scope_location_stays_in_scope() -> None:
    assert excluded_location_match("United States · Technology, Information and Internet", EXCLUDED) is None
    assert excluded_location_match("Denver, Colorado, United States · Software Development", EXCLUDED) is None


def test_an_empty_or_missing_list_excludes_nobody() -> None:
    assert excluded_location_match("Austin, Texas, United States", []) is None
    assert excluded_location_match("Austin, Texas, United States", None) is None


# ---------------------------------------------------------------------------
# The seed field rides to the gate on every path
# ---------------------------------------------------------------------------


def _seed(**extra) -> dict:
    seed = {
        "name": "territory-seed",
        "role_title": "Strategic Project Lead",
        "linkedin_project": "Test project",
        "linkedin_project_id": "123",
        "permanent_filters": {"Location": "United States"},
        "employer_blacklist": ["Babbage"],
        "alumni_blacklist": ["Babbage"],
        "excluded_locations": list(EXCLUDED),
        "instructions": ["THE TARGET: an operations leader who can read code."],
        "jd": "# The Role\n\nBabbage is hiring a Strategic Project Lead.",
    }
    seed.update(extra)
    return seed


def test_preflight_carries_the_list_into_the_generated_brief() -> None:
    from shared.brief_loader import _load_v2_brief, normalize_brief
    from shared.preflight_v2 import preflight_overrides_from_brief

    seed = normalize_brief(_seed())
    overrides = preflight_overrides_from_brief(seed, {}, geography="United States")

    assert overrides["excluded_locations"] == EXCLUDED


def test_a_resumed_session_reloads_the_list_from_the_seed() -> None:
    from shared.brief_loader import _load_v2_brief, normalize_brief
    from shared.preflight_v2 import overlay_seed_onto_resumed_brief

    frozen = json.loads(
        (Path(__file__).parent / "fixtures" / "field_engineering_resume" / "preflight_v2_brief.json").read_text()
    )
    merged = overlay_seed_onto_resumed_brief(
        copy.deepcopy(frozen), normalize_brief(_seed()), emit=lambda _line: None
    )

    assert _load_v2_brief(merged).raw["excluded_locations"] == EXCLUDED


def test_a_malformed_list_is_not_carried() -> None:
    from shared.brief_loader import normalize_brief
    from shared.preflight_v2 import preflight_overrides_from_brief

    seed = normalize_brief(_seed(excluded_locations="Texas"))

    assert "excluded_locations" not in preflight_overrides_from_brief(
        seed, {}, geography="United States"
    )


def test_the_v2_schema_recognizes_the_field() -> None:
    from shared.brief_v2_schema import RECOGNIZED_V2_KEYS

    assert "excluded_locations" in RECOGNIZED_V2_KEYS


# ---------------------------------------------------------------------------
# LinkedIn facial gate, serial and batch
# ---------------------------------------------------------------------------


def _texas_snippet(rank: int = 1):
    from tests.test_full_eval_pipelining import _snippet

    snippet = _snippet("Tex Operator", rank)
    snippet.location = "Austin, Texas, United States"
    return snippet


def test_the_gate_names_the_entry_and_the_location() -> None:
    from linkedin.candidate_review import TERRITORY_GATE_PATH, CandidateReviewer

    reviewer = object.__new__(CandidateReviewer)
    reviewer.context = SimpleNamespace(
        brief_obj=SimpleNamespace(
            employer_blacklist=[], alumni_blacklist=[], raw={"excluded_locations": EXCLUDED}
        )
    )

    path, (blocked, field, value) = reviewer._facial_employer_gate(_texas_snippet())

    assert path == TERRITORY_GATE_PATH
    assert (blocked, field, value) == ("Texas", "location", "Austin, Texas, United States")


def test_a_brief_without_a_raw_dict_never_fires_the_gate() -> None:
    from linkedin.candidate_review import CandidateReviewer

    reviewer = object.__new__(CandidateReviewer)
    reviewer.context = SimpleNamespace(
        brief_obj=SimpleNamespace(employer_blacklist=[], alumni_blacklist=[], raw=MagicMock())
    )

    assert reviewer._facial_employer_gate(_texas_snippet()) is None


def test_serial_facial_gate_settles_an_excluded_location_without_a_facial_call() -> None:
    from tests.test_full_eval_pipelining import _make_pipeline

    with tempfile.TemporaryDirectory() as td:
        pipeline = _make_pipeline(td)
        pipeline.state.context.brief_obj.alumni_blacklist = []
        pipeline.state.context.brief_obj.raw = {"excluded_locations": list(EXCLUDED)}
        snippet = _texas_snippet()
        facial = MagicMock(side_effect=AssertionError("facial judge must not run"))
        with patch("linkedin.candidate_review.facial_judge", facial):
            decision = asyncio.run(
                pipeline.state.reviewer._evaluate_snippet(snippet, None, None)
            )

        facial.assert_not_called()
        assert decision.decision == "FACIAL_NO"
        assert decision.path == "excluded_location"
        assert decision.rationale == "Excluded location: Texas (location)"
        assert pipeline.state.context.stats["excluded_location_skips"] == 1
        assert pipeline.state.candidates._prior_outcomes[snippet.profile_url] == "FACIAL_NO"


def test_batch_facial_gate_keeps_an_excluded_location_from_the_batch_judge() -> None:
    from shared.schemas import OpusDecision, SearchString
    from tests.test_full_eval_pipelining import _make_pipeline, _snippet

    with tempfile.TemporaryDirectory() as td:
        pipeline = _make_pipeline(td)
        brief = pipeline.state.context.brief_obj
        brief.has_v2_schema = True
        brief._new_brief = None
        brief.alumni_blacklist = []
        brief.raw = {"excluded_locations": list(EXCLUDED)}
        pipeline.state.reviewer._tightening_prefix = ""
        pipeline.state.reviewer._triage_tightened = False
        excluded = _texas_snippet(rank=1)
        clean = _snippet("Clean Two", 2)
        seen: list[str] = []

        def batch(snippets, *args, **kwargs):
            seen.extend(s.name for s in snippets)
            return [
                OpusDecision(
                    stage="facial",
                    decision="FACIAL_NO",
                    path="none",
                    confidence=1.0,
                    rationale="not relevant",
                    candidate_name=s.name,
                    profile_url=s.profile_url,
                )
                for s in snippets
            ]

        pipeline.state.context.browser.get_card_slot_count = AsyncMock(return_value=2)
        pipeline.state.pages._extract_card_snippet = AsyncMock(side_effect=[excluded, clean])
        pipeline.state.reviewer._run_full_queue = AsyncMock(return_value=[])
        all_candidates: list[dict] = []
        string_stats = pipeline.state.candidates._fresh_string_stats()
        search_string = SearchString(id=7, name="test string", boolean="engineer")
        with patch("shared.judger.facial_judge_batch", side_effect=batch), patch(
            "linkedin.evaluation_loop.config.LINKEDIN_FACIAL_CONCURRENCY_ENABLED", False
        ):
            asyncio.run(
                pipeline.state.pages._review_page_batch(
                    search_string, 1, 0, MagicMock(), all_candidates, string_stats, None
                )
            )

        assert "Tex Operator" not in seen, "an excluded location reached the facial judge"
        assert "Clean Two" in seen
        row = next(c for c in all_candidates if c["name"] == "Tex Operator")
        assert row["outcome"] == "territory_skip"
        assert row["rationale"] == "Excluded location: Texas (location)"
        assert pipeline.state.context.stats["excluded_location_skips"] == 1
        assert pipeline.state.pages._page_observation()["skipped_blacklist"] == 1
        # Only the judged card counts as facial_no.
        assert string_stats["facial_no"] == 1
        assert pipeline.state.context.stats["facial_no"] == 1


# ---------------------------------------------------------------------------
# A territory skip is not noise: the adaptation layer never learns from it
# ---------------------------------------------------------------------------


def test_a_page_of_territory_skips_yields_no_noise_anchors_or_not_terms() -> None:
    """Rows in the hubs must not become noise anchors, NOT terms, or a
    'dominated by facial-no' pattern: those steer drift rescues away from the
    very titles and companies that also surface in Sam's territory."""
    from linkedin.search_policy import _build_page_insights

    all_candidates = [
        {
            "name": f"Hub Fit {i}",
            "title": "Strategic Projects Lead",
            "company": "Scale AI",
            "outcome": "territory_skip",
            "rationale": "Excluded location: San Francisco Bay Area (location)",
            "page": 1,
        }
        for i in range(6)
    ]
    insights = _build_page_insights(
        page_num=1,
        result_count=300,
        preview_snippets=[],
        all_candidates=all_candidates,
        glance_result=None,
    )

    assert insights.noise_anchors == []
    assert insights.dominant_non_fit_patterns == []


def test_serial_territory_skip_stays_out_of_the_facial_no_counters() -> None:
    from tests.test_full_eval_pipelining import _make_pipeline

    with tempfile.TemporaryDirectory() as td:
        pipeline = _make_pipeline(td)
        pipeline.state.context.brief_obj.alumni_blacklist = []
        pipeline.state.context.brief_obj.raw = {"excluded_locations": list(EXCLUDED)}
        string_stats = pipeline.state.candidates._fresh_string_stats()
        with patch("linkedin.candidate_review.facial_judge", MagicMock()):
            decision = asyncio.run(
                pipeline.state.reviewer._evaluate_snippet(
                    _texas_snippet(), None, None, string_stats=string_stats
                )
            )

        assert decision.path == "excluded_location"
        assert string_stats["facial_no"] == 0
        assert pipeline.state.context.stats.get("facial_no", 0) == 0


def test_a_resumed_run_keeps_territory_skips_out_of_the_facial_no_counters() -> None:
    """The resume hydration rebuilds counters from canonical stage rows; a
    territory row is FACIAL_NO there, so it must be read back by its path."""
    from shared.schemas import OpusDecision, Progress, SearchString
    from tests.test_linkedin_pipeline import _make_pipeline, _make_snippet

    with tempfile.TemporaryDirectory() as td:
        p = _make_pipeline(td)
        owner = SearchString(id=6, name="owner", boolean="ml", status="done", pages_reviewed=1)
        progress = Progress(brief_name="test", strings=[owner])
        p.state.context._runtime_run_id, _ = p.state.context._runtime_bridge.start_or_resume_run(
            resume=False,
            initial_progress=progress,
        )
        for name, path in (("Hub Card", "excluded_location"), ("Judged No", "none")):
            snippet = _make_snippet(
                name=name,
                profile_url=f"/talent/profile/{name.replace(' ', '-').lower()}",
                source_string_id=owner.id,
                source_string_name=owner.name,
                page=1,
            )
            p.state.reviewer.attempts._record_runtime_snippet(owner, snippet)
            attempt_id = p.state.reviewer.attempts._start_runtime_stage_attempt(
                search_string=owner, snippet=snippet, stage="facial",
            )
            p.state.reviewer._finish_runtime_stage_success(
                attempt_id=attempt_id,
                stage="facial",
                snippet=snippet,
                decision=OpusDecision(
                    stage="facial", decision="FACIAL_NO", path=path, confidence=1.0,
                    rationale="x", candidate_name=name, profile_url=snippet.profile_url,
                ),
            )

        p.state.candidates._hydrate_resume_funnel_from_runtime(
            progress,
            store=p.state.context._runtime_state,
            run_id=p.state.context._runtime_run_id,
            brief_id=p.state.context._brief_id,
            stats=p.state.context.stats,
        )

        assert owner.facial_no_count == 1
        assert p.state.context.stats["facial_no"] == 1
