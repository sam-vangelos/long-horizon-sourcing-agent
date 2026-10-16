"""Alumni blacklist: anyone who has EVER worked at the hiring company is out.

Operator ruling, 2026-10-01: Babbage is the hiring company, and anyone
who has ever worked there (current or past, employee or contractor) is never
saved, surfaced, or judged as a lead in any lane. The ruling supersedes the
campaign plan's earlier alumni rule for Babbage only; OpenAI and Anthropic
alumni stay in scope, and only their CURRENT employees stay blocked.

The incident behind it: two former Babbage employees were saved to the
campaign's Recruiter project and messaged, because the employer blacklist
keys on the current employer only. The fixtures below keep the shape of those
two profiles with synthetic names and employers.

A brief's ``alumni_blacklist`` names the employers whose alumni are blocked.
These tests pin the matcher, the brief plumbing, every gate that sees a
candidate (LinkedIn facial, LinkedIn post-profile before the judge and the
save, Exa before the judge, the Exa slate and save replay), the judge prompt,
the generation lint, and the constraint manifest.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import shared.employer_blacklist as employer_blacklist
from shared.employer_blacklist import match_current_employer
from shared.schemas import (
    CandidateProfileSummary,
    CandidateSnippet,
    Experience,
)

EMPLOYER_BLACKLIST = ["Babbage", "OpenAI", "Anthropic"]
ALUMNI_BLACKLIST = ["Babbage"]


def match_any_employer(*args, **kwargs):
    # Resolved per call so each test reports its own red before the matcher exists.
    return employer_blacklist.match_any_employer(*args, **kwargs)


# ---------------------------------------------------------------------------
# Fixtures shaped like the two candidates from the incident (synthetic)
# ---------------------------------------------------------------------------


AVERY_HEADLINE = (
    "Engineering Leader | AI, LLMs & Agent-Based Systems | Scaling Production AI Teams"
)


def _avery_snippet(*, entries: list[str] | None = None, rank: int = 1) -> CandidateSnippet:
    return CandidateSnippet(
        name="Avery Quinn",
        headline=AVERY_HEADLINE,
        current_title="Sr. Manager, Forward Deploy Engineering",
        current_company="Labelstream AI",
        location="United States",
        education_snippet="",
        profile_url="/talent/profile/avery-quinn",
        source_string_id=7,
        source_string_name="test string",
        page=1,
        result_rank=rank,
        card_index=rank - 1,
        experience_entries=list(
            entries
            if entries is not None
            else [
                "Sr. Manager, Forward Deploy Engineering at Labelstream AI (2026 – Present)",
                "Director, Software Engineering - AI Agents & RL Environments at Babbage (2025 – 2026)",
                "Senior Software Engineering Manager at Northbeam Learning (2025 – 2025)",
            ]
        ),
    )


def _avery_profile() -> CandidateProfileSummary:
    return CandidateProfileSummary(
        name="Avery Quinn",
        profile_url="/talent/profile/avery-quinn",
        headline=AVERY_HEADLINE,
        experiences=[
            Experience(
                title="Sr. Manager, Forward Deploy Engineering",
                company="Labelstream AI · Full-time",
                start="Aug 2026",
                end="Present",
            ),
            Experience(
                title="Director, Software Engineering - AI Agents & RL Environments",
                company="Babbage · Full-time",
                start="Dec 2025",
                end="Jul 2026",
            ),
            Experience(
                title="Senior Software Engineering Manager",
                company="Northbeam Learning · Full-time",
                start="Feb 2025",
                end="Dec 2025",
            ),
            Experience(
                title="Director of Software Engineering",
                company="Ledgerline · Full-time",
                start="Jun 2018",
                end="Dec 2021",
            ),
        ],
    )


def _rowan_profile() -> CandidateProfileSummary:
    return CandidateProfileSummary(
        name="Rowan Ellis",
        profile_url="/talent/profile/rowan-ellis",
        headline="Post-Training Data & Eval | Data Science Faculty @ State University",
        experiences=[
            Experience(
                title="Head Teaching Assistant & Lecturer",
                company="State University College of Engineering",
                start="Sep 2022",
                end="Present",
            ),
            Experience(
                title="Founding Technical Program Manager, Research & Evals",
                company="Babbage",
                start="Jan 2025",
                end="Mar 2026",
            ),
        ],
    )


def _companies(profile: CandidateProfileSummary) -> list[str]:
    return [experience.company for experience in profile.experiences]


# ---------------------------------------------------------------------------
# (a) a past Babbage role anywhere in the history blocks
# ---------------------------------------------------------------------------


def test_past_babbage_role_in_profile_experiences_blocks() -> None:
    profile = _avery_profile()
    hit = match_any_employer(
        ALUMNI_BLACKLIST,
        current_company="Labelstream AI",
        headline=profile.headline,
        experience_companies=_companies(profile),
    )
    assert hit is not None
    assert hit[0] == "Babbage"
    assert hit[1] == "experience"
    assert hit[2] == "Babbage · Full-time"


def test_founding_role_with_bare_company_blocks() -> None:
    profile = _rowan_profile()
    hit = match_any_employer(
        ALUMNI_BLACKLIST,
        current_company="State University College of Engineering",
        headline=profile.headline,
        experience_companies=_companies(profile),
    )
    assert hit == ("Babbage", "experience", "Babbage")


def test_past_babbage_role_in_snippet_experience_entries_blocks() -> None:
    snippet = _avery_snippet()
    hit = match_any_employer(
        ALUMNI_BLACKLIST,
        current_company=snippet.current_company,
        headline=snippet.headline,
        experience_entries=snippet.experience_entries,
    )
    assert hit is not None
    assert hit[:2] == ("Babbage", "experience_entry")
    assert "at Babbage (2025 – 2026)" in hit[2]


def test_current_employer_blacklist_alone_still_lets_the_alum_through() -> None:
    # The pre-existing rule is unchanged: it keys on the CURRENT employer.
    snippet = _avery_snippet()
    assert (
        match_current_employer(
            snippet.current_company, snippet.headline, EMPLOYER_BLACKLIST
        )
        is None
    )


def test_current_babbage_employee_is_also_an_alum() -> None:
    hit = match_any_employer(ALUMNI_BLACKLIST, current_company="Babbage")
    assert hit == ("Babbage", "current_company", "Babbage")


def test_empty_alumni_blacklist_blocks_nobody() -> None:
    profile = _avery_profile()
    assert match_any_employer([], experience_companies=_companies(profile)) is None
    assert match_any_employer(None, experience_companies=_companies(profile)) is None


# ---------------------------------------------------------------------------
# (b) past-marker headlines: blocked by alumni_blacklist, never by
#     employer_blacklist alone
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "headline",
    [
        "ex-Babbage",
        "AI Engineer | ex-Babbage | LLM evals",
        "Ex Babbage, now building agents",
        "Former Director at Babbage",
        "Formerly @ Babbage | Data Lead",
        "Previously Engineering Manager at Babbage",
        "Babbage alum | Post-training",
        "Prev. Babbage",
    ],
)
def test_past_marker_headline_is_blocked_by_alumni_blacklist(headline: str) -> None:
    hit = match_any_employer(ALUMNI_BLACKLIST, headline=headline)
    assert hit is not None and hit[:2] == ("Babbage", "headline"), headline
    assert match_current_employer("", headline, EMPLOYER_BLACKLIST) is None, headline


def test_headline_bare_mention_without_employer_context_is_not_an_alum() -> None:
    # A bare mention is a tool/skill/topic, exactly as for the current rule.
    assert match_any_employer(ALUMNI_BLACKLIST, headline="Babbage | Upwork | RLHF") is None


def test_about_text_naming_a_past_stint_blocks() -> None:
    hit = match_any_employer(
        ALUMNI_BLACKLIST,
        about="I previously led RL environment delivery at Babbage for two frontier labs.",
    )
    assert hit is not None and hit[:2] == ("Babbage", "about")


# ---------------------------------------------------------------------------
# (c) institutions and concepts named after Charles Babbage never match
# ---------------------------------------------------------------------------


NOT_THE_COMPANY = [
    "Babbage School of Engineering",
    "Babbage Award",
    "Charles Babbage Institute",
    "The Charles Babbage Institute",
    "Babbage engine",
    "Babbage-engine",
    "Charles Babbage",
]


@pytest.mark.parametrize("name", NOT_THE_COMPANY)
def test_named_after_charles_babbage_never_matches_as_a_company(name: str) -> None:
    assert match_any_employer(ALUMNI_BLACKLIST, current_company=name) is None
    assert (
        match_any_employer(ALUMNI_BLACKLIST, experience_companies=[name]) is None
    )
    assert match_current_employer(name, "", EMPLOYER_BLACKLIST) is None


@pytest.mark.parametrize(
    "headline",
    [
        "Research Fellow at The Charles Babbage Institute",
        "Research Fellow at Charles Babbage Institute | ex-DeepMind",
        "Instructor at Babbage School of Engineering",
        "Engineer working on Babbage engine replicas",
        "Building Babbage-engine emulators at Acme",
        "Formerly Curator at Charles Babbage Archive",
        "Organizer at Babbage Society",
    ],
)
def test_named_after_charles_babbage_never_matches_in_a_headline(headline: str) -> None:
    assert match_any_employer(ALUMNI_BLACKLIST, headline=headline) is None, headline
    assert match_current_employer("", headline, EMPLOYER_BLACKLIST) is None, headline


def test_named_after_charles_babbage_never_matches_in_snippet_entries() -> None:
    entries = [
        "Research Fellow at The Charles Babbage Institute (2021 – 2023)",
        "Instructor at Babbage School of Engineering (2019 – 2020)",
    ]
    assert match_any_employer(ALUMNI_BLACKLIST, experience_entries=entries) is None


def test_word_boundary_holds_for_history() -> None:
    assert (
        match_any_employer(
            ALUMNI_BLACKLIST,
            experience_companies=["Centurion Systems", "BabbageBots"],
        )
        is None
    )


# ---------------------------------------------------------------------------
# (d) OpenAI / Anthropic alumni stay in scope
# ---------------------------------------------------------------------------


def test_openai_alum_is_not_blocked_by_either_rule() -> None:
    profile = CandidateProfileSummary(
        name="Ada Ex",
        profile_url="/talent/profile/ada-ex",
        headline="Research Engineer | ex-OpenAI | building evals",
        experiences=[
            Experience(title="Founder", company="Stealth", start="2025", end="Present"),
            Experience(title="Member of Technical Staff", company="OpenAI", start="2021", end="2024"),
            Experience(title="Research Engineer", company="Anthropic", start="2019", end="2021"),
        ],
    )
    assert (
        match_any_employer(
            ALUMNI_BLACKLIST,
            current_company="Stealth",
            headline=profile.headline,
            experience_companies=_companies(profile),
            about="I spent three years at OpenAI and two at Anthropic.",
        )
        is None
    )
    assert match_current_employer("Stealth", profile.headline, EMPLOYER_BLACKLIST) is None


# ---------------------------------------------------------------------------
# Brief plumbing: all three construction paths, schema, preflight, lint
# ---------------------------------------------------------------------------


def _legacy_seed(**overrides) -> dict:
    seed = {
        "name": "spl-swe",
        "role_title": "Strategic Project Lead",
        "linkedin_project_id": "9000000003",
        "jd": "The Role\n\nBabbage is hiring a Strategic Project Lead.",
        "employer_blacklist": list(EMPLOYER_BLACKLIST),
        "alumni_blacklist": list(ALUMNI_BLACKLIST),
    }
    seed.update(overrides)
    return seed


def _v2_brief(**overrides) -> dict:
    brief = {
        "role_title": "Strategic Project Lead",
        "role_summary": "Owns delivery of SWE data programs.",
        "engagement_context": {"selectivity_posture": "selective"},
        "capability_areas": [
            {"name": "Program delivery", "description": "Owns delivery."}
        ],
        "depth_distinction": {
            "builder_definition": "Owns the program.",
            "user_definition": "Attends the program.",
            "edge_case_guidance": "Borderline = full eval.",
        },
        "employer_blacklist": list(EMPLOYER_BLACKLIST),
        "alumni_blacklist": list(ALUMNI_BLACKLIST),
    }
    brief.update(overrides)
    return brief


def test_legacy_seed_carries_alumni_blacklist() -> None:
    from shared.brief_loader import normalize_brief

    brief = normalize_brief(_legacy_seed())
    assert brief.alumni_blacklist == ["Babbage"]
    assert normalize_brief(_legacy_seed(alumni_blacklist=None)).alumni_blacklist == []


def test_v2_brief_carries_alumni_blacklist_on_both_dataclasses() -> None:
    from shared.brief_loader import _load_v2_brief

    brief = _load_v2_brief(_v2_brief())
    assert brief.alumni_blacklist == ["Babbage"]
    assert brief._new_brief.alumni_blacklist == ["Babbage"]
    # Detached: the compat mirror never shares the list with _new_brief.
    assert brief.alumni_blacklist is not brief._new_brief.alumni_blacklist


def test_load_brief_reads_alumni_blacklist_from_disk(tmp_path: Path) -> None:
    from shared.brief_loader import load_brief

    path = tmp_path / "brief.json"
    path.write_text(json.dumps(_legacy_seed()))
    assert load_brief(path).alumni_blacklist == ["Babbage"]
    path.write_text(json.dumps(_v2_brief()))
    assert load_brief(path).alumni_blacklist == ["Babbage"]


@pytest.mark.parametrize("bad", ["Babbage", [""], [3], [["Babbage"]], {"name": "Babbage"}])
def test_malformed_alumni_blacklist_fails_closed_at_load(bad) -> None:
    from shared.brief_loader import _load_v2_brief, normalize_brief
    from shared.brief_v2_schema import BriefSchemaError

    with pytest.raises(BriefSchemaError):
        _load_v2_brief(_v2_brief(alumni_blacklist=bad))
    with pytest.raises(BriefSchemaError):
        normalize_brief(_legacy_seed(alumni_blacklist=bad))


def test_v2_schema_recognizes_and_validates_alumni_blacklist() -> None:
    from shared.brief_v2_schema import (
        RECOGNIZED_V2_KEYS,
        BriefSchemaError,
        validate_v2_brief,
    )

    assert "alumni_blacklist" in RECOGNIZED_V2_KEYS
    validate_v2_brief(_v2_brief())
    with pytest.raises(BriefSchemaError):
        validate_v2_brief(_v2_brief(alumni_blacklist="Babbage"))


def test_preflight_overrides_carry_seed_alumni_into_generated_brief() -> None:
    from shared.brief_loader import normalize_brief
    from shared.preflight_v2 import preflight_overrides_from_brief

    seed = normalize_brief(_legacy_seed())
    overrides = preflight_overrides_from_brief(seed, {}, geography="")
    assert overrides["alumni_blacklist"] == ["Babbage"]


def test_lint_allows_alumni_names_in_alumni_blacklist_and_protects_them_in_tiers() -> None:
    from shared.brief_lint import blocking_findings, lint_generated_brief
    from tests.test_brief_lint import _clean_brief

    data = _clean_brief()
    data["alumni_blacklist"] = ["Babbage"]
    assert blocking_findings(lint_generated_brief(data)) == []

    data["employer_signal_rules"][0]["employer_patterns"] = ["Babbage alumni"]
    codes = {f.code for f in blocking_findings(lint_generated_brief(data))}
    assert "employer_conflict" in codes

    seeded = _clean_brief()
    seeded["employer_signal_rules"][0]["employer_patterns"] = ["Babbage"]
    codes = {
        f.code
        for f in blocking_findings(
            lint_generated_brief(seeded, seed_alumni_blacklist=["Babbage"])
        )
    }
    assert "employer_conflict" in codes


def test_lint_rejects_a_malformed_alumni_blacklist() -> None:
    from shared.brief_lint import blocking_findings, lint_generated_brief
    from tests.test_brief_lint import _clean_brief

    data = _clean_brief()
    data["alumni_blacklist"] = "Babbage"
    codes = {f.code for f in blocking_findings(lint_generated_brief(data))}
    assert "alumni_blacklist_invalid" in codes


# ---------------------------------------------------------------------------
# The judge prompt and the constraint manifest name the rule
# ---------------------------------------------------------------------------


def test_employer_signal_block_names_the_alumni_rule_only_when_set() -> None:
    from shared.brief_loader import _load_v2_brief

    with_alumni = _load_v2_brief(_v2_brief())._new_brief.employer_signal_block()
    assert "ALUMNI BLACKLIST" in with_alumni
    assert "EVER worked at" in with_alumni
    assert "Babbage" in with_alumni.split("ALUMNI BLACKLIST", 1)[1]

    without = _load_v2_brief(_v2_brief(alumni_blacklist=[]))._new_brief.employer_signal_block()
    assert "ALUMNI BLACKLIST" not in without
    # The current-employer wording is byte-identical when no alumni list is set.
    assert (
        "Past employment at a blacklisted company is not disqualifying by itself; "
        "weigh that evidence like any other."
    ) in without


def test_constraint_manifest_owns_the_alumni_blacklist() -> None:
    from shared.constraint_manifest import build_constraint_manifest

    manifest = build_constraint_manifest(
        SimpleNamespace(
            permanent_filters={},
            employer_blacklist=list(EMPLOYER_BLACKLIST),
            alumni_blacklist=list(ALUMNI_BLACKLIST),
            instructions=[],
            intake_notes="",
            _new_brief=None,
        )
    )
    entry = manifest["classes"]["alumni_blacklist"]
    assert entry["status"] == "owned"
    assert entry["stated_value"] == "Babbage"
    assert "alumni_blacklist" in entry["verify_method"]


# ---------------------------------------------------------------------------
# LinkedIn facial gate: snippet history blocks with no model call
# ---------------------------------------------------------------------------


def test_facial_gate_matches_snippet_history() -> None:
    from linkedin.candidate_review import CandidateReviewer

    hit = CandidateReviewer._alumni_blacklist_match(_avery_snippet(), ALUMNI_BLACKLIST)
    assert hit is not None and hit[:2] == ("Babbage", "experience_entry")
    # SimpleNamespace snippets (no experience_entries) stay supported.
    assert (
        CandidateReviewer._alumni_blacklist_match(
            SimpleNamespace(headline="ex-Babbage", current_company="Acme"),
            ALUMNI_BLACKLIST,
        )
        is not None
    )


def test_serial_facial_gate_blocks_alum_without_a_facial_call() -> None:
    from tests.test_full_eval_pipelining import _make_pipeline

    with tempfile.TemporaryDirectory() as td:
        pipeline = _make_pipeline(td)
        pipeline.state.context.brief_obj.alumni_blacklist = list(ALUMNI_BLACKLIST)
        snippet = _avery_snippet()
        facial = MagicMock(side_effect=AssertionError("facial judge must not run"))
        with patch("linkedin.candidate_review.facial_judge", facial):
            decision = asyncio.run(
                pipeline.state.reviewer._evaluate_snippet(snippet, None, None)
            )
        facial.assert_not_called()
        assert decision.stage == "facial"
        assert decision.decision == "FACIAL_NO"
        assert decision.path == "alumni_blacklist"
        assert "Babbage" in decision.rationale
        assert pipeline.state.context.stats["alumni_blacklist_skips"] == 1
        assert pipeline.state.candidates._prior_outcomes[snippet.profile_url] == "FACIAL_NO"


def test_current_employer_gate_keeps_its_own_path_when_both_lists_match() -> None:
    from tests.test_full_eval_pipelining import _make_pipeline

    with tempfile.TemporaryDirectory() as td:
        pipeline = _make_pipeline(td)
        pipeline.state.context.brief_obj.employer_blacklist = list(EMPLOYER_BLACKLIST)
        pipeline.state.context.brief_obj.alumni_blacklist = list(ALUMNI_BLACKLIST)
        snippet = _avery_snippet()
        snippet.current_company = "Babbage"
        with patch(
            "linkedin.candidate_review.facial_judge",
            MagicMock(side_effect=AssertionError("facial judge must not run")),
        ):
            decision = asyncio.run(
                pipeline.state.reviewer._evaluate_snippet(snippet, None, None)
            )
        assert decision.path == "employer_blacklist"


def test_batch_facial_gate_blocks_alum_before_the_batch_judge() -> None:
    from shared.schemas import OpusDecision, SearchString
    from tests.test_full_eval_pipelining import _make_pipeline, _snippet

    with tempfile.TemporaryDirectory() as td:
        pipeline = _make_pipeline(td)
        brief = pipeline.state.context.brief_obj
        brief.has_v2_schema = True
        brief._new_brief = None
        brief.alumni_blacklist = list(ALUMNI_BLACKLIST)
        pipeline.state.reviewer._tightening_prefix = ""
        pipeline.state.reviewer._triage_tightened = False
        alum = _avery_snippet(rank=1)
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
        pipeline.state.pages._extract_card_snippet = AsyncMock(side_effect=[alum, clean])
        pipeline.state.reviewer._run_full_queue = AsyncMock(return_value=[])
        page_report = MagicMock()
        all_candidates: list[dict] = []
        string_stats = pipeline.state.candidates._fresh_string_stats()
        search_string = SearchString(id=7, name="test string", boolean="engineer")
        with patch("shared.judger.facial_judge_batch", side_effect=batch), patch(
            "linkedin.evaluation_loop.config.LINKEDIN_FACIAL_CONCURRENCY_ENABLED", False
        ):
            asyncio.run(
                pipeline.state.pages._review_page_batch(
                    search_string, 1, 0, page_report, all_candidates, string_stats, None
                )
            )

        assert "Avery Quinn" not in seen, "an alum reached the facial judge"
        row = next(c for c in all_candidates if c["name"] == "Avery Quinn")
        assert row["outcome"] == "facial_no"
        assert row["rationale"].startswith("Alumni blacklist: Babbage")
        assert pipeline.state.context.stats["alumni_blacklist_skips"] == 1
        assert pipeline.state.pages._page_observation()["skipped_blacklist"] == 1


# ---------------------------------------------------------------------------
# (e) LinkedIn post-profile gate: runs on the FULL profile, before the judge
#     and before any save or note write
# ---------------------------------------------------------------------------


def _card_only_current_role() -> CandidateSnippet:
    # The card shows only the current role, so the facial gate cannot see
    # the Babbage stint; only the full profile can.
    return _avery_snippet(
        entries=["Sr. Manager, Forward Deploy Engineering at Labelstream AI (2026 – Present)"]
    )


@pytest.mark.parametrize("window", [1, 2])
def test_post_profile_gate_blocks_alum_before_judge_and_save(window: int) -> None:
    from shared.schemas import SearchString
    from tests.test_full_eval_pipelining import (
        FakePageReport,
        _candidate_rows,
        _full_attempt_rows,
        _make_pipeline,
        _string_stats,
        _wire_common_pipeline,
        _wire_real_runtime,
    )

    with tempfile.TemporaryDirectory() as td:
        pipeline = _make_pipeline(td)
        pipeline.state.context.brief_obj.alumni_blacklist = list(ALUMNI_BLACKLIST)
        snippet = _card_only_current_role()
        assert (
            _snippet_alumni_hit(snippet) is None
        ), "fixture must get past the facial gate"
        _wire_common_pipeline(pipeline)
        search_string, _ = _wire_real_runtime(pipeline, output_dir=td, snippets=[snippet])

        async def extract(snippet_arg, *, interest: float = 0.5):
            pipeline.state.context.browser.show_profile(snippet_arg.profile_url)
            return SimpleNamespace(profile_summary=_avery_profile())

        pipeline.state.reviewer.acquisition.extract_profile_summary = AsyncMock(
            side_effect=extract
        )
        judge = MagicMock(side_effect=AssertionError("full judge must not run"))
        page_report = FakePageReport()
        stats = _string_stats()
        pipeline.state.candidates._record_facial_funnel_outcome(
            snippet=snippet,
            decision="FACIAL_YES",
            search_string=search_string,
            string_stats=stats,
            stats=pipeline.state.context.stats,
        )
        with patch("linkedin.candidate_review.full_judge", judge), patch(
            "linkedin.evaluation_loop.config.LINKEDIN_EXTERNAL_EVIDENCE_ENABLED", False
        ), patch("linkedin.candidate_review.human_delay_correlated", return_value=0.0):
            decisions = asyncio.run(
                pipeline.state.reviewer._run_full_queue(
                    snippets=[snippet],
                    window=window,
                    origin="page",
                    page_report=page_report,
                    search_string=search_string,
                    all_candidates=_candidate_rows([snippet]),
                    string_stats=stats,
                    progress=None,
                    page_num=1,
                )
            )

        judge.assert_not_called()
        pipeline.state.reviewer.side_effects.handle_save_decision.assert_not_awaited()
        browser_events = pipeline.state.context.browser.events
        assert not any(kind in {"save_candidate", "write_candidate_note"} for kind, _ in browser_events)
        assert ("back", "results") in browser_events, "the profile panel was left open"

        assert len(decisions) == 1
        decision = decisions[0]
        assert decision.stage == "full"
        assert decision.decision == "REJECT"
        assert decision.path == "alumni_blacklist"
        assert decision.reject_reason == "HARD_GATE"
        assert "Babbage" in decision.rationale
        assert ("REJECT", "Avery Quinn") in page_report.events
        assert pipeline.state.context.stats["alumni_blacklist_rejects"] == 1

        rows = _full_attempt_rows(pipeline)
        assert len(rows) == 1
        assert rows[0]["status"] == "succeeded"
        payload = json.loads(rows[0]["payload_json"])
        assert payload["gate"] == "alumni_blacklist"
        assert payload["gate_match"]["field"] == "experience"
        assert pipeline.state.candidates._prior_outcomes[snippet.profile_url] == "REJECT"


def test_post_profile_gate_lets_a_clean_profile_reach_the_judge() -> None:
    from shared.schemas import OpusDecision, SearchString
    from tests.test_full_eval_pipelining import (
        _candidate_rows,
        _make_pipeline,
        _string_stats,
        _wire_common_pipeline,
    )

    with tempfile.TemporaryDirectory() as td:
        pipeline = _make_pipeline(td)
        pipeline.state.context.brief_obj.alumni_blacklist = list(ALUMNI_BLACKLIST)
        snippet = _card_only_current_role()
        _wire_common_pipeline(pipeline)
        clean = _avery_profile()
        clean.experiences = [e for e in clean.experiences if "Babbage" not in e.company]
        pipeline.state.reviewer.acquisition.extract_profile_summary = AsyncMock(
            return_value=SimpleNamespace(profile_summary=clean)
        )
        judged: list[str] = []

        def judge(summary, brief, lane_context=None):
            judged.append(summary.name)
            return OpusDecision(
                stage="full",
                decision="REJECT",
                path="none",
                confidence=0.8,
                rationale="not a fit",
                candidate_name=summary.name,
                profile_url=summary.profile_url,
            )

        with patch("linkedin.candidate_review.full_judge", side_effect=judge), patch(
            "linkedin.evaluation_loop.config.LINKEDIN_EXTERNAL_EVIDENCE_ENABLED", False
        ), patch("linkedin.candidate_review.human_delay_correlated", return_value=0.0):
            decisions = asyncio.run(
                pipeline.state.reviewer._run_full_queue(
                    snippets=[snippet],
                    window=1,
                    origin="page",
                    page_report=None,
                    search_string=SearchString(id=7, name="test string", boolean="x"),
                    all_candidates=_candidate_rows([snippet]),
                    string_stats=_string_stats(),
                    progress=None,
                    page_num=1,
                )
            )
        assert judged == ["Avery Quinn"]
        assert decisions[0].path == "none"


def test_serial_full_evaluate_blocks_alum_before_judge_and_save() -> None:
    from tests.test_full_eval_pipelining import (
        _make_pipeline,
        _wire_common_pipeline,
        _wire_real_runtime,
    )

    with tempfile.TemporaryDirectory() as td:
        pipeline = _make_pipeline(td)
        pipeline.state.context.brief_obj.alumni_blacklist = list(ALUMNI_BLACKLIST)
        snippet = _card_only_current_role()
        _wire_common_pipeline(pipeline)
        search_string, _ = _wire_real_runtime(pipeline, output_dir=td, snippets=[snippet])
        pipeline.state.reviewer.acquisition.extract_profile_summary = AsyncMock(
            return_value=SimpleNamespace(profile_summary=_avery_profile())
        )
        judge = MagicMock(side_effect=AssertionError("full judge must not run"))
        with patch("linkedin.candidate_review.full_judge", judge), patch(
            "linkedin.evaluation_loop.config.LINKEDIN_EXTERNAL_EVIDENCE_ENABLED", False
        ), patch("linkedin.candidate_review.human_delay_correlated", return_value=0.0):
            decision = asyncio.run(
                pipeline.state.reviewer._full_evaluate(snippet, None, search_string)
            )
        judge.assert_not_called()
        pipeline.state.reviewer.side_effects.handle_save_decision.assert_not_awaited()
        assert decision.decision == "REJECT" and decision.path == "alumni_blacklist"


def _snippet_alumni_hit(snippet):
    from linkedin.candidate_review import CandidateReviewer

    return CandidateReviewer._alumni_blacklist_match(snippet, ALUMNI_BLACKLIST)


def test_resume_overlays_the_seed_alumni_blacklist_on_the_frozen_generated_brief() -> None:
    from shared.brief_loader import normalize_brief
    from shared.preflight_v2 import overlay_seed_alumni_blacklist

    generated = _v2_brief(alumni_blacklist=[])
    generated.pop("alumni_blacklist")
    seed = normalize_brief(_legacy_seed())
    merged, added = overlay_seed_alumni_blacklist(generated, seed)
    assert merged["alumni_blacklist"] == ["Babbage"]
    assert added == ["Babbage"]
    # Union, never a narrowing; idempotent.
    merged_again, added_again = overlay_seed_alumni_blacklist(merged, seed)
    assert merged_again["alumni_blacklist"] == ["Babbage"]
    assert added_again == []


# ---------------------------------------------------------------------------
# Exa lane: full-history gate before the judge, slate exclusion, replay
# ---------------------------------------------------------------------------


_BABBAGE_ALUM_RENDER = """# Avery Quinn
Engineering Leader | AI, LLMs & Agent-Based Systems
San Francisco, California, United States
500 connections

## Experience
### Sr. Manager, Forward Deploy Engineering - [Labelstream AI](https://www.linkedin.com/company/labelstream) (Current)
Aug 2026 - Present (2 months)
### Director, Software Engineering - [Babbage](https://www.linkedin.com/company/babbage)
Dec 2025 - Jul 2026 (8 months)
### Director of Software Engineering - [Ledgerline](https://www.linkedin.com/company/ledgerline)
Jun 2018 - Dec 2021 (3 years)
"""

_OPENAI_ALUM_RENDER = """# Ada Ex
Research Engineer | ex-OpenAI
San Francisco, California, United States
500 connections

## Experience
### Founder - [Stealth](https://www.linkedin.com/company/stealth) (Current)
Jan 2025 - Present (1 year)
### Member of Technical Staff - [OpenAI](https://www.linkedin.com/company/openai)
Jan 2021 - Dec 2024 (4 years)
"""


def _exa_people():
    from tests.test_exa_blacklist_gate import _render_person

    return {
        "Benchmark": [
            _render_person(
                "averyquinn",
                "Avery Quinn",
                "Avery Quinn - Sr. Manager - Labelstream AI | LinkedIn",
                _BABBAGE_ALUM_RENDER,
            ),
            _render_person(
                "adaex",
                "Ada Ex",
                "Ada Ex - Founder - Stealth | LinkedIn",
                _OPENAI_ALUM_RENDER,
            ),
        ]
    }


def test_exa_gate_rejects_a_babbage_alum_before_the_judge(tmp_path: Path, capsys) -> None:
    from tests.test_exa_blacklist_gate import _terminal
    from tests.test_exa_orchestrator import (
        ScriptedJudge,
        _jsonl,
        _pipeline,
        _state_dir,
        _write_brief,
    )

    brief_path = _write_brief(
        tmp_path,
        employer_blacklist=list(EMPLOYER_BLACKLIST),
        alumni_blacklist=list(ALUMNI_BLACKLIST),
    )
    judge = ScriptedJudge({"Avery Quinn": "SAVE", "Ada Ex": "SAVE"})
    pipeline, _ = _pipeline(tmp_path, brief_path=brief_path, results=_exa_people(), judge=judge)

    stats = asyncio.run(pipeline.run())

    assert judge.seen == ["Ada Ex"], "a Babbage alum reached the judge, or an OpenAI alum did not"
    avery = _terminal(tmp_path, pipeline, "averyquinn")
    assert avery["terminal_decision"] == "REJECT"
    payload = json.loads(avery["terminal_payload_json"])
    assert payload["full_decision"]["path"] == "alumni_blacklist"
    assert payload["full_decision"]["reject_reason"] == "HARD_GATE"

    rows = {row["name"]: row for row in _jsonl(_state_dir(tmp_path) / "judgments.jsonl")}
    assert rows["Avery Quinn"]["gate"] == "alumni_blacklist"
    assert "alumni blacklist (Babbage)" in rows["Avery Quinn"]["rationale"]
    assert stats.alumni_gated == 1
    assert stats.blacklist_gated == 0
    assert stats.judged == 1
    out = capsys.readouterr().out
    assert any(
        "SKIP   Avery Quinn" in line and "alumni blacklist (Babbage)" in line
        for line in out.splitlines()
    )

    slate = (_state_dir(tmp_path) / "exa-slate.md").read_text()
    assert "1 person rejected before judgment" in slate
    assert "gate alumni_blacklist" in slate


def test_exa_slate_pulls_a_stored_babbage_alum_save_off_the_short_list(tmp_path: Path) -> None:
    from tests.test_exa_orchestrator import ScriptedJudge, _pipeline, _state_dir, _write_brief

    first_judge = ScriptedJudge({"Avery Quinn": "SAVE", "Ada Ex": "SAVE"})
    first, _ = _pipeline(
        tmp_path,
        brief_path=_write_brief(tmp_path, employer_blacklist=list(EMPLOYER_BLACKLIST)),
        results=_exa_people(),
        judge=first_judge,
    )
    asyncio.run(first.run())
    assert sorted(first_judge.seen) == ["Ada Ex", "Avery Quinn"]

    second_judge = ScriptedJudge({})
    second, _ = _pipeline(
        tmp_path,
        brief_path=_write_brief(
            tmp_path,
            employer_blacklist=list(EMPLOYER_BLACKLIST),
            alumni_blacklist=list(ALUMNI_BLACKLIST),
        ),
        results=_exa_people(),
        judge=second_judge,
    )
    asyncio.run(second.run(resume=True))
    assert second_judge.seen == []

    slate = (_state_dir(tmp_path) / "exa-slate.md").read_text()
    short_list = slate.split("## The short list", 1)[1].split("\n## ", 1)[0]
    assert "Ada Ex" in short_list
    assert "Avery Quinn" not in short_list, "a Babbage alum stayed on the short list"
    excluded = slate.split("## Excluded", 1)[1].split("\n## ", 1)[0]
    assert "**Avery Quinn**" in excluded
    assert "alumni blacklist (Babbage)" in excluded


def test_exa_replay_never_saves_a_babbage_alum(tmp_path: Path, capsys) -> None:
    from exa.orchestrator import ExaRunStats
    from tests.test_exa_orchestrator import _pipeline, _write_brief

    pipeline, _ = _pipeline(
        tmp_path,
        brief_path=_write_brief(tmp_path, alumni_blacklist=list(ALUMNI_BLACKLIST)),
    )
    entry = SimpleNamespace(
        is_terminal=True,
        terminal_decision="SAVE",
        identity_key="linkedin.com/in/averyquinn",
        last_attempt_id=1,
    )
    pipeline.bridge = MagicMock()
    pipeline.bridge.succeeded_side_effect_keys.return_value = set()
    pipeline.bridge.list_campaign_candidates.return_value = [entry]
    pipeline._save_to_workspace = MagicMock(return_value=True)
    pipeline._replay_envelope = MagicMock()

    from exa.schemas import ExaCandidate

    alum = ExaCandidate(
        identity_key="linkedin.com/in/averyquinn",
        name="Avery Quinn",
        linkedin_url="https://www.linkedin.com/in/averyquinn",
        current_title="Sr. Manager",
        current_firm="Labelstream AI",
        profile_render={
            "name": "Avery Quinn",
            "headline": "Engineering Leader",
            "experiences": [
                {"title": "Sr. Manager", "company": "Labelstream AI", "is_current": True},
                {"title": "Director, Software Engineering", "company": "Babbage"},
            ],
        },
    )
    with patch("exa.orchestrator._candidate_from_record", return_value=alum), patch(
        "exa.orchestrator._decision_from_entry", return_value=MagicMock(decision="SAVE")
    ):
        stats = ExaRunStats()
        pipeline._replay_interrupted_saves(
            run_id=1, progress=MagicMock(), engine=MagicMock(), stats=stats
        )
    pipeline._save_to_workspace.assert_not_called()
    assert stats.replayed_saves == 0
    assert "Avery Quinn" in capsys.readouterr().out


def test_exa_lane_compiler_refuses_an_alumni_blacklisted_target_firm() -> None:
    from exa.lane_compiler import lint_exa_brief

    brief = SimpleNamespace(
        exa_target_firms=[{"name": "Babbage"}],
        exa_query_lanes=["people_search"],
        exa_title_terms=["Director"],
        exa_archetype_queries=[],
        employer_blacklist=[],
        alumni_blacklist=list(ALUMNI_BLACKLIST),
        engagement_context={},
        raw={},
    )
    codes = {finding.code for finding in lint_exa_brief(brief)}
    assert "employer_conflict" in codes


def test_strategy_scrubs_alumni_blacklisted_employers_from_targeting() -> None:
    from linkedin.strategy import _blacklisted_employer_keys

    brief = SimpleNamespace(employer_blacklist=["OpenAI"], alumni_blacklist=["Babbage"])
    assert _blacklisted_employer_keys(brief) == {"openai", "babbage"}


def test_profile_gate_reads_a_position_title_that_names_the_employer() -> None:
    from linkedin.candidate_review import CandidateReviewer

    profile = CandidateProfileSummary(
        name="Pat Kim",
        profile_url="/talent/profile/pat-kim",
        headline="Delivery leader",
        experiences=[
            Experience(title="Program Lead", company="Acme", start="2026", end="Present"),
            # Extraction sometimes folds the company into the title.
            Experience(title="Engineering Manager at Babbage", company="", start="2024", end="2025"),
        ],
    )
    snippet = _avery_snippet(entries=[])
    hit = CandidateReviewer._profile_alumni_blacklist_match(snippet, profile, ALUMNI_BLACKLIST)
    assert hit == ("Babbage", "experience_entry", "Engineering Manager at Babbage")
    # A title that merely mentions the name is a topic, not a job.
    profile.experiences[1].title = "Researcher, Babbage engine replicas"
    assert (
        CandidateReviewer._profile_alumni_blacklist_match(snippet, profile, ALUMNI_BLACKLIST)
        is None
    )


# Real card lines from project 9000000003 / 9000000008 artifacts (read-only
# replay, 2026-10-01): an online school, not the hiring company.
@pytest.mark.parametrize(
    "entry",
    [
        "Academic Board Member at Babbage College (2023 – 2024)",
        "Visiting Lecturer at Babbage University (2019 – 2020)",
    ],
)
def test_babbage_named_schools_are_not_the_company(entry: str) -> None:
    assert match_any_employer(ALUMNI_BLACKLIST, experience_entries=[entry]) is None
    company = entry.split(" at ", 1)[1].split(" (", 1)[0]
    assert match_any_employer(ALUMNI_BLACKLIST, experience_companies=[company]) is None
    assert match_current_employer(company, "", EMPLOYER_BLACKLIST) is None


def test_morgan_hale_shape_is_caught_only_by_the_post_profile_gate() -> None:
    # Saved in run-4 of 9000000003: the Babbage stint is his fifth position,
    # below the three the search card shows.
    from linkedin.candidate_review import CandidateReviewer

    snippet = _avery_snippet(
        entries=[
            "Operations Manager (Human Data) at Meridian AI Solutions (2025 – Present)",
            "Expert Human Data Annotator at Stealth AI Startup (2025 – 2026)",
            "Human Data Manager at Datawell (2025 – 2025)",
        ]
    )
    snippet.current_company = "Meridian AI Solutions"
    snippet.headline = "Human Data Ops · Evaluation Architecture · Rubric Design"
    assert CandidateReviewer._alumni_blacklist_match(snippet, ALUMNI_BLACKLIST) is None
    profile = CandidateProfileSummary(
        name="Morgan Hale",
        profile_url=snippet.profile_url,
        headline=snippet.headline,
        experiences=[
            Experience(title="Operations Manager ( Human Data)", company="Meridian AI Solutions"),
            Experience(title="Expert Human Data Annotator", company="Stealth AI Startup"),
            Experience(title="Human Data Manager", company="Datawell"),
            Experience(title="Final Adjudicator (Chemistry)", company="Labelstream AI"),
            Experience(title="LLM Delivery QA Lead (STEM)", company="Babbage"),
        ],
    )
    hit = CandidateReviewer._profile_alumni_blacklist_match(snippet, profile, ALUMNI_BLACKLIST)
    assert hit == ("Babbage", "experience", "Babbage")
