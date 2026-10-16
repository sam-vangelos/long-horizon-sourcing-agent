"""The LinkedIn gates enforce a brief's ``employer_allowlist`` with no model call.

On the card and again on the full profile, a person whose CURRENT employer is
positively identified and is not a listed employer is skipped as out of scope
before any judge runs: tagged ``employer_scope``, counted, and printed. A
person with no identified current employer passes to the judge. The employer
and alumni blacklists keep precedence, and a brief without the field is
untouched.
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

from shared.schemas import CandidateProfileSummary, CandidateSnippet, Experience

ALLOWLIST = {
    "facet_groups": [["Mercor"], ["Handshake"], ["Scale AI"], ["Appen", "Labelbox"]],
    "aliases": {"Outlier": "Scale AI", "Mercor Intelligence": "Mercor"},
}


def _card(name: str = "Off Scope", *, company: str = "Google", entries=None, rank: int = 1) -> CandidateSnippet:
    slug = name.lower().replace(" ", "-")
    return CandidateSnippet(
        name=name,
        headline="Program Manager",
        current_title="Program Manager",
        current_company=company,
        location="United States",
        education_snippet="",
        profile_url=f"/talent/profile/{slug}",
        source_string_id=7,
        source_string_name="test string",
        page=1,
        result_rank=rank,
        card_index=rank - 1,
        experience_entries=list(entries or []),
    )


def _reviewer(raw):
    from linkedin.candidate_review import CandidateReviewer

    reviewer = object.__new__(CandidateReviewer)
    reviewer.context = SimpleNamespace(
        brief_obj=SimpleNamespace(employer_blacklist=["Babbage", "OpenAI"], alumni_blacklist=["Babbage"], raw=raw)
    )
    return reviewer


# ---------------------------------------------------------------------------
# The card gate
# ---------------------------------------------------------------------------


def test_an_identified_unlisted_current_employer_is_out_of_scope_on_the_card() -> None:
    from linkedin.candidate_review import EMPLOYER_SCOPE_GATE_PATH

    path, (blocked, field, value) = _reviewer({"employer_allowlist": ALLOWLIST})._facial_employer_gate(_card())

    assert path == EMPLOYER_SCOPE_GATE_PATH == "employer_scope"
    assert (blocked, field, value) == ("Google", "current_company", "Google")


@pytest.mark.parametrize(
    "company, entries",
    [
        ("Mercor Intelligence", []),
        ("Outlier", []),
        ("", []),
        ("Self-employed", []),
        ("Startup", []),
        ("Stealth", []),
        # A current competitor seat on the card's history line, beside a day job.
        ("Stanford University", ["AI Trainer at Outlier (2024 – Present)"]),
    ],
)
def test_a_listed_or_unidentified_current_employer_passes_the_card(company, entries) -> None:
    gate = _reviewer({"employer_allowlist": ALLOWLIST})._facial_employer_gate(
        _card(company=company, entries=entries)
    )

    assert gate is None


def test_a_past_competitor_on_the_card_does_not_put_a_person_in_scope() -> None:
    card = _card(entries=["Program Manager at Google (2025 – Present)", "SPL at Scale AI (2023 – 2025)"])

    path, _hit = _reviewer({"employer_allowlist": ALLOWLIST})._facial_employer_gate(card)

    assert path == "employer_scope"


def test_the_blacklists_keep_precedence_over_the_scope_gate() -> None:
    path, _hit = _reviewer({"employer_allowlist": ALLOWLIST})._facial_employer_gate(_card(company="OpenAI"))
    assert path == "employer_blacklist"

    alum = _card(company="Google", entries=["PM at Google (2025 – Present)", "Lead at Babbage (2023 – 2025)"])
    path, _hit = _reviewer({"employer_allowlist": ALLOWLIST})._facial_employer_gate(alum)
    assert path == "alumni_blacklist"


@pytest.mark.parametrize("raw", [{}, MagicMock(), {"employer_allowlist": {"facet_groups": "Mercor"}}])
def test_a_brief_without_a_well_formed_allowlist_never_fires_the_scope_gate(raw) -> None:
    assert _reviewer(raw)._facial_employer_gate(_card()) is None


def test_serial_facial_gate_settles_an_out_of_scope_card_without_a_facial_call() -> None:
    from tests.test_full_eval_pipelining import _make_pipeline

    with tempfile.TemporaryDirectory() as td:
        pipeline = _make_pipeline(td)
        pipeline.state.context.brief_obj.alumni_blacklist = []
        pipeline.state.context.brief_obj.raw = {"employer_allowlist": copy.deepcopy(ALLOWLIST)}
        snippet = _card()
        facial = MagicMock(side_effect=AssertionError("facial judge must not run"))
        with patch("linkedin.candidate_review.facial_judge", facial):
            decision = asyncio.run(pipeline.state.reviewer._evaluate_snippet(snippet, None, None))

        facial.assert_not_called()
        assert decision.decision == "FACIAL_NO"
        assert decision.path == "employer_scope"
        assert decision.rationale == "Outside the competitor scope: Google (current_company)"
        assert pipeline.state.context.stats["employer_scope_skips"] == 1
        assert pipeline.state.candidates._prior_outcomes[snippet.profile_url] == "FACIAL_NO"


def test_batch_facial_gate_keeps_an_out_of_scope_card_from_the_batch_judge(capsys) -> None:
    from shared.schemas import OpusDecision, SearchString
    from tests.test_full_eval_pipelining import _make_pipeline

    with tempfile.TemporaryDirectory() as td:
        pipeline = _make_pipeline(td)
        brief = pipeline.state.context.brief_obj
        brief.has_v2_schema = True
        brief._new_brief = None
        brief.alumni_blacklist = []
        brief.raw = {"employer_allowlist": copy.deepcopy(ALLOWLIST)}
        pipeline.state.reviewer._tightening_prefix = ""
        pipeline.state.reviewer._triage_tightened = False
        off = _card("Off Scope", rank=1)
        on = _card("On Scope", company="Mercor", rank=2)
        seen: list[str] = []

        def batch(snippets, *args, **kwargs):
            seen.extend(s.name for s in snippets)
            return [
                OpusDecision(
                    stage="facial", decision="FACIAL_NO", path="none", confidence=1.0,
                    rationale="not relevant", candidate_name=s.name, profile_url=s.profile_url,
                )
                for s in snippets
            ]

        pipeline.state.context.browser.get_card_slot_count = AsyncMock(return_value=2)
        pipeline.state.pages._extract_card_snippet = AsyncMock(side_effect=[off, on])
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

        assert "Off Scope" not in seen, "an out-of-scope card reached the facial judge"
        assert "On Scope" in seen
        row = next(c for c in all_candidates if c["name"] == "Off Scope")
        assert row["outcome"] == "employer_scope_skip"
        assert row["rationale"] == "Outside the competitor scope: Google (current_company)"
        assert pipeline.state.context.stats["employer_scope_skips"] == 1
        assert pipeline.state.pages._page_observation()["skipped_blacklist"] == 1
        # Only the judged card counts as facial_no.
        assert string_stats["facial_no"] == 1
        assert pipeline.state.context.stats["facial_no"] == 1
        out = capsys.readouterr().out
        assert "[OUT-OF-SCOPE] Off Scope — current employer 'Google' is not a listed competitor" in out


# ---------------------------------------------------------------------------
# The post-profile gate: the full work history names the current employer
# ---------------------------------------------------------------------------


def _profile(name: str, current_company: str) -> CandidateProfileSummary:
    return CandidateProfileSummary(
        name=name,
        profile_url=f"/talent/profile/{name.lower().replace(' ', '-')}",
        headline="Operations",
        experiences=[
            Experience(title="Program Manager", company=f"{current_company} · Full-time", start="Jan 2026", end="Present"),
            Experience(title="Strategic Projects Lead", company="Scale AI", start="Jan 2023", end="Dec 2025"),
        ],
    )


def _run_full(pipeline, snippet, profile, judge, *, all_candidates=None, string_stats=None):
    from shared.schemas import SearchString
    from tests.test_full_eval_pipelining import _candidate_rows, _string_stats, _wire_common_pipeline

    _wire_common_pipeline(pipeline)
    pipeline.state.reviewer.acquisition.extract_profile_summary = AsyncMock(
        return_value=SimpleNamespace(profile_summary=profile)
    )
    with patch("linkedin.candidate_review.full_judge", side_effect=judge), patch(
        "linkedin.evaluation_loop.config.LINKEDIN_EXTERNAL_EVIDENCE_ENABLED", False
    ), patch("linkedin.candidate_review.human_delay_correlated", return_value=0.0):
        return asyncio.run(
            pipeline.state.reviewer._run_full_queue(
                snippets=[snippet],
                window=1,
                origin="page",
                page_report=None,
                search_string=SearchString(id=7, name="test string", boolean="x"),
                all_candidates=(
                    _candidate_rows([snippet]) if all_candidates is None else all_candidates
                ),
                string_stats=_string_stats() if string_stats is None else string_stats,
                progress=None,
                page_num=1,
            )
        )


def test_post_profile_gate_rejects_an_unlisted_current_employer_before_the_judge() -> None:
    from tests.test_full_eval_pipelining import _make_pipeline

    with tempfile.TemporaryDirectory() as td:
        pipeline = _make_pipeline(td)
        pipeline.state.context.brief_obj.alumni_blacklist = []
        pipeline.state.context.brief_obj.raw = {"employer_allowlist": copy.deepcopy(ALLOWLIST)}
        # The card names no employer, so it passed the facial gate.
        snippet = _card("Ex Scale", company="")
        judge = MagicMock(side_effect=AssertionError("full judge must not run"))

        decisions = _run_full(pipeline, snippet, _profile("Ex Scale", "Google"), judge)

        judge.assert_not_called()
        pipeline.state.reviewer.side_effects.handle_save_decision.assert_not_awaited()
        assert len(decisions) == 1
        decision = decisions[0]
        assert (decision.stage, decision.decision, decision.path) == ("full", "REJECT", "employer_scope")
        assert decision.reject_reason == "HARD_GATE"
        assert "Outside the competitor scope" in decision.rationale and "Google" in decision.rationale
        assert pipeline.state.context.stats["employer_scope_rejects"] == 1


def test_post_profile_gate_lets_a_listed_current_employer_reach_the_judge() -> None:
    from shared.schemas import OpusDecision
    from tests.test_full_eval_pipelining import _make_pipeline

    with tempfile.TemporaryDirectory() as td:
        pipeline = _make_pipeline(td)
        pipeline.state.context.brief_obj.alumni_blacklist = []
        pipeline.state.context.brief_obj.raw = {"employer_allowlist": copy.deepcopy(ALLOWLIST)}
        snippet = _card("On Scope", company="")
        judged: list[str] = []

        def judge(summary, brief, lane_context=None):
            judged.append(summary.name)
            return OpusDecision(
                stage="full", decision="REJECT", path="none", confidence=0.8,
                rationale="not a fit", candidate_name=summary.name, profile_url=summary.profile_url,
            )

        decisions = _run_full(pipeline, snippet, _profile("On Scope", "Mercor"), judge)

        assert judged == ["On Scope"]
        assert decisions[0].path == "none"


# ---------------------------------------------------------------------------
# An out-of-scope skip is not noise: the adaptation layer never learns from it
# ---------------------------------------------------------------------------
#
# The Companies facet matches current OR past employees, so a competitor string
# surfaces that competitor's alumni: the right seat at the wrong current
# employer. Counting them as facial_no would read a working vocabulary as noise
# and spend the mutation budget rewriting it. Same rule as the territory gate.


def test_a_page_of_out_of_scope_cards_feeds_no_noise_signal() -> None:
    from linkedin.search_policy import _build_page_insights, _scout_metrics
    from shared.schemas import SearchString
    from tests.test_full_eval_pipelining import _make_pipeline

    with tempfile.TemporaryDirectory() as td:
        pipeline = _make_pipeline(td)
        brief = pipeline.state.context.brief_obj
        brief.has_v2_schema = True
        brief._new_brief = None
        brief.alumni_blacklist = []
        brief.raw = {"employer_allowlist": copy.deepcopy(ALLOWLIST)}
        pipeline.state.reviewer._tightening_prefix = ""
        pipeline.state.reviewer._triage_tightened = False
        cards = [_card(f"Scale Alum {i}", company="Google", rank=i + 1) for i in range(20)]
        batch = MagicMock(side_effect=AssertionError("no card may reach the facial judge"))

        pipeline.state.context.browser.get_card_slot_count = AsyncMock(return_value=len(cards))
        pipeline.state.pages._extract_card_snippet = AsyncMock(side_effect=cards)
        pipeline.state.reviewer._run_full_queue = AsyncMock(return_value=[])
        all_candidates: list[dict] = []
        string_stats = pipeline.state.candidates._fresh_string_stats()
        before = dict(string_stats)
        search_string = SearchString(id=7, name="test string", boolean="engineer")
        with patch("shared.judger.facial_judge_batch", batch), patch(
            "linkedin.evaluation_loop.config.LINKEDIN_FACIAL_CONCURRENCY_ENABLED", False
        ):
            asyncio.run(
                pipeline.state.pages._review_page_batch(
                    search_string, 1, 0, MagicMock(), all_candidates, string_stats, None
                )
            )

        assert {row["outcome"] for row in all_candidates} == {"employer_scope_skip"}
        assert pipeline.state.context.stats["employer_scope_skips"] == 20
        assert string_stats["facial_no"] == 0
        assert pipeline.state.context.stats.get("facial_no", 0) == 0
        insights = _build_page_insights(
            page_num=1,
            result_count=900,
            preview_snippets=cards,
            all_candidates=all_candidates,
            glance_result=None,
        )
        assert insights.noise_anchors == []
        assert insights.dominant_non_fit_patterns == []
        page_stats = pipeline.state.candidates._page_stat_delta(before, string_stats)
        assert _scout_metrics(page_stats=page_stats, page_insights=insights)["noise_dominant"] is False


def test_serial_out_of_scope_skip_stays_out_of_the_facial_no_counters() -> None:
    from tests.test_full_eval_pipelining import _make_pipeline

    with tempfile.TemporaryDirectory() as td:
        pipeline = _make_pipeline(td)
        pipeline.state.context.brief_obj.alumni_blacklist = []
        pipeline.state.context.brief_obj.raw = {"employer_allowlist": copy.deepcopy(ALLOWLIST)}
        string_stats = pipeline.state.candidates._fresh_string_stats()
        with patch("linkedin.candidate_review.facial_judge", MagicMock()):
            decision = asyncio.run(
                pipeline.state.reviewer._evaluate_snippet(_card(), None, None, string_stats=string_stats)
            )

        assert decision.path == "employer_scope"
        assert pipeline.state.context.stats["employer_scope_skips"] == 1
        assert string_stats["facial_no"] == 0
        assert pipeline.state.context.stats.get("facial_no", 0) == 0


def test_a_profile_out_of_scope_reject_is_not_a_full_reject_or_a_noise_anchor() -> None:
    from linkedin.search_policy import _build_page_insights
    from tests.test_full_eval_pipelining import _candidate_rows, _make_pipeline, _string_stats

    with tempfile.TemporaryDirectory() as td:
        pipeline = _make_pipeline(td)
        pipeline.state.context.brief_obj.alumni_blacklist = []
        pipeline.state.context.brief_obj.raw = {"employer_allowlist": copy.deepcopy(ALLOWLIST)}
        snippet = _card("Ex Scale", company="")
        rows = _candidate_rows([snippet])
        string_stats = _string_stats()
        judge = MagicMock(side_effect=AssertionError("full judge must not run"))

        _run_full(
            pipeline, snippet, _profile("Ex Scale", "Google"), judge,
            all_candidates=rows, string_stats=string_stats,
        )

        assert rows[0]["outcome"] == "employer_scope_skip"
        assert string_stats.get("full_reject", 0) == 0
        assert string_stats.get("full_reviewed", 0) == 0
        assert string_stats.get("rejects", 0) == 0
        assert pipeline.state.context.stats.get("full_reject", 0) == 0
        assert pipeline.state.context.stats["employer_scope_rejects"] == 1
        insights = _build_page_insights(
            page_num=1, result_count=900, preview_snippets=[], all_candidates=rows, glance_result=None,
        )
        assert insights.noise_anchors == []


def test_the_sequential_page_loop_records_scope_skips_in_their_own_bucket() -> None:
    """A v1 brief's card-by-card loop maps both scope outcomes the same way."""
    from shared.schemas import OpusDecision, SearchString
    from tests.test_full_eval_pipelining import _make_pipeline

    with tempfile.TemporaryDirectory() as td:
        pipeline = _make_pipeline(td)
        pipeline.state.context.brief_obj.has_v2_schema = False
        card = _card("Card Alum", rank=1)
        profile = _card("Profile Alum", company="", rank=2)
        decisions = [
            OpusDecision(
                stage="facial", decision="FACIAL_NO", path="employer_scope", confidence=1.0,
                rationale="Outside the competitor scope: Google (current_company)",
                candidate_name=card.name, profile_url=card.profile_url,
            ),
            OpusDecision(
                stage="full", decision="REJECT", path="employer_scope", confidence=1.0,
                rationale="Outside the competitor scope", candidate_name=profile.name,
                profile_url=profile.profile_url, reject_reason="HARD_GATE",
            ),
        ]
        pipeline.state.context.browser.get_card_slot_count = AsyncMock(return_value=2)
        pipeline.state.pages._extract_card_snippet = AsyncMock(side_effect=[card, profile])
        pipeline.state.reviewer._evaluate_snippet = AsyncMock(side_effect=decisions)
        pipeline.state.pages._preview_skip_pause = AsyncMock()
        all_candidates: list[dict] = []
        string_stats = pipeline.state.candidates._fresh_string_stats()
        search_string = SearchString(id=7, name="test string", boolean="engineer")

        asyncio.run(
            pipeline.state.pages._review_page_sequentially(
                search_string, 1, 0, MagicMock(), all_candidates, string_stats, None
            )
        )

        assert [row["outcome"] for row in all_candidates] == ["employer_scope_skip"] * 2
        assert string_stats["facial_no"] == 0
        assert string_stats.get("full_reject", 0) == 0
        assert string_stats.get("full_reviewed", 0) == 0


def test_a_resumed_run_keeps_scope_skips_out_of_the_funnel_counters() -> None:
    """Resume rebuilds counters from canonical stage rows, where a scope skip is
    FACIAL_NO or REJECT; the row's path keeps it in its own bucket."""
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
        stages = (
            ("Card Alum", (("facial", "FACIAL_NO", "employer_scope"),)),
            ("Judged No", (("facial", "FACIAL_NO", "none"),)),
            ("Profile Alum", (("facial", "FACIAL_YES", "none"), ("full", "REJECT", "employer_scope"))),
            ("Judged Reject", (("facial", "FACIAL_YES", "none"), ("full", "REJECT", "none"))),
        )
        for name, rows in stages:
            snippet = _make_snippet(
                name=name,
                profile_url=f"/talent/profile/{name.replace(' ', '-').lower()}",
                source_string_id=owner.id,
                source_string_name=owner.name,
                page=1,
            )
            p.state.reviewer.attempts._record_runtime_snippet(owner, snippet)
            for stage, verdict, path in rows:
                attempt_id = p.state.reviewer.attempts._start_runtime_stage_attempt(
                    search_string=owner, snippet=snippet, stage=stage,
                )
                p.state.reviewer._finish_runtime_stage_success(
                    attempt_id=attempt_id,
                    stage=stage,
                    snippet=snippet,
                    decision=OpusDecision(
                        stage=stage, decision=verdict, path=path, confidence=1.0,
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
        assert owner.full_reject_count == 1
        assert owner.full_reviewed_count == 1
        stats = p.state.context.stats
        assert stats["facial_no"] == 1
        assert stats["full_reject"] == 1
        assert stats["full_reviewed"] == 1
        # The run total of rejects still names both: the live settle counts a
        # profile scope reject there too.
        assert stats["rejected"] == 2


# ---------------------------------------------------------------------------
# The seed field rides to the gates on every path
# ---------------------------------------------------------------------------


def _seed(**extra) -> dict:
    seed = {
        "name": "scope-seed",
        "role_title": "Strategic Project Lead",
        "linkedin_project": "Test project",
        "linkedin_project_id": "123",
        "permanent_filters": {"Location": "United States"},
        "employer_blacklist": ["Babbage"],
        "alumni_blacklist": ["Babbage"],
        "employer_allowlist": copy.deepcopy(ALLOWLIST),
        "instructions": ["THE TARGET: an operations leader who can read code."],
        "jd": "# The Role\n\nBabbage is hiring a Strategic Project Lead.",
    }
    seed.update(extra)
    return seed


def test_preflight_carries_the_allowlist_into_the_generated_brief() -> None:
    from shared.brief_loader import normalize_brief
    from shared.preflight_v2 import preflight_overrides_from_brief

    overrides = preflight_overrides_from_brief(normalize_brief(_seed()), {}, geography="United States")

    assert overrides["employer_allowlist"] == ALLOWLIST


def test_a_resumed_session_reloads_the_allowlist_from_the_seed() -> None:
    from shared.brief_loader import _load_v2_brief, normalize_brief
    from shared.employer_allowlist import employer_allowlist_from_brief
    from shared.preflight_v2 import overlay_seed_onto_resumed_brief

    frozen = json.loads(
        (Path(__file__).parent / "fixtures" / "field_engineering_resume" / "preflight_v2_brief.json").read_text()
    )
    merged = overlay_seed_onto_resumed_brief(copy.deepcopy(frozen), normalize_brief(_seed()), emit=lambda _l: None)
    loaded = _load_v2_brief(merged)

    assert loaded.raw["employer_allowlist"] == ALLOWLIST
    assert employer_allowlist_from_brief(loaded).names == ("Mercor", "Handshake", "Scale AI", "Appen", "Labelbox")


def test_a_malformed_allowlist_is_not_carried() -> None:
    from shared.brief_loader import normalize_brief
    from shared.preflight_v2 import preflight_overrides_from_brief

    seed = normalize_brief(_seed(employer_allowlist={"facet_groups": "Mercor"}))

    assert "employer_allowlist" not in preflight_overrides_from_brief(seed, {}, geography="United States")


def test_the_v2_schema_recognizes_the_field() -> None:
    from shared.brief_v2_schema import RECOGNIZED_V2_KEYS

    assert "employer_allowlist" in RECOGNIZED_V2_KEYS
