"""Client-side employers in an ``employer_allowlist`` count only in data seats.

Req 1238's ideal candidate profile of 2026-10-14 brings the AI-data vendors'
clients into scope (Meta, Amazon, xAI, SpaceX, Google DeepMind), but only on
their human-data, post-training and evaluation side. Those companies employ
tens of thousands of engineers, so a client facet without a title bound would
flood the queue. The allowlist's optional ``title_restricted`` block names
those employers and the title families that put a person there in scope, and
every deterministic path honours it:

- the card and full-profile gates keep a client employee only when a current
  title (or the headline) falls in the families;
- the per-string stamp never hands a client group to an unfaceted string,
  strips clients from a facet that mixes them with competitors, and puts the
  family titles on every client string's title facet;
- a client string whose title facet is empty, or landed a title outside the
  families, is skipped;
- a variant may not promote a client company, demote a client string's
  titles, or promote a title outside the families onto one, and a client
  variant whose titles fail to apply is refused rather than run on the
  company facet alone.

A brief without the block is untouched (the 2026-10-08 competitor tests pin
that behaviour).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared.employer_allowlist import parse_employer_allowlist
from shared.schemas import CandidateProfileSummary, CandidateSnippet, Experience, SearchString

CLIENTS = ["Meta", "Amazon", "xAI", "SpaceX", "Google DeepMind"]
FACET_TITLES = [
    "Human Data Manager",
    "Human Data Lead",
    "Human Data Engineer",
    "Data Operations Manager",
    "Post-training Program Manager",
    "Evaluation Program Manager",
]
FIELD = {
    "facet_groups": [["Mercor"], ["Handshake"], ["Scale AI", "Snorkel AI"], list(CLIENTS)],
    "aliases": {
        "Outlier": "Scale AI",
        "Meta Superintelligence Labs": "Meta",
        "Amazon AGI": "Amazon",
        "SpaceXAI": "SpaceX",
        "DeepMind": "Google DeepMind",
    },
    "title_restricted": {
        "employers": list(CLIENTS),
        "facet_titles": list(FACET_TITLES),
        "title_markers": [
            "human data", "human data engineer", "human data engineering",
            "data operations", "post training", "posttraining", "evaluation",
            "evaluations", "evals", "rlhf", "annotation", "training data",
        ],
        "title_exclusions": [
            "engineer", "engineering", "scientist", "researcher", "developer",
            "software", "director", "general manager", "product manager",
        ],
        "_why": "operator note, ignored by the parser",
    },
}


def _allowlist():
    allowlist = parse_employer_allowlist(FIELD)
    assert allowlist is not None
    return allowlist


def _field(**restricted_changes):
    import copy

    field = copy.deepcopy(FIELD)
    field["title_restricted"].update(restricted_changes)
    return field


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_the_block_parses_into_restricted_employers_and_title_families() -> None:
    allowlist = _allowlist()

    assert set(allowlist.restricted) == set(CLIENTS)
    assert list(allowlist.facet_titles) == FACET_TITLES
    assert allowlist.is_restricted("Meta") and not allowlist.is_restricted("Mercor")
    assert not allowlist.is_restricted(None)
    for alias, client in (
        ("Meta Superintelligence Labs", "Meta"), ("Amazon AGI", "Amazon"),
        ("SpaceXAI", "SpaceX"), ("SpaceX AI", "SpaceX"), ("DeepMind", "Google DeepMind"),
        ("Meta Platforms", "Meta"), ("xAI", "xAI"),
    ):
        assert allowlist.canonical_for(alias) == client, alias
    assert allowlist.canonical_for("Google") is None


def test_a_brief_without_the_block_restricts_nobody() -> None:
    field = {key: value for key, value in FIELD.items() if key != "title_restricted"}
    allowlist = parse_employer_allowlist(field)

    assert allowlist is not None
    assert allowlist.restricted == frozenset()
    assert allowlist.is_restricted("Meta") is False


@pytest.mark.parametrize(
    "change",
    [
        {"employers": []},
        {"employers": ["Meta", "Netflix"]},  # not a listed employer
        {"facet_titles": []},
        {"facet_titles": ["Human Data Manager", 7]},
        # A facet title outside its own families would open the client's whole staff.
        {"facet_titles": ["Human Data Manager", "Software Engineer"]},
        {"facet_titles": [f"Human Data Role {i}" for i in range(13)]},
        {"title_markers": []},
        {"title_exclusions": "engineer"},
        {"unknown_key": True},
    ],
)
def test_a_malformed_block_makes_the_whole_field_malformed(change) -> None:
    """Malformed means the launcher's preflight refuses the seed outright."""

    assert parse_employer_allowlist(_field(**change)) is None


def test_a_client_group_may_not_mix_with_competitors() -> None:
    import copy

    field = copy.deepcopy(FIELD)
    field["facet_groups"] = [["Mercor", "Meta"], ["Amazon", "xAI", "SpaceX", "Google DeepMind"]]

    assert parse_employer_allowlist(field) is None


@pytest.mark.parametrize(
    "title",
    [
        "Human Data Manager",
        "Human Data Engineer",
        "Human Data Engineering Lead",
        "Technical Program Manager, Post-Training",
        "Program Manager, Model Evaluations",
        "Data Operations Lead",
        "RLHF Program Manager",
        "Posttraining Ops",
    ],
)
def test_a_data_program_title_is_in_the_families(title) -> None:
    assert _allowlist().title_in_family(title) is True


@pytest.mark.parametrize(
    "title",
    [
        "Software Engineer",
        "Software Engineer, Evaluations",
        "Research Scientist, Post-training",
        "Machine Learning Engineer, RLHF",
        "Director, Human Data",
        "Product Manager, Human Data",
        "Technical Program Manager",
        "",
        None,
    ],
)
def test_an_engineering_research_or_leadership_title_is_not(title) -> None:
    assert _allowlist().title_in_family(title) is False


# ---------------------------------------------------------------------------
# The scope verdict
# ---------------------------------------------------------------------------


def test_a_client_employee_in_a_data_seat_is_in_scope() -> None:
    from shared.employer_allowlist import current_employer_scope

    verdict = current_employer_scope(
        companies=["Meta"], headlines=["Building post-training data"],
        titles=["Program Manager, Human Data"], allowlist=_allowlist(),
    )

    assert verdict.in_scope is True and verdict.employer == "Meta"


def test_a_client_engineer_is_out_of_scope_and_named_by_title() -> None:
    from shared.employer_allowlist import current_employer_scope

    verdict = current_employer_scope(
        companies=["Meta"], headlines=["Software Engineer at Meta"],
        titles=["Software Engineer"], allowlist=_allowlist(),
    )

    assert verdict.in_scope is False
    assert (verdict.field, verdict.value) == ("current_title", "Software Engineer at Meta")


def test_a_client_employee_with_no_title_on_record_goes_to_the_judge() -> None:
    from shared.employer_allowlist import current_employer_scope

    verdict = current_employer_scope(companies=["Amazon"], headlines=[""], titles=[""], allowlist=_allowlist())

    assert verdict.in_scope is None


def test_a_headline_in_the_families_keeps_a_client_employee_in_scope() -> None:
    from shared.employer_allowlist import current_employer_scope

    verdict = current_employer_scope(
        companies=["xAI"], headlines=["Human Data Lead @ xAI"], titles=["Member of Technical Staff"],
        allowlist=_allowlist(),
    )

    assert verdict.in_scope is True


def test_a_competitor_seat_needs_no_title() -> None:
    from shared.employer_allowlist import current_employer_scope

    verdict = current_employer_scope(
        companies=["Meta", "Outlier"], headlines=[], titles=["Software Engineer"], allowlist=_allowlist(),
    )

    assert verdict.in_scope is True and verdict.employer == "Scale AI"


@pytest.mark.parametrize(
    "entry, title",
    [
        ("Human Data Manager at xAI (2025 – Present)", "Human Data Manager"),
        ("Software Engineer at Meta · 2024 – Present", "Software Engineer"),
        ("Program Manager at Google (2021 – 2024)", None),
    ],
)
def test_a_card_entry_names_its_title_only_when_it_is_current(entry, title) -> None:
    from shared.employer_allowlist import present_entry_title

    assert present_entry_title(entry) == title


# ---------------------------------------------------------------------------
# The card and profile gates
# ---------------------------------------------------------------------------


def _card(*, company: str, title: str, headline: str = "", entries=None) -> CandidateSnippet:
    return CandidateSnippet(
        name="Client Person",
        headline=headline or f"{title} at {company}",
        current_title=title,
        current_company=company,
        location="United States",
        education_snippet="",
        profile_url="/talent/profile/client-person",
        source_string_id=9,
        source_string_name="client string",
        page=1,
        result_rank=1,
        card_index=0,
        experience_entries=list(entries or []),
    )


def _reviewer():
    from linkedin.candidate_review import CandidateReviewer

    reviewer = object.__new__(CandidateReviewer)
    reviewer.context = SimpleNamespace(
        brief_obj=SimpleNamespace(
            employer_blacklist=["Babbage", "OpenAI", "Anthropic"],
            alumni_blacklist=["Babbage"],
            raw={"employer_allowlist": FIELD},
        )
    )
    return reviewer


def test_the_card_gate_skips_a_client_engineer_before_any_judge() -> None:
    from linkedin.candidate_review import EMPLOYER_SCOPE_GATE_PATH, facial_gate_console_line

    path, (blocked, field, value) = _reviewer()._facial_employer_gate(
        _card(company="Meta", title="Software Engineer")
    )

    assert path == EMPLOYER_SCOPE_GATE_PATH
    assert field == "current_title" and value == "Software Engineer at Meta"
    line = facial_gate_console_line(path, "Client Person", blocked, field, value)
    assert "outside the client-side title families" in line


def test_the_card_gate_passes_a_client_data_seat() -> None:
    assert _reviewer()._facial_employer_gate(
        _card(company="SpaceX", title="Human Data Engineer")
    ) is None


def test_the_profile_gate_reads_present_position_titles() -> None:
    card = _card(company="Amazon", title="", headline="Amazon")
    summary = CandidateProfileSummary(
        name="Client Person",
        profile_url=card.profile_url,
        headline="Amazon",
        experiences=[
            Experience(title="Applied Scientist", company="Amazon", end="Present"),
            Experience(title="Human Data Lead", company="Scale AI", end="2023"),
        ],
    )

    hit = _reviewer()._employer_scope_match(card, summary, _reviewer().context.brief_obj)

    assert hit is not None and hit[1] == "current_title" and "Applied Scientist" in hit[2]


# ---------------------------------------------------------------------------
# The stamp
# ---------------------------------------------------------------------------


def _string(sid: int, boolean: str = '("program manager")', **filters) -> SearchString:
    return SearchString(id=sid, name=f"s{sid}", boolean=boolean, structured_filters=dict(filters))


def test_an_unfaceted_string_is_never_stamped_with_the_client_group() -> None:
    from linkedin.competitor_scope import stamp_competitor_facets

    queue = [_string(i, '("human data" AND Meta)') for i in range(1, 9)]
    stamp_competitor_facets(queue, _allowlist(), queue=queue)

    for ss in queue:
        assert not set(ss.structured_filters["companies"]) & set(CLIENTS), ss.structured_filters


def test_a_client_string_gets_the_family_titles_on_its_title_facet() -> None:
    from linkedin.competitor_scope import stamp_competitor_facet

    ss = _string(1, companies=["Meta Superintelligence Labs", "Amazon AGI"])
    record = stamp_competitor_facet(ss, _allowlist(), queue=[ss])

    assert record is not None
    assert ss.structured_filters["companies"] == ["Meta", "Amazon"]
    assert ss.structured_filters["titles"] == FACET_TITLES
    assert record["titles_after"] == FACET_TITLES


def test_a_client_string_keeps_its_own_family_titles_and_loses_the_rest() -> None:
    from linkedin.competitor_scope import stamp_competitor_facet

    ss = _string(1, companies=["xAI"], titles=["Human Data Manager", "Software Engineer"])
    stamp_competitor_facet(ss, _allowlist(), queue=[ss])

    assert ss.structured_filters["titles"] == ["Human Data Manager"]


def test_a_facet_mixing_clients_and_competitors_keeps_only_the_competitors() -> None:
    from linkedin.competitor_scope import stamp_competitor_facet

    ss = _string(1, companies=["Mercor", "Meta"], titles=["Program Manager"])
    stamp_competitor_facet(ss, _allowlist(), queue=[ss])

    assert ss.structured_filters["companies"] == ["Mercor"]
    assert ss.structured_filters["titles"] == ["Program Manager"]


def test_a_client_string_already_in_shape_is_untouched() -> None:
    from linkedin.competitor_scope import stamp_competitor_facet

    ss = _string(1, companies=["xAI"], titles=["Human Data Manager"])

    assert stamp_competitor_facet(ss, _allowlist(), queue=[ss]) is None


# ---------------------------------------------------------------------------
# The skip
# ---------------------------------------------------------------------------


def _receipt(companies, titles):
    return {
        "event": "linkedin_facet_reconciliation",
        "desired": {"companies": list(companies), "titles": list(titles)},
        "after": {"companies": list(companies), "titles": list(titles)},
        "unapplied": {"companies": [], "titles": []},
        "substituted": {"companies": {}, "titles": {}},
        "failure_gate": "",
    }


def test_a_client_string_whose_title_facet_did_not_land_is_skipped() -> None:
    from linkedin.competitor_scope import competitor_facet_skip_reason

    reason = competitor_facet_skip_reason(
        ["Meta"], _receipt(["Meta"], []), _allowlist(), requested_titles=["Human Data Manager"]
    )

    assert reason and "title facet" in reason


def test_a_client_string_whose_title_landed_as_a_near_miss_is_skipped() -> None:
    from linkedin.competitor_scope import competitor_facet_skip_reason

    reason = competitor_facet_skip_reason(
        ["Meta"], _receipt(["Meta"], ["Human Data Manager", "Program Manager"]), _allowlist(),
        requested_titles=["Human Data Manager", "Post-training Program Manager"],
    )

    assert reason and "Program Manager" in reason


def test_a_client_string_requesting_no_family_title_is_skipped_before_it_opens() -> None:
    from linkedin.competitor_scope import competitor_facet_skip_reason

    reason = competitor_facet_skip_reason(["Meta"], None, _allowlist(), requested_titles=[])

    assert reason


def test_a_client_string_whose_family_titles_landed_runs() -> None:
    from linkedin.competitor_scope import competitor_facet_skip_reason

    assert competitor_facet_skip_reason(
        ["Meta"], _receipt(["Meta"], ["Human Data Manager"]), _allowlist(),
        requested_titles=["Human Data Manager"],
    ) is None


def test_a_competitor_string_needs_no_title_facet() -> None:
    from linkedin.competitor_scope import competitor_facet_skip_reason

    assert competitor_facet_skip_reason(
        ["Mercor"], _receipt(["Mercor"], []), _allowlist(), requested_titles=[]
    ) is None


def test_the_string_driver_passes_the_variant_titles_to_the_skip(capsys) -> None:
    from linkedin.string_driver import StringDriver

    driver = object.__new__(StringDriver)
    driver.context = SimpleNamespace(brief_obj=SimpleNamespace(raw={"employer_allowlist": FIELD}), log_path=None)
    driver.surface = SimpleNamespace(last_opening_facet_receipt=_receipt(["Meta"], []))
    state = SimpleNamespace(
        active_variant=SimpleNamespace(
            structured_filters=SimpleNamespace(companies=["Meta"], titles=["Human Data Manager"])
        )
    )

    reason = driver._competitor_facet_skip(_string(4, companies=["Meta"]), state)

    assert reason and "title facet" in reason


def test_the_scope_prompt_block_states_the_client_title_rule() -> None:
    from linkedin.competitor_scope import competitor_scope_prompt_block

    block = competitor_scope_prompt_block(SimpleNamespace(raw={"employer_allowlist": FIELD}))

    assert "Meta, Amazon, xAI, SpaceX, Google DeepMind" in block
    assert "Human Data Manager" in block
    assert "never with engineering" in block


# ---------------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------------


def _resolve(raw_variant, inherited, **kwargs):
    from linkedin import search_policy

    return search_policy._resolve_structured_controls(raw_variant, inherited, **kwargs)


def test_a_variant_may_not_promote_a_client_company() -> None:
    from linkedin.search_intelligence import LinkedInStructuredFilters

    refused: list[str] = []
    filters, *_ = _resolve(
        {"surface": "hybrid", "structured_controls": {"companies": ["Meta", "Handshake"]}},
        LinkedInStructuredFilters(companies=["Mercor"]),
        company_scope=_allowlist(),
        refused=refused,
    )

    assert filters.companies == ["Mercor", "Handshake"]
    assert refused == ["promote:Meta"]


def test_a_client_variant_keeps_its_titles_and_takes_only_family_titles() -> None:
    from linkedin.search_intelligence import LinkedInStructuredFilters

    refused: list[str] = []
    filters, *_ = _resolve(
        {
            "surface": "hybrid",
            "structured_controls": {
                "titles": ["Software Engineer", "Evaluation Program Manager"],
                "demote": ["titles"],
            },
        },
        LinkedInStructuredFilters(companies=["Meta"], titles=["Human Data Manager", "Human Data Lead"]),
        company_scope=_allowlist(),
        refused=refused,
    )

    assert "Software Engineer" not in filters.titles
    assert "Human Data Manager" in filters.titles
    assert "promote:Software Engineer" in refused and "demote:titles" in refused


def _browser(apply_result):
    browser = MagicMock()
    browser.go_back_to_results = AsyncMock()
    browser.apply_advanced_search_plan = AsyncMock(return_value=apply_result)
    browser.enter_search_string = AsyncMock()
    browser.get_results_count_text = AsyncMock(return_value="80")
    browser.get_results_count = AsyncMock(return_value=80)
    browser.get_card_snapshot = AsyncMock(return_value={})
    return browser


def test_a_client_variant_whose_titles_did_not_apply_is_refused() -> None:
    from linkedin.advanced_search import ControlApplicationResult
    from linkedin.search_intelligence import (
        LinkedInExperimentState,
        LinkedInSearchIntent,
        LinkedInSearchVariant,
        LinkedInStructuredFilters,
    )
    from linkedin.search_mutation import LinkedInSearchMutationDeps, LinkedInSearchMutationExecutor

    browser = _browser(
        ControlApplicationResult(
            success=True,
            applied_controls=["keywords", "companies"],
            failed_controls=["job_titles"],
            reason="structured_controls_dropped_keyword_only",
            plan_fully_applied=False,
        )
    )
    state = LinkedInExperimentState(root_string_id=1, intent=LinkedInSearchIntent(root_boolean="x"), mode="experiment")
    variant = LinkedInSearchVariant(
        variant_id="v1",
        parent_variant_id="root",
        root_string_id=1,
        boolean='"program"',
        surface="hybrid",
        structured_filters=LinkedInStructuredFilters(companies=["Meta"], titles=["Human Data Manager"]),
    )
    state.variants[variant.variant_id] = variant
    ss = _string(1, companies=["Meta"], titles=["Human Data Manager"])
    executor = LinkedInSearchMutationExecutor(
        LinkedInSearchMutationDeps(
            browser=browser,
            log_path=Path("test.log"),
            get_input_mode=lambda: "concurrent",
            get_runtime_run_id=lambda: None,
            get_runtime_state=lambda: MagicMock(),
            get_search_mutation_budget_used=lambda: 0,
            set_search_mutation_budget_used=lambda _v: None,
            get_company_scope=_allowlist,
        )
    )
    with patch("linkedin.search_mutation.human_delay_correlated", return_value=0), patch(
        "linkedin.search_mutation.log_event"
    ):
        result = asyncio.run(
            executor.apply_variant(
                search_string=ss, experiment_state=state, variant=variant, acquisition_mode="linkedin_hybrid"
            )
        )

    assert result.applied is False
    assert result.blocked_reason == "client_title_facet_not_applied"
    assert variant.structured_filters.titles == ["Human Data Manager"]
    browser.enter_search_string.assert_not_awaited()
