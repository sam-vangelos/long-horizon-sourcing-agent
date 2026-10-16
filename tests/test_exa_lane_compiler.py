"""Exa source module S2 — lane compiler and lint coverage.

Briefs are built as raw V2 dicts and run through the real loader, so
these exercise the same Brief objects a launch would hand the compiler
(the compat Brief, whose geography lives in ``permanent_filters``, not
in a ``geography`` attribute).

The lint half encodes two rulings and one field lesson: CLO-115 makes a
blacklisted target firm an error rather than a warning; the word-
boundary rule (copied from ``shared/brief_lint.py``) keeps "Ramp" from
matching "Rampart AI"; and the namesake check is the Accel/Accel-KKR
noise class the 2026-08-28 CoS slate hit.
"""

from __future__ import annotations

from exa.lane_compiler import (
    blocking_findings,
    compile_exa_queries,
    lint_exa_brief,
)
from shared.brief_loader import _load_v2_brief


def _brief(**overrides):
    raw = {
        "role_title": "Chief of Staff to the CEO",
        "role_level": "L6",
        "role_summary": "Runs the founder's operating cadence.",
        "geography": "SF Bay Area",
        "linkedin_project": "Test Project",
        "capability_areas": [
            {
                "name": "Operating leverage",
                "description": "Has run the founder's operating cadence.",
            }
        ],
        "depth_distinction": {
            "builder_definition": "Ran the cadence.",
            "user_definition": "Attended the cadence.",
            "edge_case_guidance": "Borderline = full eval.",
        },
        "non_fit_patterns": [],
        "employer_signal_rules": [],
        "engagement_context": {"selectivity_posture": "selective"},
        "exa_target_firms": [
            {"name": "Benchmark", "domain": "benchmark.com"},
            {"name": "Accel", "exclude_terms": ["Accel-KKR"]},
        ],
        "exa_title_terms": ["Senior Associate", "Principal"],
    }
    raw.update(overrides)
    return _load_v2_brief(raw)


# ---------------------------------------------------------------------------
# compile_exa_queries
# ---------------------------------------------------------------------------


def test_compile_emits_people_search_per_firm_with_linkedin_domain() -> None:
    queries = [q for q in compile_exa_queries(_brief()) if q.lane == "people_search"]

    assert [q.firm for q in queries] == ["Benchmark", "Accel"]
    for query in queries:
        assert query.category == "people"
        assert query.include_domains == ["linkedin.com"]
    assert (
        queries[0].query
        == "Senior Associate or Principal at Benchmark SF Bay Area"
    )
    # The namesake guard rides on the query it guards.
    assert queries[0].exclude_terms == []
    assert queries[1].exclude_terms == ["Accel-KKR"]


def test_compile_team_page_uses_firm_domain_when_present() -> None:
    queries = [q for q in compile_exa_queries(_brief()) if q.lane == "team_page"]

    assert queries[0].firm == "Benchmark"
    assert queries[0].query == "Benchmark team page investors"
    assert queries[0].include_domains == ["benchmark.com"]
    assert queries[0].category is None
    # Accel carries no domain — no allowlist rather than a fabricated one.
    assert queries[1].firm == "Accel"
    assert queries[1].include_domains == []


def test_compile_operating_signal_one_per_firm() -> None:
    queries = [
        q for q in compile_exa_queries(_brief()) if q.lane == "operating_signal"
    ]

    assert [q.firm for q in queries] == ["Benchmark", "Accel"]
    # EVERY title term, not just the first — dropping the rest narrowed
    # the lane to one seniority band without saying so.
    assert queries[0].query == (
        "Benchmark Senior Associate or Principal former founder OR operator "
        "OR chief of staff"
    )
    for term in ("Senior Associate", "Principal"):
        assert term in queries[0].query
    assert all(q.category is None for q in queries)


def test_compile_includes_firm_aliases_in_query() -> None:
    """A firm the market calls something else must be searched by both names.

    ``aliases`` was in the brief schema and read only by the lint, so
    declaring one changed nothing about what was searched for. A profile
    headline carries "Andreessen Horowitz" or "a16z", essentially never
    both, and the 08-28 sweep's a16z results came back under the short
    name — half the firm's people were unreachable by the long one.
    """

    brief = _brief(
        exa_target_firms=[
            {
                "name": "Andreessen Horowitz",
                "domain": "a16z.com",
                "aliases": ["a16z"],
            }
        ]
    )
    by_lane = {q.lane: q for q in compile_exa_queries(brief)}

    assert by_lane["people_search"].query == (
        "Senior Associate or Principal at Andreessen Horowitz (a16z) SF Bay Area"
    )
    assert by_lane["operating_signal"].query.startswith(
        "Andreessen Horowitz (a16z) Senior Associate or Principal"
    )
    # The team-page lane is domain-scoped, so the alias buys nothing
    # there and the query stays the firm's own name.
    assert by_lane["team_page"].query == "Andreessen Horowitz team page investors"
    assert by_lane["team_page"].include_domains == ["a16z.com"]

    # `firm` stays the canonical name on EVERY lane — it is the grouping
    # key the slate's coverage rows and the work-unit ids are built from,
    # not display text. An alias leaking into it would split one firm's
    # coverage across two rows.
    assert {q.firm for q in compile_exa_queries(brief)} == {"Andreessen Horowitz"}


def test_compile_without_aliases_is_unchanged() -> None:
    """No aliases, no parenthetical — the common case keeps its exact query."""

    queries = [q for q in compile_exa_queries(_brief()) if q.lane == "people_search"]

    assert queries[0].query == (
        "Senior Associate or Principal at Benchmark SF Bay Area"
    )
    assert "(" not in queries[0].query


def test_compile_alias_equal_to_the_firm_name_is_not_repeated() -> None:
    """A redundant alias must not produce "Accel (Accel)"."""

    brief = _brief(
        exa_target_firms=[{"name": "Accel", "aliases": ["accel", "Accel", ""]}]
    )
    people = [q for q in compile_exa_queries(brief) if q.lane == "people_search"]

    assert people[0].query == "Senior Associate or Principal at Accel SF Bay Area"


def test_compile_respects_query_lane_subset() -> None:
    queries = compile_exa_queries(_brief(exa_query_lanes=["team_page"]))

    assert {q.lane for q in queries} == {"team_page"}
    assert len(queries) == 2


def test_compile_no_firms_falls_back_to_title_terms_with_warning() -> None:
    brief = _brief(exa_target_firms=[])
    queries = compile_exa_queries(brief)

    assert [q.query for q in queries] == [
        "Senior Associate SF Bay Area",
        "Principal SF Bay Area",
    ]
    assert all(q.firm == "" for q in queries)
    assert all(q.lane == "people_search" for q in queries)

    codes = [f.code for f in lint_exa_brief(brief)]
    assert codes.count("no_target_firms") == 1
    assert [f.severity for f in lint_exa_brief(brief) if f.code == "no_target_firms"] == [
        "warning"
    ]


def test_compile_query_ids_are_sequential_and_names_readable() -> None:
    queries = compile_exa_queries(_brief())

    assert [q.id for q in queries] == list(range(1, len(queries) + 1))
    assert queries[0].name == "people_search · Benchmark"
    assert [q.name for q in queries if q.lane == "team_page"] == [
        "team_page · Benchmark",
        "team_page · Accel",
    ]


def test_compile_geography_absent_leaves_no_double_space() -> None:
    brief = _brief(geography="")
    people = [q for q in compile_exa_queries(brief) if q.lane == "people_search"]
    assert people[0].query == "Senior Associate or Principal at Benchmark"


# ---------------------------------------------------------------------------
# The archetype lane (S8b)
# ---------------------------------------------------------------------------


_ARCHETYPES = [
    {
        "id": "a01-mbb-to-vc",
        "axis": "consulting_to_investing",
        "text": (
            "Former McKinsey, Bain, or BCG consultants in the SF Bay Area who "
            "moved into venture capital and now sit on a deal team."
        ),
        "num_results": 40,
        "include_domains": ["linkedin.com", "news.ycombinator.com"],
    },
    {
        "id": "a02-investor-to-cos",
        "text": (
            "Former SF Bay Area venture investors now Chief of Staff at an AI "
            "company."
        ),
    },
]


def test_compile_emits_archetype_queries_after_firm_lanes() -> None:
    """Appended last, in brief order, with the firm queries' ids untouched.

    Order is not cosmetic. The firm lanes are the precise half of the
    sweep and should spend the budget first, and appending means adding
    an archetype to a live brief does not renumber the work units a
    resume is matching against.
    """

    brief = _brief(exa_archetype_queries=_ARCHETYPES)
    queries = compile_exa_queries(brief)

    firm_queries = [q for q in queries if q.lane != "archetype"]
    archetypes = [q for q in queries if q.lane == "archetype"]

    assert len(firm_queries) == 6  # 3 firm lanes × 2 firms, unchanged
    assert queries[:6] == firm_queries, "archetypes come after, never between"
    assert [q.id for q in queries] == list(range(1, 9))
    assert [q.id for q in archetypes] == [7, 8]

    assert [q.name for q in archetypes] == [
        "archetype · a01-mbb-to-vc",
        "archetype · a02-investor-to-cos",
    ]
    # The query is the author's sentence, verbatim — no geography
    # appended, no title terms spliced in, no firm name.
    assert archetypes[0].query == _ARCHETYPES[0]["text"]
    assert all(q.firm == "" for q in archetypes)
    assert all(q.category == "people" for q in archetypes)

    # The author's own num_results and domains ride through.
    assert archetypes[0].num_results == 40
    assert archetypes[0].include_domains == ["linkedin.com", "news.ycombinator.com"]

    # And a brief that names no archetypes compiles exactly what it did
    # before the lane existed.
    assert compile_exa_queries(_brief()) == firm_queries


def test_archetype_query_defaults_25_results_linkedin_domain() -> None:
    """An entry that names neither gets the prior art's numbers.

    25 rather than the firm lanes' 10 because an archetype query has no
    firm allowlist narrowing it — its top 10 is a thinner slice of its
    own population. LinkedIn because the lane takes the people_search
    path, which drops every non-profile URL: widening the domains buys
    pages that are then thrown away as `not_profile`.
    """

    brief = _brief(exa_archetype_queries=_ARCHETYPES)
    bare = next(
        q for q in compile_exa_queries(brief) if q.name.endswith("a02-investor-to-cos")
    )

    assert bare.num_results == 25
    assert bare.include_domains == ["linkedin.com"]

    # An explicit empty domain list reads as "unset", not "the open web".
    empty = _brief(
        exa_archetype_queries=[{"id": "a03", "text": "x", "include_domains": []}]
    )
    only = next(q for q in compile_exa_queries(empty) if q.lane == "archetype")
    assert only.include_domains == ["linkedin.com"]

    # An out-of-range count falls back rather than raising. The loader
    # refuses this brief outright (`validate_exa_fields` is the gate),
    # so the only way in is a caller holding a hand-built Brief — and a
    # compiler that raised on one would turn an authoring typo into a
    # crashed launch instead of a lint finding.
    class _Stub:
        exa_target_firms: list = []
        exa_query_lanes = ["archetype"]
        exa_title_terms: list = []
        geography = "SF Bay Area"
        exa_archetype_queries = [
            {"id": "a03", "text": "Bay Area investors.", "num_results": 9000},
            {"id": "a04", "text": "Bay Area operators.", "num_results": "25"},
        ]

    fallen_back = compile_exa_queries(_Stub())
    assert [q.num_results for q in fallen_back] == [25, 25]


def test_archetype_lane_subset_selection() -> None:
    """``exa_query_lanes`` gates the archetype lane like any other."""

    brief = _brief(exa_archetype_queries=_ARCHETYPES, exa_query_lanes=["archetype"])
    queries = compile_exa_queries(brief)

    assert {q.lane for q in queries} == {"archetype"}
    assert [q.id for q in queries] == [1, 2]

    # Excluded: the archetypes are authored but the brief does not run them.
    firm_only = _brief(
        exa_archetype_queries=_ARCHETYPES,
        exa_query_lanes=["people_search", "team_page"],
    )
    assert "archetype" not in {q.lane for q in compile_exa_queries(firm_only)}

    # Absent `exa_query_lanes` means every lane, archetype included.
    assert "archetype" in {
        q.lane for q in compile_exa_queries(_brief(exa_archetype_queries=_ARCHETYPES))
    }


def test_lint_warns_when_archetype_omits_geography() -> None:
    """The compiler never appends the location, so the author must write it.

    Splicing the brief's geography onto the end of someone's prose
    produces "…now Senior Associates on a deal team SF Bay Area", so
    the archetype lane leaves the text alone and this is the check that
    the author anchored it themselves. A warning, not an error: a
    deliberately global archetype is a legitimate thing to write.
    """

    brief = _brief(
        exa_archetype_queries=[
            {"id": "a01-anchored", "text": "Investors in the SF Bay Area who consulted."},
            {"id": "a02-global", "text": "Investors anywhere who consulted."},
        ]
    )
    findings = [
        f for f in lint_exa_brief(brief) if f.code == "archetype_missing_geography"
    ]

    assert len(findings) == 1
    assert findings[0].severity == "warning"
    assert "a02-global" in findings[0].message
    assert blocking_findings(lint_exa_brief(brief)) == []

    # "Bay" alone anchors it — any token of the geography counts.
    partial = _brief(
        exa_archetype_queries=[{"id": "a03", "text": "Bay Area investors."}]
    )
    assert not [
        f for f in lint_exa_brief(partial) if f.code == "archetype_missing_geography"
    ]

    # The connective words do not count: "area" is in the stop list, so
    # "areas of focus" must not read as an anchor.
    decoy = _brief(
        exa_archetype_queries=[{"id": "a04", "text": "Investors across many areas."}]
    )
    assert [
        f for f in lint_exa_brief(decoy) if f.code == "archetype_missing_geography"
    ]

    # A brief with no geography of its own has nothing to check against.
    nowhere = _brief(
        geography="",
        exa_archetype_queries=[{"id": "a05", "text": "Investors anywhere."}],
    )
    assert not [
        f for f in lint_exa_brief(nowhere) if f.code == "archetype_missing_geography"
    ]

    # And the lane has to be active for the check to fire at all.
    off = _brief(
        exa_query_lanes=["people_search"],
        exa_archetype_queries=[{"id": "a06", "text": "Investors anywhere."}],
    )
    assert not [
        f for f in lint_exa_brief(off) if f.code == "archetype_missing_geography"
    ]


def test_brief_with_only_archetypes_compiles_and_is_launchable() -> None:
    """No firms, no title terms, six sentences — that is a whole campaign.

    ``no_queries_compiled`` asks the compiler rather than re-deriving
    its rules, so archetypes count as compiled queries and this brief
    launches. ``missing_title_terms`` stays silent too: the archetype
    lane is not built out of title terms.
    """

    brief = _brief(
        exa_target_firms=[],
        exa_title_terms=[],
        exa_query_lanes=["archetype"],
        exa_archetype_queries=_ARCHETYPES,
    )
    queries = compile_exa_queries(brief)

    assert [q.lane for q in queries] == ["archetype", "archetype"]
    findings = lint_exa_brief(brief)
    assert blocking_findings(findings) == []
    codes = {f.code for f in findings}
    assert "no_queries_compiled" not in codes
    assert "missing_title_terms" not in codes
    # The firm allowlist really is off, and the recruiter is told so.
    assert "no_target_firms" in codes

    # The archetypes also survive the no-firms people_search fallback,
    # which used to return before anything else could be appended.
    with_fallback = _brief(
        exa_target_firms=[],
        exa_archetype_queries=_ARCHETYPES,
    )
    lanes = [q.lane for q in compile_exa_queries(with_fallback)]
    assert lanes == ["people_search", "people_search", "archetype", "archetype"]


# ---------------------------------------------------------------------------
# lint_exa_brief
# ---------------------------------------------------------------------------


def test_lint_employer_conflict_on_blacklisted_firm() -> None:
    brief = _brief(
        employer_blacklist=["OpenAI"],
        exa_target_firms=[{"name": "OpenAI"}],
    )
    findings = [f for f in lint_exa_brief(brief) if f.code == "employer_conflict"]

    assert len(findings) == 1
    assert findings[0].severity == "error"
    assert findings[0].source == "exa"


def test_lint_employer_conflict_matches_an_alias() -> None:
    brief = _brief(
        employer_blacklist=["Anthropic"],
        exa_target_firms=[{"name": "Ant Ventures", "aliases": ["Anthropic"]}],
    )
    codes = [f.code for f in lint_exa_brief(brief)]
    assert codes.count("employer_conflict") == 1


def test_lint_employer_conflict_uses_word_boundaries() -> None:
    """"Ramp" must not match "Rampart AI" — the raw-substring bug."""

    brief = _brief(
        employer_blacklist=["Ramp"],
        exa_target_firms=[{"name": "Rampart AI"}],
    )
    assert [f for f in lint_exa_brief(brief) if f.code == "employer_conflict"] == []


def test_lint_employer_conflict_on_punctuated_name() -> None:
    r"""``\b`` finds no boundary after "!", so "Yahoo!" missed itself."""

    brief = _brief(
        employer_blacklist=["Yahoo!"],
        exa_target_firms=[{"name": "Yahoo!"}],
    )
    findings = [f for f in lint_exa_brief(brief) if f.code == "employer_conflict"]

    assert len(findings) == 1
    assert findings[0].severity == "error"


def test_lint_hiring_company_as_target_is_conflict() -> None:
    brief = _brief(
        engagement_context={
            "selectivity_posture": "selective",
            "hiring_company": "Babbage",
        },
        exa_target_firms=[{"name": "Babbage"}, {"name": "Benchmark"}],
    )
    findings = [f for f in lint_exa_brief(brief) if f.code == "employer_conflict"]

    assert len(findings) == 1
    assert findings[0].severity == "error"
    assert "Babbage" in findings[0].message


def test_lint_hiring_company_conflict_on_structured_brief() -> None:
    """The V2 Brief must see the conflict the compat Brief sees.

    A brief may carry `hiring_company` at the top level rather than
    inside `engagement_context`. The loader now mirrors it in, so the
    lint reads the same value from `_new_brief` — which is what a
    consumer holding the structured Brief would pass.
    """

    brief = _brief(
        hiring_company="Babbage",
        exa_target_firms=[{"name": "Babbage"}, {"name": "Benchmark"}],
    )

    assert brief._new_brief.engagement_context["hiring_company"] == "Babbage"
    findings = [
        f for f in lint_exa_brief(brief._new_brief) if f.code == "employer_conflict"
    ]
    assert len(findings) == 1
    assert findings[0].severity == "error"


def test_loader_never_overwrites_an_authored_hiring_company() -> None:
    brief = _brief(
        hiring_company="Top Level Co",
        engagement_context={
            "selectivity_posture": "selective",
            "hiring_company": "Authored Co",
        },
    )
    assert brief._new_brief.engagement_context["hiring_company"] == "Authored Co"


def test_lint_clean_brief_has_no_errors() -> None:
    assert blocking_findings(lint_exa_brief(_brief())) == []


def test_lint_missing_title_terms_is_error() -> None:
    brief = _brief(exa_title_terms=[])
    findings = [f for f in lint_exa_brief(brief) if f.code == "missing_title_terms"]

    assert len(findings) == 1
    assert findings[0].severity == "error"


def test_lint_missing_title_terms_silent_when_no_title_lane_is_active() -> None:
    brief = _brief(exa_title_terms=[], exa_query_lanes=["team_page"])
    codes = [f.code for f in lint_exa_brief(brief)]
    assert "missing_title_terms" not in codes


def test_lint_blocks_when_nothing_would_compile() -> None:
    """No firms + a lane subset without people_search compiles zero queries.

    A warning let that launch proceed and return nobody. It blocks.
    """

    brief = _brief(exa_target_firms=[], exa_query_lanes=["team_page"])
    assert compile_exa_queries(brief) == []

    findings = lint_exa_brief(brief)
    blocking = blocking_findings(findings)
    assert "no_queries_compiled" in {f.code for f in blocking}
    # The softer warning stays where it was.
    assert "no_target_firms" in {f.code for f in findings}

    # A brief that DOES compile queries carries no such finding.
    assert "no_queries_compiled" not in {f.code for f in lint_exa_brief(_brief())}


def test_lint_namesake_risk_info_for_short_firm_without_exclude_terms() -> None:
    brief = _brief(
        exa_target_firms=[
            {"name": "Accel"},  # short, one token, unguarded
            {"name": "Benchmark"},  # long enough to be distinctive
            {"name": "NEA", "exclude_terms": ["NEA teachers union"]},  # guarded
        ]
    )
    findings = [f for f in lint_exa_brief(brief) if f.code == "namesake_risk"]

    assert len(findings) == 1
    assert findings[0].severity == "info"
    assert "Accel" in findings[0].message


# ---------------------------------------------------------------------------
# blocking_findings
# ---------------------------------------------------------------------------


def test_blocking_findings_filters_errors_only() -> None:
    brief = _brief(
        employer_blacklist=["OpenAI"],
        exa_target_firms=[{"name": "OpenAI"}],
        exa_title_terms=[],
    )
    findings = lint_exa_brief(brief)

    assert {f.severity for f in findings} >= {"error", "info"}
    blocking = blocking_findings(findings)
    assert {f.severity for f in blocking} == {"error"}
    assert {f.code for f in blocking} == {"employer_conflict", "missing_title_terms"}
