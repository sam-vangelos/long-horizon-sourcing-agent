"""Exa source module S2 — V2 brief schema coverage.

Mirrors :mod:`tests.test_brief_v2_schema_oss_maintainers`. Asserts that:

- ``validate_v2_brief`` accepts a brief carrying ``exa_target_firms``,
  ``exa_query_lanes``, ``exa_title_terms``, ``exa_archetype_queries``,
  and the source-agnostic ``do_not_save`` list — and still accepts one carrying none of them,
  which is the behavior-preserving case for every existing brief.
- The validator rejects the malformed shapes that would degrade
  silently at run time: a firm entry with no name, a non-dict firm
  entry, an unknown query lane, non-string title terms, and a
  do-not-save row with no name.
- The loader hydrates all five onto BOTH Brief dataclasses.
- Brief polish preserves all five (the polish step returns the LLM's
  draft wholesale, so an unechoed field is a lost field).
"""

from __future__ import annotations

import json

import pytest

from market_intelligence.brief_polish import _exa_fields_drift
from shared.brief_loader import load_brief
from shared.brief_v2_schema import (
    EXA_QUERY_LANES,
    RECOGNIZED_EXA_QUERY_LANES,
    BriefSchemaError,
    validate_v2_brief,
)


def _minimal_valid_v2_brief() -> dict:
    return {
        "role_title": "Chief of Staff to the CEO",
        "capability_areas": [
            {
                "name": "Operating leverage",
                "description": "Has run the founder's operating cadence.",
            }
        ],
        "depth_distinction": {
            "builder_definition": "Ran the cadence, didn't just attend it.",
            "user_definition": "Sat in the meeting and took notes.",
            "edge_case_guidance": "Borderline = full eval.",
        },
    }


def _brief_with_exa_fields() -> dict:
    brief = _minimal_valid_v2_brief()
    brief["exa_target_firms"] = [
        {
            "name": "Accel",
            "domain": "accel.com",
            "aliases": ["Accel Partners"],
            "exclude_terms": ["Accel-KKR"],
        },
        {"name": "Benchmark"},
    ]
    brief["exa_query_lanes"] = ["people_search", "team_page"]
    brief["exa_title_terms"] = ["Senior Associate", "Principal"]
    brief["exa_archetype_queries"] = [
        {
            "id": "a01-mbb-to-vc",
            "axis": "consulting_to_investing",
            "text": (
                "Former McKinsey, Bain, or BCG consultants in the San "
                "Francisco Bay Area who moved into venture capital."
            ),
            "num_results": 25,
            "include_domains": ["linkedin.com"],
        },
        {
            "id": "a02-investor-to-cos",
            "text": (
                "Former Bay Area venture investors now Chief of Staff at an "
                "AI company."
            ),
        },
    ]
    brief["do_not_save"] = [
        {"name": "Jordan Reeves", "linkedin_url": "https://www.linkedin.com/in/jreeves"},
        {"name": "Casey Lin", "reason": "Already in process elsewhere."},
    ]
    return brief


# ---------------------------------------------------------------------------
# Recognized values
# ---------------------------------------------------------------------------


@pytest.mark.structural
def test_recognized_exa_query_lanes_match_the_slate_sweeps() -> None:
    # `archetype` is last, and that ordering is load-bearing: the
    # compiler emits lanes in this order and appends the archetype
    # queries after every firm lane's, so a brief that adds one does
    # not renumber the firm queries a resume is matching against.
    assert EXA_QUERY_LANES == (
        "people_search",
        "team_page",
        "operating_signal",
        "archetype",
    )
    assert RECOGNIZED_EXA_QUERY_LANES == frozenset(EXA_QUERY_LANES)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_exa_fields_absent_is_valid_and_defaults_empty(tmp_path) -> None:
    """Behavior-preserving: a brief that never mentions Exa still loads."""

    brief = _minimal_valid_v2_brief()
    validate_v2_brief(brief)

    path = tmp_path / "brief.json"
    path.write_text(json.dumps(brief))
    loaded = load_brief(path)

    assert loaded.exa_target_firms == []
    assert loaded.exa_query_lanes == []
    assert loaded.exa_title_terms == []
    assert loaded.exa_archetype_queries == []
    assert loaded.do_not_save == []
    assert loaded._new_brief.exa_target_firms == []
    assert loaded._new_brief.exa_query_lanes == []
    assert loaded._new_brief.exa_title_terms == []
    assert loaded._new_brief.exa_archetype_queries == []
    assert loaded._new_brief.do_not_save == []


def test_validate_accepts_brief_with_exa_fields() -> None:
    validate_v2_brief(_brief_with_exa_fields())


def test_validate_accepts_each_recognized_query_lane() -> None:
    for lane in EXA_QUERY_LANES:
        brief = _minimal_valid_v2_brief()
        brief["exa_query_lanes"] = [lane]
        validate_v2_brief(brief)


def test_exa_target_firms_requires_name() -> None:
    brief = _minimal_valid_v2_brief()
    brief["exa_target_firms"] = [{"domain": "accel.com"}]
    with pytest.raises(BriefSchemaError) as exc_info:
        validate_v2_brief(brief)
    assert "exa_target_firms" in exc_info.value.invalid_keys


def test_exa_target_firms_rejects_blank_name() -> None:
    brief = _minimal_valid_v2_brief()
    brief["exa_target_firms"] = [{"name": "   "}]
    with pytest.raises(BriefSchemaError) as exc_info:
        validate_v2_brief(brief)
    assert "exa_target_firms" in exc_info.value.invalid_keys


def test_exa_target_firms_rejects_non_dict_entries() -> None:
    brief = _minimal_valid_v2_brief()
    brief["exa_target_firms"] = ["Accel", "Benchmark"]
    with pytest.raises(BriefSchemaError) as exc_info:
        validate_v2_brief(brief)
    assert "exa_target_firms" in exc_info.value.invalid_keys


def test_exa_target_firms_rejects_non_list_exclude_terms() -> None:
    """The namesake guard has to be a list — a bare string would iterate chars."""

    brief = _minimal_valid_v2_brief()
    brief["exa_target_firms"] = [{"name": "Accel", "exclude_terms": "Accel-KKR"}]
    with pytest.raises(BriefSchemaError) as exc_info:
        validate_v2_brief(brief)
    assert "exa_target_firms" in exc_info.value.invalid_keys


def test_exa_query_lanes_rejects_unknown_lane() -> None:
    brief = _minimal_valid_v2_brief()
    brief["exa_query_lanes"] = ["people_search", "podcast_mining"]
    with pytest.raises(BriefSchemaError) as exc_info:
        validate_v2_brief(brief)
    assert "exa_query_lanes" in exc_info.value.invalid_keys


def test_exa_title_terms_rejects_non_strings() -> None:
    brief = _minimal_valid_v2_brief()
    brief["exa_title_terms"] = ["Senior Associate", 42]
    with pytest.raises(BriefSchemaError) as exc_info:
        validate_v2_brief(brief)
    assert "exa_title_terms" in exc_info.value.invalid_keys


def test_do_not_save_requires_name() -> None:
    brief = _minimal_valid_v2_brief()
    brief["do_not_save"] = [{"reason": "Already in process."}]
    with pytest.raises(BriefSchemaError) as exc_info:
        validate_v2_brief(brief)
    assert "do_not_save" in exc_info.value.invalid_keys


def test_do_not_save_rejects_non_dict_entries() -> None:
    brief = _minimal_valid_v2_brief()
    brief["do_not_save"] = ["Jordan Reeves"]
    with pytest.raises(BriefSchemaError) as exc_info:
        validate_v2_brief(brief)
    assert "do_not_save" in exc_info.value.invalid_keys


def test_exa_archetype_queries_absent_defaults_empty(tmp_path) -> None:
    """S8b is additive: a brief that names no archetype still loads.

    Every brief written before this field existed carries no
    ``exa_archetype_queries`` key, and each must keep compiling exactly
    the firm × lane sweep it compiled yesterday.
    """

    brief = _minimal_valid_v2_brief()
    brief["exa_target_firms"] = [{"name": "Benchmark"}]
    validate_v2_brief(brief)

    path = tmp_path / "brief.json"
    path.write_text(json.dumps(brief))
    loaded = load_brief(path)

    assert loaded.exa_archetype_queries == []
    assert loaded._new_brief.exa_archetype_queries == []

    # An explicit empty list is the same statement, not an error.
    brief["exa_archetype_queries"] = []
    validate_v2_brief(brief)


def test_exa_archetype_queries_require_id_and_text() -> None:
    """Both halves are required, and the id has to be a slug.

    An entry with no ``text`` searches for nothing; one with no ``id``
    cannot be named on the slate or keyed in the store. And a
    non-slug id renders three different ways across the console line,
    the coverage row, and the work-unit id.
    """

    for bad in (
        {"text": "Bay Area investors who were consultants first."},
        {"id": "a01-mbb-to-vc"},
        {"id": "   ", "text": "Bay Area investors."},
        {"id": "a01-mbb-to-vc", "text": "   "},
        {"id": "A01 MBB To VC", "text": "Bay Area investors."},
        {"id": "a01-mbb-to-vc", "text": 42},
        "a01-mbb-to-vc",
    ):
        brief = _minimal_valid_v2_brief()
        brief["exa_archetype_queries"] = [bad]
        with pytest.raises(BriefSchemaError) as exc_info:
            validate_v2_brief(brief)
        assert "exa_archetype_queries" in exc_info.value.invalid_keys, bad

    # Two entries under one id collapse to one row on the slate and one
    # work-unit id in the store — the second query's people would read
    # as the first query's.
    brief = _minimal_valid_v2_brief()
    brief["exa_archetype_queries"] = [
        {"id": "a01-mbb-to-vc", "text": "Bay Area investors who consulted."},
        {"id": "a01-mbb-to-vc", "text": "Bay Area operators who consulted."},
    ]
    with pytest.raises(BriefSchemaError) as exc_info:
        validate_v2_brief(brief)
    assert "exa_archetype_queries" in exc_info.value.invalid_keys

    # The minimum well-formed entry: an id and a sentence.
    brief = _minimal_valid_v2_brief()
    brief["exa_archetype_queries"] = [
        {"id": "a01-mbb-to-vc", "text": "Bay Area investors who consulted first."}
    ]
    validate_v2_brief(brief)


def test_exa_archetype_queries_num_results_bounds() -> None:
    """1-100 integers only — Exa's own per-search range.

    ``0`` is a query that asks for nothing and would look like a firm
    with no bench; a bool is an int in Python and would silently mean
    "1"; a float or a string reaches the API as a type it rejects, one
    query into a paid run.
    """

    for bad in (0, -5, 101, 1000, True, False, 25.0, "25", None):
        brief = _minimal_valid_v2_brief()
        brief["exa_archetype_queries"] = [
            {"id": "a01", "text": "Bay Area investors.", "num_results": bad}
        ]
        with pytest.raises(BriefSchemaError) as exc_info:
            validate_v2_brief(brief)
        assert "exa_archetype_queries" in exc_info.value.invalid_keys, bad

    for good in (1, 25, 100):
        brief = _minimal_valid_v2_brief()
        brief["exa_archetype_queries"] = [
            {"id": "a01", "text": "Bay Area investors.", "num_results": good}
        ]
        validate_v2_brief(brief)

    # include_domains has to be a list of strings — a bare string would
    # iterate into single characters as domain names.
    brief = _minimal_valid_v2_brief()
    brief["exa_archetype_queries"] = [
        {"id": "a01", "text": "Bay Area investors.", "include_domains": "linkedin.com"}
    ]
    with pytest.raises(BriefSchemaError) as exc_info:
        validate_v2_brief(brief)
    assert "exa_archetype_queries" in exc_info.value.invalid_keys


def test_validate_accepts_empty_exa_fields() -> None:
    """Empty lists and explicit nulls read as 'the recruiter cleared it'."""

    brief = _minimal_valid_v2_brief()
    brief["exa_target_firms"] = []
    brief["exa_query_lanes"] = []
    brief["exa_title_terms"] = []
    brief["exa_archetype_queries"] = []
    brief["do_not_save"] = None
    validate_v2_brief(brief)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def test_loaded_brief_carries_exa_fields_on_both_dataclasses(tmp_path) -> None:
    path = tmp_path / "brief.json"
    path.write_text(json.dumps(_brief_with_exa_fields()))
    loaded = load_brief(path)

    for brief in (loaded, loaded._new_brief):
        assert [f["name"] for f in brief.exa_target_firms] == ["Accel", "Benchmark"]
        assert brief.exa_target_firms[0]["exclude_terms"] == ["Accel-KKR"]
        assert brief.exa_query_lanes == ["people_search", "team_page"]
        assert brief.exa_title_terms == ["Senior Associate", "Principal"]
        assert [e["name"] for e in brief.do_not_save] == ["Jordan Reeves", "Casey Lin"]

    # The compat mirror must not share mutable state with _new_brief.
    loaded.exa_target_firms[0]["name"] = "MUTATED"
    assert loaded._new_brief.exa_target_firms[0]["name"] == "Accel"


def test_loaded_brief_carries_archetype_queries_on_both_dataclasses(
    tmp_path,
) -> None:
    """The compiler reads whichever Brief it is handed, so both carry it."""

    path = tmp_path / "brief.json"
    path.write_text(json.dumps(_brief_with_exa_fields()))
    loaded = load_brief(path)

    for brief in (loaded, loaded._new_brief):
        entries = brief.exa_archetype_queries
        assert [e["id"] for e in entries] == ["a01-mbb-to-vc", "a02-investor-to-cos"]
        assert entries[0]["axis"] == "consulting_to_investing"
        assert entries[0]["num_results"] == 25
        assert entries[0]["include_domains"] == ["linkedin.com"]
        # The optional fields stay ABSENT rather than being filled in at
        # load: the defaults belong to the compiler, and a loader that
        # wrote them in would make "the author said 25" and "the author
        # said nothing" indistinguishable to brief polish.
        assert set(entries[1]) == {"id", "text"}

    # The compat mirror must not share mutable state with _new_brief.
    loaded.exa_archetype_queries[0]["id"] = "MUTATED"
    assert loaded._new_brief.exa_archetype_queries[0]["id"] == "a01-mbb-to-vc"


def test_load_brief_raises_on_malformed_archetype_queries(tmp_path) -> None:
    """Fail closed at LAUNCH for this field too.

    A text-less entry that merely filtered out during hydration would
    make a brief that names six archetypes run five, and the console
    would report the smaller plan as if it were the whole one.
    """

    brief = _brief_with_exa_fields()
    brief["exa_archetype_queries"].append({"id": "a03-ghost"})
    path = tmp_path / "brief.json"
    path.write_text(json.dumps(brief))

    with pytest.raises(BriefSchemaError) as exc_info:
        load_brief(path)
    assert "exa_archetype_queries" in exc_info.value.invalid_keys


def test_load_brief_raises_on_malformed_do_not_save(tmp_path) -> None:
    """Fail closed at LAUNCH, not just at authoring.

    ``load_brief`` never calls ``validate_v2_brief``. If a nameless
    do-not-save row merely filtered out during hydration, the person
    the recruiter ruled out would walk into the run with nothing
    logged. The load has to refuse.
    """

    brief = _brief_with_exa_fields()
    brief["do_not_save"].append(
        {"linkedin_url": "https://www.linkedin.com/in/ghost", "reason": "active"}
    )
    path = tmp_path / "brief.json"
    path.write_text(json.dumps(brief))

    with pytest.raises(BriefSchemaError) as exc_info:
        load_brief(path)
    assert "do_not_save" in exc_info.value.invalid_keys


def test_load_brief_raises_on_malformed_exa_target_firms(tmp_path) -> None:
    brief = _brief_with_exa_fields()
    brief["exa_target_firms"].append({"domain": "ghost.com"})
    path = tmp_path / "brief.json"
    path.write_text(json.dumps(brief))

    with pytest.raises(BriefSchemaError) as exc_info:
        load_brief(path)
    assert "exa_target_firms" in exc_info.value.invalid_keys


def test_load_brief_still_tolerant_for_non_exa_fields(tmp_path) -> None:
    """The fail-closed guard is narrow — it covers the four Exa keys only."""

    brief = _minimal_valid_v2_brief()
    brief["target_projects"] = ["kubernetes/kubernetes", 42]  # invalid at authoring
    path = tmp_path / "brief.json"
    path.write_text(json.dumps(brief))

    loaded = load_brief(path)  # does not raise
    assert loaded.target_projects == ["kubernetes/kubernetes"]


# ---------------------------------------------------------------------------
# Brief polish preservation
# ---------------------------------------------------------------------------


def test_brief_polish_preserves_exa_fields() -> None:
    """The four Exa fields survive polish, or the cascade falls back.

    ``BriefPolishBackend._polish_core`` returns the LLM's draft
    wholesale on success, so a top-level key the model didn't echo is
    dropped. ``_exa_fields_drift`` is the preservation route: it fires
    on any add or drop, which sends the recruiter's scaffolded draft
    through instead of the lossy polish.
    """

    seed = _brief_with_exa_fields()

    # Nothing to preserve — a brief with no Exa fields sails through.
    assert _exa_fields_drift(seeded=_minimal_valid_v2_brief(), polished={}) is None

    # Echoed intact (order and case are not contract-bearing).
    polished = dict(seed)
    polished["exa_query_lanes"] = ["team_page", "people_search"]
    polished["exa_title_terms"] = ["principal", "  Senior Associate  "]
    assert _exa_fields_drift(seeded=seed, polished=polished) is None

    # Every field, dropped wholesale — the ordinary "LLM didn't echo it" case.
    descriptor = _exa_fields_drift(seeded=seed, polished=_minimal_valid_v2_brief())
    assert descriptor is not None
    for field_name in (
        "exa_target_firms",
        "exa_query_lanes",
        "exa_title_terms",
        "do_not_save",
    ):
        assert field_name in descriptor

    # A single dropped exclusion is drift — this is the one that matters.
    lost_exclusion = dict(seed)
    lost_exclusion["do_not_save"] = seed["do_not_save"][:1]  # Casey Lin removed
    descriptor = _exa_fields_drift(seeded=seed, polished=lost_exclusion)
    assert descriptor is not None
    assert "do_not_save:dropped=['casey lin']" in descriptor

    # Keeping the name but losing the profile URL is drift too — the
    # URL is how the gate recognizes the person it has to skip.
    lost_url = dict(seed)
    lost_url["do_not_save"] = [
        {"name": "Jordan Reeves"},
        seed["do_not_save"][1],
    ]
    descriptor = _exa_fields_drift(seeded=seed, polished=lost_url)
    assert descriptor is not None
    assert "jordan reeves(jreeves)" in descriptor

    # The model is not authorized to ADD a firm the recruiter didn't name.
    added_firm = dict(seed)
    added_firm["exa_target_firms"] = seed["exa_target_firms"] + [{"name": "Sequoia"}]
    descriptor = _exa_fields_drift(seeded=seed, polished=added_firm)
    assert descriptor is not None
    assert "added=['sequoia']" in descriptor


def test_polish_drift_detects_dropped_exclude_terms() -> None:
    """The namesake guard is contract, not decoration.

    Name-set comparison called this no drift; a firm that keeps its
    name and loses ``["Accel-KKR"]`` searches something else.
    """

    seed = {"exa_target_firms": [{"name": "Accel", "exclude_terms": ["Accel-KKR"]}]}
    polished = {"exa_target_firms": [{"name": "Accel"}]}

    descriptor = _exa_fields_drift(seeded=seed, polished=polished)
    assert descriptor is not None
    assert "exa_target_firms" in descriptor
    assert "accel-kkr" in descriptor


def test_polish_drift_detects_changed_domain() -> None:
    seed = {"exa_target_firms": [{"name": "Benchmark", "domain": "benchmark.com"}]}
    polished = {"exa_target_firms": [{"name": "Benchmark", "domain": "benchmark.vc"}]}

    descriptor = _exa_fields_drift(seeded=seed, polished=polished)
    assert descriptor is not None
    assert "benchmark.com" in descriptor
    assert "benchmark.vc" in descriptor


def test_polish_drift_detects_added_lanes_on_explicit_empty() -> None:
    """An explicit ``[]`` is the recruiter saying "run every lane"."""

    seed = {"exa_query_lanes": []}
    polished = {"exa_query_lanes": ["team_page"]}

    descriptor = _exa_fields_drift(seeded=seed, polished=polished)
    assert descriptor is not None
    assert "exa_query_lanes:added=['team_page']" in descriptor


def test_polish_drift_flags_invented_fields() -> None:
    """A key the seed never carried may not appear in the output."""

    descriptor = _exa_fields_drift(
        seeded={"role_title": "Chief of Staff"},
        polished={"do_not_save": [{"name": "Someone The Model Made Up"}]},
    )
    assert descriptor is not None
    assert "do_not_save:invented=" in descriptor


def test_polish_drift_ignores_reason_prose_edits() -> None:
    """`reason` is prose, and editing prose is what polish is for."""

    seed = {
        "do_not_save": [
            {
                "name": "Jordan Reeves",
                "linkedin_url": "https://www.linkedin.com/in/JReeves/",
                "reason": "already in process elsewhere lol",
            }
        ]
    }
    polished = {
        "do_not_save": [
            {
                "name": "Jordan Reeves",
                # Same handle, different URL spelling — the identity
                # normalizer resolves both to `jreeves`.
                "linkedin_url": "linkedin.com/in/jreeves?trk=public",
                "reason": "Already in process on another search.",
            }
        ]
    }

    assert _exa_fields_drift(seeded=seed, polished=polished) is None


def test_brief_polish_preserves_archetype_queries() -> None:
    """An archetype query IS its prose, so the text is in the contract.

    ``_polish_core`` returns the model's draft wholesale. A model that
    keeps every id and rewrites the sentences has replaced the search
    while leaving the id-set identical — id-only comparison would call
    that no drift, which is the same failure ``exclude_terms`` taught
    on the firm entries.
    """

    seed = _brief_with_exa_fields()

    # Echoed intact — whitespace and case are not contract-bearing.
    polished = dict(seed)
    polished["exa_archetype_queries"] = [
        {
            **seed["exa_archetype_queries"][0],
            "text": "  " + seed["exa_archetype_queries"][0]["text"].upper() + " ",
        },
        seed["exa_archetype_queries"][1],
    ]
    assert _exa_fields_drift(seeded=seed, polished=polished) is None

    # A reworded query is a different query.
    reworded = dict(seed)
    reworded["exa_archetype_queries"] = [
        {
            **seed["exa_archetype_queries"][0],
            "text": "Consultants who became investors.",
        },
        seed["exa_archetype_queries"][1],
    ]
    descriptor = _exa_fields_drift(seeded=seed, polished=reworded)
    assert descriptor is not None
    assert "exa_archetype_queries" in descriptor
    assert "consultants who became investors." in descriptor

    # Dropped wholesale — the ordinary "LLM didn't echo it" case.
    descriptor = _exa_fields_drift(seeded=seed, polished=_minimal_valid_v2_brief())
    assert descriptor is not None
    assert "exa_archetype_queries" in descriptor

    # A widened result count is drift: it is what the query costs.
    widened = dict(seed)
    widened["exa_archetype_queries"] = [
        {**seed["exa_archetype_queries"][0], "num_results": 100},
        seed["exa_archetype_queries"][1],
    ]
    assert _exa_fields_drift(seeded=seed, polished=widened) is not None

    # The model may not invent one.
    descriptor = _exa_fields_drift(
        seeded={"role_title": "Chief of Staff"},
        polished={"exa_archetype_queries": [{"id": "a09", "text": "Anyone at all."}]},
    )
    assert descriptor is not None
    assert "exa_archetype_queries:invented=" in descriptor


def test_polish_drift_flags_an_optional_the_polish_wrote_in() -> None:
    """Omitting ``num_results`` and writing 25 are NOT the same query.

    An omitted field follows the compiler's default wherever that
    default goes next; a written one pins this query to 25 forever. The
    canonicalizer treated them as equal, which let a polish step "helpfully"
    spell out defaults and silently take that decision away from the
    recruiter — for ``include_domains`` especially, where the value is a
    search SCOPE they never chose.
    """

    seed = {"exa_archetype_queries": [{"id": "a01", "text": "Bay Area investors."}]}
    polished = {
        "exa_archetype_queries": [
            {
                "id": "a01",
                "text": "Bay Area investors.",
                "num_results": 25,
                "include_domains": ["linkedin.com"],
            }
        ]
    }

    drift = _exa_fields_drift(seeded=seed, polished=polished)
    assert drift is not None
    assert "exa_archetype_queries" in drift
    # And the round trip that changes nothing is still clean.
    assert _exa_fields_drift(seeded=seed, polished=seed) is None
    assert _exa_fields_drift(seeded=polished, polished=polished) is None


def test_polish_drift_flags_axis_relabelling() -> None:
    """``axis`` groups the slate — and the polish step did not author it.

    It steers no search, which is why it was excluded from the key at
    first. But a model free to rewrite ``consulting`` as ``Consulting →
    VC`` is a model quietly regrouping the recruiter's report, and the
    preservation contract covers what the recruiter WROTE, not only what
    the compiler reads.
    """

    seed = {
        "exa_archetype_queries": [
            {"id": "a01", "text": "Bay Area investors.", "axis": "consulting"}
        ]
    }
    polished = {
        "exa_archetype_queries": [
            {"id": "a01", "text": "Bay Area investors.", "axis": "Consulting → VC"}
        ]
    }

    assert _exa_fields_drift(seeded=seed, polished=polished) is not None
    # Case and whitespace are not authorship.
    same = {
        "exa_archetype_queries": [
            {"id": "a01", "text": "Bay Area investors.", "axis": "  Consulting  "}
        ]
    }
    assert (
        _exa_fields_drift(
            seeded=same,
            polished={
                "exa_archetype_queries": [
                    {"id": "a01", "text": "Bay Area investors.", "axis": "consulting"}
                ]
            },
        )
        is None
    )
