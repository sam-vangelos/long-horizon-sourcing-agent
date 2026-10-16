"""Exa source module S5 — the slate and the cross-campaign lookup.

The slate is the run's product: the CoS document of 2026-08-28 is what a
recruiter actually read, and this renderer has to reproduce its shape
without acquiring its most dangerous habit — prose that sounds authored.
Two properties are pinned here rather than trusted:

* a missing rationale is STATED, never filled in;
* the verify-before-outreach list is a section with entries in it, not a
  closing sentence a reader can skim past.

``load_known_people`` gets a real sibling store, built through the real
lifecycle guards, because the whole question is whether it reads another
campaign's canonical rows correctly.
"""

from __future__ import annotations

from pathlib import Path

from exa.known import load_known_people
from exa.report import (
    EXCLUDED_HEADING,
    LONG_SHOTS_HEADING,
    NO_RATIONALE,
    FirmCoverage,
    RunView,
    SlateEntry,
    render_slate,
    structured_report_payload,
)
from exa.schemas import ExaCandidate
from shared.runtime_state.store import RuntimeStateStore


SIBLING_BRIEF_ID = "linkedin-cos"


def _entry(name: str, **overrides) -> SlateEntry:
    payload = {
        "name": name,
        "title": "Principal",
        "firm": "Greylock",
        "location": "SF",
        "linkedin_url": f"https://www.linkedin.com/in/{name.lower().replace(' ', '')}",
        "evidence_urls": ["https://greylock.com/team/x"],
        "rationale": "Ran the founder's cadence at a Series B.",
        "decision": "SAVE",
        "confidence": 0.82,
    }
    payload.update(overrides)
    return SlateEntry(**payload)


def _view(**overrides) -> RunView:
    payload = {
        "brief_name": "Chief of Staff to the CEO",
        "run_id": 7,
        "short_list": [_entry("Avery Holt")],
        "worth_a_look": [
            _entry(
                "Mira Lorne",
                decision="REVIEW_FLAGGED",
                review_reason_code="BAND_ABOVE",
                rationale="",
            )
        ],
        "already_known": [],
        "coverage": [
            FirmCoverage(firm="Greylock", queries=2, results=14, candidates=3),
            FirmCoverage(
                firm="Accel", queries=1, results=9, candidates=0, drops={"namesake": 8}
            ),
        ],
        "queries_run": 3,
        "results_total": 23,
        "candidates_total": 3,
        "dropped": {"namesake": 8, "not_profile": 2},
        "skipped": {"already_seen": 1},
        "rejected": 1,
        "campaign_exa_cost_usd": 0.42,
        "campaign_llm_cost_usd": 1.37,
        "session_exa_cost_usd": 0.11,
        "session_llm_cost_usd": 0.29,
        "session_judged": 2,
    }
    payload.update(overrides)
    return RunView(**payload)


# ---------------------------------------------------------------------------


def test_slate_has_all_sections_in_order() -> None:
    markdown = render_slate(_view())

    positions = [
        markdown.index(heading)
        for heading in (
            "## The short list",
            "## Worth a look",
            "## Already known",
            "## Coverage notes",
            "## Verify before outreach",
        )
    ]
    assert positions == sorted(positions)
    assert markdown.startswith(
        "# Chief of Staff to the CEO — Exa web-sourcing slate (run 7)"
    )

    # The lede leads with the CAMPAIGN and names the session as a
    # trailing clause — a resumed run that judged two people does not get
    # to present itself as the whole search.
    lede = markdown.splitlines()[2]
    assert lede.startswith("Campaign to date: ")
    assert "3 queries returned 23 results and produced 3 candidates" in lede
    assert "dropped 11 (namesake 8, not_profile 2)" in lede
    assert "short list of 1" in lede
    assert "spend $0.42 Exa + $1.37 LLM" in lede
    assert "This session (run 7): 2 judged, $0.11 Exa + $0.29 LLM." in lede
    assert lede.index("Campaign to date") < lede.index("This session")

    # Coverage notes name the namesake class and the zero-candidate firm.
    assert "**Accel** — 1 query, 9 results, 0 candidates; drops: namesake 8." in markdown
    assert "Namesake noise: 8" in markdown
    assert "Zero candidates" in markdown


def test_short_list_entries_carry_links_and_rationale() -> None:
    markdown = render_slate(_view())

    assert "1. **Avery Holt** — Principal, Greylock (SF)." in markdown
    assert "Ran the founder's cadence at a Series B." in markdown
    assert (
        "Links: https://www.linkedin.com/in/averyholt ; "
        "https://greylock.com/team/x" in markdown
    )
    # Worth-a-look states the ONE reason, from the structured code.
    assert "REVIEW_FLAGGED · BAND_ABOVE" in markdown


def test_verify_section_lists_every_flagged_entry_in_every_bucket() -> None:
    """The flag is structural, and it is not a proxy for a missing URL.

    Every Exa record is web-derived, so every entry carries the flag —
    including the already-known ones, whose title is exactly as stale as
    anybody else's. Listing only the URL-less entries would let a
    present LinkedIn URL read as verification, which is the inference the
    flag exists to block.
    """

    view = _view(
        short_list=[
            _entry("Avery Holt"),
            _entry("Ghost Person", linkedin_url="", firm="Benchmark"),
        ],
        already_known=[_entry("Known Person", known_label="9000000006")],
    )

    markdown = render_slate(view)
    section = markdown.split("## Verify before outreach", 1)[1]

    assert markdown.count("## Verify before outreach — 4") == 1
    for name in ("Avery Holt", "Ghost Person", "Mira Lorne", "Known Person"):
        assert f"- **{name}**" in section, f"{name} carries the flag and must be listed"
    assert "- **Ghost Person** — Benchmark — no LinkedIn URL found." in section
    assert "1 entry marked" in section
    # The entry that HAS a URL is listed without the marker.
    avery = next(
        line for line in section.splitlines() if line.startswith("- **Avery Holt**")
    )
    assert "no LinkedIn URL found" not in avery


def test_verify_section_omits_an_entry_whose_flag_is_off() -> None:
    """The list follows the record's flag, not the renderer's opinion."""

    view = _view(
        short_list=[_entry("Avery Holt", verify_before_outreach=False)],
        worth_a_look=[],
        already_known=[],
    )

    section = render_slate(view).split("## Verify before outreach", 1)[1]

    assert "## Verify before outreach — 0" in render_slate(view)
    assert "Avery Holt" not in section
    assert "(none — no entry on this slate carries the flag)" in section


def test_no_rationale_is_stated_not_invented() -> None:
    view = _view(short_list=[_entry("Avery Holt", rationale="   ")])

    markdown = render_slate(view)

    assert NO_RATIONALE in markdown
    # Nothing was composed in its place: the only line under the entry is
    # the stated absence, then the links.
    body = markdown.split("## The short list", 1)[1].split("## Worth a look", 1)[0]
    lines = [line.strip() for line in body.splitlines() if line.strip()]
    assert lines[0] == "— 1"
    assert lines[1].startswith("1. **Avery Holt**")
    assert lines[2] == NO_RATIONALE
    assert lines[3].startswith("Links:")
    assert len(lines) == 4, "no extra prose was composed around the absence"


def test_already_known_bucket_from_sibling_state_dir(tmp_path: Path) -> None:
    """A sibling campaign's SAVE surfaces the person as already known."""

    sibling = tmp_path / "linkedin" / "9000000006"
    sibling.mkdir(parents=True)
    store = RuntimeStateStore(sibling / "runtime_state.sqlite3")
    run_id = store.start_run(
        source="linkedin",
        brief_id=SIBLING_BRIEF_ID,
        output_dir=str(sibling),
        mode="fresh",
    )
    store.ensure_candidate(
        source="linkedin",
        brief_id=SIBLING_BRIEF_ID,
        identity_key="avery-l-holt",
        display_name="Avery L. Holt",
        profile_url="https://www.linkedin.com/in/averylinholt",
    )
    for state in ("snippet_extracted", "full_started"):
        store.set_candidate_state(
            run_id=run_id,
            source="linkedin",
            brief_id=SIBLING_BRIEF_ID,
            identity_key="avery-l-holt",
            new_state=state,
        )
    store.set_candidate_state(
        run_id=run_id,
        source="linkedin",
        brief_id=SIBLING_BRIEF_ID,
        identity_key="avery-l-holt",
        new_state="full_terminal",
        terminal_decision="SAVE",
    )
    # A REJECTED person in the same campaign is NOT "already known" —
    # importing another brief's rejection is not this brief's business.
    store.ensure_candidate(
        source="linkedin",
        brief_id=SIBLING_BRIEF_ID,
        identity_key="dropped-person",
        display_name="Dropped Person",
        profile_url="https://www.linkedin.com/in/droppedperson",
    )
    for state in ("snippet_extracted", "full_started"):
        store.set_candidate_state(
            run_id=run_id,
            source="linkedin",
            brief_id=SIBLING_BRIEF_ID,
            identity_key="dropped-person",
            new_state=state,
        )
    store.set_candidate_state(
        run_id=run_id,
        source="linkedin",
        brief_id=SIBLING_BRIEF_ID,
        identity_key="dropped-person",
        new_state="full_terminal",
        terminal_decision="REJECT",
    )

    known = load_known_people([sibling])

    # The Exa record writes the name a different way; the name-key
    # normalizer is what makes the two meet.
    match = ExaCandidate(identity_key="x", name="Avery Holt")
    assert known.match(match) == "9000000006"
    assert known.match(ExaCandidate(identity_key="y", name="Dropped Person")) is None
    assert known.match(ExaCandidate(identity_key="z", name="Nobody Here")) is None

    view = _view(
        already_known=[
            _entry("Avery Holt", known_label="9000000006", decision="SAVE")
        ]
    )
    markdown = render_slate(view)
    assert "## Already known — 1" in markdown
    assert "- **Avery Holt** — 9000000006" in markdown


def test_known_handle_mismatch_does_not_fall_back_to_name(tmp_path: Path) -> None:
    """A public handle is the identity — a miss on it is a real answer.

    Two people named Avery Holt exist. The sibling campaign holds
    ``/in/averylinholt``; this run found ``/in/avery-holt-vc``. Falling
    through to the name index would file a net-new candidate under
    "already known, no action" — the one outcome the cross-campaign
    check must never produce.
    """

    sibling = tmp_path / "linkedin" / "9000000006"
    sibling.mkdir(parents=True)
    store = RuntimeStateStore(sibling / "runtime_state.sqlite3")
    run_id = store.start_run(
        source="linkedin",
        brief_id=SIBLING_BRIEF_ID,
        output_dir=str(sibling),
        mode="fresh",
    )
    store.ensure_candidate(
        source="linkedin",
        brief_id=SIBLING_BRIEF_ID,
        identity_key="avery-holt",
        display_name="Avery Holt",
        profile_url="https://www.linkedin.com/in/averylinholt",
    )
    for state in ("snippet_extracted", "full_started"):
        store.set_candidate_state(
            run_id=run_id,
            source="linkedin",
            brief_id=SIBLING_BRIEF_ID,
            identity_key="avery-holt",
            new_state=state,
        )
    store.set_candidate_state(
        run_id=run_id,
        source="linkedin",
        brief_id=SIBLING_BRIEF_ID,
        identity_key="avery-holt",
        new_state="full_terminal",
        terminal_decision="SAVE",
    )

    known = load_known_people([sibling])

    different_person = ExaCandidate(
        identity_key="linkedin.com/in/avery-holt-vc",
        name="Avery Holt",
        linkedin_url="https://www.linkedin.com/in/avery-holt-vc",
    )
    assert known.match(different_person) is None

    same_person = ExaCandidate(
        identity_key="linkedin.com/in/averylinholt",
        name="Someone Else Entirely",
        linkedin_url="https://www.linkedin.com/in/averylinholt",
    )
    assert known.match(same_person) == "9000000006", (
        "the handle matches even when the display names disagree"
    )

    # Only a candidate with NO handle falls back to the name index.
    no_handle = ExaCandidate(identity_key="avery-holt-benchmark", name="Avery Holt")
    assert known.match(no_handle) == "9000000006"


def test_missing_sibling_state_dir_is_silent(tmp_path: Path) -> None:
    """Best effort: an absent or unreadable sibling yields nothing."""

    known = load_known_people([tmp_path / "not-there"])

    assert not known
    assert known.match(ExaCandidate(identity_key="x", name="Avery Holt")) is None


# ---------------------------------------------------------------------------
# Already known: the name fallback for rows with no public handle (CLO-330)
# ---------------------------------------------------------------------------
#
# A LinkedIn Recruiter save carries a /talent/ URL, which holds no public
# handle, and an Exa find always carries one. Handle-only matching meant the
# Exa slate for one campaign reported "Already known — 0" while 5 of its 45 saves
# were on the LinkedIn slate. The names below are stand-ins for those five.


def _recruiter_url(slug: str) -> str:
    return f"https://www.linkedin.com/talent/profile/AEMAA{slug}"


def _sibling_with_saves(tmp_path: Path, people: list[tuple[str, str]]) -> Path:
    """A sibling LinkedIn campaign holding ``(display_name, profile_url)`` saves."""

    sibling = tmp_path / "linkedin" / "9000000008"
    sibling.mkdir(parents=True)
    store = RuntimeStateStore(sibling / "runtime_state.sqlite3")
    run_id = store.start_run(
        source="linkedin",
        brief_id=SIBLING_BRIEF_ID,
        output_dir=str(sibling),
        mode="fresh",
    )
    for index, (display_name, profile_url) in enumerate(people):
        identity_key = f"recruiter-{index}"
        store.ensure_candidate(
            source="linkedin",
            brief_id=SIBLING_BRIEF_ID,
            identity_key=identity_key,
            display_name=display_name,
            profile_url=profile_url,
        )
        for state in ("snippet_extracted", "full_started", "full_terminal"):
            store.set_candidate_state(
                run_id=run_id,
                source="linkedin",
                brief_id=SIBLING_BRIEF_ID,
                identity_key=identity_key,
                new_state=state,
                terminal_decision="SAVE" if state == "full_terminal" else None,
            )
    return sibling


def _exa(name: str, handle: str = "") -> ExaCandidate:
    if not handle:
        return ExaCandidate(identity_key=f"exa-{name}", name=name)
    return ExaCandidate(
        identity_key=f"linkedin.com/in/{handle}",
        name=name,
        linkedin_url=f"https://www.linkedin.com/in/{handle}",
    )


def test_recruiter_rows_match_an_exa_find_by_name(tmp_path: Path) -> None:
    sibling = _sibling_with_saves(
        tmp_path,
        [
            ("Lena Powell", _recruiter_url("1")),
            ("Larissa M.", _recruiter_url("2")),
            ("Alexa Bartos, PhD", _recruiter_url("3")),
            ("Mei-lin Chou", _recruiter_url("4")),
            ("Zoë Adler", _recruiter_url("5")),
        ],
    )
    known = load_known_people([sibling])

    for exa_name, handle in [
        ("Lena Powell", "lenapowell"),
        ("Larissa Moreau", "larissamoreau"),
        ("Alexa Bartos", "alexabartos"),
        ("Mei-Lin Chou", "meilinchou"),
        ("Zoe  Adler", "zoeadler"),
    ]:
        assert known.match(_exa(exa_name, handle)) == "9000000008", exa_name


def test_a_known_handle_still_outranks_the_name(tmp_path: Path) -> None:
    """Both sides carry a handle and they differ: a different person."""

    sibling = _sibling_with_saves(
        tmp_path,
        [("Lena Powell", "https://www.linkedin.com/in/lenapowell-sf")],
    )
    known = load_known_people([sibling])

    assert known.match(_exa("Lena Powell", "lenapowell")) is None
    assert known.match(_exa("Lena Powell")) == "9000000008"


def test_an_initial_needs_the_same_first_name_and_initial(tmp_path: Path) -> None:
    sibling = _sibling_with_saves(tmp_path, [("Larissa M.", _recruiter_url("2"))])
    known = load_known_people([sibling])

    assert known.match(_exa("Lara Moreau", "laramoreau")) is None
    assert known.match(_exa("Larissa Kline", "larissakline")) is None
    assert known.match(_exa("Larissa", "larissa")) is None


def test_an_initial_is_ambiguous_when_two_finds_fit_it(tmp_path: Path, caplog) -> None:
    sibling = _sibling_with_saves(tmp_path, [("Larissa M.", _recruiter_url("2"))])
    known = load_known_people([sibling])
    known.set_pool(["Larissa Moreau", "Larissa Mendes", "Lena Powell"])

    with caplog.at_level("INFO", logger="exa.known"):
        assert known.match(_exa("Larissa Moreau", "larissamoreau")) is None
        assert known.match(_exa("Larissa Mendes", "larissamendes")) is None
    assert known.ambiguous and "Larissa M." in known.ambiguous[0]
    assert any("Larissa M." in record.getMessage() for record in caplog.records)

    known.set_pool(["Larissa Moreau", "Lena Powell"])
    assert known.match(_exa("Larissa Moreau", "larissamoreau")) == "9000000008"


def test_an_initial_on_the_find_needs_one_known_full_name(tmp_path: Path) -> None:
    one = load_known_people(
        [_sibling_with_saves(tmp_path / "one", [("Larissa Moreau", _recruiter_url("2"))])]
    )
    assert one.match(_exa("Larissa M.")) == "9000000008"

    two = load_known_people(
        [
            _sibling_with_saves(
                tmp_path / "two",
                [
                    ("Larissa Moreau", _recruiter_url("2")),
                    ("Larissa Mendes", _recruiter_url("3")),
                ],
            )
        ]
    )
    assert two.match(_exa("Larissa M.")) is None
    assert two.ambiguous


# ---------------------------------------------------------------------------
# Attainability: reachable first, long shots marked and grouped
# ---------------------------------------------------------------------------


def _long_shot(name: str, reason: str, **overrides) -> SlateEntry:
    return _entry(name, attainability="long_shot", attainability_reason=reason, **overrides)


def _attainability_view() -> RunView:
    return _view(
        short_list=[
            _long_shot("Lab Person", "current employer: OpenAI", firm="OpenAI"),
            _entry("Avery Holt"),
        ],
        worth_a_look=[
            _long_shot(
                "Founder Person",
                "title: Co-Founder",
                title="Co-Founder",
                decision="REVIEW_FLAGGED",
                review_reason_code="BAND_ABOVE",
            ),
            _entry("Mira Lorne", decision="REVIEW_FLAGGED", review_reason_code="BAND_ABOVE"),
        ],
        excluded_employer=[_entry("Ezra Tam", excluded_reason="blacklisted")],
    )


def test_long_shots_sort_last_and_carry_an_inline_marker() -> None:
    markdown = render_slate(_attainability_view())

    short_list = markdown.split("## The short list", 1)[1].split("\n## ", 1)[0]
    assert "1. **Avery Holt** — Principal, Greylock (SF)." in short_list
    assert (
        "2. **Lab Person** — Principal, OpenAI (SF) · long shot "
        "(current employer: OpenAI)." in short_list
    )
    worth = markdown.split("## Worth a look", 1)[1].split("\n## ", 1)[0]
    assert worth.index("**Mira Lorne**") < worth.index("**Founder Person**")
    assert "**Founder Person** — Co-Founder, Greylock (SF) · long shot (title: Co-Founder)." in worth
    avery = next(line for line in short_list.splitlines() if "Avery Holt" in line)
    assert "long shot" not in avery


def test_long_shot_summary_sits_between_worth_a_look_and_excluded() -> None:
    markdown = render_slate(_attainability_view())

    assert markdown.count(f"{LONG_SHOTS_HEADING} — 2") == 1
    positions = [
        markdown.index(heading)
        for heading in ("## Worth a look", LONG_SHOTS_HEADING, EXCLUDED_HEADING, "## Already known")
    ]
    assert positions == sorted(positions)
    section = markdown.split(LONG_SHOTS_HEADING, 1)[1].split("\n## ", 1)[0]
    assert "- **Lab Person** — Principal, OpenAI (SF). Current employer: OpenAI; on the short list." in section
    assert "- **Founder Person** — Co-Founder, Greylock (SF). Title: Co-Founder; worth a look." in section
    assert "Avery Holt" not in section


def test_slate_without_long_shots_has_no_summary_section() -> None:
    markdown = render_slate(_view())

    assert LONG_SHOTS_HEADING not in markdown
    assert "long shot" not in markdown


def test_run_report_counts_long_shots() -> None:
    payload = structured_report_payload(_attainability_view())

    assert payload["metrics_summary"]["long_shots"] == 2
    assert structured_report_payload(_view())["metrics_summary"]["long_shots"] == 0


def test_pipeline_slate_marks_a_current_long_shot_employer(tmp_path: Path) -> None:
    """End to end: the tier is read off the stored record at render time."""

    import asyncio
    import json

    from tests.test_exa_orchestrator import (
        ScriptedJudge,
        _person,
        _pipeline,
        _state_dir,
    )

    results = {
        "Benchmark": [
            _person("labperson", "Lab Person", "Research Engineer", "OpenAI"),
            _person("averylinholt", "Avery Holt", "Investor", "Benchmark"),
        ]
    }
    pipeline, _ = _pipeline(
        tmp_path,
        results=results,
        judge=ScriptedJudge({"Lab Person": "SAVE", "Avery Holt": "SAVE"}),
    )

    asyncio.run(pipeline.run())

    slate = (_state_dir(tmp_path) / "exa-slate.md").read_text()
    short_list = slate.split("## The short list", 1)[1].split("\n## ", 1)[0]
    assert short_list.index("**Avery Holt**") < short_list.index("**Lab Person**")
    assert "· long shot (current employer: OpenAI)." in short_list
    assert f"{LONG_SHOTS_HEADING} — 1" in slate
    report = json.loads((_state_dir(tmp_path) / "run-report.json").read_text())
    assert report["metrics_summary"]["long_shots"] == 1
    assert report["metrics_summary"]["saved"] == 2
