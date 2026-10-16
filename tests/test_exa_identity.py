"""Exa source module S3 — person keys, do-not-save, and dedupe.

Two things are being pinned. First the KEY SHAPE, because the whole
convergence goal rides on it: the 2026-08-28 slate found 53 people, 12 of
whom the LinkedIn campaign had already saved, and those twelve must be
detected as the same humans rather than saved twice. Second the
do-not-save gate, which fails closed on a name match whenever the name is all
it has (a namesake with a different public handle passes) and matches
across the forms one human writes one name in — the list is typed by a
recruiter, not generated.
"""

from __future__ import annotations

from exa.identity import (
    DoNotSaveGate,
    dedupe_candidates,
    exa_person_key,
)
from exa.schemas import ExaCandidate, derive_identity_key
from shared.person_identity import IdentityEvidence, merge_candidates


def _candidate(
    name: str, linkedin_url: str = "", firm: str = "", title: str = ""
) -> ExaCandidate:
    return ExaCandidate(
        identity_key=derive_identity_key(name, linkedin_url, firm),
        name=name,
        current_title=title,
        current_firm=firm,
        linkedin_url=linkedin_url,
    )


# ---------------------------------------------------------------------------
# Person keys
# ---------------------------------------------------------------------------


def test_exa_person_key_is_stable_for_same_handle() -> None:
    canonical = _candidate(
        "Avery Holt", "https://www.linkedin.com/in/averylinholt", "Benchmark"
    )
    # Same human, URL written four different ways, and a firm the second
    # page got wrong.
    variants = [
        _candidate("Avery Holt", "http://linkedin.com/in/averylinholt", "Benchmark"),
        _candidate(
            "Avery L. Holt",
            "https://www.linkedin.com/in/Averylinholt/",
            "Benchmark Capital",
        ),
        _candidate(
            "Avery Holt",
            "https://uk.linkedin.com/in/averylinholt?trk=public_profile",
            "",
        ),
    ]

    assert all(
        exa_person_key(variant) == exa_person_key(canonical) for variant in variants
    )


def test_exa_person_key_shape_documented() -> None:
    """The concrete shape, asserted rather than described.

    The hub follows the evidence. A public LinkedIn handle IS LinkedIn
    evidence, so it keys ``linkedin:<handle>`` — the same string a
    backfill over the LinkedIn campaign's own rows would mint for the
    same human. ``_derive_key`` emits ``<hub>:<handle>`` for any group
    without a ``github``-hub record
    (`shared/person_identity.py:135-141`).
    """

    strong = _candidate(
        "Avery Holt", "https://www.linkedin.com/in/averylinholt", "Benchmark"
    )
    assert exa_person_key(strong) == "linkedin:averylinholt"

    # A team-page harvest with no LinkedIn link has no LinkedIn evidence
    # to claim.
    weak = _candidate("Talia Brennan", "", "Cedarline Capital")
    assert exa_person_key(weak) == "exa:talia-brennan-cedarline-capital"

    # The handle is also recovered from the identity key alone, for a
    # record rehydrated without its URL.
    rehydrated = ExaCandidate(
        identity_key="linkedin.com/in/averylinholt", name="Avery Holt"
    )
    assert exa_person_key(rehydrated) == "linkedin:averylinholt"


def test_merge_candidates_still_cannot_join_two_hubs() -> None:
    """Why the hub choice above matters, stated as a fact about the merge.

    ``merge_candidates`` merges on same-hub-same-handle, a declared
    GitHub login, or a shared verified commit email
    (`shared/person_identity.py:153-196`). ``linkedin_url`` is carried on
    the evidence and read by no merge rule at all — so hub-hopping is
    not something the merge would have repaired. Minting the LinkedIn
    hub up front is what makes the two lanes land on one key.
    """

    keys = merge_candidates(
        [
            IdentityEvidence(
                hub="exa",
                handle="linkedin.com/in/averylinholt",
                linkedin_url="https://www.linkedin.com/in/averylinholt",
            ),
            IdentityEvidence(
                hub="linkedin",
                handle="averylinholt",
                linkedin_url="https://www.linkedin.com/in/averylinholt",
            ),
        ]
    )
    assert [person.key for person in keys] == [
        "exa:linkedin.com/in/averylinholt",
        "linkedin:averylinholt",
    ]

    # Two exa-side reads of one human, on the other hand, converge — the
    # identity key is handle-anchored on both.
    assert derive_identity_key(
        "Avery Holt", "https://www.linkedin.com/in/averylinholt", "Benchmark"
    ) == derive_identity_key(
        "Avery Holt", "https://linkedin.com/in/Averylinholt/", ""
    )


# ---------------------------------------------------------------------------
# Do-not-save gate
# ---------------------------------------------------------------------------


def test_do_not_save_blocks_on_handle() -> None:
    gate = DoNotSaveGate(
        [
            {
                "name": "Felix Brandt",
                "linkedin_url": "https://www.linkedin.com/in/felixbrandt",
                "reason": "already in process elsewhere",
            }
        ]
    )

    # A different name on the same handle is still the same human.
    blocked = _candidate("F. Brandt", "https://linkedin.com/in/Felixbrandt/", "a16z")
    reason = gate.blocks(blocked)
    assert reason is not None
    assert "already in process elsewhere" in reason

    assert gate.blocks(_candidate("Dana Okafor", "", "Northwind Capital")) is None


def test_do_not_save_blocks_on_name_alone() -> None:
    """Fail closed. The recruiter ruled the name out."""

    gate = DoNotSaveGate([{"name": "Mateo Ruiz Albarran"}])

    # No LinkedIn URL at all — a team-page harvest, the case where the
    # handle check cannot help.
    reason = gate.blocks(_candidate("Mateo Ruiz Albarran", "", "a16z"))
    assert reason is not None
    assert reason.startswith("Mateo Ruiz Albarran:")

    assert gate.blocks(_candidate("mateo  ruiz albarran", "", "")) is not None
    # The asymmetry, stated: an unrelated human with the same name is
    # blocked too, and that is the cheap error next to reaching out to
    # someone the recruiter ruled out.
    assert gate.blocks(_candidate("Mateo Ruiz Albarran", "", "Sequoia"))


def test_do_not_save_lets_a_namesake_with_a_different_handle_through() -> None:
    """Two public handles that differ are two people.

    The entry names its person by URL, so a candidate whose own public URL
    is different is a namesake, not the ruled-out person, and is judged.
    """

    gate = DoNotSaveGate(
        [
            {
                "name": "Silas Grant",
                "linkedin_url": "https://www.linkedin.com/in/silas-grant-surge",
                "reason": "hard stop",
            }
        ]
    )

    # A different public handle: judged, under any name form.
    for name in ("Silas Grant", "Silas J. Grant", "Grant, Silas"):
        assert gate.blocks(
            _candidate(name, "https://www.linkedin.com/in/silasgrant-pm", "Mercor")
        ) is None, name
    # The entry's own handle still blocks, whatever the name.
    assert gate.blocks(
        _candidate("A. Moore", "https://linkedin.com/in/Silas-grant-surge/", "Mercor")
    ) is not None
    # No public handle on the candidate: the name is all there is, and it blocks.
    assert gate.blocks(_candidate("Silas Grant", "", "Mercor")) is not None
    assert gate.blocks(
        _candidate("Silas Grant", "https://www.linkedin.com/sales/lead/ACwAAA123", "Mercor")
    ) is not None

    # A record rehydrated without its URL still carries its handle in the
    # identity key, and is read the same way: its own handle blocks, a
    # different one passes.
    rehydrated = ExaCandidate(identity_key="linkedin.com/in/silas-grant-surge", name="Drew M.")
    assert gate.blocks(rehydrated) is not None
    namesake = ExaCandidate(identity_key="linkedin.com/in/silasgrant-pm", name="Silas Grant")
    assert gate.blocks(namesake) is None


def test_do_not_save_name_only_entry_still_blocks_every_namesake() -> None:
    """An entry without a URL cannot tell its person from a namesake: fail closed."""

    gate = DoNotSaveGate(
        [
            {"name": "Avery Holt", "reason": "client conflict"},
            {
                "name": "Avery Holt",
                "linkedin_url": "https://www.linkedin.com/in/averylinholt",
                "reason": "already in process",
            },
        ]
    )

    reason = gate.blocks(
        _candidate("Avery Holt", "https://www.linkedin.com/in/avery-holt-2", "Benchmark")
    )
    assert reason == "Avery Holt: client conflict"
    assert gate.blocks(
        _candidate("Avery L. Holt", "https://www.linkedin.com/in/avery-holt-2", "")
    ) is not None


def test_do_not_save_blocks_middle_initial_variant() -> None:
    gate = DoNotSaveGate([{"name": "Avery Holt", "reason": "client conflict"}])

    assert gate.blocks(_candidate("Avery L. Holt")) is not None
    assert gate.blocks(_candidate("Avery Lin Holt")) is not None
    # A different human with a different surname is untouched.
    assert gate.blocks(_candidate("Avery Holton")) is None


def test_do_not_save_blocks_last_first_form() -> None:
    """A list pasted out of a spreadsheet comes in surname-first."""

    gate = DoNotSaveGate([{"name": "Holt, Avery"}])

    assert gate.blocks(_candidate("Avery Holt")) is not None
    assert gate.blocks(_candidate("Avery L. Holt")) is not None


def test_do_not_save_blocks_diacritic_variant() -> None:
    gate = DoNotSaveGate([{"name": "Avery Holt"}])

    assert gate.blocks(_candidate("Ávery Hólt")) is not None

    reversed_gate = DoNotSaveGate([{"name": "Ávery Hólt"}])
    assert reversed_gate.blocks(_candidate("Avery Holt")) is not None


def test_do_not_save_empty_never_blocks() -> None:
    for gate in (DoNotSaveGate(), DoNotSaveGate([]), DoNotSaveGate(None)):
        assert not gate
        assert len(gate) == 0
        assert (
            gate.blocks(
                _candidate(
                    "Dana Okafor",
                    "https://www.linkedin.com/in/dana-okafor-chiefofstaff",
                    "Northwind Capital",
                )
            )
            is None
        )


def test_do_not_save_from_brief_reads_the_field() -> None:
    class _Brief:
        do_not_save = [{"name": "Felix Brandt", "reason": "client conflict"}]

    class _NoField:
        pass

    gate = DoNotSaveGate.from_brief(_Brief())
    assert gate.blocks(_candidate("Felix Brandt")) == "Felix Brandt: client conflict"
    assert not DoNotSaveGate.from_brief(_NoField())


# ---------------------------------------------------------------------------
# Dedupe
# ---------------------------------------------------------------------------


def test_dedupe_skips_seen_and_blocked_and_batch_duplicates() -> None:
    fresh = _candidate(
        "Dana Okafor",
        "https://www.linkedin.com/in/dana-okafor-chiefofstaff",
        "Northwind Capital",
    )
    seen = _candidate(
        "Avery Holt", "https://www.linkedin.com/in/averylinholt", "Benchmark"
    )
    blocked_key = _candidate(
        "Theo Marsh", "https://www.linkedin.com/in/theo-r-marsh", "Kiln Ventures"
    )
    vetoed = _candidate("Felix Brandt", "https://www.linkedin.com/in/felixbrandt", "a16z")
    repeat = _candidate(
        "Dana Okafor", "https://linkedin.com/in/Dana-Okafor-ChiefOfStaff/", ""
    )

    result = dedupe_candidates(
        [fresh, seen, blocked_key, vetoed, repeat],
        seen_identity_keys={"linkedin.com/in/averylinholt"},
        blocked_person_keys={"linkedin:theo-r-marsh"},
        gate=DoNotSaveGate([{"name": "Felix Brandt", "reason": "client conflict"}]),
    )

    assert [c.name for c in result.kept] == ["Dana Okafor"]
    assert [(c.name, reason) for c, reason in result.skipped] == [
        ("Avery Holt", "already_seen"),
        ("Theo Marsh", "blocked_person_key"),
        ("Felix Brandt", "do_not_save:Felix Brandt: client conflict"),
        ("Dana Okafor", "duplicate_in_batch"),
    ]
    assert result.flagged == []


def test_dedupe_flags_weak_key_duplicates() -> None:
    """A name+firm slug folds when the reads agree, and says it was a slug."""

    first = _candidate("Talia Brennan", "", "Cedarline Capital")
    second = _candidate("Talia  Brennan", "", "Cedarline Capital")
    strong = _candidate(
        "Dana Okafor",
        "https://www.linkedin.com/in/dana-okafor-chiefofstaff",
        "Northwind Capital",
    )
    strong_repeat = _candidate(
        "Dana Okafor",
        "https://www.linkedin.com/in/dana-okafor-chiefofstaff",
        "Northwind Capital",
    )

    result = dedupe_candidates(
        [first, second, strong, strong_repeat],
        seen_identity_keys=set(),
        blocked_person_keys=set(),
        gate=DoNotSaveGate(),
    )

    assert [c.name for c in result.kept] == ["Talia Brennan", "Dana Okafor"]
    assert [reason for _, reason in result.skipped] == [
        "duplicate_in_batch:weak_key",
        "duplicate_in_batch",
    ]


def test_weak_key_distinct_titles_both_kept_and_flagged() -> None:
    """Two titles under one slug is evidence of two people.

    Folding them would delete a real candidate on the strength of a
    name+firm slug; keeping both costs one duplicate a human can see in
    the report.
    """

    associate = _candidate(
        "Talia Brennan", "", "Cedarline Capital", title="Senior Associate"
    )
    chief = _candidate(
        "Talia Brennan", "", "Cedarline Capital", title="Chief of Staff"
    )

    result = dedupe_candidates(
        [associate, chief],
        seen_identity_keys=set(),
        blocked_person_keys=set(),
        gate=DoNotSaveGate(),
    )

    assert [c.current_title for c in result.kept] == [
        "Senior Associate",
        "Chief of Staff",
    ]
    assert result.skipped == []
    assert len(result.flagged) == 1
    flagged_candidate, reason = result.flagged[0]
    assert flagged_candidate is chief
    assert reason == "weak_key_collision:Talia Brennan / Senior Associate"

    # A strong key is a handle, not a guess, so two titles there are one
    # person whose record disagrees with itself — still a fold.
    strong_a = _candidate(
        "Dana Okafor",
        "https://www.linkedin.com/in/dana-okafor-chiefofstaff",
        title="Chief of Staff",
    )
    strong_b = _candidate(
        "Dana Okafor",
        "https://www.linkedin.com/in/dana-okafor-chiefofstaff",
        title="Operating Partner",
    )
    strong_result = dedupe_candidates(
        [strong_a, strong_b],
        seen_identity_keys=set(),
        blocked_person_keys=set(),
        gate=DoNotSaveGate(),
    )
    assert len(strong_result.kept) == 1
    assert strong_result.flagged == []
    assert strong_result.skipped[0][1] == "duplicate_in_batch"

    # One blank title is an absence, not a disagreement.
    blank_result = dedupe_candidates(
        [_candidate("Talia Brennan", "", "Cedarline Capital"), chief],
        seen_identity_keys=set(),
        blocked_person_keys=set(),
        gate=DoNotSaveGate(),
    )
    assert len(blank_result.kept) == 1
    assert blank_result.flagged == []


def test_dedupe_preserves_input_order() -> None:
    candidates = [
        _candidate(f"Person {index}", f"https://www.linkedin.com/in/person-{index}")
        for index in range(6)
    ]

    result = dedupe_candidates(
        candidates,
        seen_identity_keys={"linkedin.com/in/person-2"},
        blocked_person_keys=set(),
        gate=DoNotSaveGate(),
    )

    assert [c.name for c in result.kept] == [
        "Person 0",
        "Person 1",
        "Person 3",
        "Person 4",
        "Person 5",
    ]
