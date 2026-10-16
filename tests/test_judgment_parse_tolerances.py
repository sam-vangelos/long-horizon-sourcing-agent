"""Parser tolerances for a judge that is not Opus.

The full-evaluation prose template was tuned against claude-opus-5 for
months, so its ambiguities were pre-resolved by the only model that ever
read them. A DeepSeek-V4.1-Flash campaign on the same brief lost roughly
a fifth of its full evaluations to PARSE_FAILURE, a class of loss the
Opus runs never produced. Almost all of those were well-formed verdicts
the parser or the validator refused on a format ambiguity — the judge
had already opened the profile and reached a decision, and the decision
was thrown away.

The template's own ambiguity is parentheses doing two jobs in one visual
slot: ``N/A (if DIRECT match)`` is a condition addressed to the reader,
while a brief's capability area can carry a parenthetical as part of its
name. This judge resolves that the opposite way from Opus in both
directions, copying the spec's parenthetical into a value and stripping
the brief's parenthetical out of an area name.

Every fixture here is SYNTHETIC — invented people, employers and
capability areas. The real corpus carries live candidate data and never
enters the repo.

Six tolerances:

1. an enum value annotated with a trailing justification
2. ``N/A (DIRECT match)`` — the spec line's parenthetical, copied in
3. a capability area written without the brief's OWN parenthetical
4. ``STEP_3_EVIDENCE`` omitted on a DIRECT match already marked N/A
5. a decided REJECT with N/A transferability on a non-direct match
6. ``N/A`` recency on a NONE-match verdict that is not a save, which
   names no capability whose recency could be dated

Both of the last two are the same rule: a moot field is moot only where
it decides nothing. Transferability decides saves, so it may read N/A
only off them; recency decides saves too — PRIORITY requires CURRENT —
so it may read N/A only off them as well, and only where the match
named no capability at all.

And the locks that keep them from over-reaching: an ambiguous area
prefix does not snap, a non-DIRECT verdict missing ``STEP_3_EVIDENCE``
still raises, a SAVE-family verdict with N/A transferability still
raises, N/A recency raises both off a NONE match and on any save-family
decision, and the DECISION token still refuses coercion — leading-token
in the legacy path, exact-token in v2.
"""

from __future__ import annotations

import pytest

from shared.judgment.templates import parse_full_evaluation_response
from shared.judgment.tool_contracts import (
    JudgmentToolContractError,
    validate_full_evaluation_semantics,
)

# Invented areas. "Fleet telemetry ingestion" and "Fleet telemetry
# dashboards" share a leading phrase on purpose: that shared phrase is the
# ambiguity the prefix-snap rule must refuse.
AREAS = (
    "Fleet telemetry ingestion (the pipeline)",
    "Fleet telemetry dashboards",
    "Harbor scheduling mechanics",
)

_FIELD_ORDER = (
    "STEP_1_MATCH",
    "STEP_1_AREA",
    "STEP_1_EVIDENCE",
    "STEP_2_DEPTH",
    "STEP_2_EVIDENCE",
    "STEP_3_TRANSFERABILITY",
    "STEP_3_EVIDENCE",
    "STEP_1_RECENCY",
    "STEP_4_LEVEL",
    "STEP_4_EVIDENCE",
    "STEP_5_COHERENCE",
    "STEP_5_DRIVER",
    "STEP_6_CALIBER",
    "STEP_6_EVIDENCE",
    "STEP_7_NEAR_MISS",
    "CASE_FOR",
    "CASE_AGAINST",
    "DECISION",
    "CONFIDENCE",
    "REJECT_REASON",
    "OUTREACH_TIER",
    "POST_SAVE_MODIFIER",
    "SUMMARY",
)

_BASE_REJECT = {
    "STEP_1_MATCH": "ADJACENT",
    "STEP_1_AREA": AREAS[0],
    "STEP_1_EVIDENCE": "Ran ingestion for a fleet of dockside sensors at Kelp Harbor Systems.",
    "STEP_2_DEPTH": "BUILDER",
    "STEP_2_EVIDENCE": "Named, first-person artifacts: the intake schema and the replay tool.",
    "STEP_3_TRANSFERABILITY": "TRANSFERABLE",
    "STEP_3_EVIDENCE": "The ingestion methodology ports to the brief's pipeline work.",
    "STEP_1_RECENCY": "CURRENT",
    "STEP_4_LEVEL": "ALIGNED",
    "STEP_4_EVIDENCE": "Senior-IC scope with one direct report.",
    "STEP_5_COHERENCE": "COHERENT",
    "STEP_5_DRIVER": "N/A",
    "STEP_6_CALIBER": "SOLID",
    "STEP_6_EVIDENCE": "Two quantified outcomes and one named platform.",
    "STEP_7_NEAR_MISS": "NONE",
    "CASE_FOR": "Real ingestion construction at a comparable fleet size.",
    "CASE_AGAINST": "Nothing in the record touches the brief's scheduling gate.",
    "DECISION": "REJECT",
    "CONFIDENCE": "0.2",
    "REJECT_REASON": "NON_FIT",
    "OUTREACH_TIER": "NONE",
    "POST_SAVE_MODIFIER": "NONE",
    "SUMMARY": "Capable ingestion builder, outside the brief's scheduling scope.",
}

_BASE_SAVE = {
    **_BASE_REJECT,
    "STEP_1_MATCH": "DIRECT",
    "STEP_3_TRANSFERABILITY": "N/A",
    "STEP_3_EVIDENCE": "N/A",
    "DECISION": "SAVE",
    "CONFIDENCE": "0.71",
    "REJECT_REASON": "NONE",
    "OUTREACH_TIER": "STANDARD",
    "SUMMARY": "Direct ingestion builder at the brief's fleet scale.",
}


def _verdict(base: dict, *, omit: tuple[str, ...] = (), **overrides: str) -> str:
    """Render a synthetic full-evaluation verdict in template order."""

    fields = {**base, **overrides}
    return "\n".join(
        f"{name}: {fields[name]}"
        for name in _FIELD_ORDER
        if name in fields and name not in omit
    )


def _parse(raw: str):
    return parse_full_evaluation_response(
        raw,
        require_semantic_v2=True,
        capability_areas=AREAS,
    )


def _semantics(**overrides):
    payload = {
        "decision": "REJECT",
        "match_type": "ADJACENT",
        "capability_area": AREAS[0],
        "depth": "BUILDER",
        "transferability": "N/A",
        "evidence_recency": "CURRENT",
        "level_alignment": "ALIGNED",
        "opportunity_coherence": "COHERENT",
        "caliber": "SOLID",
        "outreach_tier": None,
        "reject_reason": "NON_FIT",
        "confidence": 0.2,
        "post_save_modifier": "NONE",
        "review_reason_code": None,
        "review_structural_evidence": [],
        "review_recommended_next_step": None,
    }
    payload.update(overrides)
    return validate_full_evaluation_semantics(
        **payload,
        capability_areas=AREAS,
        post_save_modifiers=(),
    )


# ---------------------------------------------------------------------------
# Annotated enum values
# ---------------------------------------------------------------------------


def test_annotated_enum_recovers_leading_token():
    """``CURRENT (…), but …`` is a CURRENT verdict, not a parse failure."""

    raw = _verdict(
        _BASE_REJECT,
        STEP_1_RECENCY=(
            "CURRENT (partnership-titled role), but the qualifying "
            "capability is absent at any recency."
        ),
    )

    result = _parse(raw)

    assert result.decision == "REJECT"
    assert result.evidence_recency == "CURRENT"


@pytest.mark.parametrize(
    ("field", "value", "attribute", "expected"),
    [
        ("STEP_4_LEVEL", "ALIGNED (senior IC, one report)", "level_alignment", "ALIGNED"),
        (
            "STEP_5_COHERENCE",
            "UNCLEAR — the last two moves point away from the domain",
            "opportunity_coherence",
            "UNCLEAR",
        ),
        ("STEP_6_CALIBER", "SOLID (two quantified outcomes)", "caliber", "SOLID"),
        ("REJECT_REASON", "NON_FIT (scheduling gate absent)", "reject_reason", "NON_FIT"),
    ],
)
def test_annotated_enum_recovers_across_every_tolerated_field(
    field, value, attribute, expected
):
    """The tolerance is one rule applied to the whole enum family."""

    result = _parse(_verdict(_BASE_REJECT, **{field: value}))

    assert result.decision == "REJECT"
    assert getattr(result, attribute) == expected


def test_na_direct_match_parenthetical_recovers_to_na():
    """The spec line's own parenthetical, copied into the value.

    ``STEP_3_TRANSFERABILITY: TRANSFERABLE or NOT_TRANSFERABLE or N/A (if
    DIRECT match)`` uses parentheses for a condition addressed to the
    reader; the brief's areas use them for part of a value. DeepSeek
    resolves that ambiguity by copying the condition in.
    """

    raw = _verdict(_BASE_SAVE, STEP_3_TRANSFERABILITY="N/A (DIRECT match)")

    result = _parse(raw)

    assert result.decision == "SAVE"
    assert result.transferability == "N/A"


def test_annotated_enum_without_a_clean_leading_token_is_left_alone():
    """Conservative by design: no leading token means no guess."""

    raw = _verdict(_BASE_REJECT, STEP_6_CALIBER="— unreadable, see the case against")

    result = _parse(raw)

    assert result.decision == "PARSE_FAILURE"
    assert "caliber" in result.summary


# ---------------------------------------------------------------------------
# Capability-area prefix snap, and its ambiguity guard
# ---------------------------------------------------------------------------


def test_area_missing_the_briefs_parenthetical_snaps_to_canonical():
    """DeepSeek drops the brief's OWN trailing parenthetical from an area."""

    raw = _verdict(_BASE_REJECT, STEP_1_AREA="Fleet telemetry ingestion")

    result = _parse(raw)

    assert result.decision == "REJECT"
    assert result.capability_area == "Fleet telemetry ingestion (the pipeline)"


def test_area_that_prefixes_two_canonical_names_does_not_snap():
    """The uniqueness guard: an ambiguous prefix is the validator's to refuse."""

    raw = _verdict(_BASE_REJECT, STEP_1_AREA="Fleet telemetry")

    result = _parse(raw)

    assert result.decision == "PARSE_FAILURE"
    assert "invalid_capability_area" in result.summary


def test_area_that_is_neither_contained_nor_a_prefix_does_not_snap():
    """The three rules are exact-match, containment and prefix — nothing else.

    "telemetry ingestion" names the right area to a human reader: it is the
    canonical name with its first word dropped. It is not an exact match, it
    does not contain a canonical name, and it does not prefix one, so the
    parser leaves it for the validator rather than reconstructing intent.
    """

    raw = _verdict(_BASE_REJECT, STEP_1_AREA="telemetry ingestion")

    result = _parse(raw)

    assert result.decision == "PARSE_FAILURE"
    assert "invalid_capability_area" in result.summary


def test_area_that_is_only_a_list_index_never_claims_a_one_area_brief():
    """The empty-prefix trap: everything prefixes "".

    ``STEP_1_AREA: 3.`` survives the NONE/N/A collapse, and the list-index
    strip then reduces it to nothing. On a brief carrying a single area, an
    unguarded prefix rule would hand that area to a value naming no area at
    all.
    """

    raw = _verdict(_BASE_REJECT, STEP_1_AREA="3.")

    result = parse_full_evaluation_response(
        raw,
        require_semantic_v2=True,
        capability_areas=(AREAS[0],),
    )

    assert result.decision == "PARSE_FAILURE"
    assert "invalid_capability_area" in result.summary


# ---------------------------------------------------------------------------
# The omitted N/A transferability evidence
# ---------------------------------------------------------------------------


def test_direct_match_with_na_transferability_may_omit_step_3_evidence():
    """The largest shape: having written N/A once, the judge skips the echo."""

    raw = _verdict(_BASE_SAVE, omit=("STEP_3_EVIDENCE",))

    result = _parse(raw)

    assert result.decision == "SAVE"
    assert result.transferability == "N/A"
    assert result.transferability_evidence == "N/A"


def test_direct_match_with_annotated_na_may_also_omit_step_3_evidence():
    """The two tolerances compose — that pair appears in the real corpus."""

    raw = _verdict(
        _BASE_SAVE,
        omit=("STEP_3_EVIDENCE",),
        STEP_3_TRANSFERABILITY="N/A (DIRECT match)",
    )

    result = _parse(raw)

    assert result.decision == "SAVE"
    assert result.transferability_evidence == "N/A"


def test_non_direct_match_missing_step_3_evidence_still_raises():
    """No over-reach: transferability is load-bearing off a DIRECT match."""

    raw = _verdict(_BASE_REJECT, omit=("STEP_3_EVIDENCE",))

    result = _parse(raw)

    assert result.decision == "PARSE_FAILURE"
    assert "missing v2 evidence field" in result.summary


def test_direct_match_with_real_transferability_missing_evidence_still_raises():
    """The default is keyed to N/A, not to DIRECT alone."""

    raw = _verdict(
        _BASE_SAVE,
        omit=("STEP_3_EVIDENCE",),
        STEP_3_TRANSFERABILITY="TRANSFERABLE",
    )

    result = _parse(raw)

    assert result.decision == "PARSE_FAILURE"
    assert "missing v2 evidence field" in result.summary


def test_missing_case_against_still_raises_on_a_direct_na_verdict():
    """Only the transferability evidence is defaulted; nothing else is."""

    raw = _verdict(_BASE_SAVE, omit=("STEP_3_EVIDENCE", "CASE_AGAINST"))

    result = _parse(raw)

    assert result.decision == "PARSE_FAILURE"
    assert "missing v2 evidence field" in result.summary


# ---------------------------------------------------------------------------
# Non-direct N/A transferability is a save-path rule
# ---------------------------------------------------------------------------


def test_non_direct_na_transferability_on_a_decided_reject_is_accepted():
    """Transferability is the gate for rescuing a non-direct candidate INTO
    a save. A judge that is rejecting correctly marks it moot."""

    semantics = _semantics()

    assert semantics.decision == "REJECT"
    assert semantics.transferability == "N/A"


def test_non_direct_na_transferability_reject_parses_end_to_end():
    """The same shape through the production parser, not just the validator."""

    raw = _verdict(
        _BASE_REJECT,
        STEP_3_TRANSFERABILITY="N/A",
        STEP_3_EVIDENCE="N/A",
    )

    result = _parse(raw)

    assert result.decision == "REJECT"
    assert result.match_type == "ADJACENT"
    assert result.transferability == "N/A"


@pytest.mark.parametrize(
    "decision", ["SAVE", "INFERENTIAL_SAVE", "TRANSFERABLE_SAVE", "SIGNAL_SAVE"]
)
def test_non_direct_na_transferability_on_a_save_still_raises(decision):
    """The genuinely incoherent combination the rule exists for."""

    with pytest.raises(JudgmentToolContractError) as excinfo:
        _semantics(
            decision=decision,
            outreach_tier="STANDARD",
            reject_reason=None,
        )

    assert excinfo.value.reason == "missing_non_direct_transferability"


def test_non_direct_na_transferability_on_a_save_still_raises_end_to_end():
    raw = _verdict(
        _BASE_REJECT,
        STEP_3_TRANSFERABILITY="N/A",
        STEP_3_EVIDENCE="N/A",
        DECISION="SAVE",
        REJECT_REASON="NONE",
        OUTREACH_TIER="STANDARD",
    )

    result = _parse(raw)

    assert result.decision == "PARSE_FAILURE"
    assert "missing_non_direct_transferability" in result.summary


# ---------------------------------------------------------------------------
# STEP_7_NEAR_MISS is optional and round-trips
# ---------------------------------------------------------------------------


def test_verdict_without_step_7_still_parses():
    """A newly REQUIRED field is the bug class this slice exists to fix."""

    raw = _verdict(_BASE_REJECT, omit=("STEP_7_NEAR_MISS",))

    result = _parse(raw)

    assert result.decision == "REJECT"
    assert result.near_miss == "NONE"


def test_empty_step_7_defaults_to_none():
    raw = _verdict(_BASE_REJECT, STEP_7_NEAR_MISS="")

    result = _parse(raw)

    assert result.decision == "REJECT"
    assert result.near_miss == "NONE"


def test_step_7_value_round_trips():
    raw = _verdict(_BASE_REJECT, STEP_7_NEAR_MISS="LEVEL")

    result = _parse(raw)

    assert result.decision == "REJECT"
    assert result.near_miss == "LEVEL"


def test_step_7_prose_does_not_bleed_into_a_neighbouring_field():
    """STEP_7 must be in the extractor's boundary set.

    A judge reorders fields; this one wrote Step 7 after CASE_FOR. A label
    the extractor does not know is not a boundary, so CASE_FOR would absorb
    the whole line as continuation prose.
    """

    raw = (
        _verdict(_BASE_REJECT, omit=("STEP_7_NEAR_MISS",)).replace(
            f"CASE_FOR: {_BASE_REJECT['CASE_FOR']}",
            f"CASE_FOR: {_BASE_REJECT['CASE_FOR']}\nSTEP_7_NEAR_MISS: CALIBER",
        )
    )

    result = _parse(raw)

    assert result.near_miss == "CALIBER"
    assert result.case_for == _BASE_REJECT["CASE_FOR"]


def test_unrecognized_step_7_value_collapses_to_none():
    """A diagnostic field never fails a verdict."""

    raw = _verdict(_BASE_REJECT, STEP_7_NEAR_MISS="SOMETHING ELSE ENTIRELY")

    result = _parse(raw)

    assert result.decision == "REJECT"
    assert result.near_miss == "NONE"


# The near-miss checkpoint is a measurable field, not a dead one. The
# fixed semantic-evidence field set is restated here so the only-when-present
# shape is proven, not assumed.
_SEMANTIC_EVIDENCE_FIELDS = (
    "match_type",
    "capability_area",
    "capability_evidence",
    "depth",
    "depth_evidence",
    "transferability",
    "transferability_evidence",
    "evidence_recency",
    "level_alignment",
    "opportunity_coherence",
    "caliber",
    "case_for",
    "case_against",
)


def test_named_near_miss_reaches_the_semantic_evidence_payload():
    """A checkpoint nobody can count is a checkpoint that does not exist.

    ``near_miss`` is what makes the inferential-save rate measurable against
    a declared near miss, so it has to survive into the payload the run
    writes to ``final_judgments.jsonl`` and the runtime store.
    """

    from shared.judger import _full_semantic_evidence

    result = _parse(_verdict(_BASE_REJECT, STEP_7_NEAR_MISS="LEVEL"))

    payload = _full_semantic_evidence(result)

    assert payload["near_miss"] == "LEVEL"
    assert tuple(payload) == (*_SEMANTIC_EVIDENCE_FIELDS, "near_miss")


def test_verdict_without_a_near_miss_leaves_the_payload_field_set_alone():
    """Only-when-present: no declared near miss, no new key."""

    from shared.judger import _full_semantic_evidence

    payload = _full_semantic_evidence(_parse(_verdict(_BASE_REJECT)))

    assert "near_miss" not in payload
    assert tuple(payload) == _SEMANTIC_EVIDENCE_FIELDS


# ---------------------------------------------------------------------------
# Recency has no legal answer on a NONE match
# ---------------------------------------------------------------------------


def test_none_match_with_na_recency_validates():
    """The third instance of the moot-field class.

    Recency dates the strongest qualifying capability. A NONE match has just
    said no qualifying capability exists, so there is nothing to date and the
    honest answer is N/A — which the enum used not to offer, destroying a
    decided REJECT over a field the verdict did not rest on.
    """

    semantics = _semantics(
        match_type="NONE",
        capability_area=None,
        transferability="NOT_TRANSFERABLE",
        evidence_recency="N/A",
    )

    assert semantics.decision == "REJECT"
    assert semantics.evidence_recency == "N/A"


def test_none_match_with_na_recency_parses_end_to_end():
    """The same shape through the production parser, as the corpus wrote it."""

    raw = _verdict(
        _BASE_REJECT,
        STEP_1_MATCH="NONE",
        STEP_1_AREA="N/A",
        STEP_3_TRANSFERABILITY="NOT_TRANSFERABLE",
        STEP_3_EVIDENCE="None of the direct-selling methodology ports across.",
        STEP_1_RECENCY="N/A",
    )

    result = _parse(raw)

    assert result.decision == "REJECT"
    assert result.match_type == "NONE"
    assert result.capability_area is None
    assert result.evidence_recency == "N/A"


def test_none_match_with_annotated_na_recency_parses_end_to_end():
    """The two tolerances compose, which is how the real verdict was written."""

    raw = _verdict(
        _BASE_REJECT,
        STEP_1_MATCH="NONE",
        STEP_1_AREA="N/A",
        STEP_3_TRANSFERABILITY="NOT_TRANSFERABLE",
        STEP_3_EVIDENCE="None of the direct-selling methodology ports across.",
        STEP_1_RECENCY=(
            "N/A (no qualifying scheduling evidence exists in any position, "
            "current or past)"
        ),
    )

    result = _parse(raw)

    assert result.decision == "REJECT"
    assert result.evidence_recency == "N/A"


@pytest.mark.parametrize(
    "decision", ["SAVE", "INFERENTIAL_SAVE", "TRANSFERABLE_SAVE", "SIGNAL_SAVE"]
)
def test_na_recency_on_a_none_match_save_still_raises(decision):
    """A moot field is moot only where it decides nothing.

    Recency decides saves — PRIORITY requires CURRENT, and stale-only
    evidence cannot carry a save on its own — so the NONE-match tolerance
    stops at the save-family boundary, which is the same boundary the
    transferability tolerance stops at from the other side.
    """

    with pytest.raises(JudgmentToolContractError) as excinfo:
        _semantics(
            decision=decision,
            match_type="NONE",
            capability_area=None,
            transferability="TRANSFERABLE",
            evidence_recency="N/A",
            outreach_tier="STANDARD",
            reject_reason=None,
        )

    assert excinfo.value.reason == "invalid_evidence_recency"


@pytest.mark.parametrize("match_type", ["DIRECT", "ADJACENT"])
def test_na_recency_off_a_none_match_still_raises(match_type):
    """No leak: a match that named an area named something datable."""

    with pytest.raises(JudgmentToolContractError) as excinfo:
        _semantics(match_type=match_type, evidence_recency="N/A")

    assert excinfo.value.reason == "invalid_evidence_recency"


@pytest.mark.parametrize("match_type", ["DIRECT", "ADJACENT"])
def test_na_recency_off_a_none_match_still_fails_the_parse(match_type):
    """And the parser refuses it too, rather than passing the token down."""

    raw = _verdict(
        _BASE_SAVE if match_type == "DIRECT" else _BASE_REJECT,
        STEP_1_MATCH=match_type,
        STEP_1_RECENCY="N/A",
    )

    result = _parse(raw)

    assert result.decision == "PARSE_FAILURE"
    assert "evidence_recency" in result.summary


# ---------------------------------------------------------------------------
# The DECISION token still refuses coercion
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "decision_line",
    [
        "REJECT — the co-sell case is real but not worth a SAVE",
        "REJECT (a SAVE would need the scheduling gate)",
        "REJECT, and no SAVE channel is open to it",
    ],
)
def test_decision_token_still_refuses_trailing_save_prose(decision_line):
    """R1 must survive this change untouched.

    ``_decision_line_value`` reads the DECISION line's OWN value and
    ``_resolve_decision`` prefers the LEADING decision token, so trailing
    rationale prose can never bleed a SAVE-family substring into the verdict.
    """

    raw = _verdict(_BASE_REJECT, DECISION=decision_line)

    assert parse_full_evaluation_response(raw).decision == "REJECT"


def test_v2_mode_holds_decision_to_an_exact_token():
    """The enum tolerance is deliberately NOT applied to DECISION.

    V2 is a wire contract: the DECISION value is the whole line, matched
    exactly. An annotated verdict is refused rather than trimmed to its
    leading token, so no annotation can ever pick the verdict.
    """

    raw = _verdict(_BASE_REJECT, DECISION="REJECT — not worth a SAVE")

    result = _parse(raw)

    assert result.decision == "PARSE_FAILURE"
    assert "invalid_full_decision" in result.summary


def test_v2_mode_refuses_an_annotated_save_just_as_hard():
    """Strictness cuts both ways — an annotated SAVE is refused too."""

    raw = _verdict(_BASE_SAVE, DECISION="SAVE (pending a recruiter screen)")

    result = _parse(raw)

    assert result.decision == "PARSE_FAILURE"
    assert "invalid_full_decision" in result.summary


# ---------------------------------------------------------------------------
# The tolerance is scoped to the v2 wire contract
# ---------------------------------------------------------------------------


def test_the_historical_substring_ladder_keeps_the_whole_value():
    """The enum tolerance is v2-only, and that scoping is deliberate.

    Off the v2 path the parser reads each enum by substring over the whole
    value, so ``STEP_2_DEPTH: NOT A BUILDER`` resolves to BUILDER — asserted
    directly by ``test_strict_legacy_v2_rejects_substring_enum_near_matches``
    in tests/test_linkedin_judgment_tool_contracts.py. Every refused verdict
    this module exists for came through the v2 path, so the tolerance goes
    there and the substring ladder is left alone.
    """

    raw = _verdict(_BASE_REJECT, STEP_2_DEPTH="NOT A BUILDER")

    assert parse_full_evaluation_response(raw).depth == "BUILDER"
    assert _parse(raw).decision == "PARSE_FAILURE"
