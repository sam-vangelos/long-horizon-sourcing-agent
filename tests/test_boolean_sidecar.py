"""The enricher sidecar behind its flag, in the two places plans are finalized.

Off, the flag must be inert: a finalized plan serializes to exactly what it
serialized to before the sidecar existed. On, a proposal is kept only when the
scorer rates the whole string higher, and every way the exchange can fail —
a hung sidecar, a 5xx, a body that is not JSON, a group that does not parse,
a group that scores worse, a string that grows past the field cap — leaves
the plan exactly as the flag-off path does.
"""

from __future__ import annotations

import io
import json
import urllib.error
import urllib.request
from pathlib import Path

import pytest

import shared.config as config
from linkedin import boolean_enrichment
from linkedin import boolean_sidecar
from linkedin import strategy
from linkedin.boolean_score.parser import parse
from linkedin.boolean_score.score import score
from shared.schemas import AdaptationResponse, ExecutionPlan

# One single-concept group the lexicon maps, beside a two-concept sweep; the
# first group is a register request, and the string already has its anchor.
# The sweep is a mixed group, so its two concept runs ("RLHF", "DPO") are sent
# too; no fake below improves them.
THIN = '("evaluation") AND ("RLHF" OR "DPO")'
REGISTER_GROUP = '("evaluation" OR "evaluations" OR "evals")'
RICH = '("evaluation" OR "evaluations" OR "evals") AND ("RLHF" OR "DPO")'
# A lone sweep of named benchmarks with nothing general pinning it.
SWEEP = '("IFEval" OR "RewardBench" OR "MT-Bench")'
ANCHOR_GROUP = '("evaluation" OR "evaluations" OR "evals")'


@pytest.fixture
def brief():
    from tests.test_lane_key_integrity import _make_brief

    return _make_brief()


def _plan(boolean: str = THIN) -> ExecutionPlan:
    return ExecutionPlan.from_dict(
        {
            "strategy_rationale": "t",
            "generated_strings": [
                {
                    "boolean": boolean,
                    "rationale": "r",
                    "family_key": "f",
                    "domain_lane": "general",
                }
            ],
            "coverage_gaps": [
                {
                    "gap": "g",
                    "suggested_boolean": '("post-training" OR "RLHF")',
                    "rationale": "r",
                }
            ],
        }
    )


@pytest.fixture(autouse=True)
def _forget_the_cached_lexicon():
    boolean_enrichment.load_role_lexicon.cache_clear()
    yield
    boolean_enrichment.load_role_lexicon.cache_clear()


@pytest.fixture(autouse=True)
def _lexicon_enricher_off(monkeypatch):
    """The operator's ``.env`` may turn the deterministic enricher on; these
    tests are about the sidecar alone."""
    monkeypatch.setattr(config, "LINKEDIN_LEXICON_ENRICHMENT_ENABLED", False)


def _sidecar(monkeypatch, on: bool) -> None:
    monkeypatch.setattr(config, "LINKEDIN_ENRICHER_SIDECAR_ENABLED", on)
    monkeypatch.setattr(config, "LINKEDIN_ENRICHER_SIDECAR_URL", "http://sidecar.test")
    monkeypatch.setattr(config, "LINKEDIN_ENRICHER_SIDECAR_TIMEOUT_SECONDS", 1.0)
    monkeypatch.setattr(config, "LINKEDIN_ENRICHER_SIDECAR_SAMPLES", 4)


class _Response:
    def __init__(self, body: bytes, status: int = 200) -> None:
        self.status = status
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        return None


class _FakeSidecar:
    """Stands in for ``urlopen``. Answers per task, records every request."""

    def __init__(self, groups: dict[str, list[str]] | None = None, *, fail=None):
        self.groups = groups or {}
        self.fail = fail
        self.requests: list[dict] = []

    def __call__(self, request, timeout=None):
        body = json.loads(request.data.decode("utf-8"))
        self.requests.append(
            {
                "url": request.full_url,
                "method": request.get_method(),
                "content_type": request.get_header("Content-type"),
                "timeout": timeout,
                **body,
            }
        )
        if self.fail is not None:
            return self.fail(request)
        candidates = [
            {"text": group, "group": group}
            for group in self.groups.get(body["task"], [])
        ]
        candidates.append({"text": "no group here", "group": None})
        payload = {"adapter": "re-v3-3b-ck100", "candidates": candidates}
        return _Response(json.dumps(payload).encode("utf-8"))


def _serve(monkeypatch, fake: _FakeSidecar) -> _FakeSidecar:
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    return fake


def _flag_off_payload(brief, plan_factory=_plan) -> str:
    return json.dumps(
        strategy._finalize_execution_plan(brief, plan_factory(), None).to_dict(),
        sort_keys=True,
    )


def _payload_with(monkeypatch, brief, fake, plan_factory=_plan) -> str:
    _serve(monkeypatch, fake)
    _sidecar(monkeypatch, True)
    return json.dumps(
        strategy._finalize_execution_plan(brief, plan_factory(), None).to_dict(),
        sort_keys=True,
    )


# --- the flag is inert by default ------------------------------------------


@pytest.mark.structural
def test_the_flag_defaults_off():
    """The CODE default is off; the operator's ``.env`` may say otherwise."""
    source = Path(config.__file__).read_text()
    assert '_env_flag(\n    "LINKEDIN_ENRICHER_SIDECAR_ENABLED", "false"' in source


def test_a_finalized_plan_is_byte_identical_with_the_flag_off(monkeypatch, brief):
    """The gate is proved by breaking what it gates: the client raises on
    contact, so a flag that is not read fails here."""
    _sidecar(monkeypatch, False)
    untouched = json.dumps(_plan().to_dict(), sort_keys=True)

    def _explode(*args, **kwargs):
        raise AssertionError("the sidecar was asked with the flag off")

    monkeypatch.setattr(boolean_sidecar, "request_candidates", _explode)

    finalized = strategy._finalize_execution_plan(brief, _plan(), None)
    payload = finalized.to_dict()
    assert payload["generated_strings"][0]["boolean"] == THIN
    blob = json.dumps(payload)
    for key in ("boolean_sidecar", "boolean_original", "suggested_boolean_sidecar"):
        assert key not in blob

    items = [{"boolean": THIN}]
    assert boolean_sidecar.enrich_items_with_sidecar(items, "boolean") == 0
    assert items == [{"boolean": THIN}]
    assert json.dumps(_plan().to_dict(), sort_keys=True) == untouched


# --- what the flag turns on -------------------------------------------------


def test_a_better_register_group_is_kept_with_its_receipt(monkeypatch, brief):
    fake = _serve(monkeypatch, _FakeSidecar({"register": [REGISTER_GROUP]}))
    _sidecar(monkeypatch, True)

    finalized = strategy._finalize_execution_plan(brief, _plan(), None)
    item = finalized.to_dict()["generated_strings"][0]

    assert item["boolean"] == RICH
    assert item["boolean_original"] == THIN
    receipt = item["boolean_sidecar"]
    assert receipt["adapter"] == "re-v3-3b-ck100"
    # The whole group, then the sweep's two runs.
    assert receipt["requests"] == 3
    assert len(receipt["kept"]) == 1
    kept = receipt["kept"][0]
    assert kept["task"] == "register"
    assert kept["group_index"] == 0
    assert kept["before"] == THIN
    assert kept["after"] == RICH
    assert kept["score_after"] > kept["score_before"] + 0.01

    lexicon = boolean_enrichment.load_role_lexicon()
    assert item["boolean_score"] == boolean_enrichment._score_summary(RICH, lexicon)
    assert score(RICH, lexicon).log_composite > score(THIN, lexicon).log_composite

    register = [r for r in fake.requests if r["task"] == "register"]
    assert register[0]["given"] == '("evaluation")'
    assert register[0]["url"] == "http://sidecar.test/enrich"
    assert register[0]["method"] == "POST"
    assert register[0]["content_type"] == "application/json"
    assert register[0]["n"] == 4
    assert register[0]["temperature"] == 0.8
    assert register[0]["timeout"] == 1.0


def test_a_bare_term_in_an_and_is_widened_inside_parentheses(monkeypatch):
    _serve(monkeypatch, _FakeSidecar({"register": [REGISTER_GROUP]}))
    _sidecar(monkeypatch, True)
    items = [{"boolean": '"evaluation" AND ("RLHF" OR "DPO")'}]
    assert boolean_sidecar.enrich_items_with_sidecar(items, "boolean") == 1
    assert items[0]["boolean"] == RICH
    assert parse(items[0]["boolean"]).legal


def test_a_lone_sweep_gains_an_anchor(monkeypatch, brief):
    fake = _serve(monkeypatch, _FakeSidecar({"anchor": [ANCHOR_GROUP]}))
    _sidecar(monkeypatch, True)

    finalized = strategy._finalize_execution_plan(brief, _plan(SWEEP), None)
    item = finalized.to_dict()["generated_strings"][0]

    assert item["boolean"] == f"{ANCHOR_GROUP} AND {SWEEP}"
    assert item["boolean_original"] == SWEEP
    kept = item["boolean_sidecar"]["kept"]
    assert [entry["task"] for entry in kept] == ["anchor"]
    assert kept[0]["group_index"] == "anchor"
    lexicon = boolean_enrichment.load_role_lexicon()
    assert score(item["boolean"], lexicon).anchor_present
    assert not score(SWEEP, lexicon).anchor_present
    assert item["boolean_score"]["anchor_present"] is True
    anchor = [r for r in fake.requests if r["task"] == "anchor"]
    assert [r["given"] for r in anchor] == [SWEEP]


def test_an_unparenthesised_sweep_is_wrapped_before_the_anchor(monkeypatch):
    _serve(monkeypatch, _FakeSidecar({"anchor": [ANCHOR_GROUP]}))
    _sidecar(monkeypatch, True)
    bare = '"IFEval" OR "RewardBench" OR "MT-Bench"'
    items = [{"boolean": bare}]
    boolean_sidecar.enrich_items_with_sidecar(items, "boolean")
    assert items[0]["boolean"] == f"{ANCHOR_GROUP} AND ({bare})"


def test_coverage_gaps_get_their_own_receipt(monkeypatch, brief):
    _serve(monkeypatch, _FakeSidecar({"register": [REGISTER_GROUP]}))
    _sidecar(monkeypatch, True)

    def _gap_plan() -> ExecutionPlan:
        plan = _plan()
        plan.coverage_gaps[0]["suggested_boolean"] = THIN
        return plan

    finalized = strategy._finalize_execution_plan(brief, _gap_plan(), None)
    gap = finalized.to_dict()["coverage_gaps"][0]
    assert gap["suggested_boolean"] == RICH
    assert gap["suggested_boolean_original"] == THIN
    assert gap["suggested_boolean_sidecar"]["kept"][0]["task"] == "register"
    assert "boolean_score" not in gap


# --- every failure leaves the plan as the flag-off path does ---------------


def test_a_timeout_is_retried_once_then_leaves_the_plan_alone(monkeypatch, brief):
    _sidecar(monkeypatch, False)
    expected = _flag_off_payload(brief)

    def _hang(request):
        raise TimeoutError("timed out")

    fake = _FakeSidecar(fail=_hang)
    assert _payload_with(monkeypatch, brief, fake) == expected
    # The plan string's first request, sent twice; the breaker then stops the
    # rest of the string and the gap.
    assert len(fake.requests) == 2


def test_a_refused_connection_leaves_the_plan_alone(monkeypatch, brief):
    _sidecar(monkeypatch, False)
    expected = _flag_off_payload(brief)

    def _refuse(request):
        raise urllib.error.URLError(ConnectionRefusedError(61, "refused"))

    assert _payload_with(monkeypatch, brief, _FakeSidecar(fail=_refuse)) == expected


def test_a_server_error_is_not_retried_and_leaves_the_plan_alone(monkeypatch, brief):
    _sidecar(monkeypatch, False)
    expected = _flag_off_payload(brief)

    def _explode(request):
        raise urllib.error.HTTPError(
            request.full_url, 500, "boom", {}, io.BytesIO(b'{"detail": "x"}')
        )

    fake = _FakeSidecar(fail=_explode)
    assert _payload_with(monkeypatch, brief, fake) == expected
    # Each request once: the string's group and its two runs, the gap's two runs.
    assert [r["given"] for r in fake.requests] == [
        '("evaluation")',
        '("RLHF")',
        '("DPO")',
        '("post-training")',
        '("RLHF")',
    ]


def test_a_garbage_body_leaves_the_plan_alone(monkeypatch, brief):
    _sidecar(monkeypatch, False)
    expected = _flag_off_payload(brief)
    fake = _FakeSidecar(fail=lambda request: _Response(b"<html>not json"))
    assert _payload_with(monkeypatch, brief, fake) == expected


def test_a_wrongly_shaped_body_leaves_the_plan_alone(monkeypatch, brief):
    _sidecar(monkeypatch, False)
    expected = _flag_off_payload(brief)
    fake = _FakeSidecar(
        fail=lambda request: _Response(b'{"adapter": "x", "candidates": "nope"}')
    )
    assert _payload_with(monkeypatch, brief, fake) == expected


def test_a_candidate_that_does_not_parse_is_dropped(monkeypatch, brief):
    _sidecar(monkeypatch, False)
    expected = _flag_off_payload(brief)
    illegal = '("evaluation" OR "evals"'
    assert not parse(illegal).legal
    fake = _FakeSidecar({"register": [illegal]})
    assert _payload_with(monkeypatch, brief, fake) == expected


def test_a_candidate_that_scores_lower_is_dropped(monkeypatch, brief):
    _sidecar(monkeypatch, False)
    expected = _flag_off_payload(brief)
    worse = '("evaluation" OR "LLM")'
    lexicon = boolean_enrichment.load_role_lexicon()
    assert (
        score(THIN.replace('("evaluation")', worse), lexicon).log_composite
        < score(THIN, lexicon).log_composite
    )
    fake = _FakeSidecar({"register": [worse]})
    assert _payload_with(monkeypatch, brief, fake) == expected


def test_a_candidate_past_the_field_cap_is_dropped(monkeypatch, brief):
    _sidecar(monkeypatch, False)
    expected = _flag_off_payload(brief)
    padding = " OR ".join(f'"evaluation item {i}"' for i in range(120))
    huge = f'("evaluation" OR "evaluations" OR "evals" OR {padding})'
    assert len(THIN.replace('("evaluation")', huge)) > 2000
    fake = _FakeSidecar({"register": [huge]})
    assert _payload_with(monkeypatch, brief, fake) == expected


def test_the_length_cap_is_enforced_by_the_client_not_only_the_scorer(monkeypatch):
    """The scorer charges a long string too, so the case above cannot tell the
    client's own cap from the scorer's. Lower the cap under a candidate the
    scorer would keep and the client must still refuse it."""
    _serve(monkeypatch, _FakeSidecar({"register": [REGISTER_GROUP]}))
    _sidecar(monkeypatch, True)
    monkeypatch.setattr(boolean_sidecar, "_MAX_CHARS", len(RICH) - 1)
    items = [{"boolean": THIN}]
    assert boolean_sidecar.enrich_items_with_sidecar(items, "boolean") == 0
    assert items == [{"boolean": THIN}]


def test_an_illegal_string_is_never_sent(monkeypatch):
    fake = _serve(monkeypatch, _FakeSidecar({"register": [REGISTER_GROUP]}))
    _sidecar(monkeypatch, True)
    items = [{"boolean": '("evaluation" AND'}]
    boolean_sidecar.enrich_items_with_sidecar(items, "boolean")
    assert items == [{"boolean": '("evaluation" AND'}]
    assert fake.requests == []


# --- adaptation runs twice; the second pass must change nothing -------------


def test_the_adaptation_hook_does_not_run_twice(monkeypatch, brief):
    fake = _serve(monkeypatch, _FakeSidecar({"register": [REGISTER_GROUP]}))
    _sidecar(monkeypatch, True)
    adaptation = AdaptationResponse.from_dict(
        {
            "new_strings": [
                {
                    "boolean": THIN,
                    "rationale": "r",
                    "family_key": "f",
                    "domain_lane": "general",
                }
            ]
        }
    )
    strategy._annotate_adaptation_metadata(brief, adaptation, None)
    first = json.loads(json.dumps(adaptation.new_strings[0]))
    asked = len(fake.requests)
    assert first["boolean"] == RICH
    assert first["boolean_original"] == THIN

    strategy._annotate_adaptation_metadata(brief, adaptation, None)
    second = adaptation.new_strings[0]
    assert len(fake.requests) == asked
    assert second["boolean_sidecar"] == first["boolean_sidecar"]
    assert second["boolean_original"] == THIN
    assert second["boolean"] == RICH


def test_an_original_the_lexicon_enricher_left_is_preserved(monkeypatch):
    _serve(monkeypatch, _FakeSidecar({"register": [REGISTER_GROUP]}))
    _sidecar(monkeypatch, True)
    items = [{"boolean": THIN, "boolean_original": "what the model wrote"}]
    boolean_sidecar.enrich_items_with_sidecar(items, "boolean")
    assert items[0]["boolean"] == RICH
    assert items[0]["boolean_original"] == "what the model wrote"


def test_the_sidecar_never_prints(monkeypatch, capsys):
    """The run console is a fixed surface; the sidecar is not on it."""
    _serve(monkeypatch, _FakeSidecar({"register": [REGISTER_GROUP]}))
    _sidecar(monkeypatch, True)
    plan = _plan()
    boolean_sidecar.enrich_plan_with_sidecar(plan)
    assert plan.generated_strings[0]["boolean"] == RICH

    def _refuse(request):
        raise urllib.error.URLError("refused")

    _serve(monkeypatch, _FakeSidecar(fail=_refuse))
    boolean_sidecar.enrich_plan_with_sidecar(_plan())
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


# --- a dead sidecar is asked once per plan, not once per string -------------


def _three_thin_strings() -> ExecutionPlan:
    plan = _plan()
    plan.generated_strings = [
        {"boolean": THIN, "rationale": "r"},
        {"boolean": '("evaluation") AND ("SFT" OR "DPO")', "rationale": "r"},
        {"boolean": '("evaluation") AND ("RLHF" OR "PPO")', "rationale": "r"},
    ]
    return plan


def test_the_first_timeout_stops_every_later_request(monkeypatch):
    _sidecar(monkeypatch, True)

    def _hang(request):
        raise TimeoutError("timed out")

    fake = _serve(monkeypatch, _FakeSidecar(fail=_hang))
    plan = _three_thin_strings()
    before = json.dumps(plan.to_dict(), sort_keys=True)
    boolean_sidecar.enrich_plan_with_sidecar(plan)
    # One request and its one retry; the other two strings are never sent.
    assert len(fake.requests) == 2
    assert json.dumps(plan.to_dict(), sort_keys=True) == before


def test_a_change_made_before_the_sidecar_died_is_kept(monkeypatch):
    _sidecar(monkeypatch, True)
    asked: list[dict] = []

    def _first_then_hang(request, timeout=None):
        asked.append(json.loads(request.data.decode("utf-8")))
        # The first string's three requests are answered: its group, two runs.
        if len(asked) <= 3:
            groups = [{"text": REGISTER_GROUP, "group": REGISTER_GROUP}]
            payload = {"adapter": "a", "candidates": groups}
            return _Response(json.dumps(payload).encode("utf-8"))
        raise TimeoutError("timed out")

    monkeypatch.setattr(urllib.request, "urlopen", _first_then_hang)
    plan = _three_thin_strings()
    boolean_sidecar.enrich_plan_with_sidecar(plan)
    first, second, third = plan.generated_strings
    assert first["boolean"] == RICH
    assert first["boolean_sidecar"]["kept"][0]["task"] == "register"
    assert second["boolean"] == '("evaluation") AND ("SFT" OR "DPO")'
    assert third["boolean"] == '("evaluation") AND ("RLHF" OR "PPO")'
    assert "boolean_sidecar" not in second and "boolean_sidecar" not in third
    # The first string's three, the second's first twice (its retry), the
    # third never.
    assert [r["given"] for r in asked] == [
        '("evaluation")',
        '("RLHF")',
        '("DPO")',
        '("evaluation")',
        '("evaluation")',
    ]


def test_a_refusal_with_an_answer_does_not_trip_the_breaker(monkeypatch):
    """A 500 is the sidecar answering; later strings are still asked."""
    _sidecar(monkeypatch, True)

    def _explode(request):
        raise urllib.error.HTTPError(request.full_url, 500, "boom", {}, None)

    fake = _serve(monkeypatch, _FakeSidecar(fail=_explode))
    boolean_sidecar.enrich_plan_with_sidecar(_three_thin_strings())
    # Every string is asked, its group and then its runs ("PPO" maps to
    # nothing), and so is the gap.
    assert [r["given"] for r in fake.requests] == [
        '("evaluation")',
        '("RLHF")',
        '("DPO")',
        '("evaluation")',
        '("SFT")',
        '("DPO")',
        '("evaluation")',
        '("RLHF")',
        '("post-training")',
        '("RLHF")',
    ]


def test_the_client_marks_an_exhausted_retry_as_failed(monkeypatch):
    def _hang(request):
        raise TimeoutError("timed out")

    _serve(monkeypatch, _FakeSidecar(fail=_hang))
    found = boolean_sidecar.request_candidates(
        "register", '"x"', url="http://sidecar.test", timeout=1.0, n=1
    )
    assert found == [] and found.failed is True

    _serve(monkeypatch, _FakeSidecar())
    found = boolean_sidecar.request_candidates(
        "register", '"x"', url="http://sidecar.test", timeout=1.0, n=1
    )
    assert found == [] and found.failed is False


# --- padding a good group with near-misspellings earns nothing --------------


def _kept_from(monkeypatch, register: list[str], boolean: str = THIN) -> str:
    _serve(monkeypatch, _FakeSidecar({"register": register}))
    _sidecar(monkeypatch, True)
    items = [{"boolean": boolean}]
    boolean_sidecar.enrich_items_with_sidecar(items, "boolean")
    return items[0]["boolean"]


def test_a_misspelled_padding_term_rejects_the_candidate(monkeypatch):
    padded = '("evaluation" OR "evaluations" OR "evalution")'
    assert _kept_from(monkeypatch, [padded]) == THIN


def test_the_clean_candidate_is_kept_over_the_misspelled_one(monkeypatch):
    padded = '("evaluation" OR "evaluations" OR "evalution")'
    clean = '("evaluation" OR "evaluations")'
    kept = _kept_from(monkeypatch, [padded, clean])
    assert kept == THIN.replace('("evaluation")', clean)


def test_within_the_band_the_candidate_with_fewer_unmapped_terms_wins(monkeypatch):
    unmapped = '("evaluation" OR "evaluations" OR "evals" OR "evaluation framework")'
    lexicon = boolean_enrichment.load_role_lexicon()
    assert lexicon.concept_for("evaluation framework") is None
    tie = score(THIN.replace('("evaluation")', unmapped), lexicon).log_composite
    assert abs(tie - score(RICH, lexicon).log_composite) <= 0.01
    assert _kept_from(monkeypatch, [unmapped, REGISTER_GROUP]) == RICH
    assert _kept_from(monkeypatch, [unmapped]) == THIN.replace('("evaluation")', unmapped)


def _padded(proposal: str, authored: str = '("evaluation")') -> bool:
    return boolean_sidecar._padded(
        boolean_sidecar._candidate_terms(proposal),
        boolean_sidecar._candidate_terms(authored),
        boolean_enrichment.load_role_lexicon(),
    )


def test_a_new_unmapped_form_far_from_every_term_is_not_padding():
    """The root rule refuses this proposal (see below); the padding rule on
    its own must not, or it would refuse rooted forms far from every term."""
    assert _padded('("evaluation" OR "evaluations" OR "evals" OR "avaliacao")') is False


def test_an_anchor_padded_with_its_own_near_duplicate_is_rejected(monkeypatch):
    _serve(
        monkeypatch,
        _FakeSidecar({"anchor": ['("evaluation" OR "evaluations" OR "evaluatons")']}),
    )
    _sidecar(monkeypatch, True)
    items = [{"boolean": SWEEP}]
    boolean_sidecar.enrich_items_with_sidecar(items, "boolean")
    assert items == [{"boolean": SWEEP}]


def test_an_unmapped_compound_of_a_group_term_rejects_the_candidate(monkeypatch):
    fused = '("evaluation" OR "evaluations" OR "evals" OR "evaluationsuite")'
    lexicon = boolean_enrichment.load_role_lexicon()
    assert lexicon.concept_for("evaluationsuite") is None
    assert _kept_from(monkeypatch, [fused]) == THIN


def test_short_unmapped_terms_one_edit_apart_are_not_padding():
    """Two- and three-letter acronyms sit one edit from each other by chance;
    that is not a misspelling. (The lexicon maps "ML" and "RL", so an
    unmapped pair stands in for them.) Added to this group they are off its
    root, so only the padding rule is read here."""
    short = '("evaluation" OR "evaluations" OR "evals" OR "QA" OR "QE")'
    lexicon = boolean_enrichment.load_role_lexicon()
    assert lexicon.concept_for("QA") is None and lexicon.concept_for("QE") is None
    assert _padded(short) is False
    assert _padded('("evaluation" OR "evaluations" OR "evalution")') is True


def test_a_separated_phrase_holding_a_group_term_is_not_a_compound(monkeypatch):
    phrase = '("evaluation" OR "evaluations" OR "evals" OR "evaluation framework")'
    lexicon = boolean_enrichment.load_role_lexicon()
    assert lexicon.concept_for("evaluation framework") is None
    assert _kept_from(monkeypatch, [phrase]) == THIN.replace('("evaluation")', phrase)


def test_a_hyphenated_compound_on_a_mapped_word_is_not_fused(monkeypatch):
    """"evaluation" sits fused inside "evaluations", but that word is itself a
    mapped form and a hyphen bounds it; the rule is for fused junk only."""
    hyphen = '("evaluation" OR "evaluations" OR "evals" OR "evaluations-suite")'
    assert _kept_from(monkeypatch, [hyphen]) == THIN.replace('("evaluation")', hyphen)


# --- a proposal keeps each term once and adds nothing off the group's root ---
# The strings and proposals below are the ones a live probe of the sidecar
# returned; each proposal is the group text alone, as the sidecar sends it.

EVAL_SWEEP = '("SWE-bench" OR "HumanEval" OR "LiveCodeBench")'
RUBRIC_STRING = '("ML engineer" OR "research engineer") AND ("code generation") AND ("rubric")'
AGENTS_STRING = '("coding agents") AND ("benchmark" OR "benchmarks")'
PRESALES_STRING = '("pre-sales" OR "presales") AND ("evals" OR "evaluation")'


def _improved(boolean: str, answers: dict[str, list[str]]):
    """``_improve`` on the real lexicon, the sidecar answering by the group
    it was given."""
    lexicon = boolean_enrichment.load_role_lexicon()
    return boolean_sidecar._improve(
        boolean, lexicon, lambda task, given: list(answers.get(given, []))
    )


def test_an_anchor_carrying_an_unrooted_unmapped_word_is_refused():
    proposal = '("evaluation" OR "evals" OR "evaluating" OR "evaluations" OR "metrics")'
    lexicon = boolean_enrichment.load_role_lexicon()
    assert lexicon.concept_for("metrics") is None
    outcome = _improved(EVAL_SWEEP, {EVAL_SWEEP: [proposal]})
    assert outcome.boolean == EVAL_SWEEP
    assert outcome.kept == []


def test_the_same_anchor_without_the_unrooted_word_is_kept():
    clean = '("evaluation" OR "evals" OR "evaluating" OR "evaluations")'
    outcome = _improved(EVAL_SWEEP, {EVAL_SWEEP: [clean]})
    assert outcome.boolean == f"{clean} AND {EVAL_SWEEP}"


def test_a_register_proposal_adding_an_off_root_word_is_refused():
    proposal = (
        '("rubric" OR "rubrics" OR "rubric" OR "rubrics" OR "scorecard" OR "scoring rubric")'
    )
    outcome = _improved(RUBRIC_STRING, {'("rubric")': [proposal]})
    assert outcome.boolean == RUBRIC_STRING
    assert outcome.kept == []


def test_the_off_root_word_alone_is_enough_to_refuse():
    """Without the duplicates the proposal still scores a gain today; the
    unmapped "scorecard" is what refuses it."""
    proposal = '("rubric" OR "rubrics" OR "scorecard")'
    outcome = _improved(RUBRIC_STRING, {'("rubric")': [proposal]})
    assert outcome.boolean == RUBRIC_STRING


def test_a_proposal_repeating_a_term_is_kept_with_each_term_once():
    outcome = _improved(RUBRIC_STRING, {'("rubric")': ['("rubric" OR "rubrics" OR "rubric")']})
    assert outcome.boolean == RUBRIC_STRING.replace('("rubric")', '("rubric" OR "rubrics")')
    assert outcome.kept[0]["score_after"] > outcome.kept[0]["score_before"] + 0.01


def test_a_proposal_that_only_repeats_the_authored_group_changes_nothing():
    outcome = _improved(RUBRIC_STRING, {'("rubric")': ['("rubric" OR "rubric")']})
    assert outcome.boolean == RUBRIC_STRING
    assert outcome.kept == []


def test_rooted_new_forms_are_kept_once_and_an_off_concept_group_is_refused():
    agents = (
        '("coding agents" OR "coding agent" OR "coding agents" OR "agents that code"'
        ' OR "code agents" OR "agent coders")'
    )
    benchmarks = (
        '("benchmark" OR "benchmarks" OR "bench" OR "benchmarks" OR "swe-bench"'
        ' OR "benchmarks-measure")'
    )
    outcome = _improved(
        AGENTS_STRING,
        {'("coding agents")': [agents], '("benchmark" OR "benchmarks")': [benchmarks]},
    )
    assert outcome.boolean == (
        '("coding agents" OR "coding agent" OR "agents that code" OR "code agents"'
        ' OR "agent coders") AND ("benchmark" OR "benchmarks")'
    )
    assert [(entry["task"], entry["group_index"]) for entry in outcome.kept] == [
        ("register", 0)
    ]


def test_a_term_mapped_to_another_concept_is_off_the_group_s_root():
    """"swe-bench" maps, but to its own concept, and no word of it starts
    with "benchmark"; alone it would score a gain."""
    lexicon = boolean_enrichment.load_role_lexicon()
    assert lexicon.concept_for("swe-bench").id != lexicon.concept_for("benchmark").id
    proposal = '("benchmark" OR "benchmarks" OR "swe-bench")'
    outcome = _improved(AGENTS_STRING, {'("benchmark" OR "benchmarks")': [proposal]})
    assert outcome.boolean == AGENTS_STRING


def test_a_clipped_word_is_off_the_group_s_root():
    """"bench" scores no gain on its own, so the guard is read directly: the
    stem of "benchmark" is the whole word, and "bench" does not start with it."""
    lexicon = boolean_enrichment.load_role_lexicon()
    authored = boolean_sidecar._candidate_terms('("benchmark" OR "benchmarks")')
    for added, refused in (("bench", True), ("benchmarks-measure", False)):
        proposed = boolean_sidecar._candidate_terms(
            f'("benchmark" OR "benchmarks" OR "{added}")'
        )
        assert boolean_sidecar._off_root(proposed, authored, lexicon) is refused


def test_forms_of_the_same_concept_are_still_kept():
    proposal = '("evals" OR "evaluation" OR "evaluations" OR "eval" OR "evaluating")'
    outcome = _improved(PRESALES_STRING, {'("evals" OR "evaluation")': [proposal]})
    assert outcome.boolean == PRESALES_STRING.replace('("evals" OR "evaluation")', proposal)


def test_an_unmapped_form_sharing_no_stem_is_refused_however_far_it_sits(monkeypatch):
    far = '("evaluation" OR "evaluations" OR "evals" OR "avaliacao")'
    assert _kept_from(monkeypatch, [far]) == THIN


@pytest.mark.parametrize(
    ("word", "stem"),
    [
        ("benchmark", "benchmark"),
        ("benchmarks", "benchmark"),
        ("evals", "eval"),
        ("evaluation", "evalu"),
        ("evaluations", "evalu"),
        ("agents", "agent"),
        ("coding", "coding"),
        ("rubric", "rubric"),
    ],
)
def test_a_stem_strips_one_suffix_and_keeps_at_least_four_letters(word, stem):
    assert boolean_sidecar._stem(word) == stem


def test_words_split_on_space_hyphen_and_slash():
    assert boolean_sidecar._words("agents that-code/ml  ") == ["agents", "that", "code", "ml"]


def test_a_root_matches_at_the_start_of_a_whole_word_only():
    assert boolean_sidecar._rooted("storage", {"rag"}) is False
    assert boolean_sidecar._rooted("rag pipelines", {"rag"}) is True
    assert boolean_sidecar._rooted("agent coders", {"coding", "agent"}) is True


# --- concept runs inside a mixed group --------------------------------------
# The strategy model writes OR groups that mix concepts and carry names the
# lexicon cannot map. Each concept's terms in such a group are a run; a run is
# sent alone, and only the forms it lacks are added, after its last term.

EVALS = '("evaluation" OR "evaluations" OR "eval" OR "evals")'
MIXED = f'("SWE-bench" OR "Scale AI" OR "code generation") AND {EVALS}'
MIXED_RICHER = f'("SWE-bench" OR "Scale AI" OR "code generation" OR "code gen") AND {EVALS}'
CODE_GEN_RUN = '("code generation")'


def _asking(boolean: str, answers: dict[str, list[str]]):
    """``_improve`` on the real lexicon; the sidecar answers by the group it
    was given, and every request is recorded as (task, given)."""
    lexicon = boolean_enrichment.load_role_lexicon()
    asked: list[tuple[str, str]] = []

    def ask(task: str, given: str) -> list[str]:
        asked.append((task, given))
        return list(answers.get(given, []))

    return boolean_sidecar._improve(boolean, lexicon, ask), asked


def test_a_mixed_group_gains_new_forms_after_its_run():
    outcome, asked = _asking(MIXED, {CODE_GEN_RUN: ['("code generation" OR "code gen")']})
    assert outcome.boolean == MIXED_RICHER
    kept = outcome.kept
    assert [entry["task"] for entry in kept] == ["register_run"]
    assert kept[0]["group_index"] == 0
    assert kept[0]["concept"] == "code-generation"
    assert kept[0]["added"] == ["code gen"]
    assert kept[0]["before"] == MIXED
    assert kept[0]["after"] == MIXED_RICHER
    assert kept[0]["score_after"] > kept[0]["score_before"] + 0.01
    # The whole-group request for the evals group, then the two runs.
    assert outcome.requests == 3 == len(asked)


def test_new_forms_go_after_the_run_s_last_term_not_the_group_s():
    boolean = f'("code generation" OR "Scale AI" OR "SWE-bench") AND {EVALS}'
    outcome, _ = _asking(boolean, {CODE_GEN_RUN: ['("code generation" OR "code gen")']})
    assert outcome.boolean == (
        f'("code generation" OR "code gen" OR "Scale AI" OR "SWE-bench") AND {EVALS}'
    )


def test_an_unmapped_term_is_never_sent():
    lexicon = boolean_enrichment.load_role_lexicon()
    assert lexicon.concept_for("Scale AI") is None
    _, asked = _asking(MIXED, {})
    assert [given for _, given in asked] == [EVALS, '("SWE-bench")', CODE_GEN_RUN]
    assert all(task == "register" for task, _ in asked)
    assert not any("Scale AI" in given for _, given in asked)


def test_a_run_is_sent_as_authored_in_one_group():
    boolean = f'("code generation" OR "Scale AI" OR "code gen" OR "SWE-bench") AND {EVALS}'
    _, asked = _asking(boolean, {})
    assert ("register", '("code generation" OR "code gen")') in asked


def test_a_mixed_group_already_holding_every_candidate_term_changes_nothing():
    """A candidate that only restates terms the group has is not a proposal.
    The scorer counts a term as often as it is written, so the echo written
    twice would score a gain: only the new-terms rule refuses it."""
    lexicon = boolean_enrichment.load_role_lexicon()
    doubled = MIXED.replace('"code generation")', '"code generation" OR "code generation")')
    assert score(doubled, lexicon).log_composite > score(MIXED, lexicon).log_composite + 0.01
    for echo in [CODE_GEN_RUN, '("code generation" OR "Scale AI" OR "SWE-bench")']:
        outcome, _ = _asking(MIXED, {CODE_GEN_RUN: [echo]})
        assert outcome.boolean == MIXED
        assert outcome.kept == []


def test_terms_already_elsewhere_in_the_group_are_not_added_again():
    outcome, _ = _asking(
        MIXED, {CODE_GEN_RUN: ['("code generation" OR "Scale AI" OR "code gen")']}
    )
    assert outcome.boolean == MIXED_RICHER
    assert outcome.kept[0]["added"] == ["code gen"]


def test_a_run_candidate_adding_an_off_root_term_is_refused_whole():
    lexicon = boolean_enrichment.load_role_lexicon()
    assert lexicon.concept_for("program writing") is None
    proposal = '("code generation" OR "code gen" OR "program writing")'
    outcome, _ = _asking(MIXED, {CODE_GEN_RUN: [proposal]})
    assert outcome.boolean == MIXED
    assert outcome.kept == []


def test_a_run_candidate_mapped_to_the_run_s_concept_is_on_its_root():
    """"program synthesis" shares no word stem with "code generation", but the
    lexicon maps it to the run's concept."""
    proposal = '("code generation" OR "program synthesis")'
    outcome, _ = _asking(MIXED, {CODE_GEN_RUN: [proposal]})
    assert outcome.boolean == MIXED.replace(
        '"code generation")', '"code generation" OR "program synthesis")'
    )


def test_a_run_candidate_padded_with_a_misspelling_is_refused_whole():
    proposal = '("code generation" OR "code gen" OR "code generaton")'
    outcome, _ = _asking(MIXED, {CODE_GEN_RUN: [proposal]})
    assert outcome.boolean == MIXED
    assert outcome.kept == []


def test_a_run_splice_past_the_field_cap_is_refused(monkeypatch):
    monkeypatch.setattr(boolean_sidecar, "_MAX_CHARS", len(MIXED_RICHER) - 1)
    outcome, _ = _asking(MIXED, {CODE_GEN_RUN: ['("code generation" OR "code gen")']})
    assert outcome.boolean == MIXED
    assert outcome.kept == []


@pytest.mark.parametrize(
    "leak",
    [
        # A new AND gate: every group after it moves.
        '"code gen") AND ("RLHF"',
        # One group more, split off the run's own group.
        '"code gen") OR ("code gen"',
        # The same groups, but the run's group gains a term it was not given.
        '"code gen" OR "evals"',
    ],
)
def test_a_run_splice_that_reshapes_the_groups_is_refused(monkeypatch, leak):
    """A renderer that leaked an operator or an extra term would reshape the
    string. Each leaked string parses and fits the cap, so the signature
    check is the only thing refusing it."""
    lexicon = boolean_enrichment.load_role_lexicon()
    leaked = MIXED.replace('"code generation")', f'"code generation" OR {leak})')
    assert parse(leaked).legal and len(leaked) <= 2000
    proposal = ['("code generation" OR "code gen")']

    honest = list(boolean_sidecar._run_trials(MIXED, 0, "code-generation", proposal, lexicon))
    assert honest == [(MIXED_RICHER, ("code gen",))]

    monkeypatch.setattr(boolean_sidecar, "_render_addition", lambda surface, quoted: leak)
    assert list(boolean_sidecar._run_trials(MIXED, 0, "code-generation", proposal, lexicon)) == []


def test_an_insertion_that_could_reassociate_is_parenthesised():
    """The group ends where an AND begins, so ``_closes_safely`` cannot vouch
    for a bare insertion and the group is closed in parentheses."""
    boolean = '"SWE-bench" OR "code generation" OR "Scale AI" AND "LLM"'
    lexicon = boolean_enrichment.load_role_lexicon()
    outcome, _ = _asking(boolean, {CODE_GEN_RUN: ['("code generation" OR "code gen")']})
    assert outcome.boolean == (
        '("SWE-bench" OR "code generation" OR "code gen") OR "Scale AI" AND "LLM"'
    )
    assert score(outcome.boolean, lexicon).legal


def test_at_most_three_runs_are_sent_per_string_thinnest_first():
    """Five runs across two groups. The two-form code-generation run comes
    first but is the thickest; of the four one-form runs the first three by
    position are sent, and the requests go left to right."""
    boolean = (
        '("code generation" OR "code gen" OR "SWE-bench") AND '
        '("RLHF" OR "DPO" OR "reward model")'
    )
    outcome, asked = _asking(boolean, {})
    assert asked == [
        ("register", '("SWE-bench")'),
        ("register", '("RLHF")'),
        ("register", '("DPO")'),
    ]
    assert outcome.requests == 3
    assert boolean_sidecar.MAX_CONCEPT_RUNS_PER_STRING == 3


def test_a_run_already_carrying_four_forms_is_skipped_and_three_is_not():
    boolean = (
        '("evaluation" OR "evaluations" OR "eval" OR "evals" OR "SWE-bench") AND '
        '("code generation" OR "code gen" OR "program synthesis" OR "Scale AI")'
    )
    _, asked = _asking(boolean, {})
    assert asked == [
        ("register", '("SWE-bench")'),
        ("register", '("code generation" OR "code gen" OR "program synthesis")'),
    ]
    assert boolean_sidecar.RICH_RUN_FORMS == 4


def test_a_single_concept_group_keeps_the_whole_group_path():
    boolean = '("evaluation") AND ("SWE-bench" OR "Scale AI" OR "code generation")'
    outcome, asked = _asking(
        boolean,
        {
            '("evaluation")': [REGISTER_GROUP],
            CODE_GEN_RUN: ['("code generation" OR "code gen")'],
        },
    )
    assert outcome.boolean == (
        f'{REGISTER_GROUP} AND ("SWE-bench" OR "Scale AI" OR "code generation" OR "code gen")'
    )
    assert [(e["task"], e["group_index"]) for e in outcome.kept] == [
        ("register", 0),
        ("register_run", 1),
    ]
    assert "concept" not in outcome.kept[0]
    assert asked[0] == ("register", '("evaluation")')


def test_one_change_per_run_and_later_runs_see_the_string_as_it_stands():
    boolean = f'("code generation" OR "Scale AI" OR "benchmark") AND {EVALS}'
    outcome, asked = _asking(
        boolean,
        {
            CODE_GEN_RUN: ['("code generation" OR "code gen")'],
            '("benchmark")': ['("benchmark" OR "benchmarks")'],
        },
    )
    assert outcome.boolean == (
        '("code generation" OR "code gen" OR "Scale AI" OR "benchmark" OR "benchmarks")'
        f" AND {EVALS}"
    )
    assert [(e["concept"], e["added"]) for e in outcome.kept] == [
        ("code-generation", ["code gen"]),
        ("benchmark", ["benchmarks"]),
    ]
    assert outcome.kept[1]["before"] == outcome.kept[0]["after"]
    assert [given for _, given in asked][1:] == [CODE_GEN_RUN, '("benchmark")']


def test_a_mixed_plan_is_byte_identical_with_the_flag_off(monkeypatch, brief):
    _sidecar(monkeypatch, False)

    def _explode(*args, **kwargs):
        raise AssertionError("the sidecar was asked with the flag off")

    monkeypatch.setattr(boolean_sidecar, "request_candidates", _explode)
    untouched = json.dumps(_plan(MIXED).to_dict(), sort_keys=True)
    finalized = strategy._finalize_execution_plan(brief, _plan(MIXED), None)
    assert finalized.to_dict()["generated_strings"][0]["boolean"] == MIXED
    assert "sidecar" not in json.dumps(finalized.to_dict())
    assert json.dumps(_plan(MIXED).to_dict(), sort_keys=True) == untouched


def test_a_mixed_group_gains_its_run_through_the_plan_hook(monkeypatch, brief):
    fake = _serve(
        monkeypatch, _FakeSidecar({"register": ['("code generation" OR "code gen")']})
    )
    _sidecar(monkeypatch, True)
    finalized = strategy._finalize_execution_plan(brief, _plan(MIXED), None)
    item = finalized.to_dict()["generated_strings"][0]
    assert item["boolean"] == MIXED_RICHER
    assert item["boolean_original"] == MIXED
    receipt = item["boolean_sidecar"]
    assert [entry["task"] for entry in receipt["kept"]] == ["register_run"]
    assert receipt["requests"] == 3
    assert finalized.to_dict()["coverage_gaps"][0]["suggested_boolean"] == (
        '("post-training" OR "RLHF")'
    )
    assert ('("post-training")', '("RLHF")') == tuple(r["given"] for r in fake.requests[3:])


def test_a_dead_sidecar_leaves_a_mixed_plan_alone(monkeypatch):
    _sidecar(monkeypatch, True)

    def _hang(request):
        raise TimeoutError("timed out")

    fake = _serve(monkeypatch, _FakeSidecar(fail=_hang))
    plan = _plan(MIXED)
    plan.generated_strings.append({"boolean": MIXED, "rationale": "r"})
    before = json.dumps(plan.to_dict(), sort_keys=True)
    boolean_sidecar.enrich_plan_with_sidecar(plan)
    assert len(fake.requests) == 2
    assert json.dumps(plan.to_dict(), sort_keys=True) == before


# --- a term mapped to another concept is off the root, whatever it starts with
# A live replay widened an eval-harness run with forms of "evaluation": every
# one starts with the stem "eval", and each turned a specific group into a
# generic one. The root test is for terms the lexicon cannot map.

CODE_GEN_GROUP = '("code generation" OR "code gen")'
HARNESS_RUN = f'("eval harness" OR "Scale AI") AND {CODE_GEN_GROUP}'
HARNESS_GROUP = f'("eval harness") AND {CODE_GEN_GROUP}'
DRIFT = ['("eval harness" OR "evaluations")', '("eval harness" OR "evaluations" OR "evaluating" OR "evals")']


def _with_added(boolean: str, added: str) -> str:
    return boolean.replace('"eval harness"', f'"eval harness" OR {added}', 1)


@pytest.mark.parametrize("boolean", [HARNESS_RUN, HARNESS_GROUP])
def test_a_rooted_term_mapped_to_another_concept_is_refused(boolean):
    lexicon = boolean_enrichment.load_role_lexicon()
    assert lexicon.concept_for("eval harness").id == "eval-harness"
    for term in ("evaluations", "evaluating", "evals"):
        assert lexicon.concept_for(term).id == "evaluation"
    # Each drift scores a gain today, so only the concept rule refuses it.
    base = score(boolean, lexicon).log_composite
    assert score(_with_added(boolean, '"evaluations"'), lexicon).log_composite > base + 0.01
    for proposal in DRIFT:
        outcome, _ = _asking(boolean, {'("eval harness")': [proposal]})
        assert outcome.boolean == boolean
        assert outcome.kept == []


@pytest.mark.parametrize(
    ("boolean", "task"), [(HARNESS_RUN, "register_run"), (HARNESS_GROUP, "register")]
)
def test_a_form_mapped_to_the_same_concept_is_still_kept(boolean, task):
    lexicon = boolean_enrichment.load_role_lexicon()
    assert lexicon.concept_for("evaluation harness").id == "eval-harness"
    proposal = '("eval harness" OR "evaluation harness")'
    outcome, _ = _asking(boolean, {'("eval harness")': [proposal]})
    assert outcome.boolean == _with_added(boolean, '"evaluation harness"')
    assert [entry["task"] for entry in outcome.kept] == [task]


def test_the_root_test_reads_unmapped_terms_only():
    lexicon = boolean_enrichment.load_role_lexicon()
    authored = boolean_sidecar._candidate_terms('("eval harness")')
    for added, refused in (
        ("evaluations", True),  # rooted, but mapped to "evaluation"
        ("evaluation harness", False),  # mapped to the group's own concept
        ("eval harnesses", False),  # unmapped and rooted
        ("grading", True),  # unmapped and unrooted
    ):
        proposed = boolean_sidecar._candidate_terms(f'("eval harness" OR "{added}")')
        assert boolean_sidecar._off_root(proposed, authored, lexicon) is refused, added


# --- an unmapped term with a lexicon form fused inside is padding ----------
# A live replay kept "evaluationsuite" on an eval-harness run and "LLMjudging"
# on an llm-as-a-judge run. The scorer ignores both, so they earn nothing and
# cost reach nobody searches for.

JUDGE_RUN = f'("LLM-as-a-judge" OR "Scale AI") AND {CODE_GEN_GROUP}'
JUDGE_GROUP = f'("LLM-as-a-judge") AND {CODE_GEN_GROUP}'


def _judge_with(boolean: str, added: str) -> str:
    return boolean.replace('"LLM-as-a-judge"', f'"LLM-as-a-judge" OR {added}', 1)


@pytest.mark.parametrize("boolean", [HARNESS_RUN, HARNESS_GROUP])
def test_a_form_of_another_concept_fused_into_a_word_is_refused(boolean):
    """"evaluation" is a form of a concept that is neither the run's nor in
    the proposal, so only a lexicon-wide look finds it inside the word."""
    lexicon = boolean_enrichment.load_role_lexicon()
    assert lexicon.concept_for("evaluationsuite") is None
    assert lexicon.concept_for("evaluation").id == "evaluation"
    padded = '("eval harness" OR "evaluationsuite" OR "evaluation harness")'
    base = score(boolean, lexicon).log_composite
    trial = _with_added(boolean, '"evaluationsuite" OR "evaluation harness"')
    assert score(trial, lexicon).log_composite > base + 0.01
    outcome, _ = _asking(boolean, {'("eval harness")': [padded]})
    assert outcome.boolean == boolean
    assert outcome.kept == []


@pytest.mark.parametrize("boolean", [JUDGE_RUN, JUDGE_GROUP])
def test_a_spaced_term_run_together_is_refused(boolean):
    """"LLMjudging" is "LLM judge" with its space dropped and an inflection
    added. No form of five letters sits inside it, so this is the
    near-duplicate test read without separators."""
    lexicon = boolean_enrichment.load_role_lexicon()
    assert lexicon.concept_for("LLMjudging") is None
    padded = '("LLM-as-a-judge" OR "LLM judge" OR "LLMjudging")'
    base = score(boolean, lexicon).log_composite
    trial = _judge_with(boolean, '"LLM judge" OR "LLMjudging"')
    assert score(trial, lexicon).log_composite > base + 0.01
    outcome, _ = _asking(boolean, {'("LLM-as-a-judge")': [padded]})
    assert outcome.boolean == boolean
    assert outcome.kept == []


@pytest.mark.parametrize(
    ("boolean", "given", "proposal", "expected"),
    [
        (JUDGE_RUN, '("LLM-as-a-judge")', '("LLM-as-a-judge" OR "LLM judge")',
         _judge_with(JUDGE_RUN, '"LLM judge"')),
        (JUDGE_GROUP, '("LLM-as-a-judge")', '("LLM-as-a-judge" OR "LLM judge")',
         _judge_with(JUDGE_GROUP, '"LLM judge"')),
        (HARNESS_RUN, '("eval harness")', '("eval harness" OR "evaluation harness")',
         _with_added(HARNESS_RUN, '"evaluation harness"')),
    ],
)
def test_the_same_proposal_without_the_fused_word_is_kept(boolean, given, proposal, expected):
    outcome, _ = _asking(boolean, {given: [proposal]})
    assert outcome.boolean == expected


def test_an_anchor_carrying_a_fused_form_of_an_absent_concept_is_refused():
    """"evalbenchmark" is rooted on "eval", so the anchor's root rule lets it
    by; "benchmark" is fused inside it and is in neither the sweep nor the
    anchor."""
    fused = '("evaluation" OR "evals" OR "evaluating" OR "evaluations" OR "evalbenchmark")'
    outcome = _improved(EVAL_SWEEP, {EVAL_SWEEP: [fused]})
    assert outcome.boolean == EVAL_SWEEP


@pytest.mark.parametrize(
    ("added", "beside", "refused"),
    [
        ("evaluationsuite", '"eval harness"', True),
        ("LLMjudging", '"LLM-as-a-judge" OR "LLM judge"', True),
        ("evalharness", '"eval harness"', True),
        ("evaluation harness", '"eval harness"', False),
        ("evaluations-suite", '"eval harness"', False),
        ("evaluation framework", '"eval harness"', False),
        ("judging", '"LLM-as-a-judge" OR "LLM judge"', False),
        # A hyphen marks the words: a compound, not a word run together.
        ("LLM-judging", '"LLM-as-a-judge" OR "LLM judge"', False),
        ("LLMOps", '"LLM-as-a-judge" OR "LLM judge"', False),
        ("evaluator", '"eval harness"', False),
        ("evalbenchmark", '"eval harness"', True),
        # One inflection suffix left over is a form of the word, not a fusion.
        # Each sits beside a group that lacks its base, which a one- or
        # two-letter ending would otherwise sit within edit distance of.
        ("benchmarking", '"eval harness"', False),
        ("reward models", '"eval harness"', False),
        ("evaluation harnesses", '"eval harness"', False),
        ("eval harnesses", '"evaluation harness"', False),
        ("language modeling", '"eval harness"', False),
        ("large language modeling", '"eval harness"', False),
    ],
)
def test_what_counts_as_a_fused_word(added, beside, refused):
    lexicon = boolean_enrichment.load_role_lexicon()
    authored = boolean_sidecar._candidate_terms(f"({beside})")
    proposed = boolean_sidecar._candidate_terms(f'({beside} OR "{added}")')
    assert boolean_sidecar._padded(proposed, authored, lexicon) is refused



# --- a short word gives no stem ---------------------------------------------
# "LLM-as-a-judge" splits into "llm", "as", "a" and "judge". Taken whole, the
# short ones made every word starting with "a" look rooted in the group.


def test_a_short_word_in_the_run_admits_nothing():
    lexicon = boolean_enrichment.load_role_lexicon()
    assert lexicon.concept_for("annotation") is None
    base = score(JUDGE_RUN, lexicon).log_composite
    assert score(_judge_with(JUDGE_RUN, '"LLM judge" OR "annotation"'), lexicon).log_composite > (
        base + 0.01
    )
    for added in ("annotation", "accuracy"):
        proposal = f'("LLM-as-a-judge" OR "LLM judge" OR "{added}")'
        outcome, _ = _asking(JUDGE_RUN, {'("LLM-as-a-judge")': [proposal]})
        assert outcome.boolean == JUDGE_RUN, added
        assert outcome.kept == []


@pytest.mark.parametrize(
    ("added", "beside", "refused"),
    [
        ("annotation", '"LLM-as-a-judge"', True),
        ("accuracy", '"LLM-as-a-judge"', True),
        ("ab testing", '"LLM-as-a-judge"', True),
        ("LLM judges", '"LLM-as-a-judge"', False),  # rooted on "judge"
        ("rubrics", '"rubric"', False),  # mapped to the group's concept
        ("scoring rubric", '"rubric"', False),  # rooted on "rubric"
        # A run of short words only admits what maps to its concept.
        ("LLMOps", '"LLM"', True),
        ("LLMs", '"LLM"', False),
        # Four letters is long enough: "eval" and "rlhf" still give stems.
        ("evaluator", '"eval"', False),
        ("RLHF pipelines", '"RLHF"', False),
    ],
)
def test_what_a_run_s_words_admit(added, beside, refused):
    lexicon = boolean_enrichment.load_role_lexicon()
    authored = boolean_sidecar._candidate_terms(f"({beside})")
    proposed = boolean_sidecar._candidate_terms(f'({beside} OR "{added}")')
    assert boolean_sidecar._off_root(proposed, authored, lexicon) is refused


# --- an unmapped term that drops a qualifier is refused ---------------------
# A live replay widened a frontier-model run with the bare "frontier": one
# word of the authored term, so it passes the stem test, and far broader
# (an airline, a phone company) than the term it came from.

FRONTIER_RUN = f'("frontier model" OR "Scale AI") AND {CODE_GEN_GROUP}'
FRONTIER_GROUP = f'("frontier model") AND {CODE_GEN_GROUP}'


def _frontier_with(boolean: str, added: str) -> str:
    return boolean.replace('"frontier model"', f'"frontier model" OR {added}', 1)


@pytest.mark.parametrize("boolean", [FRONTIER_RUN, FRONTIER_GROUP])
def test_a_term_dropping_a_qualifier_refuses_the_candidate(boolean):
    lexicon = boolean_enrichment.load_role_lexicon()
    assert lexicon.concept_for("frontier") is None
    base = score(boolean, lexicon).log_composite
    trial = _frontier_with(boolean, '"frontier models" OR "frontier"')
    assert score(trial, lexicon).log_composite > base + 0.01
    proposal = '("frontier model" OR "frontier models" OR "frontier")'
    outcome, _ = _asking(boolean, {'("frontier model")': [proposal]})
    assert outcome.boolean == boolean
    assert outcome.kept == []


@pytest.mark.parametrize("boolean", [FRONTIER_RUN, FRONTIER_GROUP])
def test_the_candidate_without_the_bare_word_is_kept(boolean):
    proposal = '("frontier model" OR "frontier models")'
    outcome, _ = _asking(boolean, {'("frontier model")': [proposal]})
    assert outcome.boolean == _frontier_with(boolean, '"frontier models"')


@pytest.mark.parametrize(
    ("added", "beside", "refused"),
    [
        ("frontier", '"frontier model"', True),
        ("model", '"frontier model"', True),
        ("judge", '"LLM-as-a-judge"', True),
        # Inflected, the dropped qualifier is still dropped.
        ("frontiers", '"frontier model"', True),
        # The same words, or more of them, keep every qualifier.
        ("frontier modeling", '"frontier model"', False),
        ("frontier models", '"frontier model"', False),
        ("evaluation rubric", '"rubric"', False),
        ("scoring rubric", '"rubric"', False),
        ("LLM judges", '"LLM-as-a-judge"', False),
    ],
)
def test_what_counts_as_dropping_a_qualifier(added, beside, refused):
    lexicon = boolean_enrichment.load_role_lexicon()
    authored = boolean_sidecar._candidate_terms(f"({beside})")
    proposed = boolean_sidecar._candidate_terms(f'({beside} OR "{added}")')
    assert boolean_sidecar._off_root(proposed, authored, lexicon) is refused
