"""The enumeration channel: what it must produce, and what it must never do.

Measured 2026-07-27 on brief 3000000001. The strategy step produced 72 search
strings whose entire benchmark vocabulary was nine head-of-distribution names.
The same model, asked directly with the same capability areas, returned 292
named artifacts including 98 benchmarks — so the gap was satisficing, not
missing knowledge, and the fix is a separate call with a separate job.

Two properties here are load-bearing and easy to break with a well-intentioned
edit, so they are locked explicitly:

  * Rare-on-a-profile artifacts SURVIVE. They are the marginal candidates the
    channel exists to reach. A future "filter out the rare ones" tidy-up would
    invert the entire purpose while still passing a naive smoke test.
  * Every emitted Boolean parses, and glossed names are SPLIT rather than
    dropped. Refusing parentheses outright (an earlier guard did) silently lost
    "Model Context Protocol (MCP)" — an acronym this very JD names. Parens do
    not actually break the compiler; only double quotes do.
"""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

from linkedin.boolean_compiler import _parenthetical_groups, _quoted_terms
from shared.llm_usage import llm_usage_session
from shared.vocabulary_enumeration import (
    domain_context_from_brief,
    EnumeratedArtifact,
    artifacts_to_kit_strings,
    default_research_call,
    enumerate_domain_vocabulary,
    merge_artifacts,
)


class _Area:
    def __init__(self, name, description, key_terms):
        self.name = name
        self.description = description
        self.key_terms = key_terms


class _Brief:
    id = "test-brief"
    capability_areas = [
        _Area("Post-training for code", "Fine-tunes models for SWE capability",
              ["post-training", "RLVR"]),
    ]
    instructions = ["Domain anchor: coding agents and software engineering."]


def _payload(*rows: dict) -> str:
    return json.dumps({"artifacts": list(rows)})


def _artifact(name, family="benchmarks", certainty="certain", on_profile="common"):
    return {
        "name": name, "family": family, "year": 2024,
        "certainty": certainty, "on_profile": on_profile,
    }


# --- the reason this exists ------------------------------------------------


def test_enumeration_reaches_the_strategy_channel_as_kit_vocabulary() -> None:
    kit = enumerate_domain_vocabulary(
        _Brief(),
        llm_call=lambda s, u: _payload(
            _artifact("SWE-bench"),
            _artifact("SWE-bench Verified"),
            _artifact("SWE-smith", on_profile="occasional"),
        ),
    )

    assert kit, "enumeration produced no vocabulary"
    blob = " ".join(k.boolean for k in kit)
    assert '"SWE-bench Verified"' in blob
    assert '"SWE-smith"' in blob
    # KitString is the existing vocabulary shape form_strategy already renders.
    assert all(k.string_type == "Precision" for k in kit)


def test_rare_on_profile_artifacts_are_kept_not_filtered_out() -> None:
    # THE load-bearing product property. The register rating is a LABEL that
    # tells the strategist which names are safe anchors and which are precision
    # probes — it is not a drop filter. The rare names are the marginal
    # candidates standard sourcing misses, which is the whole point.
    kit = enumerate_domain_vocabulary(
        _Brief(),
        llm_call=lambda s, u: _payload(
            _artifact("SWE-bench", on_profile="common"),
            _artifact("SWT-Bench", on_profile="rare"),
            _artifact("BountyBench", on_profile="rare"),
        ),
    )

    blob = " ".join(k.boolean for k in kit)
    assert '"SWT-Bench"' in blob
    assert '"BountyBench"' in blob
    # ...and the register survives as a visible label, so the strategist can
    # sequence with it rather than being handed an undifferentiated bag.
    assert any("Rare on profiles" in k.subblock for k in kit)
    assert any("Common on profiles" in k.subblock for k in kit)


def test_unsure_artifacts_are_dropped_as_a_precision_control() -> None:
    # Distinct from register: this is the model's own did-I-invent-this flag.
    kit = enumerate_domain_vocabulary(
        _Brief(),
        llm_call=lambda s, u: _payload(
            _artifact("SWE-bench", certainty="certain"),
            _artifact("TotallyMadeUpBench", certainty="unsure"),
        ),
    )

    blob = " ".join(k.boolean for k in kit)
    assert '"SWE-bench"' in blob
    assert "TotallyMadeUpBench" not in blob


# --- boolean safety --------------------------------------------------------


@pytest.mark.parametrize("name", ['quote"inside', 'a"b"c', "x", "("])
def test_names_that_would_break_the_boolean_parser_are_refused(name: str) -> None:
    # Only DOUBLE QUOTES actually break it — verified against the production
    # compiler, which lints '("SWE-bench" OR "quote"inside")' as
    # unbalanced_quote + unbalanced_parenthesis. Single-character names are
    # punctuation noise. Parentheses are NOT refused; see the gloss tests below.
    kit = enumerate_domain_vocabulary(
        _Brief(),
        llm_call=lambda s, u: _payload(_artifact("SWE-bench"), _artifact(name)),
    )

    assert kit
    # Assert on the emitted TERMS, not on a substring of the group syntax —
    # a name of "(" is trivially a substring of every rendered group.
    terms = {t for ks in kit for t in _quoted_terms(ks.boolean)}
    assert "SWE-bench" in terms
    assert name not in terms


def test_a_glossed_name_becomes_both_of_its_real_surface_forms() -> None:
    # "Model Context Protocol (MCP)" matches nobody as written — a profile says
    # one or the other. Both strings are literally present in the model's
    # output, so splitting is parsing, not invention. Dropping these instead
    # (an earlier guard did) silently lost MCP, which THIS JD names directly.
    kit = enumerate_domain_vocabulary(
        _Brief(),
        llm_call=lambda s, u: _payload(
            _artifact("Model Context Protocol (MCP)"),
            _artifact("Berkeley Function-Calling Leaderboard (BFCL)"),
        ),
    )

    terms = {t for ks in kit for t in _quoted_terms(ks.boolean)}
    assert "MCP" in terms
    assert "Model Context Protocol" in terms
    assert "BFCL" in terms
    assert "Model Context Protocol (MCP)" not in terms


def test_a_stylized_paren_name_is_kept_whole_not_shredded() -> None:
    # No whitespace before the paren means it is part of the name, not a gloss.
    # Splitting would yield "BigO" and "Bench", neither of which is the tool.
    kit = enumerate_domain_vocabulary(
        _Brief(), llm_call=lambda s, u: _payload(_artifact("BigO(Bench)"))
    )

    terms = {t for ks in kit for t in _quoted_terms(ks.boolean)}
    assert "BigO(Bench)" in terms
    assert "BigO" not in terms


def test_every_emitted_group_parses_as_exactly_one_or_group() -> None:
    rows = [
        _artifact(f"Bench{i}", on_profile=("common", "occasional", "rare")[i % 3])
        for i in range(40)
    ]
    rows.append(_artifact("BigO(Bench)"))
    kit = enumerate_domain_vocabulary(_Brief(), llm_call=lambda s, u: _payload(*rows))

    assert kit
    for ks in kit:
        groups = _parenthetical_groups(ks.boolean)
        assert len(groups) == 1, ks.boolean
        terms = _quoted_terms(groups[0][2])
        assert len(terms) == ks.boolean.count(" OR ") + 1
        assert terms, ks.boolean


def test_groups_are_capped_so_one_family_cannot_become_one_giant_group() -> None:
    rows = [_artifact(f"Bench{i}") for i in range(35)]
    kit = artifacts_to_kit_strings(
        merge_artifacts(
            [
                EnumeratedArtifact(r["name"], r["family"], r["certainty"], r["on_profile"])
                for r in rows
            ]
        )
    )

    assert len(kit) >= 4
    for ks in kit:
        assert ks.boolean.count(" OR ") + 1 <= 10


# --- fail-soft -------------------------------------------------------------


def test_a_raising_provider_yields_no_vocabulary_not_an_exception() -> None:
    def boom(system, user):
        raise RuntimeError("provider down")

    assert enumerate_domain_vocabulary(_Brief(), llm_call=boom) == []


@pytest.mark.parametrize(
    "response",
    ["not json at all", "", "{}", '{"artifacts": "wrong type"}', "[]", None],
)
def test_a_malformed_response_yields_no_vocabulary(response) -> None:
    assert enumerate_domain_vocabulary(_Brief(), llm_call=lambda s, u: response) == []


def test_a_fenced_json_response_is_still_parsed() -> None:
    # Fable wrapped strategy output in ```json fences on the 2026-07-27 A/B.
    fenced = "```json\n" + _payload(_artifact("SWE-bench")) + "\n```"

    kit = enumerate_domain_vocabulary(_Brief(), llm_call=lambda s, u: fenced)

    assert '"SWE-bench"' in " ".join(k.boolean for k in kit)


def test_a_failing_research_pass_does_not_lose_the_parametric_batch() -> None:
    def boom(system, user):
        raise RuntimeError("perplexity 500")

    kit = enumerate_domain_vocabulary(
        _Brief(),
        llm_call=lambda s, u: _payload(_artifact("SWE-bench")),
        research_call=boom,
    )

    assert '"SWE-bench"' in " ".join(k.boolean for k in kit)


def test_default_research_call_records_perplexity_usage(tmp_path, monkeypatch) -> None:
    response = SimpleNamespace(
        model="",
        output_text=_payload(_artifact("SWE-bench")),
        usage=SimpleNamespace(input_tokens=120, output_tokens=30),
    )
    fake_openai = SimpleNamespace(
        OpenAI=lambda **kwargs: SimpleNamespace(
            responses=SimpleNamespace(create=lambda **kwargs: response)
        )
    )
    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    log_path = tmp_path / "token-cost-log.jsonl"

    with llm_usage_session(log_path):
        result = default_research_call("system", "user")

    record = json.loads(log_path.read_text())
    assert result == response.output_text
    assert record["provider"] == "perplexity"
    assert record["model"] == "perplexity-response-api"
    assert record["input_tokens"] == 120
    assert record["output_tokens"] == 30
    assert record["stage"] == "vocabulary_enumeration"
    assert record["estimated_cost_usd"] is None


def test_exclusions_come_from_non_fit_patterns_not_just_instructions() -> None:
    # Found by verifying the live seam: a preflight-GENERATED brief has no
    # `instructions` key at all — the seed's operator instructions are absorbed
    # during generation. Reading only that field handed the enumeration an
    # EMPTY exclusion set (measured 0 chars), leaving it free to return the
    # robotics/AV RL artifacts this role explicitly does not want. The real
    # exclusions survive in non_fit_patterns (measured 637 chars).
    class Pattern:
        def __init__(self, description):
            self.description = description

    class GeneratedBrief:
        id = "generated"
        capability_areas = _Brief.capability_areas
        instructions = []  # exactly what preflight produces
        non_fit_patterns = [
            Pattern("Builds RL systems for robotics, autonomous vehicles, games")
        ]

    _context, scope = domain_context_from_brief(GeneratedBrief())

    assert "robotics" in scope
    assert scope.startswith("OUT OF SCOPE:")

    seen: list[str] = []
    enumerate_domain_vocabulary(
        GeneratedBrief(),
        llm_call=lambda s, u: seen.append(u) or _payload(_artifact("SWE-bench")),
    )
    assert "robotics" in seen[0], "exclusions never reached the prompt"


def test_seed_instructions_and_non_fit_patterns_both_land(brief_cls=_Brief) -> None:
    class Both:
        id = "seed"
        capability_areas = _Brief.capability_areas
        instructions = ["Domain anchor: this role is coding agents."]
        non_fit_patterns = [type("P", (), {"description": "robotics RL"})()]

    _context, scope = domain_context_from_brief(Both())

    assert "Domain anchor" in scope
    assert "OUT OF SCOPE: robotics RL" in scope


def test_a_brief_without_capability_areas_makes_no_call_at_all() -> None:
    called = []

    class Empty:
        id = "x"
        capability_areas = []
        instructions = []

    assert enumerate_domain_vocabulary(Empty(), llm_call=lambda s, u: called.append(1)) == []
    assert not called, "spent a strategy-tier call on a brief with no domain"


# --- union -----------------------------------------------------------------


def test_the_two_providers_union_and_dedupe_case_insensitively() -> None:
    # Measured: Fable 292, Perplexity 222, overlap only 106 — complementary
    # enough to be worth the second call, which is why dedupe matters.
    kit = enumerate_domain_vocabulary(
        _Brief(),
        llm_call=lambda s, u: _payload(_artifact("SWE-bench"), _artifact("SWT-Bench")),
        research_call=lambda s, u: _payload(
            _artifact("swe-bench"), _artifact("CursorBench")
        ),
    )

    blob = " ".join(k.boolean for k in kit)
    assert '"CursorBench"' in blob, "research-only artifact was lost"
    assert '"SWT-Bench"' in blob, "parametric-only artifact was lost"
    terms = [t for ks in kit for t in _quoted_terms(ks.boolean)]
    assert len([t for t in terms if t.lower() == "swe-bench"]) == 1


def test_the_receipt_records_what_was_enumerated(tmp_path) -> None:
    enumerate_domain_vocabulary(
        _Brief(),
        llm_call=lambda s, u: _payload(
            _artifact("SWE-bench"), _artifact("SWT-Bench", on_profile="rare")
        ),
        artifact_dir=tmp_path,
    )

    receipt = json.loads((tmp_path / "vocabulary_enumeration.json").read_text())
    assert receipt["artifact_count"] == 2
    assert receipt["by_register"]["rare"] == 1
    assert {a["name"] for a in receipt["artifacts"]} == {"SWE-bench", "SWT-Bench"}
