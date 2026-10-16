"""Scorer components, composition, and the §7 acceptance tests.

plans/boolean-scorer.md §6 fixes the components and the weights; §7 fixes what
the numbers must do on the 76 real strings, on Sam's exemplar, and on the ten
adversarial strings. Constant-shape assertions are marked ``structural``.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import pytest

from linkedin.boolean_score import load_lexicon
from linkedin.boolean_score.lexicon import LEXICON_DIR
from linkedin.boolean_score.score import (
    BLOAT_COEFFICIENTS,
    SHAPE_SCORES,
    PARSIMONY_DECAY,
    RELEVANCE_SCORES,
    UNMAPPED_RELEVANCE,
    NEUTRAL_SCORE,
    NEUTRAL_WHEN_INAPPLICABLE,
    STRUCTURE_WEIGHTS,
    WEIGHTS,
    ScoreReport,
    score,
)

FIXTURES = Path(__file__).parent / "fixtures" / "boolean_corpus"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
CORPUS = FIXTURES / "ml_runs_76.json"
ADVERSARIAL = FIXTURES / "adversarial.json"

EXEMPLAR = (
    '("evaluation" OR "evaluations" OR "eval" OR "evals") AND '
    '("agentic coding" OR "software engineering agents" OR "coding agents" OR "code gen")'
)


@pytest.fixture(scope="module")
def lexicon():
    return load_lexicon(LEXICON_DIR / "research-engineer-post-training.json")


def _strings(path: Path) -> list[dict]:
    return json.loads(path.read_text())["strings"]


def _corpus_entry(identifier: str) -> str:
    return next(
        entry["boolean"] for entry in _strings(CORPUS) if entry["id"] == identifier
    )


@pytest.fixture(scope="module")
def corpus_scores(lexicon) -> list[float]:
    return [score(entry["boolean"], lexicon).composite for entry in _strings(CORPUS)]


# --- §7.1 malformed strings score zero --------------------------------------


@pytest.mark.parametrize(
    ("boolean", "code"),
    [
        ("", "empty_input"),
        ('("alpha" OR "beta"', "unbalanced_parenthesis"),
        ('("alpha" OR "beta)', "unclosed_quote"),
        ('"alpha" && "beta"', "malformed_operator"),
        ('"alpha" || "beta"', "malformed_operator"),
        ('"alpha" & "beta"', "malformed_operator"),
        ('("alpha") AND engineer*', "wildcard_operator"),
        ('("alpha") AND ()', "empty_group"),
        ('("alpha") AND (OR)', "empty_group"),
        ('"alpha" OR', "dangling_operator"),
        ('("alpha") NOT ("beta" AND "gamma")', "not_group_contains_and"),
    ],
)
def test_illegal_strings_score_zero(lexicon, boolean, code):
    report = score(boolean, lexicon)
    assert report.legal is False
    assert report.composite == 0.0
    assert code in {error.code for error in report.errors}


def test_lowercase_operator_is_scored_not_rejected(lexicon):
    """Scored rather than refused. The composite can still reach zero: a
    lowercase operator turns an OR group into implicit AND gates, so the
    charge compounds with over-gating and repeated concepts."""
    report = score('("evaluation" or "evals")', lexicon)
    assert report.legal is True
    assert report.errors == ()
    assert "lowercase_operator" in {finding.code for finding in report.findings}
    assert report.bloat["lowercase_operator"] == pytest.approx(0.20)


# --- components -------------------------------------------------------------


def test_register_coverage_rewards_the_full_common_set(lexicon):
    full = score('("evaluation" OR "evaluations" OR "eval" OR "evals")', lexicon)
    thin = score('("evaluation")', lexicon)
    assert full.components["register_coverage"] == pytest.approx(1.0)
    assert thin.components["register_coverage"] == pytest.approx(0.25)


def test_occasional_forms_earn_a_bonus_and_are_never_required(lexicon):
    without = score('("instruction tuning")', lexicon)
    with_extra = score('("instruction tuning" OR "instruction-tuned")', lexicon)
    assert without.components["register_coverage"] == pytest.approx(1.0)
    assert with_extra.components["register_coverage"] == pytest.approx(1.0)


def test_variant_coverage_counts_twins_in_the_same_group(lexicon):
    paired = score('("post-training" OR "post training")', lexicon)
    lone = score('("post-training")', lexicon)
    assert paired.components["variant_coverage"] == pytest.approx(1.0)
    assert lone.components["variant_coverage"] == pytest.approx(0.0)


def test_variant_coverage_is_not_applicable_without_twinned_terms(lexicon):
    report = score('("RLHF" OR "GRPO")', lexicon)
    assert report.components["variant_coverage"] is None
    assert report.applicable["variant_coverage"] is False


def test_proper_nouns_are_exempt_from_variant_coverage(lexicon):
    report = score('("SWE-bench" OR "benchmark" OR "benchmarks")', lexicon)
    assert report.components["variant_coverage"] == pytest.approx(1.0)


def test_cohesion_is_full_for_one_concept_said_several_ways(lexicon):
    report = score('("reward model" OR "reward modeling")', lexicon)
    assert report.components["cohesion"] == pytest.approx(1.0)


def test_cohesion_is_full_for_one_family_of_related_concepts(lexicon):
    """§6, Sam's ruling: a group may broadly touch INTERRELATED concepts."""
    report = score(
        '("post-training" OR "instruction tuning" OR "distillation" OR "PEFT")', lexicon
    )
    assert report.components["cohesion"] == pytest.approx(1.0)


def test_cohesion_falls_for_a_grab_bag_of_unrelated_families(lexicon):
    report = score(
        '("research engineer" OR "vector database" OR "multi-agent" OR "transformer")',
        lexicon,
    )
    assert report.components["cohesion"] < 0.5


def test_a_shared_parent_family_is_most_of_the_way_to_cohesion(lexicon):
    """Post-training methods, evaluation, data curation and RL environments
    are separate families under one practice; a group spanning them is broad,
    not a grab-bag."""
    report = score(
        '("post-training" OR "RLHF" OR "evals" OR "evaluation" OR "synthetic data" '
        'OR "RL environment" OR "fine-tuning")',
        lexicon,
    )
    assert report.components["cohesion"] == pytest.approx(0.75)


def test_the_parent_family_keeps_a_broad_group_out_of_grab_bag(lexicon):
    report = score(_corpus_entry("research_engineer_frontier-11"), lexicon)
    assert report.bloat["grab_bag"] == pytest.approx(0.0)


def test_cohesion_needs_two_mapped_terms(lexicon):
    report = score('("RLHF" OR "Acme Robotics")', lexicon)
    assert report.components["cohesion"] is None


def test_parsimony_charges_rare_expansions(lexicon):
    plain = score('("post-training" OR "RLHF")', lexicon)
    padded = score(
        '("post-training" OR "RLHF" OR "reinforcement learning from human feedback")',
        lexicon,
    )
    assert plain.components["parsimony"] == pytest.approx(1.0)
    assert padded.bloat["rare_form"] == pytest.approx(0.04)
    assert padded.components["parsimony"] == pytest.approx(0.9418, abs=1e-4)


def test_parsimony_charges_a_gate_of_ubiquitous_verbs(lexicon):
    report = score(
        '("reward model" OR "reward modeling") AND ("led" OR "managed" OR "owned")',
        lexicon,
    )
    assert report.bloat["generic_and_gate"] == pytest.approx(0.35)
    assert report.components["parsimony"] == pytest.approx(0.5916, abs=1e-4)


def test_an_umbrella_group_is_charged_only_without_a_discriminator(lexicon):
    """The compiler's bare-generic discriminator rule: naming the field is
    free when something in the string narrows it, and a gate when nothing
    does."""
    bare = score(_corpus_entry("research_engineer_frontier-15"), lexicon)
    discriminated = score(_corpus_entry("prre_code_brazil-24"), lexicon)
    assert bare.bloat["umbrella_and_gate"] == pytest.approx(0.75)
    assert discriminated.bloat["umbrella_and_gate"] == pytest.approx(0.0)
    assert bare.composite <= 0.25


def test_a_verb_gate_is_charged_even_beside_a_discriminator(lexicon):
    report = score('("RLHF" OR "DPO") AND ("led" OR "managed")', lexicon)
    assert report.bloat["generic_and_gate"] == pytest.approx(0.35)


def test_an_excluded_term_does_not_discriminate(lexicon):
    """What a string rules OUT cannot narrow what it rules in, so a mapped
    term inside a NOT list leaves the umbrella charge standing."""
    report = score('("LLM" OR "machine learning") NOT ("RLHF")', lexicon)
    assert report.bloat["umbrella_and_gate"] == pytest.approx(0.75)


def test_length_alone_costs_nothing(lexicon):
    """Length is never inherently bad: twenty keywords that all capture the
    same concept without inviting noise are more than fine."""
    core = ["evaluation", "evaluations", "eval", "evals"]
    padding = [f"internal tooling variant {index}" for index in range(16)]
    lean = score("(" + " OR ".join(f'"{t}"' for t in core) + ")", lexicon)
    long = score("(" + " OR ".join(f'"{t}"' for t in core + padding) + ")", lexicon)
    assert len(long.groups[0].terms) == 20
    assert long.composite == pytest.approx(lean.composite, abs=0.02)


def test_a_long_group_that_invites_noise_still_pays(lexicon):
    core = ["evaluation", "evaluations", "eval", "evals"]
    noisy = core + [
        "machine learning",
        "reinforcement learning from human feedback",
        "reinforcement learning from AI feedback",
        "group relative policy optimization",
        "Magpie",
        "Nectar",
    ]
    clean = score("(" + " OR ".join(f'"{t}"' for t in core) + ")", lexicon)
    invited = score("(" + " OR ".join(f'"{t}"' for t in noisy) + ")", lexicon)
    assert invited.composite < clean.composite - 0.20


def test_a_long_clean_string_pays_no_length_charge(lexicon):
    terms = [f"internal tooling variant number {index}" for index in range(45)]
    boolean = "(" + " OR ".join(f'"{t}"' for t in terms) + ")"
    assert 1200 < len(boolean) < 2000
    report = score(boolean, lexicon)
    assert "length_over_cap" not in {finding.code for finding in report.findings}


def test_a_string_past_the_field_limit_is_flagged(lexicon):
    terms = [f"internal tooling variant number {index}" for index in range(60)]
    boolean = "(" + " OR ".join(f'"{t}"' for t in terms) + ")"
    assert len(boolean) > 2000
    report = score(boolean, lexicon)
    assert "length_over_cap" in {finding.code for finding in report.findings}
    assert report.structure_charges["length_over_cap"] == pytest.approx(0.30)


def test_parsimony_charges_a_long_not_list(lexicon):
    excluded = " OR ".join(f'"exclude {index}"' for index in range(8))
    report = score(f'("post-training") NOT ({excluded})', lexicon)
    assert report.components["parsimony"] == pytest.approx(0.5488, abs=1e-4)


def test_parsimony_charges_a_concept_repeated_across_and_gates(lexicon):
    report = score('("post-training") AND ("post-training" OR "distillation")', lexicon)
    assert report.bloat["repeated_concept"] == pytest.approx(0.25)
    assert report.components["parsimony"] == pytest.approx(0.6873, abs=1e-4)


def test_structure_charges_the_warning_findings(lexicon):
    report = score('("R&D spend" OR "RLHF")', lexicon)
    assert "noop_special_character" in {finding.code for finding in report.findings}
    assert report.components["structure"] == pytest.approx(0.80)


def test_structure_caps_unquoted_runs(lexicon):
    report = score('(reward model shaping OR alpha beta gamma OR delta epsilon zeta OR eta theta iota)', lexicon)
    assert report.structure_charges["unquoted_multiword_run"] == pytest.approx(0.30)


def test_structure_charges_deep_nesting(lexicon):
    report = score('("RLHF" OR ("GRPO" AND ("DPO" OR "SFT")))', lexicon)
    assert "nesting_depth_over_2" in {finding.code for finding in report.findings}


def test_structure_charges_extra_and_gates(lexicon):
    report = score(
        '("RLHF") AND ("GRPO") AND ("DPO") AND ("SFT") AND ("PEFT")', lexicon
    )
    codes = [finding.code for finding in report.findings]
    assert codes.count("and_groups_over_3") == 3
    assert report.bloat["over_gated"] == pytest.approx(1.20)
    assert report.components["structure"] == pytest.approx(0.55)


# --- the anchor rule (§6 parsimony) -----------------------------------------


def test_an_anchor_is_a_class_of_thing_not_a_list_of_instances(lexicon):
    """§6: an anchor is a set of GENERAL terms that directly implicate a class
    of thing being searched for. Named methods and artifacts are instances,
    and a group of them is a sweep, never an anchor."""
    anchored = score(
        '("RLHF" OR "DPO" OR "SFT") AND ("post-training" OR "instruction tuning")',
        lexicon,
    )
    adrift = score('("RLHF" OR "DPO" OR "SFT") AND ("SWE-bench" OR "tau-bench")', lexicon)
    assert anchored.anchor_present is True
    assert adrift.anchor_present is False
    assert adrift.shape == "two_groups_no_anchor"
    assert anchored.composite > adrift.composite


def test_a_group_of_named_methods_is_a_sweep_not_an_anchor(lexicon):
    assert score('("RLHF" OR "DPO" OR "SFT")', lexicon).anchor_present is False
    assert score('("OpenRLHF" OR "SWE-bench")', lexicon).anchor_present is False


def test_a_title_cluster_anchors(lexicon):
    entry = _corpus_entry("research_engineer_frontier-11")
    assert score(entry, lexicon).anchor_present is True


def test_two_class_concepts_in_one_family_anchor(lexicon):
    entry = _corpus_entry("prre_code_brazil-23")
    assert score(entry, lexicon).anchor_present is True


def test_a_class_group_spanning_two_families_does_not_anchor(lexicon):
    report = score('("machine learning" OR "LLM" OR "language models")', lexicon)
    assert report.anchor_present is False


def test_named_instances_of_the_same_class_still_anchor(lexicon):
    """A class group does not stop being a class group because it names a few
    of the things in the class: this implicates "works on LLMs, including
    these"."""
    report = score(
        '("LLM" OR "large language model" OR "Llama" OR "Mistral" OR "Qwen")', lexicon
    )
    assert report.anchor_present is True
    assert report.components["cohesion"] == pytest.approx(1.0)


def test_named_benchmarks_beside_the_word_benchmark_anchor(lexicon):
    report = score('("benchmark" OR "benchmarks" OR "SWE-bench" OR "HumanEval")', lexicon)
    assert report.anchor_present is True


def test_instances_without_a_class_term_do_not_anchor(lexicon):
    assert score('("Llama" OR "Mistral" OR "Qwen")', lexicon).anchor_present is False


def test_an_instance_outside_the_class_reach_breaks_the_anchor(lexicon):
    """SWE-bench now shares the post-training practice with the LLM class, so
    the case that breaks an anchor is an instance from outside it."""
    assert score('("LLM" OR "Pinecone")', lexicon).anchor_present is False
    assert score('("LLM" OR "SWE-bench")', lexicon).anchor_present is True


def test_the_model_family_gate_is_an_anchor_again(lexicon):
    """Held at 0.80 rather than 0.85 since the redundant-superstring charge
    arrived: "large language model" contains "language model", which costs
    this string 0.05 and moves it toward the raters' 0.83, not away."""
    report = score(_corpus_entry("prre_code_brazil-19"), lexicon)
    assert report.anchor_present is True
    assert report.composite > 0.80


def test_discovery_not_forgives_only_the_long_not_list(lexicon):
    """§6: subtracting the titles you know, to surface the ones you do not, is
    a technique rather than bloat — but only when the caller says so."""
    excluded = " OR ".join(f'"exclude {index}"' for index in range(8))
    boolean = f'("reward model" OR "reward modeling") NOT ({excluded})'
    charged = score(boolean, lexicon)
    forgiven = score(boolean, lexicon, discovery_not=True)

    assert forgiven.discovery_not is True
    assert charged.discovery_not is False
    assert forgiven.composite > charged.composite
    assert charged.bloat["excess_not_term"] == pytest.approx(0.40)
    assert forgiven.bloat["excess_not_term"] == pytest.approx(0.0)
    moved = {
        name
        for name in charged.components
        if charged.components[name] != forgiven.components[name]
    }
    assert moved == {"parsimony"}
    assert forgiven.components["parsimony"] > charged.components["parsimony"]


# --- named artifacts are instances the scorer can see -----------------------


def test_a_framework_sweep_is_mapped_and_coherent(lexicon):
    """A list of named training frameworks is a legitimate sweep of
    instances, so the lexicon has to be able to see it."""
    report = score(_corpus_entry("research_engineer_frontier-00"), lexicon)
    assert all(report.groups[0].concept_ids), report.groups[0].to_dict()
    assert len(report.groups[0].terms) == 9
    assert report.components["cohesion"] == pytest.approx(1.0)
    assert report.anchor_present is False


def test_a_framework_sweep_scores_at_least_the_corpus_median(lexicon, corpus_scores):
    report = score(_corpus_entry("research_engineer_frontier-00"), lexicon)
    assert report.composite >= statistics.median(corpus_scores)


def test_a_venue_group_is_a_single_family_instance_sweep(lexicon):
    report = score('("NeurIPS" OR "ICML" OR "ICLR" OR "COLM" OR "EMNLP")', lexicon)
    assert all(report.groups[0].concept_ids)
    assert report.components["cohesion"] == pytest.approx(1.0)
    assert len({lexicon.families_for(term.text) for term in report.groups[0].terms}) == 1


def test_a_thirty_name_dump_across_four_families_stays_below_the_median(
    lexicon, corpus_scores
):
    names = [
        "SWE-bench", "HumanEval", "MBPP", "GPQA", "MT-Bench", "AlpacaEval",
        "IFEval", "LiveCodeBench",
        "OpenRLHF", "verl", "TRLX", "torchtune", "DeepSpeed", "FSDP",
        "Colossal-AI", "LLaMA-Factory",
        "Tulu", "UltraFeedback", "UltraChat", "OpenHermes", "Magpie", "Nectar",
        "HelpSteer",
        "Llama", "Mistral", "Qwen", "Gemini", "Grok", "StarCoder", "Whisper",
    ]
    assert len(names) == 30
    boolean = "(" + " OR ".join(f'"{name}"' for name in names) + ")"
    report = score(boolean, lexicon)
    assert report.bloat["grab_bag"] == pytest.approx(0.30)
    assert report.composite < statistics.median(corpus_scores)


def test_the_llm_anchor_people_actually_write_is_an_anchor(lexicon):
    """The canonical anchor spans two families — "LLM" from the umbrella and
    "fine-tuning" from the methods — and the reach they share is what makes
    it one class: people who post-train LLMs."""
    report = score(
        '("LLM" OR "large language model" OR "fine-tuning" OR "instruction tuning")',
        lexicon,
    )
    assert report.anchor_present is True


@pytest.mark.parametrize(
    "identifier",
    ["prre_code_brazil-08", "prre_code_brazil-19", "research_engineer_frontier-17",
     "prre_code_brazil-25"],
)
def test_the_strings_whose_anchors_the_scorer_could_not_see(lexicon, identifier):
    assert score(_corpus_entry(identifier), lexicon).anchor_present is True


def test_a_venue_group_anchors_published_researchers(lexicon):
    """A conference list implicates a class of person even though every term
    in it is a named instance."""
    report = score('("NeurIPS" OR "ICML" OR "ICLR" OR "COLM")', lexicon)
    assert report.anchor_present is True
    assert score('("SWE-bench" OR "HumanEval")', lexicon).anchor_present is False


def test_bare_benchmark_is_not_an_umbrella_term(lexicon):
    assert lexicon.is_umbrella("benchmark") is False
    report = score('("benchmark" OR "benchmarks" OR "evaluation harness")', lexicon)
    assert report.bloat["umbrella_in_group"] == pytest.approx(0.0)


def test_one_defect_is_charged_once(lexicon):
    """A group that is both swallowed by an umbrella term and a grab-bag has
    one thing wrong with it, not two."""
    report = score(
        '("machine learning" OR "research engineer" OR "vector database" OR "SWE-bench")',
        lexicon,
    )
    assert (
        report.bloat["umbrella_in_group"] == pytest.approx(0.0)
        or report.bloat["grab_bag"] == pytest.approx(0.0)
    )


def test_relevance_ignores_a_group_the_lexicon_cannot_read(lexicon):
    report = score(_corpus_entry("prre_code_brazil-03"), lexicon)
    assert report.components["relevance"] == pytest.approx(1.0)
    assert 0.40 <= report.composite <= 0.70, report.to_dict()


# --- what the blind raters saw that the scorer did not ----------------------


def test_excluding_the_roles_own_class_is_charged(lexicon):
    """A NOT list that removes engineers and researchers removes the people
    the role is for; that is different in kind from excluding annotators."""
    report = score(_corpus_entry("prre_code_brazil-16"), lexicon)
    assert report.bloat["self_exclusion"] == pytest.approx(0.75)
    assert report.composite <= 0.10


def test_self_exclusion_ranks_below_the_other_broken_strings(lexicon):
    ranked = {
        name: score(_corpus_entry(name), lexicon).composite
        for name in (
            "prre_code_brazil-16",
            "research_engineer_frontier-15",
            "research_engineer_frontier-19",
        )
    }
    assert ranked["prre_code_brazil-16"] < ranked["research_engineer_frontier-15"]
    assert ranked["prre_code_brazil-16"] < ranked["research_engineer_frontier-19"]


def test_excluding_unmapped_vocabulary_is_not_self_exclusion(lexicon):
    report = score(
        '("post-training" OR "RLHF") NOT ("annotator" OR "AI trainer")', lexicon
    )
    assert report.bloat["self_exclusion"] == pytest.approx(0.0)


def test_an_umbrella_term_swallows_the_group_it_sits_in(lexicon):
    """A group is as broad as its broadest term."""
    tight = score('("post-training" OR "RLHF" OR "instruction tuning")', lexicon)
    swallowed = score(
        '("post-training" OR "RLHF" OR "instruction tuning" OR "machine learning")',
        lexicon,
    )
    assert tight.bloat["umbrella_in_group"] == pytest.approx(0.0)
    assert swallowed.bloat["umbrella_in_group"] == pytest.approx(0.30)
    assert swallowed.composite < tight.composite


@pytest.mark.parametrize(
    ("identifier", "low", "high"),
    [
        ("research_engineer_frontier-12", 0.25, 0.45),
        ("research_engineer_frontier-14", 0.45, 0.85),
        ("prre_code_brazil-09", 0.40, 0.60),
        ("prre_code_brazil-26", 0.40, 0.70),
    ],
)
def test_the_rater_bands_the_slice_aims_at(lexicon, identifier, low, high):
    report = score(_corpus_entry(identifier), lexicon)
    assert low <= report.composite <= high, report.to_dict()


def test_relevance_is_the_narrowest_group_the_string_reaches(lexicon):
    """An AND string only reaches the population its narrowest group allows."""
    core = score('("post-training" OR "RLHF")', lexicon)
    gated = score('("post-training" OR "RLHF") AND ("vector database")', lexicon)
    assert core.components["relevance"] == pytest.approx(RELEVANCE_SCORES["core"])
    assert gated.components["relevance"] == pytest.approx(
        RELEVANCE_SCORES["peripheral"]
    )


def test_a_group_the_lexicon_cannot_read_is_barely_relevant(lexicon):
    report = score('("Acme Robotics" OR "Globex Research")', lexicon)
    assert report.components["relevance"] == pytest.approx(UNMAPPED_RELEVANCE)


def test_register_weighs_a_concept_by_the_forms_it_owes(lexicon):
    """One form present out of four owed cannot be averaged away by three
    concepts that owe one form each."""
    report = score('("evals" OR "red teaming" OR "model evaluation")', lexicon)
    assert report.components["register_coverage"] == pytest.approx(
        (4 * 0.25 + 1 * 1.0 + 1 * 1.0) / 6
    )


def test_an_employer_gate_does_not_stack_with_the_anchor_charge(lexicon):
    report = score(_corpus_entry("research_engineer_frontier-19"), lexicon)
    assert report.bloat["employer_gate"] == pytest.approx(0.50)
    assert report.bloat["umbrella_and_gate"] == pytest.approx(0.0)


def test_over_gating_is_charged_harder(lexicon):
    report = score(_corpus_entry("prre_code_brazil-11"), lexicon)
    assert report.bloat["over_gated"] == pytest.approx(0.40)


def test_rare_forms_are_charged_more_gently(lexicon):
    report = score(_corpus_entry("prre_code_brazil-06"), lexicon)
    assert report.bloat["rare_form"] == pytest.approx(0.12)


def test_a_sweep_is_judged_on_how_much_of_its_family_it_reaches(lexicon):
    """A recall ceiling is not a defect. A precision string is judged on how
    exhaustively it covers the concept it targets, so two coding benchmarks
    is a poor sweep and fifteen is a thorough one."""
    thin = score('("SWE-bench" OR "Terminal-Bench")', lexicon)
    thorough = score(
        '("SWE-bench" OR "Terminal-Bench" OR "HumanEval" OR "MBPP" OR "GPQA" OR '
        '"MT-Bench" OR "AlpacaEval" OR "IFEval" OR "LiveCodeBench" OR "MLE-bench" OR '
        '"BrowseComp" OR "AgentBench" OR "Mind2Web" OR "OSWorld" OR "WebArena")',
        lexicon,
    )
    assert thin.components["register_coverage"] == pytest.approx(2 / 15)
    assert thorough.components["register_coverage"] == pytest.approx(1.0)
    assert thorough.composite > thin.composite + 0.20


def test_a_long_sweep_is_not_charged_for_its_length(lexicon):
    names = [
        "SWE-bench", "Terminal-Bench", "HumanEval", "MBPP", "GPQA", "MT-Bench",
        "AlpacaEval", "IFEval", "LiveCodeBench", "MLE-bench", "BrowseComp",
        "AgentBench", "Mind2Web", "OSWorld", "WebArena", "VisualWebArena",
        "BrowserGym", "ALFWorld", "AndroidWorld", "Arena-Hard", "RewardBench",
        "WildBench", "JudgeBench", "EvalPlus", "CodeContests",
    ]
    assert len(names) == 25
    report = score("(" + " OR ".join(f'"{n}"' for n in names) + ")", lexicon)
    assert "oversize_group" not in report.bloat
    assert report.components["register_coverage"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "identifier", ["research_engineer_frontier-04", "research_engineer_frontier-00"]
)
def test_a_precision_sweep_is_not_charged_for_lacking_an_anchor(lexicon, identifier):
    report = score(_corpus_entry(identifier), lexicon)
    assert report.shape in {"lone_sweep", "anchor_and_sweep"}


# --- §7.4 unmapped vocabulary is ignored, never punished ---------------------


def test_employer_only_string_is_not_punished_on_coverage(lexicon):
    report = score(
        '("Acme Robotics" OR "Globex Research" OR "Initech Labs" OR "Umbrella AI Group")',
        lexicon,
    )
    assert report.legal is True
    assert report.components["register_coverage"] is None
    assert report.components["variant_coverage"] is None
    assert report.components["cohesion"] is None
    assert report.applicable["structure"] is True
    assert report.applicable["parsimony"] is True
    assert report.bloat["employer_gate"] == pytest.approx(0.0)


def test_unmapped_terms_do_not_dilute_cohesion(lexicon):
    report = score('("reward model" OR "reward modeling" OR "Acme Robotics")', lexicon)
    assert report.components["cohesion"] == pytest.approx(1.0)


# --- §7.2 Sam's exemplar -----------------------------------------------------


def test_exemplar_scores_at_least_point_eight_five(lexicon):
    report = score(EXEMPLAR, lexicon)
    assert report.legal is True
    assert report.composite >= 0.85, report.to_dict()


def test_exemplar_ranks_in_the_top_decile_of_the_corpus(lexicon, corpus_scores):
    exemplar = score(EXEMPLAR, lexicon).composite
    ranked = sorted(corpus_scores + [exemplar], reverse=True)
    rank = ranked.index(exemplar) + 1
    assert rank <= max(1, round(0.10 * len(ranked))), (rank, len(ranked), exemplar)


# --- §7.3 the adversarial ten -----------------------------------------------


def test_every_adversarial_string_scores_below_the_corpus_median(lexicon, corpus_scores):
    median = statistics.median(corpus_scores)
    entries = _strings(ADVERSARIAL)
    assert len(entries) == 10
    worse = {
        entry["id"]: score(entry["boolean"], lexicon).composite
        for entry in entries
    }
    assert all(value < median for value in worse.values()), (median, worse)


def test_adversarial_strings_are_all_legal(lexicon):
    for entry in _strings(ADVERSARIAL):
        report = score(entry["boolean"], lexicon)
        assert report.legal is True, (entry["id"], [e.code for e in report.errors])


# --- §7.6 determinism --------------------------------------------------------


def test_scoring_is_deterministic(lexicon):
    for entry in _strings(CORPUS):
        first = score(entry["boolean"], lexicon)
        second = score(entry["boolean"], lexicon)
        assert first == second
        assert json.dumps(first.to_dict(), sort_keys=False) == json.dumps(
            second.to_dict(), sort_keys=False
        )


def test_report_carries_everything_a_reader_needs(lexicon):
    report = score(EXEMPLAR, lexicon)
    payload = report.to_dict()
    assert payload["composite"] == report.composite
    extra = {"legality", "parsimony", "structure"}
    assert set(payload["components"]) == set(WEIGHTS) | extra
    assert set(payload["applicable"]) == set(WEIGHTS) | extra
    assert payload["weights"] == dict(WEIGHTS)
    assert payload["role"] == "research-engineer-post-training"
    assert len(payload["groups"]) == 2
    assert payload["groups"][0]["concepts"]["evaluation"] == "evaluation"
    assert isinstance(report, ScoreReport)


def test_structure_charges_multiply_rather_than_average(lexicon):
    """A malformed string should not be able to average its defect away
    against good coverage, so the structure charges join the multiplier."""
    clean = score('("post-training" OR "RLHF" OR "instruction tuning")', lexicon)
    lowercase = score('("post-training" or "RLHF" or "instruction tuning")', lexicon)
    assert "structure" not in WEIGHTS
    assert lowercase.bloat["lowercase_operator"] == pytest.approx(0.20)
    assert lowercase.components["parsimony"] < clean.components["parsimony"]
    assert lowercase.composite < clean.composite


def test_the_structure_component_is_still_reported(lexicon):
    report = score('("post-training" or "RLHF")', lexicon)
    assert report.components["structure"] is not None
    assert report.applicable["structure"] is True


DELETION_LEAK_RATCHET = 6


def test_deleting_a_group_does_not_pay(lexicon):
    """A policy trained on this reward must not learn to delete. Removing a
    positive group should only raise the composite when the group removed was
    carrying a charge of its own.

    Measured only where neither group carries a charge of its own and the
    string is not over-gated, so it counts the leak and not a legitimate
    repair: 6 of the 18 qualifying strings still pay for deletion, down from
    13 before the shape multiplier. What is left is that cohesion is a MEAN
    over groups and relevance a MIN over them. This pins the count. See §9.
    """
    import make_boolean_pairs as pairs_tool

    disqualifying = ("umbrella_in_group", "noisy_term", "rare_form",
                     "self_exclusion", "grab_bag", "sentence_fragment",
                     "over_gated")
    offenders = []
    for entry in json.loads((FIXTURES / "rater_targets.json").read_text())["strings"]:
        boolean = entry["boolean"]
        indices = pairs_tool._positive_group_indices(boolean)
        if len(indices) < 2:
            continue
        full = score(boolean, lexicon)
        for index in indices:
            start, end, _terms = pairs_tool._groups(boolean)[index]
            trimmed = (boolean[:start] + boolean[end:]).strip()
            trimmed = trimmed.replace("  ", " ").strip()
            for token in (" AND", "AND ", " NOT", "NOT "):
                if trimmed.startswith(token.strip() + " "):
                    trimmed = trimmed[len(token.strip()) + 1:]
                if trimmed.endswith(" " + token.strip()):
                    trimmed = trimmed[: -(len(token.strip()) + 1)]
            if not trimmed or not score(trimmed, lexicon).legal:
                continue
            cut = score(trimmed, lexicon)
            if any(full.bloat.get(key, 0.0) for key in disqualifying):
                continue
            if cut.composite - full.composite > 0.02:
                offenders.append((entry["id"], index, round(cut.composite - full.composite, 3)))
    assert len(offenders) <= DELETION_LEAK_RATCHET, sorted(
        offenders, key=lambda row: -row[2]
    )


# --- shape: which of the two legitimate builds this is ----------------------


@pytest.mark.structural
def test_the_shapes_are_the_declared_ones():
    assert SHAPE_SCORES == {
        "anchor_and_sweep": 1.00,
        "two_groups_no_anchor": 0.85,
        "anchored_by_filter": 1.00,
        "lone_sweep": 0.90,
        "lone_anchor": 0.75,
    }
    assert "missing_anchor" not in BLOAT_COEFFICIENTS
    assert "redundant_superstring" not in BLOAT_COEFFICIENTS


def test_an_anchor_with_nothing_to_pin_it_is_a_wide_net(lexicon):
    """A lone anchor group is neither of Sam's two shapes."""
    whole = score(EXEMPLAR, lexicon)
    anchor_only = score('("evaluation" OR "evaluations" OR "eval" OR "evals")', lexicon)
    assert whole.shape == "anchor_and_sweep"
    assert anchor_only.shape == "lone_anchor"
    assert anchor_only.composite < whole.composite


def test_deleting_the_sweep_from_a_two_group_string_costs(lexicon):
    whole = score(_corpus_entry("prre_code_brazil-19"), lexicon)
    without = score(
        '("LLM" OR "LLMs" OR "large language model" OR "language model" OR '
        '"Llama" OR "Mistral" OR "Qwen")',
        lexicon,
    )
    assert without.composite < whole.composite


def test_a_single_exhaustive_sweep_is_a_legitimate_shape(lexicon):
    report = score(_corpus_entry("research_engineer_frontier-00"), lexicon)
    assert report.shape == "lone_sweep"
    # The brief's band was 0.70-0.85; the string lands at 0.55 because its
    # sweep reaches 9 of the 15 the completeness denominator asks for and its
    # relevance is adjacent, not core. Reported in §9, not tuned.
    assert 0.50 <= report.composite <= 0.85, report.to_dict()


def test_a_declared_facet_restores_the_shape(lexicon):
    boolean = '("evaluation" OR "evaluations" OR "eval" OR "evals")'
    lone = score(boolean, lexicon)
    filtered = score(boolean, lexicon, anchored_by_filter=True)
    assert filtered.shape == "anchored_by_filter"
    assert filtered.composite > lone.composite


def test_preference_is_readable_in_log_space(lexicon):
    import math

    report = score(EXEMPLAR, lexicon)
    assert report.log_composite == pytest.approx(math.log(report.composite))
    broken = score('("alpha" && "beta")', lexicon)
    assert broken.log_composite == pytest.approx(math.log(1e-6))


# --- parsimony as the multiplier -------------------------------------------


def test_parsimony_decays_rather_than_clamping(lexicon):
    """A clamped multiplier makes every badly broken string identical, which
    gives a training policy no gradient among bad completions."""
    reports = [score(entry["boolean"], lexicon) for entry in _strings(ADVERSARIAL)]
    assert all(report.composite > 0.0 for report in reports)
    charges = [sum(report.bloat.values()) for report in reports]
    parsimonies = [report.components["parsimony"] for report in reports]
    ordered = sorted(zip(charges, parsimonies))
    assert all(
        later <= earlier + 1e-9
        for (_c1, earlier), (_c2, later) in zip(ordered, ordered[1:])
    )
    assert len({round(value, 6) for value in parsimonies}) == len(parsimonies)


def test_breadth_is_never_free(lexicon):
    """Every form of every concept in one group: perfect on every coverage
    component, and worth nothing, because parsimony multiplies."""
    dump = next(
        entry for entry in _strings(ADVERSARIAL) if "every form" in entry["defect"]
    )
    report = score(dump["boolean"], lexicon)
    assert report.components["register_coverage"] > 0.95
    assert report.composite < 0.10


def test_the_weak_reference_string_lands_where_it_should(lexicon):
    report = score('("RLHF" OR "DPO" OR "SFT") AND ("led" OR "managed")', lexicon)
    # Sam's band was 0.35-0.55 under the clamped multiplier; the prescribed
    # exp(-1.5 x bloat) decay is harsher below bloat 1.0 and moves it to 0.30.
    assert 0.25 <= report.composite <= 0.55, report.to_dict()


def test_inapplicable_coverage_is_neutral_not_free(lexicon):
    employers = score(
        '("Acme Robotics" OR "Globex Research" OR "Initech Labs" OR "Umbrella AI Group")',
        lexicon,
    )
    exemplar = score(EXEMPLAR, lexicon)
    assert employers.applicable["register_coverage"] is False
    assert employers.applicable["variant_coverage"] is False
    assert employers.composite <= 0.75
    assert employers.composite < exemplar.composite


def test_a_grab_bag_is_charged_even_as_a_single_group(lexicon):
    report = score(
        '("RLHF" OR "research engineer" OR "semantic search" OR "multi-agent")', lexicon
    )
    assert report.components["register_coverage"] == pytest.approx(1.0)
    assert report.bloat["grab_bag"] == pytest.approx(0.30)
    assert report.composite < 0.70


def test_a_group_the_lexicon_cannot_read_is_a_sweep_not_an_anchor(lexicon):
    """A group with no mapped term may well be the anchor; the lexicon simply
    cannot see it, so it is not charged for."""
    report = score(
        '("SWE-bench" OR "HumanEval") AND ("Acme Robotics" OR "Globex Research")',
        lexicon,
    )
    assert report.anchor_present is False


def test_variant_coverage_does_not_recharge_a_declared_form(lexicon):
    """A plural that is a declared form of its own concept is the register
    axis's business; charging it here too charges one absence twice."""
    report = score(EXEMPLAR, lexicon)
    assert report.components["variant_coverage"] == pytest.approx(1.0)


def test_variant_coverage_still_charges_an_undeclared_twin(lexicon):
    lone = score('("reward model")', lexicon)
    paired = score('("reward model" OR "reward models")', lexicon)
    assert lone.components["variant_coverage"] == pytest.approx(0.0)
    assert paired.components["variant_coverage"] == pytest.approx(1.0)


def test_an_ampersand_phrase_with_its_and_twin_is_not_charged(lexicon):
    """Mirrors the compiler, down to which ampersands have a twin at all: the
    helper only rewrites a free-standing `&`, so "R&D" has none and is charged
    whatever else the group holds."""
    charged = score('("R & D spend" OR "RLHF")', lexicon)
    exempt = score('("R & D spend" OR "R and D spend" OR "RLHF")', lexicon)
    welded = score('("R&D spend" OR "R and D spend" OR "RLHF")', lexicon)
    assert "noop_special_character" in {f.code for f in charged.findings}
    assert "noop_special_character" not in {f.code for f in exempt.findings}
    assert "noop_special_character" in {f.code for f in welded.findings}


def test_a_concept_may_belong_to_several_families(lexicon):
    """GRPO sits in post-training methods and in RL environments, so it hangs
    together with either neighbourhood."""
    with_methods = score('("GRPO" OR "instruction tuning" OR "distillation")', lexicon)
    with_environments = score(
        '("GRPO" OR "RL environment" OR "RL environments")', lexicon
    )
    assert with_methods.components["cohesion"] == pytest.approx(1.0)
    assert with_environments.components["cohesion"] == pytest.approx(1.0)


# --- declared constants ------------------------------------------------------


@pytest.mark.structural
def test_component_weights_are_the_declared_ones():
    """Parsimony left the weighted mean and became the multiplier, so breadth
    is never free: it scales whatever the coverage components earned."""
    assert WEIGHTS == {
        "register_coverage": 0.30,
        "variant_coverage": 0.15,
        "cohesion": 0.10,
        "relevance": 0.05,
    }
    assert "parsimony" not in WEIGHTS
    assert "structure" not in WEIGHTS
    assert sum(WEIGHTS.values()) == pytest.approx(0.60)


@pytest.mark.structural
def test_inapplicable_coverage_scores_the_population_mean():
    """A neutral above the population mean is a reward for saying less."""
    assert NEUTRAL_WHEN_INAPPLICABLE == frozenset(
        {"register_coverage", "variant_coverage"}
    )
    assert NEUTRAL_SCORE == pytest.approx(0.20)
    assert PARSIMONY_DECAY == pytest.approx(1.5)


@pytest.mark.structural
def test_the_variant_neutral_is_the_corpus_mean(lexicon):
    assert 0.25 <= lexicon.neutral_variant <= 0.45


@pytest.mark.structural
def test_bloat_coefficients_are_the_declared_ones():
    assert BLOAT_COEFFICIENTS == {
        "rare_form": 0.04,
        "repeated_concept": 0.25,
        "generic_and_gate": 0.35,
        "excess_not_term": 0.10,
        "grab_bag": 0.30,
        "umbrella_and_gate": 0.75,
        "employer_gate": 0.50,
        "employer_facet_gate": 0.10,
        "over_gated": 0.40,
        "noisy_term": 0.10,
        "self_exclusion": 0.30,
        "umbrella_in_group": 0.30,
    }


@pytest.mark.structural
def test_structure_weights_are_the_declared_ones():
    assert STRUCTURE_WEIGHTS == {
        "unquoted_multiword_run": 0.10,
        "lowercase_operator": 0.20,
        "noop_special_character": 0.20,
        "comma_numeral": 0.10,
        "mid_word_stem": 0.20,
        "nesting_depth_over_2": 0.20,
        "and_groups_over_3": 0.15,
        "sentence_fragment": 0.15,
        "length_over_cap": 0.30,
    }
