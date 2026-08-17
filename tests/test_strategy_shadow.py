"""Shadow strategist seam (shared/strategy_shadow.py) — item 19.

Pins the four properties the experiment depends on:

1. flag off (or shadow_dir None) -> dispatch is a TOTAL no-op: no executor
   is ever spun up, the shadow client is never called, no file appears.
2. flag on -> dispatch returns while the shadow call is still in flight on
   the single background worker, and ``drain_strategy_shadows`` flushes the
   queue so the artifact exists afterward with the right stage / model /
   metrics. The blocked-shadow stub is an Event, not a sleep: dispatch
   returning while ``release_shadow`` is still unset PROVES non-blocking
   dispatch (a synchronous regression would sit inside the stub until its
   bounded wait expires — deterministic red, no timing flake). Everything,
   including drain, happens INSIDE the ``with patch`` blocks so the worker
   can never touch a real client. (Pattern copied from
   tests/test_judger_shadow_async.py.)
3. ``plan_metrics`` golden values on a known 3-boolean / 2-skeleton plan.
4. an exception raised through ``opus_llm`` (how a Fable refusal surfaces:
   RuntimeError on a non-end_turn stop_reason) lands in the artifact as
   ``shadow_error`` and NEVER propagates to the dispatching caller.

NOTE: the executor-never-spun assertion in the first test relies on this
module being the only one that imports shared.strategy_shadow and on
pytest's definition-order execution — keep that test first in this file.

Run with: PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/test_strategy_shadow.py -q
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from unittest.mock import patch

import pytest


_STRATEGY_RESPONSE = json.dumps(
    {
        "strategy_rationale": "test plan",
        "generated_strings": [
            {
                "name": "ml core",
                "boolean": '("machine learning" OR "deep learning") AND (PyTorch OR JAX)',
            },
            {
                "name": "rl core",
                "boolean": '("reinforcement learning" OR RLHF) AND (TensorFlow OR Keras)',
            },
            {
                "name": "title probe",
                "boolean": '"staff engineer" NOT recruiter',
            },
        ],
    }
)

# Contains "machine learning" and "deep learning" but NOT "reinforcement
# learning" / "staff engineer" -> novelty 2/4 = 0.5 for the plan above.
_REFERENCE_SYSTEM = "Brief mentions machine learning and deep learning platforms."
_REFERENCE_USER_WITH_STAFF = "User prompt asks for staff engineer candidates."


def _artifacts(shadow_dir: Path) -> list[Path]:
    if not shadow_dir.exists():
        return []
    return sorted(shadow_dir.glob("shadow-*.json"))


def test_flag_off_dispatch_is_a_noop(tmp_path):
    """Flag off, and flag on with shadow_dir=None: no executor, no call,
    no file. Must stay FIRST in this file (see module docstring)."""
    import shared.strategy_shadow as strategy_shadow

    calls: list[tuple] = []

    def _stub(*args, **kwargs):  # pragma: no cover — must never run
        calls.append(args)
        return _STRATEGY_RESPONSE

    shadow_dir = tmp_path / "shadow_strategy"
    with patch("shared.strategy_shadow.opus_llm", side_effect=_stub):
        with patch("shared.config.SHADOW_STRATEGY_ENABLED", False):
            strategy_shadow.dispatch_strategy_shadow(
                stage="linkedin_strategy_form",
                system_prompt="system",
                user_prompt="user",
                max_tokens=16384,
                shadow_dir=shadow_dir,
                primary_meta={},
            )
        with patch("shared.config.SHADOW_STRATEGY_ENABLED", True):
            strategy_shadow.dispatch_strategy_shadow(
                stage="linkedin_strategy_form",
                system_prompt="system",
                user_prompt="user",
                max_tokens=16384,
                shadow_dir=None,
                primary_meta={},
            )
        assert strategy_shadow.drain_strategy_shadows(timeout=1) is True
        assert strategy_shadow.drain_strategy_shadows() is True

    assert strategy_shadow._shadow_executor is None  # never spun up
    assert calls == []
    assert not shadow_dir.exists()


def test_dispatch_returns_before_shadow_completes_and_drain_flushes_artifact(
    tmp_path, capsys
):
    """The required async property end to end: dispatch returns while the
    stub is still blocked; drain times out while it is blocked; after
    release + drain exactly one artifact exists with the right stage,
    model, primary_meta, and deterministic metrics."""
    import shared.strategy_shadow as strategy_shadow

    release_shadow = threading.Event()

    def _blocked_stub(system_prompt, user_prompt, **kwargs):
        # Bounded wait so a regression to synchronous dispatch fails the
        # no-artifact-yet assertion below instead of hanging the suite.
        assert release_shadow.wait(timeout=10), "shadow never released"
        assert kwargs.get("expect_json") is False
        assert kwargs.get("model_name") == "claude-fable-5"
        assert kwargs.get("usage_context", {}).get("stage") == (
            "linkedin_strategy_form_shadow"
        )
        # The worker passes a capture dict; opus_llm fills it with the
        # always-thinking model's summarized reasoning + stop_reason.
        capture = kwargs.get("capture")
        assert isinstance(capture, dict)
        capture["thinking_summary"] = "Weighed ML-core vs RL-core angles."
        capture["stop_reason"] = "end_turn"
        return _STRATEGY_RESPONSE

    shadow_dir = tmp_path / "shadow_strategy"
    with patch("shared.config.SHADOW_STRATEGY_ENABLED", True), \
         patch("shared.config.SHADOW_STRATEGY_MODEL_NAME", "claude-fable-5"), \
         patch("shared.strategy_shadow.opus_llm", side_effect=_blocked_stub):
        strategy_shadow.dispatch_strategy_shadow(
            stage="linkedin_strategy_form",
            system_prompt=_REFERENCE_SYSTEM,
            user_prompt="user prompt",
            max_tokens=16384,
            shadow_dir=shadow_dir,
            primary_meta={"primary_model": "claude-opus-4-6"},
        )

        # Dispatch already returned; the worker is blocked on the Event, so
        # nothing can have been written yet.
        assert _artifacts(shadow_dir) == []
        # Drain with the task still blocked: times out, False.
        assert strategy_shadow.drain_strategy_shadows(timeout=0.05) is False

        release_shadow.set()
        assert strategy_shadow.drain_strategy_shadows(timeout=10) is True

        files = _artifacts(shadow_dir)
        assert len(files) == 1
        assert files[0].name.startswith("shadow-linkedin_strategy_form-")
        artifact = json.loads(files[0].read_text(encoding="utf-8"))
        assert list(shadow_dir.glob("*.tmp")) == []

    assert artifact["stage"] == "linkedin_strategy_form"
    assert artifact["shadow_model"] == "claude-fable-5"
    assert artifact["shadow_error"] is None
    assert artifact["raw_response"] == _STRATEGY_RESPONSE
    assert artifact["latency_ms"] is not None
    assert artifact["primary_meta"] == {"primary_model": "claude-opus-4-6"}
    # Shadow visibility (item 19): the artifact is self-contained — it
    # carries the shadow's reasoning, its stop signal, and the exact
    # prompts the comparison ran on.
    assert artifact["thinking_summary"] == "Weighed ML-core vs RL-core angles."
    assert artifact["shadow_stop_reason"] == "end_turn"
    assert artifact["system_prompt"] == _REFERENCE_SYSTEM
    assert artifact["user_prompt"] == "user prompt"
    # Prompt-contract fields (2026-07-05): callers that don't declare a
    # context ran the default byte-identical contract.
    assert artifact["shadow_prompt_context"] == "byte-identical"
    assert artifact["primary_prompt_included_prior_run_data"] is None
    # Run-console contract (Sam, 2026-07-05, revised): EXACTLY ONE compact
    # line per completed comparison — models by name, metric comparison,
    # a pointer to the deep surfaces. Thinking and booleans stay OFF the
    # run console (they live in the artifact and the --follow feed).
    out = capsys.readouterr().out
    shadow_lines = [l for l in out.splitlines() if l.startswith("[shadow] ")]
    assert len(shadow_lines) == 1
    line = shadow_lines[0]
    assert line.startswith("[shadow] strategist(fable) linkedin_strategy_form done ")
    assert "3 strings/2 skeletons/novelty 0.50" in line
    assert "vs opus" in line
    assert "detail: feed or artifact" in line
    assert "Weighed ML-core vs RL-core angles." not in out
    assert '("machine learning" OR "deep learning")' not in out
    assert '"staff engineer" NOT recruiter' not in out
    metrics = artifact["metrics"]
    assert metrics["parse_failed"] is False
    assert metrics["n_strings"] == 3
    assert metrics["distinct_skeletons"] == 2
    assert metrics["max_skeleton_share"] == pytest.approx(0.667, abs=1e-3)
    assert metrics["vocab_novelty"] == pytest.approx(0.5)
    assert metrics["novelty_reference"] == "system+user"


def test_form_strategy_shadow_dispatch_gets_fresh_context_prompt(tmp_path):
    """Item 19 experiment-design lock (2026-07-05): the PRIMARY formation
    prompt may carry `## Prior Run Data`, search-family memory, and
    lane-feedback diffs — past-run performance the shadow experiment must
    exclude. The dispatched SHADOW user prompt carries none of it (the
    question is what the shadow model produces from the brief alone);
    the dispatch records the divergence for the artifact. Lives HERE
    rather than tests/test_linkedin_strategy.py because that module
    skips wholesale when its optional brief fixtures are absent — this
    lock must always run."""
    from shared.brief_loader import Brief
    from linkedin.strategy import form_strategy

    brief = Brief(
        id="shadow-fresh-context-test",
        role_title="Strategic Project Lead",
        role_description="Runs end-to-end delivery of AI data projects.",
        kit_url="",
        linkedin_project="test-project",
        linkedin_project_id="123",
        minimum_bar="delivery ownership",
        archetypes=[],
        noise_archetypes=[],
        hard_skips=[],
        clear_skips_from_review=[],
        known_noise_patterns=[],
        permanent_filters={},
        save_instructions={},
        experience_floor={},
        jd_text="Own end-to-end delivery of human-data projects for AI labs.",
    )
    mock_plan = {
        "strategy_rationale": "mock",
        "generated_strings": [
            {
                "boolean": '("program manager" OR "delivery lead") AND ("annotation")',
                "rationale": "mock",
                "vocabulary_sources": "mock",
            },
        ],
        "coverage_gaps": [],
        "noise_predictions": [],
    }
    primary_seen = {}

    def _primary(system, user, **kwargs):
        primary_seen["system"] = system
        primary_seen["user"] = user
        return mock_plan

    dispatched = {}

    with patch("linkedin.strategy.opus_llm", side_effect=_primary), \
         patch("shared.config.SHADOW_STRATEGY_ENABLED", True), \
         patch(
             "linkedin.strategy.dispatch_strategy_shadow",
             side_effect=lambda **kwargs: dispatched.update(kwargs),
         ):
        form_strategy(
            brief,
            [],
            prior_run_data={"total_saved": 7},
            lane_feedback=[
                {"diff_id": "d1", "action": "add", "target_type": "lane"}
            ],
            shadow_dir=tmp_path / "shadow_strategy",
        )

    # The primary really ran with past-run context...
    assert "## Prior Run Data" in primary_seen["user"]
    assert "## Lane Feedback Diffs" in primary_seen["user"]
    # ...and the shadow's prompt carries none of it.
    assert dispatched, "shadow dispatch never fired"
    assert "## Prior Run Data" not in dispatched["user_prompt"]
    assert "## Lane Feedback Diffs" not in dispatched["user_prompt"]
    assert "PROVEN VEINS" not in dispatched["user_prompt"]
    # Brief-derived content survives in the fresh prompt...
    assert "human-data projects for AI labs" in dispatched["user_prompt"]
    # ...the system prompt is shared (brief-derived), and the dispatch
    # declares which contract this comparison ran under.
    assert dispatched["system_prompt"] == primary_seen["system"]
    assert dispatched["shadow_prompt_context"] == "fresh"
    assert dispatched["primary_prompt_included_prior_run_data"] is True


def test_strategy_model_substitution_is_confined_to_strategy_calls(monkeypatch):
    from linkedin.strategy import adapt_after_block, form_strategy
    from shared import config
    from shared.brief_loader import Brief
    from shared.schemas import BlockReport

    brief = Brief(
        id="strategy-role-test",
        role_title="Strategic Project Lead",
        role_description="Runs end-to-end delivery of AI data projects.",
        kit_url="",
        linkedin_project="test-project",
        linkedin_project_id="123",
        minimum_bar="delivery ownership",
        archetypes=[],
        noise_archetypes=[],
        hard_skips=[],
        clear_skips_from_review=[],
        known_noise_patterns=[],
        permanent_filters={},
        save_instructions={},
        experience_floor={},
        jd_text="Own end-to-end delivery of human-data projects for AI labs.",
    )
    model_name = "strategy-role-test-model"
    unrelated_roles = {
        name: getattr(config, name)
        for name in (
            "CHEAP_MODEL_NAME",
            "FACIAL_MODEL_NAME",
            "FULL_EVAL_MODEL_NAME",
            "OPUS_MODEL_NAME",
        )
    }
    calls: list[dict] = []

    def fake_strategy_llm(*_args, **kwargs):
        calls.append(kwargs)
        if kwargs["usage_context"]["stage"] == "linkedin_strategy_form":
            return {
                "strategy_rationale": "mock",
                "generated_strings": [],
                "coverage_gaps": [],
                "noise_predictions": [],
            }
        return {
            "new_strings": [],
            "skip_remaining": [],
            "reorder": [],
            "noise_updates": [],
            "no_change": True,
        }

    monkeypatch.setattr(config, "STRATEGY_MODEL_NAME", model_name)
    with patch("linkedin.strategy.opus_llm", side_effect=fake_strategy_llm):
        form_strategy(brief, [], prior_run_data={})
        adapt_after_block(
            brief,
            BlockReport(
                block_name="test",
                strings_run=1,
                strings_with_saves=0,
                total_results=1,
                total_saves=0,
                string_details=[],
            ),
            [],
        )

    assert [call["model_name"] for call in calls] == [model_name, model_name]
    assert {
        name: getattr(config, name)
        for name in unrelated_roles
    } == unrelated_roles


def test_plan_metrics_golden():
    """Deterministic metrics on a known plan: 3 booleans, two skeletons
    ((G) AND (G) twice, T NOT T once), 4 distinct quoted terms. Default
    novelty records the legacy system-only reference label; a system+user
    reference excludes quoted terms that appear only in the user prompt."""
    from shared.strategy_shadow import plan_metrics

    plan = {
        "generated_strings": [
            {"boolean": '("machine learning" OR "deep learning") AND (PyTorch OR JAX)'},
            {"boolean": '("reinforcement learning" OR RLHF) AND (TensorFlow OR Keras)'},
            {"boolean": '"staff engineer" NOT recruiter'},
        ]
    }
    metrics = plan_metrics(plan, reference_text=_REFERENCE_SYSTEM)

    assert metrics["parse_failed"] is False
    assert metrics["n_strings"] == 3
    assert metrics["distinct_skeletons"] == 2
    assert metrics["skeleton_counts"] == {"(G) AND (G)": 2, "T NOT T": 1}
    assert metrics["max_skeleton_share"] == pytest.approx(0.667, abs=1e-3)
    assert metrics["mean_and_count"] == pytest.approx(0.667, abs=1e-3)
    assert metrics["not_usage_rate"] == pytest.approx(0.333, abs=1e-3)
    assert metrics["n_quoted_terms"] == 4
    assert metrics["novelty_reference"] == "system"
    # "machine learning" + "deep learning" appear in the reference;
    # "reinforcement learning" + "staff engineer" do not -> 2/4.
    assert metrics["vocab_novelty"] == pytest.approx(0.5)

    system_user_metrics = plan_metrics(
        plan,
        reference_text=_REFERENCE_SYSTEM + "\n" + _REFERENCE_USER_WITH_STAFF,
        novelty_reference="system+user",
    )

    assert system_user_metrics["novelty_reference"] == "system+user"
    assert system_user_metrics["n_quoted_terms"] == 4
    # "staff engineer" appears only in the user prompt, so it is no longer
    # novel under the system+user reference: only "reinforcement learning" is.
    assert system_user_metrics["vocab_novelty"] == pytest.approx(0.25)
    assert system_user_metrics["vocab_novelty"] < metrics["vocab_novelty"]


def test_boolean_metrics_skeletonize_is_strategy_shadow_object():
    import shared.boolean_metrics as boolean_metrics
    import shared.strategy_shadow as strategy_shadow

    assert boolean_metrics._skeletonize is strategy_shadow._skeletonize


def test_plan_metrics_unparseable_returns_parse_failed():
    from shared.strategy_shadow import plan_metrics

    assert plan_metrics("I refuse to produce boolean strings.") == {
        "parse_failed": True,
        "novelty_reference": "system",
    }
    assert plan_metrics(None, novelty_reference="system+user") == {
        "parse_failed": True,
        "novelty_reference": "system+user",
    }


def test_non_plan_response_prints_size_line_never_the_response(tmp_path, capsys):
    """A preflight-shaped shadow response (parseable JSON, zero
    generated_strings) renders as ONE line naming the response size and the
    artifact path — the verbatim multi-KB dump is exactly the flooding
    class the one-line contract exists to prevent."""
    import shared.strategy_shadow as strategy_shadow

    brief_json = json.dumps(
        {"role_title": "SPL", "capability_areas": ["delivery ops"] * 50}
    )

    shadow_dir = tmp_path / "shadow_strategy"
    with patch("shared.config.SHADOW_STRATEGY_ENABLED", True), \
         patch("shared.config.SHADOW_STRATEGY_MODEL_NAME", "claude-fable-5"), \
         patch("shared.strategy_shadow.opus_llm", return_value=brief_json):
        strategy_shadow.dispatch_strategy_shadow(
            stage="linkedin_preflight_v2",
            system_prompt="system",
            user_prompt="user",
            max_tokens=16384,
            shadow_dir=shadow_dir,
            primary_meta={"primary_model": "claude-opus-4-6"},
        )
        assert strategy_shadow.drain_strategy_shadows(timeout=10) is True

    out = capsys.readouterr().out
    shadow_lines = [l for l in out.splitlines() if l.startswith("[shadow] ")]
    assert len(shadow_lines) == 1
    line = shadow_lines[0]
    assert line.startswith("[shadow] strategist(fable) linkedin_preflight_v2 done ")
    assert "not plan-shaped" in line
    assert "KB" in line
    assert "artifact:" in line
    assert "role_title" not in out  # the response body never hits the console


def test_shadow_exception_lands_as_shadow_error_and_never_propagates(
    tmp_path, capsys
):
    """A raise through opus_llm — the RuntimeError shape a Fable refusal
    produces via a non-end_turn stop_reason — is captured in the artifact,
    and neither dispatch nor drain re-raises."""
    import shared.strategy_shadow as strategy_shadow

    def _raising_stub(system_prompt, user_prompt, **kwargs):
        raise RuntimeError(
            "Opus response truncated: stop_reason=refusal. "
            "Increase max_tokens or reduce prompt size."
        )

    shadow_dir = tmp_path / "shadow_strategy"
    with patch("shared.config.SHADOW_STRATEGY_ENABLED", True), \
         patch("shared.config.SHADOW_STRATEGY_MODEL_NAME", "claude-fable-5"), \
         patch("shared.strategy_shadow.opus_llm", side_effect=_raising_stub):
        strategy_shadow.dispatch_strategy_shadow(
            stage="linkedin_preflight_v2",
            system_prompt="system",
            user_prompt="user",
            max_tokens=16384,
            shadow_dir=shadow_dir,
            primary_meta={"primary_model": "claude-opus-4-6"},
        )
        assert strategy_shadow.drain_strategy_shadows(timeout=10) is True

        files = _artifacts(shadow_dir)
        assert len(files) == 1
        assert files[0].name.startswith("shadow-linkedin_preflight_v2-")
        artifact = json.loads(files[0].read_text(encoding="utf-8"))

    assert artifact["stage"] == "linkedin_preflight_v2"
    assert "stop_reason=refusal" in artifact["shadow_error"]
    assert artifact["raw_response"] is None
    assert artifact["metrics"] is None
    assert artifact["latency_ms"] is not None
    assert artifact["primary_meta"] == {"primary_model": "claude-opus-4-6"}
    # Prompts persist on the error path too (the artifact must stay
    # replayable); the stub raised before filling capture, so the
    # reasoning fields read as absent rather than crashing the write.
    assert artifact["system_prompt"] == "system"
    assert artifact["user_prompt"] == "user"
    assert artifact["thinking_summary"] is None
    assert artifact["shadow_stop_reason"] is None
    # The failure is named on the live console too — as ONE line carrying
    # the error and the artifact path, never a multi-line block.
    out = capsys.readouterr().out
    shadow_lines = [l for l in out.splitlines() if l.startswith("[shadow] ")]
    assert len(shadow_lines) == 1
    line = shadow_lines[0]
    assert line.startswith("[shadow] strategist(fable) linkedin_preflight_v2 ")
    assert "SHADOW ERROR" in line
    assert "stop_reason=refusal" in line
    assert "artifact:" in line
