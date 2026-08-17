"""Tests for the Reflection HITL flow — persistence + engine phases + API.

Coverage:
- Persistence (shared.runtime_state.reflection): create / get / patch /
  active-lookup / commit / discard. Phase transitions and terminal-row
  immutability.
- Engine hunk computation (market_intelligence.reflection): brief
  recommendations → hunks; per-section apply semantics.
- API endpoints: create requires brief_id; conflict on already-active;
  steering cap enforcement; commit requires accepted_hunk_ids; discard
  is idempotent.

The engine phase functions that actually call the planner / Perplexity
backends are exercised at the unit level by mocking the LLM access
helpers — the goal is to verify phase contract, not LLM output quality.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from cloris import api as cloris_api
from cloris.app import create_app
from market_intelligence import reflection as reflection_engine
from shared.runtime_state import reflection as reflection_store
from shared.runtime_state.store import RuntimeStateStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_store(tmp_path: Path) -> RuntimeStateStore:
    return RuntimeStateStore(tmp_path / "intake" / "intake_sessions.sqlite3")


@pytest.fixture
def store(tmp_path: Path) -> RuntimeStateStore:
    return _make_store(tmp_path)


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> TestClient:
    """TestClient with the intake/reflection store pointed at tmp_path.

    Monkeypatches three seams:

    - ``_intake_store`` / ``_reflection_store_factory`` — writer paths
      for POST/PATCH/DELETE handlers.
    - ``_intake_db_path`` — read-only path the migrated reflection
      GET endpoints (``/api/reflection/sessions/active``,
      ``/api/reflection/sessions/{id}``) take through ``read_models``
      to avoid instantiating the writer per request.
    """

    tmp_db_path = tmp_path / "intake" / "intake_sessions.sqlite3"
    test_store = RuntimeStateStore(tmp_db_path)
    monkeypatch.setattr(cloris_api._monolith, "_intake_store", lambda: test_store)
    monkeypatch.setattr(
        cloris_api, "_reflection_store_factory", lambda: test_store
    )
    monkeypatch.setattr(cloris_api._monolith, "_intake_db_path", lambda: tmp_db_path)
    return TestClient(create_app())


# ---------------------------------------------------------------------------
# Persistence layer
# ---------------------------------------------------------------------------


class TestReflectionPersistence:
    def test_create_initializes_planning_phase(
        self, store: RuntimeStateStore
    ) -> None:
        s = reflection_store.create_reflection_session(
            store=store, brief_id="brief-x", source_run_id=42
        )
        assert s["current_phase"] == "planning"
        assert s["steering_iterations"] == 0
        assert s["completed_at"] is None
        assert s["discarded_at"] is None
        assert s["state_json"] == {}
        assert s["brief_id"] == "brief-x"
        assert s["source_run_id"] == 42

    def test_get_active_returns_only_non_terminal(
        self, store: RuntimeStateStore
    ) -> None:
        s1 = reflection_store.create_reflection_session(
            store=store, brief_id="brief-y"
        )
        # An active session is returned by get_active.
        active = reflection_store.get_active_reflection_for_brief(
            store=store, brief_id="brief-y"
        )
        assert active is not None
        assert active["id"] == s1["id"]
        # Discard, then no active.
        reflection_store.discard_reflection(store=store, session_id=s1["id"])
        assert (
            reflection_store.get_active_reflection_for_brief(
                store=store, brief_id="brief-y"
            )
            is None
        )

    def test_patch_state_bumps_steering(self, store: RuntimeStateStore) -> None:
        s = reflection_store.create_reflection_session(
            store=store, brief_id="b"
        )
        s2 = reflection_store.patch_reflection_state(
            store=store,
            session_id=s["id"],
            state_json={"foo": 1},
            bump_steering=True,
        )
        assert s2 is not None
        assert s2["steering_iterations"] == 1
        s3 = reflection_store.patch_reflection_state(
            store=store, session_id=s["id"], bump_steering=True
        )
        assert s3 is not None
        assert s3["steering_iterations"] == 2

    def test_terminal_row_refuses_patch(self, store: RuntimeStateStore) -> None:
        s = reflection_store.create_reflection_session(
            store=store, brief_id="b"
        )
        reflection_store.commit_reflection(
            store=store,
            session_id=s["id"],
            brief_version_path="config/b/versions/x.json",
        )
        with pytest.raises(ValueError):
            reflection_store.patch_reflection_state(
                store=store, session_id=s["id"], state_json={"foo": 1}
            )

    def test_commit_is_idempotent(self, store: RuntimeStateStore) -> None:
        s = reflection_store.create_reflection_session(
            store=store, brief_id="b"
        )
        s1 = reflection_store.commit_reflection(
            store=store,
            session_id=s["id"],
            brief_version_path="path-1",
        )
        s2 = reflection_store.commit_reflection(
            store=store,
            session_id=s["id"],
            brief_version_path="path-2",
        )
        assert s1 is not None
        assert s2 is not None
        # Second commit doesn't overwrite the original timestamp / version.
        assert s1["completed_at"] == s2["completed_at"]
        assert s1["brief_version_committed"] == s2["brief_version_committed"]
        assert s1["brief_version_committed"] == "path-1"

    def test_discard_is_idempotent(self, store: RuntimeStateStore) -> None:
        s = reflection_store.create_reflection_session(
            store=store, brief_id="b"
        )
        s1 = reflection_store.discard_reflection(
            store=store, session_id=s["id"]
        )
        s2 = reflection_store.discard_reflection(
            store=store, session_id=s["id"]
        )
        assert s1 is not None and s2 is not None
        assert s1["discarded_at"] == s2["discarded_at"]


# ---------------------------------------------------------------------------
# Engine — hunk computation + application
# ---------------------------------------------------------------------------


class TestHunkComputation:
    def test_recommendations_become_hunks(self) -> None:
        artifact = {
            "brief_recommendations": [
                {
                    "recommendation_id": "rec-1",
                    "target_field": "additional_search_terms",
                    "proposal": "Stripe",
                    "reason": "Strong adjacent talent",
                    "confidence": 0.8,
                },
                {
                    "recommendation_id": "rec-2",
                    "target_field": "instructions",
                    "proposal": "Prefer recent payments-domain experience",
                    "reason": "Pattern across 5 reject signals",
                    "confidence": 0.7,
                },
            ]
        }
        brief_raw: dict = {"additional_search_terms": []}
        hunks = reflection_engine._build_hunks_from_artifact(
            artifact, brief_raw=brief_raw
        )
        assert len(hunks) == 2
        h1, h2 = hunks
        assert h1["section"] == "additional_search_terms"
        assert h1["kind"] == "add"
        assert h1["after"] == "Stripe"
        assert h1["default_approved"] is True  # 0.8 >= 0.65
        assert h2["section"] == "instructions"
        assert h2["kind"] == "add"  # no existing instructions content
        assert h2["default_approved"] is True

    def test_no_op_recommendation_is_dropped(self) -> None:
        artifact = {
            "brief_recommendations": [
                {
                    "recommendation_id": "rec-noop",
                    "target_field": "instructions",
                    "proposal": "Already on the brief",
                    "reason": "n/a",
                    "confidence": 0.7,
                },
            ]
        }
        brief_raw = {"instructions": "Already on the brief"}
        hunks = reflection_engine._build_hunks_from_artifact(
            artifact, brief_raw=brief_raw
        )
        assert hunks == []

    def test_apply_list_section_dedupes(self) -> None:
        brief = {"additional_search_terms": ["python", "go"]}
        hunk = {
            "section": "additional_search_terms",
            "kind": "add",
            "after": "python",  # already present, normalize-case match
        }
        out = reflection_engine._apply_hunk_to_brief(brief, hunk)
        assert out["additional_search_terms"] == ["python", "go"]

    def test_apply_appends_new_list_value(self) -> None:
        brief = {"additional_search_terms": ["python"]}
        hunk = {
            "section": "additional_search_terms",
            "kind": "add",
            "after": "rust",
        }
        out = reflection_engine._apply_hunk_to_brief(brief, hunk)
        assert out["additional_search_terms"] == ["python", "rust"]

    def test_apply_prose_section_appends(self) -> None:
        brief = {"instructions": "Existing guidance."}
        hunk = {
            "section": "instructions",
            "kind": "add",
            "after": "Additional guidance.",
        }
        out = reflection_engine._apply_hunk_to_brief(brief, hunk)
        assert (
            out["instructions"]
            == "Existing guidance.\n\nAdditional guidance."
        )

    def test_apply_skips_blank_after(self) -> None:
        brief = {"notes": "Existing"}
        hunk = {"section": "notes", "kind": "add", "after": "   "}
        out = reflection_engine._apply_hunk_to_brief(brief, hunk)
        assert out["notes"] == "Existing"


class TestMergeContractParity:
    """Snapshot the merge contract so it doesn't drift between
    Python (_apply_hunk_to_brief) and TypeScript (buildMergedV2).

    These tests pin the exact output shape for a representative set of
    (hunk, brief) inputs. The TypeScript side has parallel coverage in
    cloris/frontend/src/test/briefDiff.test.ts asserting identical
    shapes. When either implementation changes, BOTH sides need
    updating in the same PR — that's the contract.
    """

    def test_list_section_dedupe_append_case_insensitive(self) -> None:
        # "STRIPE" already present in case-insensitive form should be
        # treated as a duplicate. Mirrors the TS isProseEntry +
        # normalizeForDedupe path.
        brief = {"additional_search_terms": ["Stripe", "Plaid"]}
        hunk = {
            "section": "additional_search_terms",
            "kind": "add",
            "after": "stripe",
        }
        out = reflection_engine._apply_hunk_to_brief(brief, hunk)
        assert out["additional_search_terms"] == ["Stripe", "Plaid"]

    def test_list_section_appends_when_truly_new(self) -> None:
        brief = {"additional_search_terms": ["Stripe"]}
        hunk = {
            "section": "additional_search_terms",
            "kind": "add",
            "after": "Block",
        }
        out = reflection_engine._apply_hunk_to_brief(brief, hunk)
        assert out["additional_search_terms"] == ["Stripe", "Block"]

    def test_prose_section_appends_with_double_newline(self) -> None:
        # Append behavior with trailing whitespace stripped from
        # existing content. Mirrors the TS .replace(/\s+$/, "") path.
        brief = {"instructions": "First line.   "}
        hunk = {
            "section": "instructions",
            "kind": "add",
            "after": "Second line.",
        }
        out = reflection_engine._apply_hunk_to_brief(brief, hunk)
        assert out["instructions"] == "First line.\n\nSecond line."

    def test_prose_section_replaces_when_empty(self) -> None:
        brief: dict = {}
        hunk = {
            "section": "notes",
            "kind": "add",
            "after": "Fresh notes content.",
        }
        out = reflection_engine._apply_hunk_to_brief(brief, hunk)
        assert out["notes"] == "Fresh notes content."

    def test_employer_signal_rules_uses_list_semantics(self) -> None:
        brief = {"employer_signal_rules": ["FAANG ex"]}
        hunk = {
            "section": "employer_signal_rules",
            "kind": "add",
            "after": "Stripe alumni",
        }
        out = reflection_engine._apply_hunk_to_brief(brief, hunk)
        assert out["employer_signal_rules"] == ["FAANG ex", "Stripe alumni"]


# TestEditorialTranslation removed in v2 — _build_editorial_briefing
# and _build_intentions were superseded by HeuristicBriefingBackend +
# BriefingPolishBackend in market_intelligence/briefing_polish.py.
# Equivalent coverage lives in tests/test_briefing_polish.py:
# - TestHeuristicBriefing covers the editorial output shape
# - TestConfidence covers the signal-density formula
# - TestSnapshotsAgainstRealArtifacts covers behavior against real
#   on-disk fixtures
# - TestPolishCascade covers the four LLM failure routes


# ---------------------------------------------------------------------------
# API surface — happy path + structured error codes
# ---------------------------------------------------------------------------


class TestReflectionApi:
    def test_create_requires_known_brief_id(self, client: TestClient) -> None:
        # Patch the brief resolver to raise BriefIdNotFoundError so we
        # exercise the 404 path without seeding a real brief.
        from cloris.api import BriefIdNotFoundError

        with patch.object(
            cloris_api,
            "_resolve_brief_path_or_raise",
            side_effect=BriefIdNotFoundError("brief-missing"),
        ):
            resp = client.post(
                "/api/reflection/sessions",
                json={"brief_id": "brief-missing"},
            )
        assert resp.status_code == 404
        assert resp.json()["detail"]["error"] == "brief_id_not_found"

    def test_get_active_returns_null_when_none(
        self, client: TestClient
    ) -> None:
        resp = client.get(
            "/api/reflection/sessions/active",
            params={"brief_id": "anything"},
        )
        assert resp.status_code == 200
        assert resp.json()["session"] is None

    def test_steering_cap_returns_409(
        self, client: TestClient, store: RuntimeStateStore
    ) -> None:
        # Seed a session with steering_iterations at the cap.
        s = reflection_store.create_reflection_session(
            store=store,
            brief_id="b",
            initial_state={
                "phase_outputs": {"plan": {}},
                "context": {
                    "brief_path": "/tmp/brief.json",
                    "run_dir": None,
                    "mode": "post_run",
                },
                "steering_history": [],
            },
        )
        # Bump to the cap manually.
        for _ in range(reflection_engine.MAX_STEERING_ITERATIONS):
            reflection_store.patch_reflection_state(
                store=store, session_id=s["id"], bump_steering=True
            )
        resp = client.patch(
            f"/api/reflection/sessions/{s['id']}/steering",
            json={"note": "one more refinement please"},
        )
        assert resp.status_code == 409
        assert resp.json()["detail"]["error"] == "reflection_steering_capped"

    def test_steering_empty_note_is_noop(
        self, client: TestClient, store: RuntimeStateStore
    ) -> None:
        s = reflection_store.create_reflection_session(
            store=store, brief_id="b"
        )
        before_iters = s["steering_iterations"]
        resp = client.patch(
            f"/api/reflection/sessions/{s['id']}/steering",
            json={"note": "   "},
        )
        assert resp.status_code == 200
        assert resp.json()["session"]["steering_iterations"] == before_iters

    def test_commit_requires_accepted_hunks(
        self, client: TestClient, store: RuntimeStateStore
    ) -> None:
        s = reflection_store.create_reflection_session(
            store=store, brief_id="b"
        )
        reflection_store.patch_reflection_state(
            store=store,
            session_id=s["id"],
            current_phase="awaiting_diff",
        )
        resp = client.post(
            f"/api/reflection/sessions/{s['id']}/commit",
            json={"accepted_hunk_ids": []},
        )
        assert resp.status_code == 422
        assert (
            resp.json()["detail"]["error"] == "reflection_commit_no_hunks"
        )

    def test_discard_is_safe_on_terminal(
        self, client: TestClient, store: RuntimeStateStore
    ) -> None:
        s = reflection_store.create_reflection_session(
            store=store, brief_id="b"
        )
        reflection_store.discard_reflection(
            store=store, session_id=s["id"]
        )
        resp = client.post(
            f"/api/reflection/sessions/{s['id']}/discard",
            json={},
        )
        assert resp.status_code == 200
        assert resp.json()["session"]["current_phase"] == "discarded"

    def test_steering_locked_outside_planning(
        self, client: TestClient, store: RuntimeStateStore
    ) -> None:
        s = reflection_store.create_reflection_session(
            store=store, brief_id="b"
        )
        reflection_store.patch_reflection_state(
            store=store,
            session_id=s["id"],
            current_phase="researching",
        )
        resp = client.patch(
            f"/api/reflection/sessions/{s['id']}/steering",
            json={"note": "one more thing"},
        )
        assert resp.status_code == 409
        assert resp.json()["detail"]["error"] == "reflection_phase_locked"

    def test_get_session_404(self, client: TestClient) -> None:
        resp = client.get("/api/reflection/sessions/99999")
        assert resp.status_code == 404
        assert (
            resp.json()["detail"]["error"] == "reflection_session_not_found"
        )

    def test_already_active_returns_409_with_session_id(
        self, client: TestClient, store: RuntimeStateStore
    ) -> None:
        s = reflection_store.create_reflection_session(
            store=store, brief_id="brief-collision"
        )
        from cloris.api import BriefIdNotFoundError  # noqa: F401

        with patch.object(
            cloris_api,
            "_resolve_brief_path_or_raise",
            return_value=Path("/tmp/brief.json"),
        ):
            resp = client.post(
                "/api/reflection/sessions",
                json={"brief_id": "brief-collision"},
            )
        assert resp.status_code == 409
        body = resp.json()["detail"]
        assert body["error"] == "reflection_already_active"
        assert body["session_id"] == s["id"]

    def test_create_with_source_run_id_resolves_run_dir(
        self,
        client: TestClient,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """POST with source_run_id maps the run id to the run row's output_dir.

        Regression: the lookup imported a nonexistent
        ``resolve_runtime_state_path`` outside its try block, so every
        create carrying a source_run_id 500'd before the planner ran.
        """

        import shared.output_paths as output_paths

        brief_id = "brief-run-anchor"
        state_root = tmp_path / "state"
        state_dir = state_root / "linkedin" / "9999"
        state_dir.mkdir(parents=True)
        run_store = RuntimeStateStore(state_dir / "runtime_state.sqlite3")
        run_output_dir = tmp_path / "runs" / "linkedin" / "run-1"
        run_id = run_store.start_run(
            source="linkedin",
            brief_id=brief_id,
            output_dir=str(run_output_dir),
            mode="live",
        )
        monkeypatch.setattr(output_paths, "STATE_ROOT", state_root)

        captured: dict = {}

        def fake_plan(*, brief_path: Path, run_dir: Path | None, mode: str) -> dict:
            captured["run_dir"] = run_dir
            return {
                "phase_outputs": {"plan": {}},
                "context": {
                    "brief_path": str(brief_path),
                    "run_dir": str(run_dir) if run_dir else None,
                    "mode": mode,
                },
                "steering_history": [],
            }

        monkeypatch.setattr(
            reflection_engine, "reflection_phase_plan", fake_plan
        )
        with patch.object(
            cloris_api,
            "_resolve_brief_path_or_raise",
            return_value=Path("/tmp/brief.json"),
        ):
            resp = client.post(
                "/api/reflection/sessions",
                json={"brief_id": brief_id, "source_run_id": run_id},
            )
        assert resp.status_code == 201
        assert captured["run_dir"] == run_output_dir
        assert resp.json()["session"]["source_run_id"] == run_id

    def test_create_with_unknown_source_run_id_passes_none(
        self,
        client: TestClient,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A source_run_id that resolves nowhere degrades to run_dir=None."""

        import shared.output_paths as output_paths

        monkeypatch.setattr(output_paths, "STATE_ROOT", tmp_path / "empty-state")

        captured: dict = {}

        def fake_plan(*, brief_path: Path, run_dir: Path | None, mode: str) -> dict:
            captured["run_dir"] = run_dir
            return {
                "phase_outputs": {"plan": {}},
                "context": {
                    "brief_path": str(brief_path),
                    "run_dir": None,
                    "mode": mode,
                },
                "steering_history": [],
            }

        monkeypatch.setattr(
            reflection_engine, "reflection_phase_plan", fake_plan
        )
        with patch.object(
            cloris_api,
            "_resolve_brief_path_or_raise",
            return_value=Path("/tmp/brief.json"),
        ):
            resp = client.post(
                "/api/reflection/sessions",
                json={"brief_id": "brief-no-run", "source_run_id": 12345},
            )
        assert resp.status_code == 201
        assert captured["run_dir"] is None


def test_apply_hunk_refuses_to_replace_structured_section_with_prose():
    """P3.7: the fallthrough replace must not corrupt a structured section."""

    from market_intelligence.reflection import (
        StructuredSectionHunkError,
        _apply_hunk_to_brief,
    )

    brief = {"non_fit_patterns": [{"label": "IC-only", "description": "..."}]}
    with pytest.raises(StructuredSectionHunkError, match="non_fit_patterns"):
        _apply_hunk_to_brief(
            brief,
            {"section": "non_fit_patterns", "kind": "replace", "after": "just some prose"},
        )
    # A plain-string section still replaces normally.
    out = _apply_hunk_to_brief(
        {"minimum_bar": "old bar"},
        {"section": "minimum_bar", "kind": "replace", "after": "new bar"},
    )
    assert out["minimum_bar"] == "new bar"
