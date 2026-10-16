"""Offline re-judge tool: selection by latest prior decision, and the dry run."""
import json
import sys
from types import SimpleNamespace

import pytest

import tools.rejudge_review_flagged as rejudge

# One row per line, oldest first; the latest line per URL is the verdict.
JUDGMENTS = [
    ("u-flagged", "Flagged Person", "REVIEW_FLAGGED", "could not decide"),
    ("u-reject-then-save", "Upgraded Person", "REJECT", "first pass said no"),
    ("u-reject-then-save", "Upgraded Person", "SAVE", "second pass saved"),
    ("u-save-then-reject", "Downgraded Person", "SAVE", "first pass saved"),
    ("u-save-then-reject", "Downgraded Person", "REJECT", "second pass said no"),
    ("u-flagged-then-reject", "Settled Person", "REVIEW_FLAGGED", "parked"),
    ("u-flagged-then-reject", "Settled Person", "REJECT", "settled as reject"),
    ("u-inferential", "Inferred Person", "INFERENTIAL_SAVE", "inferred from team"),
    ("u-no-profile", "Missing Person", "REJECT", "no stored profile"),
]
STORED = ["u-flagged", "u-reject-then-save", "u-save-then-reject",
          "u-flagged-then-reject", "u-inferential"]


@pytest.fixture
def state_dir(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    (state / "final_judgments.jsonl").write_text("".join(
        json.dumps({"profile_url": url, "candidate_name": name,
                    "decision": decision, "rationale": rationale}) + "\n"
        for url, name, decision, rationale in JUDGMENTS
    ))
    names = {url: name for url, name, _, _ in JUDGMENTS}
    (state / "profile_summaries.jsonl").write_text("".join(
        json.dumps({"name": names[url], "profile_url": url, "headline": "h"}) + "\n"
        for url in STORED
    ))
    return state


@pytest.fixture
def calls(monkeypatch):
    """Replace every model-facing seam; record what reached it."""
    seen = {"preflight": 0, "judged": []}
    brief = SimpleNamespace(id="brief-test")

    def fake_preflight(*args, **kwargs):
        seen["preflight"] += 1
        return object()

    def fake_judge(summary, brief=None):
        seen["judged"].append(summary.profile_url)
        return SimpleNamespace(decision="SAVE", rationale=f"new for {summary.name}",
                               post_save_modifier="", confidence=0.9)

    monkeypatch.setattr(rejudge, "load_brief", lambda path: brief)
    monkeypatch.setattr(rejudge, "generate_preflight_v2_once", fake_preflight)
    monkeypatch.setattr(rejudge, "finalize_preflight_v2",
                        lambda b, g: SimpleNamespace(brief=b, brief_json={"capability_areas": []}))
    monkeypatch.setattr(rejudge, "init_judger", lambda b: None)
    monkeypatch.setattr(rejudge, "full_judge", fake_judge)
    return seen


def run(monkeypatch, state_dir, out_dir, *extra):
    monkeypatch.setattr(sys, "argv", ["rejudge_review_flagged.py", str(state_dir),
                                      "--brief", "unused.json", "--out-dir", str(out_dir), *extra])
    return rejudge.main()


def results(out_dir):
    (rescue_dir,) = out_dir.iterdir()
    rows = [json.loads(line) for line in (rescue_dir / "rejudgments.jsonl").read_text().splitlines()]
    return {row["profile_url"]: row for row in rows}


def test_default_selects_only_latest_review_flagged(monkeypatch, tmp_path, state_dir, calls):
    assert run(monkeypatch, state_dir, tmp_path / "out") == 0
    rows = results(tmp_path / "out")
    assert calls["judged"] == ["u-flagged"]
    assert rows["u-flagged"]["old_decision"] == "REVIEW_FLAGGED"
    assert rows["u-flagged"]["old_rationale"] == "could not decide"


def test_decisions_filter_on_the_latest_verdict_per_url(monkeypatch, tmp_path, state_dir, calls):
    assert run(monkeypatch, state_dir, tmp_path / "out", "--decisions", "reject,Save") == 0
    rows = results(tmp_path / "out")
    assert sorted(calls["judged"]) == ["u-flagged-then-reject", "u-reject-then-save",
                                       "u-save-then-reject"]
    assert rows["u-reject-then-save"]["old_decision"] == "SAVE"
    assert rows["u-reject-then-save"]["old_rationale"] == "second pass saved"
    assert rows["u-save-then-reject"]["old_decision"] == "REJECT"
    assert rows["u-flagged-then-reject"]["old_decision"] == "REJECT"


def test_all_selects_every_latest_verdict(monkeypatch, tmp_path, state_dir, calls):
    assert run(monkeypatch, state_dir, tmp_path / "out", "--decisions", "all") == 0
    rows = results(tmp_path / "out")
    assert sorted(calls["judged"]) == sorted(STORED)
    assert {url: row["old_decision"] for url, row in rows.items()} == {
        "u-flagged": "REVIEW_FLAGGED",
        "u-reject-then-save": "SAVE",
        "u-save-then-reject": "REJECT",
        "u-flagged-then-reject": "REJECT",
        "u-inferential": "INFERENTIAL_SAVE",
    }


def test_judge_error_rows_carry_the_real_old_decision(monkeypatch, tmp_path, state_dir, calls):
    def failing_judge(summary, brief=None):
        raise RuntimeError("api fault")

    monkeypatch.setattr(rejudge, "full_judge", failing_judge)
    assert run(monkeypatch, state_dir, tmp_path / "out", "--decisions", "INFERENTIAL_SAVE") == 0
    row = results(tmp_path / "out")["u-inferential"]
    assert row["new_decision"] == "RESCUE_ERROR"
    assert row["old_decision"] == "INFERENTIAL_SAVE"


def test_dry_run_prices_the_selection_without_model_calls_or_writes(
        monkeypatch, tmp_path, state_dir, calls, capsys):
    before = sorted((p.name, p.stat().st_mtime_ns) for p in state_dir.iterdir())
    out_dir = tmp_path / "out"
    assert run(monkeypatch, state_dir, out_dir, "--decisions", "all", "--dry-run") == 0
    assert calls == {"preflight": 0, "judged": []}
    assert not out_dir.exists()
    assert sorted((p.name, p.stat().st_mtime_ns) for p in state_dir.iterdir()) == before
    printed = capsys.readouterr().out
    assert "6 candidates selected" in printed
    assert "REJECT: 3 selected, 2 with a stored profile" in printed
    assert "judge calls a real run would make: 5" in printed


# ---------------------------------------------------------------------------
# --resume / --shard / --merge: finish a killed run on its saved criteria.
# ---------------------------------------------------------------------------

from shared.brief_loader import Brief as LoadedBrief  # noqa: E402
from shared.brief_loader import load_brief as real_load_brief  # noqa: E402

# A V2 brief as preflight_v2 saves it: loadable by the repo's own V2 loader.
SAVED_V2_BRIEF = {
    "role_title": "Saved Criteria Role",
    "role_summary": "The criteria the killed run judged against.",
    "geography": "United States",
    "linkedin_project": "saved-project",
    "minimum_years_experience": 5,
    "minimum_bar_description": "Owns the work.",
    "engagement_context": {
        "hiring_company": "ExampleCo",
        "engagement_description": "A test search.",
        "talent_bar_statement": "Ownership clears the bar.",
        "selectivity_posture": "selective",
    },
    "capability_areas": [{
        "name": "Ownership",
        "description": "Owns the work end to end.",
        "builder_signals": ["shipped it"],
        "user_signals": ["used it"],
    }],
    "depth_distinction": {
        "builder_definition": "Builds it.",
        "user_definition": "Uses it.",
        "edge_case_guidance": "Borderline = full eval.",
    },
    "provenance": {"generated_by": "preflight_v2", "reviewed": False},
}


def row(url, decision, **extra):
    return {"profile_url": url, "candidate_name": url, "old_decision": "REJECT",
            "new_decision": decision, **extra}


def make_rescue(tmp_path, rows, *, name="rescue", raw_tail=""):
    rescue = tmp_path / name
    rescue.mkdir()
    (rescue / "preflight_v2_brief.json").write_text(json.dumps(SAVED_V2_BRIEF, indent=2))
    (rescue / "rejudgments.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows) + raw_tail)
    return rescue


def run_args(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["rejudge_review_flagged.py", *map(str, args)])
    return rejudge.main()


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def snapshot(directory):
    return sorted((p.name, p.stat().st_size, p.stat().st_mtime_ns)
                  for p in directory.iterdir())


def test_resume_skips_judged_urls_and_reattempts_error_rows(
        monkeypatch, tmp_path, state_dir, calls):
    rescue = make_rescue(tmp_path, [
        row("u-flagged", "REJECT"),
        row("u-reject-then-save", "RESCUE_ERROR", error="RuntimeError: api fault"),
    ])
    assert run_args(monkeypatch, state_dir, "--decisions", "all",
                    "--resume", rescue) == 0
    assert sorted(calls["judged"]) == ["u-flagged-then-reject", "u-inferential",
                                       "u-reject-then-save", "u-save-then-reject"]
    rows = read_rows(rescue / "rejudgments.jsonl")
    # The killed run's rows survive; new rows append after them.
    assert rows[0]["profile_url"] == "u-flagged" and rows[0]["new_decision"] == "REJECT"
    assert rows[1]["new_decision"] == "RESCUE_ERROR"
    assert sorted(r["profile_url"] for r in rows[2:]) == sorted(calls["judged"])
    # No new stamped dir beside the rescue dir: everything lands in it.
    assert sorted(p.name for p in tmp_path.iterdir()) == ["rescue", "state"]
    passers = json.loads((rescue / "passers.json").read_text())
    assert sorted(p["profile_url"] for p in passers) == sorted(calls["judged"])


def test_resume_repairs_a_row_cut_off_by_the_kill(monkeypatch, tmp_path, state_dir, calls):
    rescue = make_rescue(tmp_path, [row("u-flagged", "REJECT")],
                         raw_tail='{"profile_url": "u-inferential", "new_dec')
    assert run_args(monkeypatch, state_dir, "--decisions", "all",
                    "--resume", rescue) == 0
    assert "u-inferential" in calls["judged"]
    assert "u-flagged" not in calls["judged"]
    lines = (rescue / "rejudgments.jsonl").read_text().splitlines()
    # The torn line stays on its own line; every appended row parses.
    assert lines[1] == '{"profile_url": "u-inferential", "new_dec'
    assert all(json.loads(line) for line in lines[2:])


def test_resume_makes_no_preflight_call_and_judges_on_the_saved_brief(
        monkeypatch, tmp_path, state_dir, calls):
    loaded_from = []

    def recording_load_brief(path):
        loaded_from.append(str(path))
        return real_load_brief(path)

    def no_finalize(*args, **kwargs):
        raise AssertionError("resume must not finalize a new preflight")

    briefs_seen = []

    def judge(summary, brief=None):
        briefs_seen.append(brief)
        return SimpleNamespace(decision="REJECT", rationale="r",
                               post_save_modifier="", confidence=0.5)

    monkeypatch.setattr(rejudge, "load_brief", recording_load_brief)
    monkeypatch.setattr(rejudge, "finalize_preflight_v2", no_finalize)
    monkeypatch.setattr(rejudge, "full_judge", judge)
    rescue = make_rescue(tmp_path, [])
    assert run_args(monkeypatch, state_dir, "--resume", rescue) == 0
    assert calls["preflight"] == 0
    assert loaded_from == [str(rescue / "preflight_v2_brief.json")]
    assert len(briefs_seen) == 1
    (brief,) = briefs_seen
    assert isinstance(brief, LoadedBrief)
    assert brief.role_title == "Saved Criteria Role"
    assert brief._new_brief.capability_areas[0].name == "Ownership"


def test_shards_are_disjoint_and_cover_exactly_the_remaining_set(
        monkeypatch, tmp_path, calls):
    urls = [f"https://example.test/in/person-{k:02d}" for k in range(40)]
    state = tmp_path / "state"
    state.mkdir()
    (state / "final_judgments.jsonl").write_text("".join(
        json.dumps({"profile_url": u, "candidate_name": u, "decision": "REJECT"}) + "\n"
        for u in urls))
    (state / "profile_summaries.jsonl").write_text("".join(
        json.dumps({"name": u, "profile_url": u, "headline": "h"}) + "\n" for u in urls))
    judged_before = urls[:7]
    rescue = make_rescue(tmp_path, [row(u, "REJECT") for u in judged_before]
                         + [row(urls[7], "RESCUE_ERROR")])
    main_before = (rescue / "rejudgments.jsonl").read_text()
    remaining = set(urls) - set(judged_before)

    per_shard = {}
    for i in range(1, 5):
        calls["judged"].clear()
        assert run_args(monkeypatch, state, "--decisions", "all", "--resume", rescue,
                        "--shard", f"{i}/4") == 0
        per_shard[i] = set(calls["judged"])
        written = read_rows(rescue / f"rejudgments.shard-{i}-of-4.jsonl")
        assert {r["profile_url"] for r in written} == per_shard[i]
        assert (rescue / f"passers.shard-{i}-of-4.json").exists()

    for i in per_shard:
        for j in per_shard:
            if i < j:
                assert not per_shard[i] & per_shard[j]
    assert set().union(*per_shard.values()) == remaining
    assert all(per_shard.values())  # 33 items over 4 shards: none empty
    # Shards never touch the shared file the killed run wrote.
    assert (rescue / "rejudgments.jsonl").read_text() == main_before

    # Membership is by URL hash, not list position: reversing the input keeps it.
    reversed_state = tmp_path / "reversed"
    reversed_state.mkdir()
    for name in ("final_judgments.jsonl", "profile_summaries.jsonl"):
        lines = (state / name).read_text().splitlines()
        (reversed_state / name).write_text("\n".join(reversed(lines)) + "\n")
    fresh = make_rescue(tmp_path, [row(u, "REJECT") for u in judged_before],
                        name="fresh-rescue")
    calls["judged"].clear()
    assert run_args(monkeypatch, reversed_state, "--decisions", "all",
                    "--resume", fresh, "--shard", "2/4") == 0
    assert set(calls["judged"]) == per_shard[2]


def test_shard_requires_resume_so_every_shard_shares_one_criteria_set(
        monkeypatch, tmp_path, state_dir, calls):
    with pytest.raises(SystemExit):
        run_args(monkeypatch, state_dir, "--shard", "1/4")
    assert calls == {"preflight": 0, "judged": []}


@pytest.mark.parametrize("bad", ["0/4", "5/4", "1/0", "x/4", "2"])
def test_shard_rejects_malformed_specs(monkeypatch, tmp_path, state_dir, calls, bad):
    rescue = make_rescue(tmp_path, [])
    with pytest.raises(SystemExit):
        run_args(monkeypatch, state_dir, "--resume", rescue, "--shard", bad)
    assert calls == {"preflight": 0, "judged": []}


def test_merge_dedupes_with_last_real_decision_winning_and_rebuilds_passers(
        monkeypatch, tmp_path, calls):
    rescue = make_rescue(tmp_path, [
        row("u-a", "REJECT"),
        row("u-b", "RESCUE_ERROR"),
        row("u-c", "SAVE"),
    ])
    (rescue / "rejudgments.shard-1-of-2.jsonl").write_text(
        json.dumps(row("u-b", "TRANSFERABLE_SAVE")) + "\n")
    (rescue / "rejudgments.shard-2-of-2.jsonl").write_text("".join(json.dumps(r) + "\n" for r in [
        row("u-a", "RESCUE_ERROR"),  # an error never overwrites a real decision
        row("u-c", "REJECT"),        # a later real decision does
        row("u-d", "RESCUE_ERROR"),  # never judged for real: the error row survives
    ]))
    (rescue / "passers.json").write_text("[]")
    assert run_args(monkeypatch, "--merge", rescue) == 0
    assert calls == {"preflight": 0, "judged": []}
    merged = read_rows(rescue / "rejudgments.merged.jsonl")
    assert {r["profile_url"]: r["new_decision"] for r in merged} == {
        "u-a": "REJECT", "u-b": "TRANSFERABLE_SAVE", "u-c": "REJECT",
        "u-d": "RESCUE_ERROR"}
    assert len(merged) == 4
    passers = json.loads((rescue / "passers.json").read_text())
    assert [p["profile_url"] for p in passers] == ["u-b"]


def test_dry_run_with_resume_and_shard_counts_without_model_calls_or_writes(
        monkeypatch, tmp_path, state_dir, calls, capsys):
    loaded = []
    monkeypatch.setattr(rejudge, "load_brief", lambda path: loaded.append(path))
    rescue = make_rescue(tmp_path, [
        row("u-flagged", "REJECT"),
        row("u-reject-then-save", "RESCUE_ERROR"),
    ])
    before_rescue, before_state = snapshot(rescue), snapshot(state_dir)
    assert run_args(monkeypatch, state_dir, "--decisions", "all", "--resume", rescue,
                    "--shard", "1/2", "--dry-run") == 0
    assert calls == {"preflight": 0, "judged": []}
    assert loaded == []
    assert snapshot(rescue) == before_rescue
    assert snapshot(state_dir) == before_state
    printed = capsys.readouterr().out
    assert "6 candidates selected" in printed
    assert "already judged: 1" in printed
    assert "remaining: 5" in printed
    share = rejudge.shard_of
    remaining = ["u-reject-then-save", "u-save-then-reject", "u-flagged-then-reject",
                 "u-inferential", "u-no-profile"]
    mine = [u for u in remaining if share(u, 2) == 1]
    assert f"shard 1/2: {len(mine)} of 5 remaining" in printed


# ---------------------------------------------------------------------------
# The brief's deterministic exclusions run before any judge call
# ---------------------------------------------------------------------------


def test_exclusions_gate_before_judgment_and_territory_splits_the_passers(
    monkeypatch, tmp_path, calls
):
    """A re-judge of stored profiles must hold the same hard stops the live run
    does: ex-Babbage staff, current blacklisted-employer staff, and people on the
    skip list are GATED with no judge call; a passer in another recruiter's
    market goes to the hand-off file, never to the passers.json a save pass reads."""
    state = tmp_path / "state"
    state.mkdir()
    people = {
        "u-alum": ("Former Babbage", [("Program Manager", "Acme"), ("Delivery Lead", "Babbage")]),
        "u-current": ("Current OpenAI", [("TPM", "OpenAI")]),
        "u-skip": ("Already Contacted", [("TPM", "Acme")]),
        "u-hub": ("Hub Fit", [("TPM", "Scale AI")]),
        "u-home": ("Territory Fit", [("TPM", "Snorkel AI")]),
    }
    locations = {
        "u-hub": "San Francisco Bay Area",
        "u-home": "Columbus, Ohio, United States",
    }
    (state / "final_judgments.jsonl").write_text("".join(
        json.dumps({"profile_url": url, "candidate_name": name, "decision": "REJECT",
                    "rationale": "v1 gate"}) + "\n"
        for url, (name, _) in people.items()
    ))
    (state / "profile_summaries.jsonl").write_text("".join(
        json.dumps({"name": name, "profile_url": url, "headline": "operator",
                    "experiences": [{"title": t, "company": c} for t, c in history]}) + "\n"
        for url, (name, history) in people.items()
    ))
    (state / "snippets.jsonl").write_text("".join(
        json.dumps({"profile_url": url, "name": name,
                    "current_company": history[0][1],
                    "location": locations.get(url, "Denver, Colorado, United States")}) + "\n"
        for url, (name, history) in people.items()
    ))
    brief = SimpleNamespace(
        id="brief-test",
        employer_blacklist=["Babbage", "OpenAI", "Anthropic"],
        alumni_blacklist=["Babbage"],
        prior_search=SimpleNamespace(ruled_out_urls=["u-skip"]),
        raw={"excluded_locations": ["San Francisco Bay Area", "Texas"]},
    )
    monkeypatch.setattr(rejudge, "load_brief", lambda path: brief)

    assert run(monkeypatch, state, tmp_path / "out", "--decisions", "REJECT") == 0

    assert sorted(calls["judged"]) == ["u-home", "u-hub"]
    rows = results(tmp_path / "out")
    assert rows["u-alum"]["new_decision"] == "GATED"
    assert rows["u-alum"]["new_rationale"].startswith("alumni_blacklist: Babbage")
    assert rows["u-current"]["new_rationale"].startswith("employer_blacklist: OpenAI")
    assert rows["u-skip"]["new_rationale"].startswith("prior_search")
    (rescue_dir,) = (tmp_path / "out").iterdir()
    passers = json.loads((rescue_dir / "passers.json").read_text())
    elsewhere = json.loads((rescue_dir / "passers-excluded-location.json").read_text())
    assert [r["profile_url"] for r in passers] == ["u-home"]
    assert [(r["profile_url"], r["excluded_location"]) for r in elsewhere] == [
        ("u-hub", "San Francisco Bay Area")
    ]
