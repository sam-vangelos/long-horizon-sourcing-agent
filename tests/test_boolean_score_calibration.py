"""The calibration CLI (plans/boolean-scorer.md §7).

The tool is the only way anyone reads the scorer at corpus scale, so the
contract tested here is the printed shape, the read-only promise about
``output/``, and the arithmetic of the summary.
"""

from __future__ import annotations

import json
import statistics
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "score_boolean_corpus.py"
FIXTURES = ROOT / "tests" / "fixtures" / "boolean_corpus"
CORPUS = FIXTURES / "ml_runs_76.json"
LEXICON = ROOT / "linkedin" / "boolean_score" / "lexicons" / "research-engineer-post-training.json"

sys.path.insert(0, str(ROOT / "tools"))
import score_boolean_corpus as tool  # noqa: E402


def _run(*args) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(TOOL), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )


def test_tool_reports_the_corpus(tmp_path):
    out = tmp_path / "report.json"
    completed = _run(
        "--fixture", str(CORPUS), "--lexicon", str(LEXICON), "--out", str(out)
    )
    assert completed.returncode == 0, completed.stderr
    printed = completed.stdout
    assert "strings scored: 76" in printed
    for section in ("composite", "component", "highest", "lowest", "findings"):
        assert section in printed.lower()
    for name in ("register_coverage", "variant_coverage", "cohesion", "relevance", "parsimony"):
        assert name in printed
    payload = json.loads(out.read_text())
    assert payload["count"] == 76
    assert len(payload["reports"]) == 76
    assert payload["role"] == "research-engineer-post-training"
    assert set(payload["summary"]["components"]) == {
        "register_coverage",
        "variant_coverage",
        "cohesion",
        "relevance",
        "structure",
        "parsimony",
    }


def test_tool_shows_five_highest_and_five_lowest(tmp_path):
    completed = _run("--fixture", str(CORPUS), "--lexicon", str(LEXICON))
    assert completed.returncode == 0, completed.stderr
    body = completed.stdout
    highest = body.split("highest")[1].split("lowest")[0]
    assert highest.count("composite") == 5


def test_tool_refuses_to_write_under_output(tmp_path):
    completed = _run(
        "--fixture", str(CORPUS),
        "--lexicon", str(LEXICON),
        "--out", "output/state/linkedin/should-never-happen.json",
    )
    assert completed.returncode != 0
    assert "output/" in (completed.stderr + completed.stdout)
    assert not (
        ROOT / "output" / "state" / "linkedin" / "should-never-happen.json"
    ).exists()


def test_tool_reads_generated_strings_out_of_an_execution_plan(tmp_path):
    plan_dir = tmp_path / "run-one"
    plan_dir.mkdir()
    (plan_dir / "execution_plan.json").write_text(
        json.dumps(
            {
                "strategy_rationale": "synthetic",
                "generated_strings": [
                    {"id": 1, "boolean": '("RLHF" OR "post-training")'},
                    {"id": 2, "boolean": '("evaluation" OR "evals")'},
                    {"id": 3, "notes": "no boolean here"},
                ],
            }
        )
    )
    entries = tool.read_plans(str(tmp_path / "*" / "execution_plan.json"))
    assert [entry["boolean"] for entry in entries] == [
        '("RLHF" OR "post-training")',
        '("evaluation" OR "evals")',
    ]
    assert entries[0]["id"].startswith("run-one")


def test_summary_arithmetic():
    lexicon = tool.load_lexicon(LEXICON)
    entries = tool.read_fixture(CORPUS)
    reports = [tool.score(entry["boolean"], lexicon) for entry in entries]
    summary = tool.summarize(reports)
    composites = sorted(report.composite for report in reports)
    assert summary["count"] == len(reports)
    assert summary["composite"]["mean"] == pytest.approx(
        sum(composites) / len(composites)
    )
    assert summary["composite"]["min"] == pytest.approx(composites[0])
    assert summary["composite"]["max"] == pytest.approx(composites[-1])
    assert (
        summary["composite"]["q1"]
        <= summary["composite"]["median"]
        <= summary["composite"]["q3"]
    )
    applicable = sum(
        1 for report in reports if report.components["variant_coverage"] is not None
    )
    assert summary["components"]["variant_coverage"]["applicable"] == applicable
    assert summary["components"]["variant_coverage"]["applicability_rate"] == pytest.approx(
        applicable / len(reports)
    )


def test_finding_frequencies_are_counted():
    lexicon = tool.load_lexicon(LEXICON)
    reports = [
        tool.score(entry["boolean"], lexicon) for entry in tool.read_fixture(CORPUS)
    ]
    summary = tool.summarize(reports)
    counted = sum(summary["findings"].values())
    emitted = sum(len(report.findings) for report in reports)
    assert counted == emitted


def test_missing_fixture_is_an_error(tmp_path):
    completed = _run("--fixture", str(tmp_path / "nope.json"), "--lexicon", str(LEXICON))
    assert completed.returncode != 0


# --- agreement with the blind raters (§7) -----------------------------------

RATERS = FIXTURES / "rater_targets.json"
OVERRIDES = FIXTURES / "sam_overrides.json"


def _spearman(left, right) -> float:
    def ranked(values):
        order = sorted(range(len(values)), key=lambda index: values[index])
        out = [0.0] * len(values)
        for position, index in enumerate(order):
            out[index] = float(position)
        return out

    return statistics.correlation(ranked(left), ranked(right))


def _rater_rows() -> list[dict]:
    rows = json.loads(RATERS.read_text())["strings"]
    if OVERRIDES.exists():
        replacement = {
            row["id"]: row["target"] for row in json.loads(OVERRIDES.read_text())
        }
        for row in rows:
            if row["id"] in replacement:
                row["target"] = replacement[row["id"]]
    return rows


def _rater_agreement() -> dict[str, float]:
    lexicon = tool.load_lexicon(LEXICON)
    rows = _rater_rows()
    composites = [tool.score(row["boolean"], lexicon).composite for row in rows]
    return {
        key: _spearman(composites, [row[key] for row in rows])
        for key in ("target", "fable", "opus")
    }


def test_rater_agreement_is_reported(capsys):
    """Agreement against each rater separately, for the record."""
    agreement = _rater_agreement()
    with capsys.disabled():
        print(
            "\n  Spearman(scorer, target) = {target:.4f}"
            "\n  Spearman(scorer, fable)  = {fable:.4f}"
            "\n  Spearman(scorer, opus)   = {opus:.4f}".format(**agreement)
        )
    assert set(agreement) == {"target", "fable", "opus"}


# The 52-string correlation measures CROSS-string ranking, which is dominated
# by topic choice and rater taste. A reward's job is to rank completions of
# one prompt against each other, so pairwise preference is the acceptance
# test and this is a regression floor: it reports the number and fails only
# if the scorer goes backwards.
RATER_AGREEMENT_FLOOR_SOFT = 0.55


def test_rater_agreement_does_not_regress():
    agreement = _rater_agreement()["target"]
    assert agreement >= RATER_AGREEMENT_FLOOR_SOFT, (
        f"cross-string agreement fell to {agreement:.4f}; the soft floor is "
        f"{RATER_AGREEMENT_FLOOR_SOFT}. See §9 of plans/boolean-scorer.md."
    )


# --- pairwise preference: the acceptance test a reward actually needs ------

sys.path.insert(0, str(ROOT / "tools"))
import make_boolean_pairs as pairs_tool  # noqa: E402

PAIRS = FIXTURES / "pairs_v1.json"
VERDICTS = FIXTURES / "pairs_v1_rater_verdicts.json"
OPERATOR_FLOOR = 0.80
# Operators whose expected direction is a tie, so the floor does not apply.
TIE_OPERATOR_NAMES = {"reorder_terms", "add_more_real_forms_beyond_common"}
# Measured disagreements, each a finding rather than a tuning target; see §9.
KNOWN_DISAGREEMENTS = {
    "inject_noisy_token": (
        "4/6. A noisy token added to a sweep raises sweep completeness by "
        "more than the 0.10 charge costs, so the richer-but-noisier side wins."
    ),
    "remove_anchor_group": (
        "2/6. The shape multiplier closed a third of this. The rest is that "
        "cohesion is a mean over groups and relevance a min over them, so "
        "both still improve when the weakest group goes."
    ),
}


def _pairs() -> list[dict]:
    return json.loads(PAIRS.read_text())["pairs"]


def _operator_names() -> list[str]:
    return [name for name, _operator, _direction in pairs_tool.OPERATORS]


@pytest.mark.structural
def test_the_pair_fixture_is_balanced_and_keyed():
    rows = _pairs()
    assert 70 <= len(rows) <= 90
    counts = {name: 0 for name in _operator_names()}
    for row in rows:
        assert set(row) >= {"pair_id", "A", "B", "base_id", "edit", "expected"}
        assert row["expected"] in {"A", "B", "tie"}
        assert row["A"] != row["B"]
        counts[row["edit"]] += 1
    assert all(count > 0 for count in counts.values()), counts
    assert max(counts.values()) - min(counts.values()) <= 2


@pytest.mark.parametrize(
    "operator",
    [
        pytest.param(
            name,
            marks=pytest.mark.xfail(strict=True, reason=KNOWN_DISAGREEMENTS[name]),
        )
        if name in KNOWN_DISAGREEMENTS
        else name
        for name in [
            name for name, _o, _d in pairs_tool.OPERATORS if name not in TIE_OPERATOR_NAMES
        ]
    ],
)
def test_the_scorer_prefers_the_side_the_edit_intends(operator):
    lexicon = tool.load_lexicon(LEXICON)
    rows = [row for row in _pairs() if row["edit"] == operator]
    assert rows
    hits = sum(
        1
        for row in rows
        if pairs_tool.scorer_preference(row, lexicon) == row["expected"]
    )
    assert hits / len(rows) >= OPERATOR_FLOOR, f"{hits}/{len(rows)}"


def test_reordering_a_group_changes_nothing():
    lexicon = tool.load_lexicon(LEXICON)
    rows = [row for row in _pairs() if row["edit"] == "reorder_terms"]
    assert rows
    for row in rows:
        assert pairs_tool.scorer_preference(row, lexicon) == "tie", row["pair_id"]


RATER_PAIR_FLOOR = 0.85


def _verdicts() -> dict[str, str]:
    payload = json.loads(VERDICTS.read_text())
    return {row["pair_id"]: row["verdict"] for row in payload["verdicts"]}


def _pair_accuracy(lexicon) -> tuple[int, int, dict[str, tuple[int, int]]]:
    verdicts = _verdicts()
    per_operator: dict[str, list[int]] = {}
    hits = total = 0
    for row in _pairs():
        verdict = verdicts.get(row["pair_id"])
        if verdict is None:
            continue
        agreed = pairs_tool.scorer_preference(row, lexicon) == verdict
        per_operator.setdefault(row["edit"], []).append(1 if agreed else 0)
        hits += int(agreed)
        total += 1
    return hits, total, {
        name: (sum(values), len(values)) for name, values in per_operator.items()
    }


@pytest.mark.structural
def test_the_tie_band_is_narrow_enough_to_see_one_term():
    assert pairs_tool.TIE_BAND == pytest.approx(0.01)


@pytest.mark.parametrize("pair_id", ["P013", "P019", "P061", "P049"])
def test_one_term_is_not_a_tie(pair_id):
    """A single umbrella word, rare expansion, hyphen twin or sentence
    fragment moves the composite by more than the tie band."""
    lexicon = tool.load_lexicon(LEXICON)
    row = next(item for item in _pairs() if item["pair_id"] == pair_id)
    assert pairs_tool.scorer_preference(row, lexicon) != "tie"


@pytest.mark.xfail(
    strict=True,
    reason=(
        "3/6. Both raters said the richer string wins every time, and the "
        "uncapped occasional credit and the log-space reading recovered half "
        "of it. The rest is that some added real forms are themselves "
        "umbrella terms or trip the swallowed-group charge. See §9."
    ),
)
def test_more_real_forms_of_a_present_concept_win():
    lexicon = tool.load_lexicon(LEXICON)
    rows = [row for row in _pairs() if row["edit"] == "add_more_real_forms_beyond_common"]
    assert len(rows) == 6
    for row in rows:
        richer, leaner = row["edited_side"], "A" if row["edited_side"] == "B" else "B"
        gain = score_of(row[richer], lexicon) - score_of(row[leaner], lexicon)
        assert gain >= 0.005, (row["pair_id"], round(gain, 4))


def score_of(boolean: str, lexicon) -> float:
    return tool.score(boolean, lexicon).composite


def test_the_scorer_agrees_with_the_raters_on_edit_pairs(capsys):
    """The acceptance test: a reward is judged on preference, not on
    correlation across unrelated strings."""
    lexicon = tool.load_lexicon(LEXICON)
    hits, total, per_operator = _pair_accuracy(lexicon)
    with capsys.disabled():
        print(f"\n  pair accuracy {hits}/{total} = {hits / total:.3f}")
        for name in sorted(per_operator):
            got, count = per_operator[name]
            print(f"    {name:36s} {got}/{count}")
    assert hits / total >= RATER_PAIR_FLOOR, f"{hits}/{total}"


@pytest.mark.structural
def test_the_rater_fixture_covers_the_in_role_strings():
    rows = json.loads(RATERS.read_text())["strings"]
    assert len(rows) == 52
    for row in rows:
        assert 0.0 <= row["target"] <= 1.0
        assert row["fable_rationale"] and row["opus_rationale"]


@pytest.mark.structural
def test_tool_declares_its_read_only_promise():
    source = TOOL.read_text()
    assert "read-only" in source.lower()
    assert "output/" in source
