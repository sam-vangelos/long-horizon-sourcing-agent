from shared.run_report_schema import (
    RunDebriefAnalysis,
    StructuredRunReport,
    render_run_report_markdown,
)


def _snapshot() -> dict:
    return {
        "schema_version": 1,
        "run_metadata": {
            "role_title": "Head of Applied AI Lab",
            "brief_name": "head-ai",
            "brief_version": "2.1",
            "linkedin_project": "Head of Applied AI Lab",
            "linkedin_project_id": "3000000006",
            "generated_at": "2026-04-06T12:00:00+00:00",
            "overall_summary": "Strong BFSI run with clear winning lanes.",
        },
        "metrics_summary": {
            "strings_executed": 8,
            "strings_skipped": 9,
            "total_results": 12137,
            "total_pages_reviewed": 25,
            "candidates_evaluated": 295,
            "facial_yes": 74,
            "facial_no": 221,
            "saved": 14,
            "rejected": 57,
            "overall_save_rate": 0.047,
            "facial_yes_rate": 0.251,
        },
        "string_performance": [
            {
                "string_id": 2,
                "name": "Research copilot lane",
                "status": "done",
                "result_count": 526,
                "pages_reviewed": 4,
                "saves": 8,
                "save_rate": 0.08,
                "saved_candidates": ["Mithun Azhagappan"],
                "notes": "Strong lane",
                "family_key": "research_copilot_asset_mgmt",
                "novelty_bucket": "edge_case",
                "domain_lane": "asset_management",
            }
        ],
    }


def _analysis_dict() -> dict:
    return {
        "winning_lanes": [
            {
                "lane": "Research Copilot",
                "string_ids": [2],
                "candidate_examples": ["Mithun Azhagappan"],
                "evidence": "Highest absolute save count.",
                "why_it_worked": "Workflow-specific product language gated for real builders.",
                "recommended_action": "Promote this lane early.",
            }
        ],
        "underperforming_lanes": [
            {
                "lane": "Surveillance",
                "string_ids": [8],
                "issue": "Traditional rules-engine noise.",
                "evidence": "Zero saves across two pages.",
                "recommended_action": "Only run with explicit GenAI AND-gate.",
            }
        ],
        "coverage_gaps": [
            {
                "gap": "Payments",
                "why_it_matters": "No explicit payments string was run.",
                "suggested_search_strategy": "Add transaction-banking and payment-orchestration strings.",
            }
        ],
        "noise_patterns": [
            {
                "pattern": "Product leadership without builder depth",
                "evidence": "Multiple CPTO and product-heavy AI officer profiles rejected.",
                "mitigation": "Strengthen builder verb gating.",
            }
        ],
        "saved_candidate_patterns": {
            "standout_candidates": [{"name": "Mithun Azhagappan", "why": "Goldman AI platform architect."}],
            "common_employers": [{"employer": "JPMorgan", "count": 5, "note": "Strong GenAI convert population."}],
            "common_titles": [{"title_family": "Executive Director", "count": 2, "note": "Right seniority band."}],
            "archetype_distribution": [{"archetype": "BFSI-native GenAI converts", "count": 6, "note": "Most common save type."}],
            "seniority_notes": ["Many VP-level bank builders were interesting but below full lab-leadership scope."],
        },
        "adaptation_assessment": {
            "summary": "Adaptation concentrated effort on productive workflow language.",
            "effective_refinements": ["Narrowing workflow strings improved precision."],
            "questionable_or_skipped": ["Regulatory reporting was skipped and should return."],
            "operational_notes": ["Keep tight strings over broad archetype-first nets."],
        },
        "recommendations": {
            "try_next": ["Payments and transaction-banking builders"],
            "avoid_next": ["Ungated surveillance strings"],
            "prioritize_pipeline": ["Engage Mithun Azhagappan immediately"],
        },
        "brief_iteration_hints": {
            "instructions": ["Cover payments and regulatory reporting in the first block."],
            "search_priorities": ["Payments and transaction-banking builders"],
            "additional_search_terms": ["payment orchestration", "transaction banking"],
            "intake_notes": "The latest run validated research-copilot lanes and exposed a payments gap.",
            "depth_distinction": {
                "builder_definition": "Still a BFSI executive-builder role.",
                "user_definition": "Product and strategy leaders remain non-fits.",
                "edge_case_guidance": "Bank VP profiles require extra scope scrutiny.",
            },
            "non_fit_patterns": [
                {
                    "label": "Product-heavy AI officer",
                    "description": "Executive AI product leadership without builder authorship.",
                    "why_not": "Wrong depth for the role.",
                    "examples": ["Chief Product & AI Officer"],
                }
            ],
            "minimum_bar_description": "NYC, 15+ years, BFSI, and post-2022 GenAI remain hard requirements.",
            "facial_calibration": {
                "expected_yes_rate_low": 0.1,
                "expected_yes_rate_high": 0.22,
                "fast_exit_patterns": ["Pure product history"],
                "trajectory_yes_patterns": ["Big-bank GenAI convert"],
                "trajectory_ambiguous_patterns": ["VP at smaller firm"],
                "trajectory_no_patterns": ["Vendor field CTO without build ownership"],
            },
            "employer_signal_rules": [
                {
                    "tier": "payments_builder",
                    "employer_patterns": ["Visa", "Mastercard"],
                    "evidence_required": "Still requires production builder evidence.",
                    "save_on_employer_alone": False,
                }
            ],
            "calibration_examples": {
                "strong_saves": [{"name": "Mithun Azhagappan", "why": "Strong fit."}],
                "incorrect_saves": [{"name": "Deepinder Gulati", "why": "Product-heavy."}],
                "borderline_verify": [{"name": "Peter Chung", "why": "Check scope carefully."}],
            },
            "notes": "Promote payments in the next revision.",
            "locked_field_cautions": ["Do not relax geography or years-of-experience gates."],
        },
    }


def test_run_report_schema_round_trips_and_renders_markdown():
    analysis = RunDebriefAnalysis.from_dict(_analysis_dict())
    report = StructuredRunReport.from_parts(_snapshot(), analysis)
    round_tripped = StructuredRunReport.from_dict(report.to_dict())

    markdown = render_run_report_markdown(round_tripped)

    assert round_tripped.run_metadata["role_title"] == "Head of Applied AI Lab"
    assert round_tripped.winning_lanes[0]["lane"] == "Research Copilot"
    assert "Mithun Azhagappan" in markdown
    assert "Payments" in markdown
    assert "Ungated surveillance strings" in markdown


def test_run_debrief_analysis_rejects_missing_required_keys():
    try:
        RunDebriefAnalysis.from_dict({"winning_lanes": []})
    except ValueError as exc:
        assert "missing keys" in str(exc)
    else:
        raise AssertionError("Expected RunDebriefAnalysis.from_dict to reject incomplete data")


# ---------------------------------------------------------------------------
# P4.2/P4.3/P4.5 — deterministic markdown rendering for the new snapshot
# blocks (cost_summary, run_health, adaptation_roi, facial_calibration).
# Every block must degrade to silence on missing/non-"ok" status — no
# affirmative zeros, no empty scaffolding sections.
# ---------------------------------------------------------------------------


def _report_with_metrics(extra_metrics: dict) -> StructuredRunReport:
    snapshot = _snapshot()
    snapshot["metrics_summary"] = {**snapshot["metrics_summary"], **extra_metrics}
    analysis = RunDebriefAnalysis.from_dict(_analysis_dict())
    return StructuredRunReport.from_parts(snapshot, analysis)


def test_markdown_omits_all_new_sections_when_blocks_absent():
    report = _report_with_metrics({})
    markdown = render_run_report_markdown(report)

    assert "## Run Cost" not in markdown
    assert "## Run Health" not in markdown
    assert "## Facial Calibration" not in markdown
    assert "### Measured Adaptation ROI" not in markdown
    assert "## Boolean Craft" not in markdown


def test_markdown_renders_constraint_manifest_when_present():
    """P3b (Wave 2): the manifest renders per-class ownership plus the
    defer-dimension counter; absent metrics render nothing."""
    report = _report_with_metrics(
        {
            "constraint_manifest": {
                "schema_version": 1,
                "classes": {
                    "geography": {
                        "stated_in_brief": True,
                        "stated_value": "New York City Metropolitan Area",
                        "owner_layer": "session filter",
                        "actuator": "browser.apply_location_filter",
                        "verify_method": "applied-chip confirmation",
                        "status": "owned",
                    },
                    "compensation": {
                        "stated_in_brief": False,
                        "stated_value": "",
                        "owner_layer": "",
                        "actuator": "",
                        "verify_method": "",
                        "status": "unstated",
                    },
                },
                "requested_but_unsupported": {"seniority": 3},
            }
        }
    )
    markdown = render_run_report_markdown(report)

    assert "## Constraint Manifest" in markdown
    assert "| geography | yes | owned | session filter |" in markdown
    assert "seniority×3" in markdown


def test_markdown_omits_constraint_manifest_when_absent():
    markdown = render_run_report_markdown(_report_with_metrics({}))
    assert "## Constraint Manifest" not in markdown


def test_markdown_renders_session_geography_receipt():
    """P3b (Wave 2): 'what pool did this run search' is a recorded fact in the
    report header, with the reassert count."""
    snapshot = _snapshot()
    snapshot["run_metadata"]["session_geography"] = {
        "intended": ["New York City Metropolitan Area"],
        "verified_applied": True,
        "reasserts": 1,
    }
    analysis = RunDebriefAnalysis.from_dict(_analysis_dict())
    report = StructuredRunReport.from_parts(snapshot, analysis)

    markdown = render_run_report_markdown(report)

    assert "Geography: New York City Metropolitan Area — verified applied, 1 re-assert(s)." in markdown

    plain = render_run_report_markdown(
        StructuredRunReport.from_parts(_snapshot(), analysis)
    )
    assert "Geography:" not in plain


def test_markdown_renders_unverified_geography_and_no_reassert_suffix():
    """Pin the negative branches: a receipt that never verified renders 'NOT
    verified' loudly, and zero reasserts renders no suffix (contract lens,
    slice 8 — these branches were dead-but-unpinned)."""
    snapshot = _snapshot()
    snapshot["run_metadata"]["session_geography"] = {
        "intended": ["Berlin Metropolitan Area"],
        "verified_applied": False,
        "reasserts": 0,
    }
    analysis = RunDebriefAnalysis.from_dict(_analysis_dict())
    markdown = render_run_report_markdown(
        StructuredRunReport.from_parts(snapshot, analysis)
    )

    assert "Geography: Berlin Metropolitan Area — NOT verified." in markdown
    assert "re-assert" not in markdown


def test_markdown_renders_lint_blocked_section_when_present():
    """P5 (Wave 2): strings refused at queue build render with their codes and
    repair hints — the run report is the lint's report-time consumer."""
    report = _report_with_metrics(
        {
            "lint_blocked": [
                {
                    "source": "generated",
                    "name": "too broad",
                    "family_key": "generic",
                    "boolean": '("AI") AND ("Engineer")',
                    "codes": ["ubiquitous_and_gate"],
                    "messages": ["boolean failed the ubiquitous-term AND-gate"],
                    "repair_hints": ["Anchor at least one AND group on a specific term."],
                }
            ]
        }
    )
    markdown = render_run_report_markdown(report)

    assert "## Boolean Craft" in markdown
    assert "1 string(s) blocked at queue build" in markdown
    assert "ubiquitous_and_gate" in markdown
    assert "Anchor at least one AND group" in markdown


def test_markdown_renders_cost_section_when_present():
    report = _report_with_metrics(
        {"cost_summary": {"status": "ok", "cost_usd": 12.3456, "cost_per_save_usd": 0.8811}}
    )
    markdown = render_run_report_markdown(report)

    assert "## Run Cost" in markdown
    assert "$12.3456" in markdown
    assert "$0.8811" in markdown


def test_markdown_omits_cost_per_save_line_when_no_saves():
    report = _report_with_metrics(
        {"cost_summary": {"status": "ok", "cost_usd": 5.0, "cost_per_save_usd": None}}
    )
    markdown = render_run_report_markdown(report)

    assert "## Run Cost" in markdown
    assert "$5.0000" in markdown
    assert "Cost per save" not in markdown


# ---------------------------------------------------------------------------
# GLM-5.2 shadow-judge section — only-when-present discipline, no
# fabricated 0.0%/n/a inversions.
# ---------------------------------------------------------------------------


def test_markdown_omits_shadow_judge_section_when_absent():
    report = _report_with_metrics({})
    markdown = render_run_report_markdown(report)

    assert "## Shadow Judge" not in markdown


def test_markdown_renders_shadow_judge_section_when_present():
    report = _report_with_metrics(
        {
            "shadow_facial": {
                "model": "accounts/fireworks/models/glm-5p2",
                "comparisons": 42,
                "agreement_rate": 0.75,
                "shadow_parse_failure_rate": 0.05,
                "primary_yes_rate": 0.3,
                "shadow_yes_rate": 0.28,
                "mean_latency_ms": 812.4,
            }
        }
    )
    markdown = render_run_report_markdown(report)

    assert "## Shadow Judge" in markdown
    assert "accounts/fireworks/models/glm-5p2" in markdown
    assert "42" in markdown
    assert "75.0%" in markdown
    assert "5.0%" in markdown
    assert "30.0%" in markdown
    assert "28.0%" in markdown
    assert "812ms" in markdown


def test_markdown_renders_na_for_undefined_shadow_rates_not_fabricated_zero():
    report = _report_with_metrics(
        {
            "shadow_facial": {
                "model": "accounts/fireworks/models/glm-5p2",
                "comparisons": 1,
                "agreement_rate": None,
                "shadow_parse_failure_rate": None,
                "primary_yes_rate": 1.0,
                "shadow_yes_rate": None,
                "mean_latency_ms": None,
            }
        }
    )
    markdown = render_run_report_markdown(report)

    assert "## Shadow Judge" in markdown
    # None-valued rates render as "n/a", never a fabricated "0.0%".
    lines = [l for l in markdown.splitlines() if "Agreement rate" in l]
    assert lines and "n/a" in lines[0]


def test_markdown_omits_cache_hit_rate_line_when_absent():
    """mean_cache_hit_rate key entirely absent (not None) -> no line at
    all, never a fabricated 0.0%."""
    report = _report_with_metrics(
        {
            "shadow_facial": {
                "model": "accounts/fireworks/models/glm-5p2",
                "comparisons": 5,
                "agreement_rate": 0.8,
                "shadow_parse_failure_rate": 0.0,
                "primary_yes_rate": 0.4,
                "shadow_yes_rate": 0.4,
                "mean_latency_ms": 500.0,
            }
        }
    )
    markdown = render_run_report_markdown(report)

    assert "Mean cache-hit rate" not in markdown


def test_markdown_renders_cache_hit_rate_line_when_present():
    report = _report_with_metrics(
        {
            "shadow_facial": {
                "model": "accounts/fireworks/models/glm-5p2",
                "comparisons": 5,
                "agreement_rate": 0.8,
                "shadow_parse_failure_rate": 0.0,
                "primary_yes_rate": 0.4,
                "shadow_yes_rate": 0.4,
                "mean_latency_ms": 500.0,
                "mean_cache_hit_rate": 0.62,
            }
        }
    )
    markdown = render_run_report_markdown(report)

    lines = [l for l in markdown.splitlines() if "Mean cache-hit rate" in l]
    assert lines and "62.0%" in lines[0]


# ---------------------------------------------------------------------------
# GLM-5.2 shadow-judge FULL-EVAL sub-block — same only-when-present
# discipline as the facial block, independent presence.
# ---------------------------------------------------------------------------


def test_markdown_omits_full_eval_shadow_block_when_absent():
    report = _report_with_metrics({})
    markdown = render_run_report_markdown(report)

    assert "Full Evaluation (shadow)" not in markdown


def test_markdown_renders_full_eval_shadow_block_when_present():
    report = _report_with_metrics(
        {
            "shadow_full": {
                "model": "accounts/fireworks/models/glm-5p2",
                "comparisons": 18,
                "agreement_rate": 0.8,
                "shadow_parse_failure_rate": 0.0,
                "primary_save_rate": 0.4,
                "shadow_save_rate": 0.38,
                "mean_latency_ms": 950.4,
            }
        }
    )
    markdown = render_run_report_markdown(report)

    assert "Full Evaluation (shadow)" in markdown
    assert "accounts/fireworks/models/glm-5p2" in markdown
    assert "18" in markdown
    assert "80.0%" in markdown
    assert "0.0%" in markdown
    assert "40.0%" in markdown
    assert "38.0%" in markdown
    assert "950ms" in markdown


def test_markdown_renders_full_eval_shadow_block_independent_of_facial_presence():
    """A run can have full-eval shadow comparisons with zero facial ones
    (or vice versa) — each block's presence is independently gated."""
    report = _report_with_metrics(
        {
            "shadow_full": {
                "model": "accounts/fireworks/models/glm-5p2",
                "comparisons": 3,
                "agreement_rate": None,
                "shadow_parse_failure_rate": None,
                "primary_save_rate": 1.0,
                "shadow_save_rate": None,
                "mean_latency_ms": None,
            }
        }
    )
    markdown = render_run_report_markdown(report)

    assert "## Shadow Judge" not in markdown
    assert "Full Evaluation (shadow)" in markdown
    lines = [l for l in markdown.splitlines() if "Agreement rate" in l]
    assert lines and "n/a" in lines[0]


def test_markdown_omits_cost_section_when_no_cost_data():
    report = _report_with_metrics({"cost_summary": {"status": "no_cost_data"}})
    markdown = render_run_report_markdown(report)

    assert "## Run Cost" not in markdown
    # Never an affirmative $0.00 when cost is unknown.
    assert "$0.00" not in markdown


def test_markdown_renders_run_health_quiet_line_when_healthy():
    report = _report_with_metrics(
        {
            "run_health": {
                "status": "ok",
                "degraded": False,
                "degraded_reasons": [],
            }
        }
    )
    markdown = render_run_report_markdown(report)

    assert "## Run Health" in markdown
    assert "Degraded: No" in markdown


def test_markdown_renders_run_health_reasons_when_degraded():
    report = _report_with_metrics(
        {
            "run_health": {
                "status": "ok",
                "degraded": True,
                "degraded_reasons": ["green_but_useless"],
            }
        }
    )
    markdown = render_run_report_markdown(report)

    assert "## Run Health" in markdown
    assert "Degraded: Yes" in markdown
    assert "green_but_useless" in markdown


def test_markdown_omits_run_health_when_no_runtime_state():
    report = _report_with_metrics({"run_health": {"status": "no_runtime_state"}})
    markdown = render_run_report_markdown(report)

    assert "## Run Health" not in markdown


def test_markdown_renders_adaptation_roi_numbers_when_events_present():
    report = _report_with_metrics(
        {
            "adaptation_roi": {
                "status": "ok",
                "events": [{"block": "block_1"}],
                "total_inserted_saves": 5,
                "total_displaced_saves": 2,
                "net_saves_gained": 3,
            }
        }
    )
    markdown = render_run_report_markdown(report)

    assert "### Measured Adaptation ROI" in markdown
    assert "Inserted saves: 5" in markdown
    assert "Displaced saves: 2" in markdown
    assert "Net saves gained: 3" in markdown


def test_markdown_omits_adaptation_roi_numbers_when_no_events():
    report = _report_with_metrics({"adaptation_roi": {"status": "no_adaptation_events"}})
    markdown = render_run_report_markdown(report)

    assert "### Measured Adaptation ROI" not in markdown


def test_markdown_renders_facial_calibration_line_when_ok():
    report = _report_with_metrics(
        {
            "facial_calibration": {
                "status": "ok",
                "actual_yes_rate": 0.18,
                "authored_low": 0.1,
                "authored_high": 0.22,
                "deviation_from_band": 0.0,
                "out_of_band": False,
                "calibration_drift_warning": False,
            }
        }
    )
    markdown = render_run_report_markdown(report)

    assert "## Facial Calibration" in markdown
    assert "18.0%" in markdown
    assert "10.0%" in markdown
    assert "22.0%" in markdown
    assert "DRIFT" not in markdown


def test_markdown_renders_facial_calibration_drift_warning_when_true():
    report = _report_with_metrics(
        {
            "facial_calibration": {
                "status": "ok",
                "actual_yes_rate": 0.45,
                "authored_low": 0.1,
                "authored_high": 0.22,
                "deviation_from_band": 0.23,
                "out_of_band": True,
                "calibration_drift_warning": True,
            }
        }
    )
    markdown = render_run_report_markdown(report)

    assert "## Facial Calibration" in markdown
    assert "DRIFT" in markdown


def test_markdown_omits_facial_calibration_when_no_verdicts():
    report = _report_with_metrics({"facial_calibration": {"status": "no_facial_verdicts"}})
    markdown = render_run_report_markdown(report)

    assert "## Facial Calibration" not in markdown


def test_markdown_omits_facial_calibration_when_band_not_authored():
    report = _report_with_metrics(
        {"facial_calibration": {"status": "band_not_authored", "actual_yes_rate": 0.3}}
    )
    markdown = render_run_report_markdown(report)

    assert "## Facial Calibration" not in markdown


# ---------------------------------------------------------------------------
# P9.3 — unreviewed-brief marker in run_metadata. Cross-reference: P9.5's
# brief_loader detects intake-born briefs by absence-en-bloc of calibration
# fields; this provenance stamp is the FUTURE positive detector — a brief
# that carries it is definitively machine-authored and definitively
# unreviewed, no inference required.
# ---------------------------------------------------------------------------


def _report_with_run_metadata(extra_run_metadata: dict) -> StructuredRunReport:
    snapshot = _snapshot()
    snapshot["run_metadata"] = {**snapshot["run_metadata"], **extra_run_metadata}
    analysis = RunDebriefAnalysis.from_dict(_analysis_dict())
    return StructuredRunReport.from_parts(snapshot, analysis)


def test_markdown_omits_unreviewed_marker_when_provenance_absent():
    report = _report_with_run_metadata({})
    markdown = render_run_report_markdown(report)

    assert "UNREVIEWED" not in markdown


def test_markdown_renders_unreviewed_marker_when_reviewed_is_false():
    report = _report_with_run_metadata(
        {
            "provenance": {
                "generated_by": "preflight_v2",
                "reviewed": False,
                "generated_at": "2026-07-02T00:00:00+00:00",
            }
        }
    )
    markdown = render_run_report_markdown(report)

    assert "UNREVIEWED" in markdown
    assert "preflight_v2" in markdown


def test_markdown_omits_unreviewed_marker_when_reviewed_is_true():
    report = _report_with_run_metadata(
        {
            "provenance": {
                "generated_by": "preflight_v2",
                "reviewed": True,
                "generated_at": "2026-07-02T00:00:00+00:00",
            }
        }
    )
    markdown = render_run_report_markdown(report)

    assert "UNREVIEWED" not in markdown
