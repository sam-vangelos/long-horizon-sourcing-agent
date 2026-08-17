"""
Integration example — how the reusable components wire into the orchestrator.

This is NOT a complete orchestrator. It shows the integration points where
brief_schema, judgment_templates, and bias_controls plug into the existing
pipeline loop from orchestrator.py.

The key principle: the orchestrator NEVER constructs evaluation prompts directly.
It calls assembly functions that inject Brief content into structural templates.
The orchestrator also NEVER passes cumulative session stats to the evaluator —
that's the bias controls' job.
"""

import json
from pathlib import Path

# --- Existing imports from your codebase ---
# from browser import Browser
# from extractors import extract_snippets, extract_full_profile
# from llm_clients import opus_client, cheap_client
# from storage import append_jsonl

# --- New imports: reusable components ---
from shared.brief_schema import Brief, CapabilityArea, DepthDistinction, NonFitPattern, \
    EmployerSignalRule, FacialCalibration, BiasControls, MarketDensity
from linkedin.judgment_templates import (
    assemble_facial_prompt,
    assemble_facial_prompt_batch,
    assemble_full_evaluation_prompt,
    parse_facial_response,
    parse_full_evaluation_response,
    FacialResult,
    FullEvaluationResult,
)
from shared.bias_controls import BiasMonitor, DecisionRecord


# ---------------------------------------------------------------------------
# BRIEF LOADING
# ---------------------------------------------------------------------------
# Load the brief JSON into the Brief dataclass.
# This replaces the old brief_loader.py normalization with the new schema.
# ---------------------------------------------------------------------------

def load_brief(path: str) -> Brief:
    """Load a brief JSON file into the Brief dataclass."""
    data = json.loads(Path(path).read_text())

    capability_areas = [
        CapabilityArea(
            name=ca["name"],
            description=ca["description"],
            builder_signals=ca["builder_signals"],
            user_signals=ca.get("user_signals", []),
            key_terms=ca.get("key_terms", []),
        )
        for ca in data["capability_areas"]
    ]

    depth = DepthDistinction(
        builder_definition=data["depth_distinction"]["builder_definition"],
        user_definition=data["depth_distinction"]["user_definition"],
        edge_case_guidance=data["depth_distinction"]["edge_case_guidance"],
    )

    non_fit_patterns = [
        NonFitPattern(
            label=nf["label"],
            description=nf["description"],
            why_not=nf["why_not"],
            examples=nf.get("examples", []),
        )
        for nf in data["non_fit_patterns"]
    ]

    employer_rules = [
        EmployerSignalRule(
            tier=er["tier"],
            employer_patterns=er["employer_patterns"],
            evidence_required=er["evidence_required"],
            save_on_employer_alone=er.get("save_on_employer_alone", False),
        )
        for er in data["employer_signal_rules"]
    ]

    facial = FacialCalibration(
        expected_yes_rate_low=data["facial_calibration"]["expected_yes_rate_low"],
        expected_yes_rate_high=data["facial_calibration"]["expected_yes_rate_high"],
        fast_exit_patterns=data["facial_calibration"]["fast_exit_patterns"],
        trajectory_yes_patterns=data["facial_calibration"].get("trajectory_yes_patterns", []),
        trajectory_ambiguous_patterns=data["facial_calibration"].get("trajectory_ambiguous_patterns", []),
        trajectory_no_patterns=data["facial_calibration"].get("trajectory_no_patterns", []),
    )

    bc_data = data.get("bias_controls", {})
    bias = BiasControls(
        max_consecutive_saves=bc_data.get("max_consecutive_saves", 5),
        max_consecutive_rejects=bc_data.get("max_consecutive_rejects", 20),
        parse_failure_alarm_rate=bc_data.get("parse_failure_alarm_rate", 0.03),
    )

    return Brief(
        role_title=data["role_title"],
        role_level=data["role_level"],
        role_summary=data["role_summary"],
        geography=data.get("geography", ""),
        linkedin_project=data.get("linkedin_project", ""),
        capability_areas=capability_areas,
        depth_distinction=depth,
        non_fit_patterns=non_fit_patterns,
        employer_signal_rules=employer_rules,
        minimum_years_experience=data["minimum_years_experience"],
        minimum_bar_description=data["minimum_bar_description"],
        facial_calibration=facial,
        market_density=MarketDensity(data.get("market_density", "moderate")),
        employer_blacklist=data.get("employer_blacklist", []),
        kit_url=data.get("kit_url"),
        jd_path=data.get("jd_path"),
        bias_controls=bias,
        version=data.get("version", "1.0"),
        author=data.get("author", ""),
        notes=data.get("notes", ""),
    )


# ---------------------------------------------------------------------------
# ORCHESTRATOR INTEGRATION — CANDIDATE EVALUATION
# ---------------------------------------------------------------------------
# Shows how the evaluation loop uses the reusable components.
# This replaces the direct prompt construction in the existing judger.py.
# ---------------------------------------------------------------------------

def evaluate_candidate_facial(
    brief: Brief,
    candidate_snippet: str,
    opus_client,  # Your existing Opus LLM client
) -> FacialResult:
    """
    Run facial triage on a single candidate snippet.
    The orchestrator calls this; it never constructs the prompt itself.
    """
    prompt = assemble_facial_prompt(brief, candidate_snippet)
    raw_response = opus_client.complete(prompt)
    return parse_facial_response(raw_response)


def evaluate_candidate_full(
    brief: Brief,
    candidate_profile: str,
    opus_client,
) -> FullEvaluationResult:
    """
    Run full evaluation on a candidate's profile text.
    """
    prompt = assemble_full_evaluation_prompt(brief, candidate_profile)
    raw_response = opus_client.complete(prompt)
    return parse_full_evaluation_response(raw_response)


# ---------------------------------------------------------------------------
# ORCHESTRATOR LOOP SKETCH
# ---------------------------------------------------------------------------
# Pseudocode showing the complete flow with bias monitoring integrated.
# Replace the comments with actual calls to your browser/extraction layer.
# ---------------------------------------------------------------------------

def run_string(
    string_id: str,
    boolean_query: str,
    brief: Brief,
    monitor: BiasMonitor,
    opus_client,
    cheap_client,
    # browser: Browser,
    max_pages: int = 10,
):
    """
    Process one search string through the full pipeline.
    Returns a list of saved candidates and their evaluations.
    """
    saves = []

    # --- Enter Boolean in LinkedIn Recruiter ---
    # browser.enter_search(boolean_query)

    for page_num in range(1, max_pages + 1):

        # --- Extract snippets from current page (cheap model) ---
        # page_html = browser.get_page_html()
        # snippets = extract_snippets(page_html, cheap_client)
        snippets = []  # placeholder

        for i, snippet in enumerate(snippets):
            candidate_id = f"{string_id}_p{page_num}_c{i}"

            # --- FACIAL TRIAGE ---
            facial_result = evaluate_candidate_facial(brief, snippet["text"], opus_client)

            # Record the decision for bias monitoring
            monitor.record_decision(DecisionRecord(
                candidate_id=candidate_id,
                string_id=string_id,
                stage="facial",
                decision=facial_result.decision,
                confidence=1.0 if facial_result.decision != "PARSE_FAILURE" else 0.0,
                capability_area=None,
            ))

            if facial_result.decision == "FACIAL_NO":
                continue

            # --- FULL PROFILE EXTRACTION (cheap model) ---
            # browser.open_profile_panel(snippet["element_ref"])
            # profile_html = browser.get_profile_panel_html()
            # profile_text = extract_full_profile(profile_html, cheap_client)
            profile_text = ""  # placeholder

            # --- FULL EVALUATION (Opus) ---
            eval_result = evaluate_candidate_full(brief, profile_text, opus_client)

            # Record the decision
            monitor.record_decision(DecisionRecord(
                candidate_id=candidate_id,
                string_id=string_id,
                stage="full",
                decision=eval_result.decision,
                confidence=eval_result.confidence,
                capability_area=eval_result.capability_area,
            ))

            if eval_result.decision == "SAVE":
                # --- SAVE TO LINKEDIN PROJECT ---
                # success = browser.save_candidate(snippet["element_ref"])
                saves.append({
                    "candidate_id": candidate_id,
                    "evaluation": eval_result,
                    "snippet": snippet,
                })

            # --- CHECK BIAS ALERTS (after each candidate) ---
            alerts = monitor.check_alerts(string_id)
            for alert in alerts:
                if alert.severity == "pause":
                    print(f"\n⚠ PAUSE: {alert.message}")
                    # In production: pause the string, surface to operator
                    # For now: break the inner loop, let adaptation decide
                    return saves  # Early return — string is suspect
                elif alert.severity == "flag":
                    print(f"⚡ FLAG: {alert.message}")
                elif alert.severity == "info":
                    print(f"ℹ INFO: {alert.message}")

        # --- PAGE ADAPTATION (Opus) ---
        # adaptation_decision = run_page_adaptation(brief, page_stats, opus_client)
        # if adaptation_decision == "stop" or adaptation_decision == "abandon":
        #     break
        pass  # placeholder

    return saves


# ---------------------------------------------------------------------------
# SESSION ENTRY POINT
# ---------------------------------------------------------------------------

def main():
    """
    Session entry point showing the complete initialization sequence.
    """
    # 1. Load the brief
    brief = load_brief("briefs/example_brief_fdl_brazil.json")

    # 2. Initialize bias monitor from the brief
    monitor = BiasMonitor.from_brief(brief)

    # 3. Optional: resume from checkpoint
    checkpoint_path = "checkpoints/bias_monitor.json"
    if Path(checkpoint_path).exists():
        monitor.load_checkpoint(checkpoint_path)
        print(f"Resumed bias monitor from checkpoint: {checkpoint_path}")

    # 4. Initialize LLM clients
    # opus_client = OpusClient(...)
    # cheap_client = CheapClient(...)

    # 5. Form strategy (Opus synthesizes Boolean strings from kit)
    # strings = form_strategy(brief, kit, opus_client)

    # 6. Run each string through the pipeline
    strings = []  # placeholder
    all_saves = []
    for string_id, boolean_query in strings:
        string_saves = run_string(
            string_id=string_id,
            boolean_query=boolean_query,
            brief=brief,
            monitor=monitor,
            opus_client=None,  # placeholder
            cheap_client=None,  # placeholder
        )
        all_saves.extend(string_saves)

        # Checkpoint after each string
        monitor.save_checkpoint(checkpoint_path)

    # 7. Session summary
    summary = monitor.session_summary()
    print(f"\nSession complete:")
    print(f"  Total saves: {summary['saves']}")
    print(f"  Save rate: {summary['save_rate']:.1%}")
    print(f"  Parse failures: {summary['parse_failures']} ({summary['parse_failure_rate']:.1%})")
    print(f"  Alerts fired: {len(summary['alerts_fired'])}")


if __name__ == "__main__":
    main()
