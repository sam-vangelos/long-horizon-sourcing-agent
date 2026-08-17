"""Verification script for V4 brief deployment.

Checks:
1. V4 brief loads without errors
2. PostSaveModifier dataclass populates correctly
3. additional_search_terms populates on both old and new Brief
4. Template rendering includes new placeholders
5. Parser handles POST_SAVE_MODIFIER field correctly
6. Dynamic CA count guidance works
"""

import json
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BRIEF_PATH = "config/brief-head-fde-enterprise-ai-nyc-v4.json"
PASS = 0
FAIL = 0


def check(label: str, condition: bool, detail: str = ""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS  {label}")
    else:
        FAIL += 1
        print(f"  FAIL  {label}{f' — {detail}' if detail else ''}")


def main():
    global PASS, FAIL
    print("=" * 60)
    print("V4 Brief Verification")
    print("=" * 60)

    # --- 1. JSON validity ---
    print("\n1. JSON validity")
    try:
        with open(BRIEF_PATH) as f:
            raw = json.load(f)
        check("Valid JSON", True)
    except Exception as e:
        check("Valid JSON", False, str(e))
        print("\nCannot continue — brief JSON is invalid.")
        sys.exit(1)

    check("Version is 4.0", raw.get("version") == "4.0", f"got {raw.get('version')}")
    check("Has post_save_modifiers", "post_save_modifiers" in raw)
    check("Has additional_search_terms", "additional_search_terms" in raw)
    check("Has 3 capability_areas", len(raw.get("capability_areas", [])) == 3,
          f"got {len(raw.get('capability_areas', []))}")

    # --- 2. Brief loading ---
    print("\n2. Brief loading")
    from shared.brief_loader import load_brief
    brief = load_brief(BRIEF_PATH)
    check("load_brief() succeeds", brief is not None)
    check("has_v2_schema is True", brief.has_v2_schema)

    nb = brief._new_brief
    check("NewBrief created", nb is not None)
    check("NewBrief has 3 capability_areas", len(nb.capability_areas) == 3,
          f"got {len(nb.capability_areas)}")

    # --- 3. PostSaveModifier ---
    print("\n3. PostSaveModifier")
    check("post_save_modifiers populated", len(nb.post_save_modifiers) > 0,
          f"got {len(nb.post_save_modifiers)}")
    if nb.post_save_modifiers:
        psm = nb.post_save_modifiers[0]
        check("PSM has name", bool(psm.name), psm.name)
        check("PSM has trigger", bool(psm.trigger))
        check("PSM has if_present", bool(psm.if_present))
        check("PSM has if_absent", bool(psm.if_absent))
        check("PSM has signals", len(psm.signals) > 0, f"got {len(psm.signals)}")

    # --- 4. additional_search_terms ---
    print("\n4. additional_search_terms")
    check("NewBrief additional_search_terms populated",
          len(nb.additional_search_terms) > 0,
          f"got {len(nb.additional_search_terms)}")
    check("Old Brief additional_search_terms populated",
          len(brief.additional_search_terms) > 0,
          f"got {len(brief.additional_search_terms)}")

    # --- 5. Rendering methods ---
    print("\n5. Rendering methods")
    psm_block = nb.post_save_modifiers_block()
    check("post_save_modifiers_block() non-empty", len(psm_block) > 0)
    check("PSM block contains 'POST-SAVE MODIFIERS'", "POST-SAVE MODIFIERS" in psm_block)
    check("PSM block contains modifier name",
          nb.post_save_modifiers[0].name in psm_block if nb.post_save_modifiers else True)

    ast_block = nb.additional_search_terms_block()
    check("additional_search_terms_block() non-empty", len(ast_block) > 0)
    check("AST block contains terms", "forward-deployed" in ast_block)

    guidance = nb.capability_area_stack_rank_guidance()
    check("capability_area_stack_rank_guidance() non-empty", len(guidance) > 0)
    check("Guidance references area #3 (not #4)",
          "area #3" in guidance and "area #4" not in guidance,
          f"got: {guidance[:100]}")

    # --- 6. Template assembly ---
    print("\n6. Template assembly")
    from linkedin.judgment_templates import assemble_full_evaluation_prompt
    prompt = assemble_full_evaluation_prompt(nb, "Test candidate profile")
    check("Template assembles without error", len(prompt) > 0)
    check("Template contains POST-SAVE MODIFIERS", "POST-SAVE MODIFIERS" in prompt)
    check("Template contains POST_SAVE_MODIFIER response field", "POST_SAVE_MODIFIER:" in prompt)
    check("Template does NOT contain {post_save_modifiers_block}", "{post_save_modifiers_block}" not in prompt)
    check("Template does NOT contain {capability_area_stack_rank_guidance}",
          "{capability_area_stack_rank_guidance}" not in prompt)
    # Should NOT have old hardcoded "areas ranked 1-3" in the Step 1 section
    check("No hardcoded 'areas ranked 4+' in template",
          "areas ranked 4+" not in prompt)

    # --- 7. Parser tests ---
    print("\n7. Parser tests")
    from linkedin.judgment_templates import parse_full_evaluation_response

    # Test: SAVE + modifier fired
    save_with_modifier = """STEP_1_MATCH: DIRECT
STEP_1_AREA: 1. Enterprise GenAI Delivery (HARD GATE)
STEP_1_EVIDENCE: Built production RAG pipeline serving 500 users
STEP_2_DEPTH: BUILDER
STEP_2_EVIDENCE: Specific tools: LangGraph, LlamaIndex
STEP_3_TRANSFERABILITY: N/A
STEP_3_EVIDENCE: N/A
CASE_FOR: Strong GenAI builder with enterprise deployment
CASE_AGAINST: Limited vertical exposure
DECISION: SAVE
CONFIDENCE: 0.82
POST_SAVE_MODIFIER: Client-Facing / Forward-Deployed Delivery Experience
SUMMARY: Strong GenAI builder with enterprise delivery"""

    r = parse_full_evaluation_response(save_with_modifier)
    check("SAVE+modifier: decision=SAVE", r.decision == "SAVE")
    check("SAVE+modifier: modifier parsed",
          r.post_save_modifier == "Client-Facing / Forward-Deployed Delivery Experience",
          f"got '{r.post_save_modifier}'")
    check("SAVE+modifier: confidence=0.82", r.confidence == 0.82)

    # Test: SAVE + NONE
    save_no_modifier = """STEP_1_MATCH: DIRECT
STEP_1_AREA: 1. Enterprise GenAI Delivery (HARD GATE)
STEP_1_EVIDENCE: Built production systems
STEP_2_DEPTH: BUILDER
STEP_2_EVIDENCE: Hands-on
STEP_3_TRANSFERABILITY: N/A
STEP_3_EVIDENCE: N/A
CASE_FOR: Strong builder
CASE_AGAINST: No client-facing
DECISION: SAVE
CONFIDENCE: 0.75
POST_SAVE_MODIFIER: NONE
SUMMARY: Strong internal builder"""

    r = parse_full_evaluation_response(save_no_modifier)
    check("SAVE+NONE: decision=SAVE", r.decision == "SAVE")
    check("SAVE+NONE: modifier=NONE", r.post_save_modifier == "NONE")

    # Test: REJECT + NONE
    reject = """STEP_1_MATCH: NONE
STEP_1_AREA: N/A
STEP_1_EVIDENCE: No match
STEP_2_DEPTH: USER
STEP_2_EVIDENCE: Application layer only
STEP_3_TRANSFERABILITY: NOT_TRANSFERABLE
STEP_3_EVIDENCE: No ML depth
CASE_FOR: Seniority
CASE_AGAINST: No technical depth
DECISION: REJECT
CONFIDENCE: 0.15
POST_SAVE_MODIFIER: NONE
SUMMARY: No fit"""

    r = parse_full_evaluation_response(reject)
    check("REJECT: decision=REJECT", r.decision == "REJECT")
    check("REJECT: modifier=NONE", r.post_save_modifier == "NONE")

    # Test: INFERENTIAL_SAVE + NONE
    inf_save = """STEP_1_MATCH: ADJACENT
STEP_1_AREA: 1. Enterprise GenAI Delivery (HARD GATE)
STEP_1_EVIDENCE: Sparse but strong priors
STEP_2_DEPTH: BUILDER
STEP_2_EVIDENCE: PhD + ML title
STEP_3_TRANSFERABILITY: TRANSFERABLE
STEP_3_EVIDENCE: Methodology transfers
CASE_FOR: Strong priors
CASE_AGAINST: Sparse profile
DECISION: INFERENTIAL_SAVE
CONFIDENCE: 0.42
POST_SAVE_MODIFIER: NONE
SUMMARY: Sparse but strong priors — flag for recruiter"""

    r = parse_full_evaluation_response(inf_save)
    check("INFERENTIAL_SAVE: decision correct", r.decision == "INFERENTIAL_SAVE")
    check("INFERENTIAL_SAVE: modifier=NONE", r.post_save_modifier == "NONE")

    # Test: Graceful degradation — missing POST_SAVE_MODIFIER field
    no_modifier_field = """STEP_1_MATCH: DIRECT
STEP_1_AREA: 1. Enterprise GenAI Delivery (HARD GATE)
STEP_1_EVIDENCE: Built systems
STEP_2_DEPTH: BUILDER
STEP_2_EVIDENCE: Hands-on
STEP_3_TRANSFERABILITY: N/A
STEP_3_EVIDENCE: N/A
CASE_FOR: Builder
CASE_AGAINST: None
DECISION: SAVE
CONFIDENCE: 0.80
SUMMARY: Strong fit"""

    r = parse_full_evaluation_response(no_modifier_field)
    check("Missing field: decision=SAVE", r.decision == "SAVE")
    check("Missing field: modifier defaults to NONE", r.post_save_modifier == "NONE",
          f"got '{r.post_save_modifier}'")

    # --- 8. Decision matrix dynamic CA count ---
    print("\n8. Decision matrix dynamic CA count")
    dm = nb.decision_matrix_block()
    check("Decision matrix no longer has 'areas ranked 1-3'",
          "areas ranked 1-3" not in dm)
    check("Decision matrix uses generic language",
          "higher-ranked areas" in dm)

    # --- Summary ---
    print("\n" + "=" * 60)
    print(f"Results: {PASS} passed, {FAIL} failed")
    print("=" * 60)

    if FAIL > 0:
        sys.exit(1)
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
