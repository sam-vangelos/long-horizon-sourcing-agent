"""Designer module — discovers visual designers, evaluates portfolio
imagery against a brief-encoded `BriefDesignRubric` via vision-LLM.

See `/Users/jordan.rivera/.cursor/plans/designer-module-spec_5f3d48c1.plan.md`
for the full spec. Slice 1 ships the foundation only (schema, launcher
entry, stub orchestrator, placeholder evaluator). The real pipeline
arrives in subsequent slices:

- Slice 2: Behance source adapter + discovery + real text-based
  contextualization prompt.
- Slice 3: Google CSE adapter (portfolio-host filtered) + cross-source
  dedup with Behance.
- Slice 4: design_rubric intake chapter + brief polish
  ``_design_rubric_drift`` cascade entry.
- Slice 5: Vision evaluation pipeline (image acquisition + Gemini 2.5 Pro
  + four-layer hallucination guard).
- Slice 6: HITL visual review surface (EditorialReviewCard extraction,
  VisualHunkCard, surface_type dispatch).
- Slice 7: Recruiter annotation flow (image-misrepresentative + re-render).
- Slice 8: Sonnet 4.6 cross-check on top-decile.
- Slice 9: Reflection polish for design-market intelligence.
- Slice 10: Dribbble v2 enrichment adapter (optional).
- Slice 11: End-to-end customer-launch readiness.

Hard boundaries Slice 1 honors:

- The Designer module is sourcing only. No outreach automation; no
  Cloris-authored design feedback to candidates.
- Saves write `candidates` rows with SAVE-class `terminal_decision`
  + `surface_type: "hitl_visual_review"` in `terminal_payload_json`
  (the field arrives in Slice 6). Slice 1 only ships the placeholder
  evaluator that returns the rationale stub "no judgment yet —
  designer pipeline not built."
"""
