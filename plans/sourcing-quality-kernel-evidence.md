# Sourcing Quality Kernel Evidence

Status: complete; final LinkedIn live bucket supplied and verified
Owner: Codex
Last updated: 2026-06-18

This artifact records verifier output and sample evidence for
`plans/sourcing-quality-kernel-spec.md`. The Recruiter seat tests and live-seat
cold-path probes were bucketed at the end of the goal, then supplied through
the final LinkedIn live-bucket verifier.

## Verification Summary

- `pytest tests/test_receipts.py tests/test_runtime_state_*.py -q -ra` -> 53 passed.
- `pytest tests/test_phase0_contracts.py::test_run_log_event_vocabulary_matches_current_emitters tests/test_linkedin_fallback_acquisition.py::test_record_fallback_discovery_emits_events_and_persists -q -ra` -> 2 passed.
- `pytest tests/test_phase0_contracts.py tests/test_receipts.py tests/test_linkedin_fallback_acquisition.py -q -ra` -> 36 passed, 1 skipped (optional legacy Brazil / Head-of-FDE brief fixtures absent).
- `pytest tests/test_designer_judging.py -q` -> 7 passed.
- `pytest tests/test_linkedin_judge_receipts.py tests/test_designer_judging.py tests/test_judgment_decision_anchoring.py -q` -> 26 passed.
- `pytest tests/test_judger_budget_exhaustion.py tests/test_judger_external_evidence.py tests/test_receipts.py -q` -> 29 passed.
- `pytest tests/test_langfuse_observability.py::TestByteEquivalence::test_record_llm_usage_jsonl_byte_equivalent_with_or_without_keys tests/test_llm_cost_contracts.py tests/test_llm_clients.py tests/test_run_cost_attribution.py -q` -> 17 passed.
- `pytest tests/test_cloris_anthropic_health.py tests/test_llm_cost_contracts.py tests/test_llm_clients.py -q -ra` -> 17 passed.
- `pytest tests/test_designer_vision_evaluation.py -ra` -> 39 passed.
- `pytest tests/test_designer_vision_evaluation.py tests/test_llm_cost_contracts.py tests/test_llm_clients.py -ra` -> 53 passed.
- `pytest tests/test_external_evidence.py -ra` -> 20 passed.
- `pytest tests/test_market_intelligence.py::test_perplexity_backend_packages_deep_research_into_external_research_result tests/test_market_intelligence.py::test_perplexity_backend_records_error_receipt_when_provider_fails tests/test_market_intelligence.py::test_anthropic_research_backend_records_error_receipt_when_provider_fails tests/test_market_intelligence.py::test_perplexity_backend_packages_edge_case_research_into_external_research_result tests/test_market_intelligence.py::test_perplexity_backend_retries_after_truncated_initial_payload -ra` -> 5 passed.
- `pytest tests/test_external_evidence.py tests/test_designer_vision_evaluation.py tests/test_llm_cost_contracts.py tests/test_llm_clients.py tests/test_market_intelligence.py::test_perplexity_backend_packages_deep_research_into_external_research_result tests/test_market_intelligence.py::test_perplexity_backend_records_error_receipt_when_provider_fails tests/test_market_intelligence.py::test_anthropic_research_backend_records_error_receipt_when_provider_fails tests/test_market_intelligence.py::test_perplexity_backend_packages_edge_case_research_into_external_research_result tests/test_market_intelligence.py::test_perplexity_backend_retries_after_truncated_initial_payload -ra` -> 78 passed.
- `pytest tests/test_llm_cost_contracts.py tests/test_intake_conversation_extractor.py tests/test_conversation_agent.py tests/test_conversation_voice_register.py -q -ra` -> 32 passed.
- `pytest tests/test_extractors.py tests/test_github_pipeline.py tests/test_intake_conversation_endpoint.py tests/test_intake_conversation_caps.py -q -ra` -> 51 passed, 1 skipped (optional sourcing brief fixture absent).
- `pytest tests/test_receipts.py tests/test_runtime_state_*.py tests/test_llm_cost_contracts.py tests/test_llm_clients.py tests/test_run_cost_attribution.py -q -ra` -> 68 passed.
- `pytest tests/test_designer_judging.py tests/test_researcher_strategy.py tests/test_researcher_adaptation.py tests/test_github_strategy.py tests/test_linkedin_pipeline.py::test_generate_run_report_writes_json_markdown_and_input_artifacts tests/test_linkedin_pipeline.py::test_generate_run_report_failure_is_warning_only -q -ra` -> 35 passed, 1 skipped (optional GitHub FDE brief fixture absent).
- `pytest tests/test_adaptation_signal_state.py tests/test_seam_strategy_execution.py tests/test_linkedin_empirical_register.py tests/test_linkedin_continuous_verification.py -q -ra` -> 82 passed.
- `pytest tests/test_intake_hiring_manager_image.py tests/test_intake_conversation_extractor.py -q -ra` -> 37 passed.
- `pytest tests/test_matching_contract.py -q` -> 10 passed.
- `pytest tests/test_matching_contract.py tests/test_boolean_normalizer.py -q -ra` -> 15 passed.
- `pytest tests/test_boolean_normalizer.py tests/test_seam_strategy_execution.py -q` -> 69 passed.
- `pytest tests/test_surface_receipt.py tests/test_search_string_lane_fields.py -ra` -> 16 passed.
- `pytest tests/test_surface_receipt.py tests/test_search_string_lane_fields.py tests/test_boolean_normalizer.py tests/test_seam_strategy_execution.py -ra` -> 85 passed.
- `pytest tests/test_boolean_normalizer.py tests/test_seam_strategy_execution.py tests/test_surface_receipt.py tests/test_search_string_lane_fields.py -ra` -> 87 passed.
- `pytest tests/test_surface_receipt.py tests/test_seam_strategy_execution.py tests/test_boolean_normalizer.py tests/test_search_string_lane_fields.py -ra` -> 87 passed.
- `pytest tests/test_adaptation_signal_state.py tests/test_adaptation_*.py -ra` -> 10 passed.
- `pytest tests/test_adaptation_signal_state.py tests/test_adaptation_*.py -q -ra` -> 11 passed.
- `pytest tests/test_adaptation_signal_state.py tests/test_adaptation_*.py tests/test_boolean_normalizer.py tests/test_seam_strategy_execution.py -ra` -> 81 passed.
- `pytest tests/test_boolean_normalizer.py tests/test_seam_strategy_execution.py tests/test_surface_receipt.py tests/test_search_string_lane_fields.py tests/test_adaptation_signal_state.py tests/test_adaptation_*.py -ra` -> 97 passed.
- `pytest tests/test_adaptation_signal_state.py tests/test_adaptation_*.py tests/test_linkedin_pipeline.py::test_run_block_adaptation_pivot_keeps_inserted_replacements_queued tests/test_linkedin_pipeline.py::test_run_full_rechecks_queue_after_block_adaptation tests/test_linkedin_pipeline.py::test_run_full_resume_executes_pending_block_adaptation_before_next_string tests/test_linkedin_pipeline.py::test_run_block_adaptation_defers_when_signal_gate_collects_more_signal tests/test_linkedin_pipeline.py::test_run_block_adaptation_uses_opening_checkpoint_mode tests/test_linkedin_pipeline.py::test_run_block_adaptation_treats_adaptive_followup_as_normal_checkpoint tests/test_linkedin_pipeline.py::test_run_block_adaptation_exploitation_bias_promotes_live_lane_and_demotes_dead_family -q -ra` -> 17 passed.
- `pytest tests/test_linkedin_pipeline.py -q -ra` -> passed.
- `pytest tests/test_observability_monitors.py -q` -> 3 passed.
- `pytest tests/test_adaptation_*.py -q` -> 10 passed.
- `pytest tests/test_adaptation_signal_state.py tests/test_market_intelligence.py::test_adapt_after_block_renders_market_intel_as_typed_prior -q -ra` -> 10 passed, 1 skipped (missing optional brief fixture).
- `pytest tests/test_market_intelligence_provenance.py tests/test_linkedin_continuous_verification.py -q` -> 8 passed.
- `pytest tests/test_market_intelligence.py::test_build_artifact_attaches_groundedness_without_optional_brief_fixture tests/test_market_intelligence.py::test_extract_perplexity_sources_supports_dict_payload tests/test_market_intelligence_provenance.py -ra` -> 6 passed.
- `pytest tests/test_market_intelligence_provenance.py tests/test_linkedin_continuous_verification.py tests/test_market_intelligence.py::test_build_artifact_attaches_groundedness_without_optional_brief_fixture tests/test_market_intelligence.py::test_extract_perplexity_sources_supports_dict_payload tests/test_market_intelligence.py::test_external_result_normalization_maps_provider_aliases tests/test_market_intelligence.py::test_external_research_result_is_packaged_then_applied_to_artifact -ra` -> 11 passed, 1 skipped (optional brief fixture absent).
- `pytest tests/test_linkedin_empirical_register.py tests/test_matching_contract.py tests/test_linkedin_continuous_verification.py -q -ra` -> 19 passed.
- `pytest tests/test_linkedin_empirical_register.py tests/test_linkedin_continuous_verification.py tests/test_matching_contract.py -ra` -> 20 passed.
- `pytest tests/test_linkedin_continuous_verification.py tests/test_linkedin_empirical_register.py tests/test_matching_contract.py -q -ra` -> 23 passed.
- `pytest tests/test_market_intelligence_provenance.py tests/test_market_intelligence.py::test_build_artifact_attaches_groundedness_without_optional_brief_fixture -q -ra` -> 5 passed.
- `pytest tests/test_market_intelligence_provenance.py tests/test_linkedin_continuous_verification.py tests/test_market_intelligence.py::test_build_artifact_attaches_groundedness_without_optional_brief_fixture tests/test_market_intelligence.py::test_extract_perplexity_sources_supports_dict_payload tests/test_market_intelligence.py::test_external_result_normalization_maps_provider_aliases tests/test_market_intelligence.py::test_external_research_result_is_packaged_then_applied_to_artifact -ra` -> 14 passed, 1 skipped (optional brief fixture absent).
- Expanded focused bundle across M0-M4, including seam judgment, LLM receipts, and LinkedIn judge receipts -> passed with the two pre-existing expected xfails in `tests/test_seam_judgment.py`.
- `make validate` -> passed: backend default suite, `svelte-check` with 0 errors / 0 warnings, Vitest 85 files / 924 tests.
- Fresh continuation audit on 2026-06-17: `pytest tests/test_observability_monitors.py -q -ra` -> 3 passed; current M2 rates remained `{"green_but_useless_rate":1.0,"green_but_useless_runs":50,"judge_decisions":0,"judge_parse_failure_rate":0.0,"judge_parse_failures":0,"runs_measured":50}`; `make validate` -> passed: backend default suite, `svelte-check` with 0 errors / 0 warnings, Vitest 85 files / 924 tests.
- Evidence-artifact guard on 2026-06-17: `pytest tests/test_sourcing_quality_kernel_evidence.py tests/test_linkedin_empirical_register.py -q -ra` -> 8 passed; the guard verifies this artifact has every M0-M4 evidence section and that the documented final-bucket `pending_gates` match `validate_final_linkedin_live_bucket({})`.
- Post-guard `make validate` on 2026-06-17 -> passed: backend default suite, `svelte-check` with 0 errors / 0 warnings, Vitest 85 files / 924 tests.
- Strengthened evidence-artifact guard on 2026-06-17: `pytest tests/test_sourcing_quality_kernel_evidence.py tests/test_linkedin_empirical_register.py tests/test_matching_contract.py -q -ra` -> 19 passed; the guard now also verifies the required live-bucket payload template tracks canonical matching-count keys, cold-path names, and stale-policy options.
- Post-strengthening `make validate` on 2026-06-17 -> passed: backend default suite, `svelte-check` with 0 errors / 0 warnings, Vitest 85 files / 924 tests.
- Final-bucket invalid-evidence regression on 2026-06-17: `pytest tests/test_linkedin_empirical_register.py -q -ra` -> 6 passed; malformed `profile_id_probe.verified_at` now keeps `profile_id_availability` absent instead of deriving a verified profile-ID object from incoherent evidence.
- Final-bucket guard bundle on 2026-06-17: `pytest tests/test_linkedin_empirical_register.py tests/test_sourcing_quality_kernel_evidence.py tests/test_matching_contract.py tests/test_linkedin_continuous_verification.py -q -ra` -> 26 passed.
- Post-final-bucket-regression `make validate` on 2026-06-17 -> passed: backend default suite, `svelte-check` with 0 errors / 0 warnings, Vitest 85 files / 924 tests.
- Final-bucket cold-path proof filtering on 2026-06-17: `pytest tests/test_linkedin_empirical_register.py -q -ra` -> 6 passed; malformed cold-path rows are now excluded from `cold_path_results` proof rows instead of being exposed from an invalid report.
- Final-bucket guard bundle after cold-path filtering on 2026-06-17: `pytest tests/test_linkedin_empirical_register.py tests/test_sourcing_quality_kernel_evidence.py tests/test_matching_contract.py tests/test_linkedin_continuous_verification.py -q -ra` -> 26 passed.
- Post-cold-path-filtering `make validate` on 2026-06-17 -> passed: backend default suite, `svelte-check` with 0 errors / 0 warnings, Vitest 85 files / 924 tests.
- Final-bucket duplicate cold-path rejection on 2026-06-17: `pytest tests/test_linkedin_empirical_register.py -q -ra` -> 7 passed; duplicate required cold-path rows now make the live-bucket report invalid and exclude that ambiguous path from proof rows.
- Final-bucket guard bundle after duplicate rejection on 2026-06-17: `pytest tests/test_linkedin_empirical_register.py tests/test_sourcing_quality_kernel_evidence.py tests/test_matching_contract.py tests/test_linkedin_continuous_verification.py -q -ra` -> 27 passed.
- Post-duplicate-rejection `make validate` on 2026-06-17 -> passed: backend default suite, `svelte-check` with 0 errors / 0 warnings, Vitest 85 files / 924 tests.
- Final-bucket unknown cold-path rejection on 2026-06-17: `pytest tests/test_linkedin_empirical_register.py -q -ra` -> 8 passed; cold-path rows whose names are not in the canonical `save`, `drawer`, `pagination`, `fallback` registry now make the live-bucket report invalid.
- Final-bucket guard bundle after unknown-name rejection on 2026-06-17: `pytest tests/test_linkedin_empirical_register.py tests/test_sourcing_quality_kernel_evidence.py tests/test_matching_contract.py tests/test_linkedin_continuous_verification.py -q -ra` -> 28 passed.
- Post-unknown-name-rejection `make validate` on 2026-06-17 -> passed: backend default suite, `svelte-check` with 0 errors / 0 warnings, Vitest 85 files / 924 tests.
- Final-bucket unknown matching-count rejection on 2026-06-17: `pytest tests/test_linkedin_empirical_register.py tests/test_matching_contract.py -q -ra` -> 19 passed; matching-count payloads with keys outside the canonical required seat-test registry now make the live-bucket report invalid.
- Final-bucket guard bundle after unknown-count rejection on 2026-06-17: `pytest tests/test_linkedin_empirical_register.py tests/test_sourcing_quality_kernel_evidence.py tests/test_matching_contract.py tests/test_linkedin_continuous_verification.py -q -ra` -> 29 passed.
- Post-unknown-count-rejection `make validate` on 2026-06-17 -> passed: backend default suite, `svelte-check` with 0 errors / 0 warnings, Vitest 85 files / 924 tests.
- Final-bucket evidence-ref type hardening on 2026-06-17: `pytest tests/test_linkedin_empirical_register.py -q -ra` -> 10 passed; matching-count, profile-ID, and cold-path evidence refs must now be non-empty strings instead of arbitrary values that stringify.
- Final-bucket guard bundle after evidence-ref type hardening on 2026-06-17: `pytest tests/test_linkedin_empirical_register.py tests/test_sourcing_quality_kernel_evidence.py tests/test_matching_contract.py tests/test_linkedin_continuous_verification.py -q -ra` -> 30 passed.
- Post-evidence-ref-hardening `make validate` on 2026-06-17 -> passed: backend default suite, `svelte-check` with 0 errors / 0 warnings, Vitest 85 files / 924 tests.
- Final-bucket top-level payload hardening on 2026-06-17: `pytest tests/test_linkedin_empirical_register.py -q -ra` -> 11 passed; non-object final live-bucket payloads now return a typed invalid report instead of crashing the verifier.
- Final-bucket guard bundle after top-level payload hardening on 2026-06-17: `pytest tests/test_linkedin_empirical_register.py tests/test_sourcing_quality_kernel_evidence.py tests/test_matching_contract.py tests/test_linkedin_continuous_verification.py -q -ra` -> 31 passed.
- Post-top-level-payload-hardening `make validate` on 2026-06-17 -> passed: backend default suite, `svelte-check` with 0 errors / 0 warnings, Vitest 85 files / 924 tests.
- Final-bucket timestamp type hardening on 2026-06-17: `pytest tests/test_linkedin_empirical_register.py -q -ra` -> 12 passed; final-bucket timestamps must now be strings before ISO parsing, so numeric or structured values cannot satisfy live verification timestamps by stringification.
- Final-bucket guard bundle after timestamp type hardening on 2026-06-17: `pytest tests/test_linkedin_empirical_register.py tests/test_sourcing_quality_kernel_evidence.py tests/test_matching_contract.py tests/test_linkedin_continuous_verification.py -q -ra` -> 32 passed.
- Post-timestamp-type-hardening `make validate` on 2026-06-17 -> passed: backend default suite, `svelte-check` with 0 errors / 0 warnings, Vitest 85 files / 924 tests.
- Final-bucket schema-label type hardening on 2026-06-17: `pytest tests/test_linkedin_empirical_register.py -q -ra` -> 13 passed; cold-path names, cold-path statuses, and stale-policy choices must now be actual strings instead of arbitrary values that stringify to canonical labels.
- Final-bucket guard bundle after schema-label type hardening on 2026-06-17: `pytest tests/test_linkedin_empirical_register.py tests/test_sourcing_quality_kernel_evidence.py tests/test_matching_contract.py tests/test_linkedin_continuous_verification.py -q -ra` -> 33 passed.
- Post-schema-label-type-hardening `make validate` on 2026-06-17 -> passed: backend default suite, `svelte-check` with 0 errors / 0 warnings, Vitest 85 files / 924 tests.
- Matching-count key type hardening on 2026-06-17: `pytest tests/test_matching_contract.py tests/test_linkedin_empirical_register.py -q -ra` -> 24 passed; non-string Recruiter seat-test count keys now raise typed evidence errors instead of crashing the final-bucket verifier, and direct matching-contract `verified_at` inputs must be string ISO timestamps.
- Final-bucket guard bundle after matching-count key type hardening on 2026-06-17: `pytest tests/test_linkedin_empirical_register.py tests/test_sourcing_quality_kernel_evidence.py tests/test_matching_contract.py tests/test_linkedin_continuous_verification.py -q -ra` -> 34 passed.
- Post-matching-count-key-type-hardening `make validate` on 2026-06-17 -> passed: backend default suite, `svelte-check` with 0 errors / 0 warnings, Vitest 85 files / 924 tests.
- Final-bucket cold-path list type hardening on 2026-06-17: `pytest tests/test_linkedin_empirical_register.py -q -ra` -> 15 passed; `cold_path_results` must now be a real list instead of any non-string sequence, keeping the final live payload JSON-shaped.
- Final-bucket guard bundle after cold-path list type hardening on 2026-06-17: `pytest tests/test_linkedin_empirical_register.py tests/test_sourcing_quality_kernel_evidence.py tests/test_matching_contract.py tests/test_linkedin_continuous_verification.py -q -ra` -> 35 passed.
- Post-cold-path-list-type-hardening `make validate` on 2026-06-17 -> passed: backend default suite, `svelte-check` with 0 errors / 0 warnings, Vitest 85 files / 924 tests.
- Post-unused-helper-cleanup `make validate` on 2026-06-17 -> passed: backend default suite, `svelte-check` with 0 errors / 0 warnings, Vitest 85 files / 924 tests.
- Cold-path fixture outcome type hardening on 2026-06-17: `pytest tests/test_linkedin_continuous_verification.py -q -ra` -> 8 passed; due fixture probe outcomes now reject non-string `status`, `evidence_ref`, and `error` fields instead of stringifying them into proof.
- M4 local guard bundle after fixture outcome hardening on 2026-06-17: `pytest tests/test_market_intelligence_provenance.py tests/test_linkedin_continuous_verification.py tests/test_linkedin_empirical_register.py -q -ra` -> 27 passed.
- Post-fixture-outcome-type-hardening `make validate` on 2026-06-17 -> passed: backend default suite, `svelte-check` with 0 errors / 0 warnings, Vitest 85 files / 924 tests.
- Market-evidence-ref type hardening on 2026-06-17: `pytest tests/test_market_intelligence.py::test_build_artifact_attaches_groundedness_without_optional_brief_fixture tests/test_market_intelligence.py::test_extract_perplexity_sources_supports_dict_payload tests/test_market_intelligence_provenance.py -q -ra` -> 7 passed; mapping-form evidence refs now require string `source_id`, `source_type`, `locator`, and `quote` fields instead of accepting arbitrary stringifiable values.
- M4 local guard bundle after market-evidence-ref type hardening on 2026-06-17: `pytest tests/test_market_intelligence_provenance.py tests/test_linkedin_continuous_verification.py tests/test_linkedin_empirical_register.py -q -ra` -> 28 passed.
- Post-market-evidence-ref-type-hardening `make validate` on 2026-06-17 -> passed: backend default suite, `svelte-check` with 0 errors / 0 warnings, Vitest 85 files / 924 tests.
- Market-claim mapping type hardening on 2026-06-17: `pytest tests/test_market_intelligence.py::test_build_artifact_attaches_groundedness_without_optional_brief_fixture tests/test_market_intelligence.py::test_extract_perplexity_sources_supports_dict_payload tests/test_market_intelligence_provenance.py -q -ra` -> 9 passed; mapping-form market claims now require string claim IDs/text and object metadata, and invalid evidence-ref entries raise typed `ValueError`s instead of incidental parser errors.
- M4 local guard bundle after market-claim mapping type hardening on 2026-06-17: `pytest tests/test_market_intelligence_provenance.py tests/test_linkedin_continuous_verification.py tests/test_linkedin_empirical_register.py -q -ra` -> 30 passed.
- Post-market-claim-mapping-type-hardening `make validate` on 2026-06-17 -> passed: backend default suite, `svelte-check` with 0 errors / 0 warnings, Vitest 85 files / 924 tests.
- Market-evidence support-text hardening on 2026-06-17: `pytest tests/test_market_intelligence.py::test_build_artifact_attaches_groundedness_without_optional_brief_fixture tests/test_market_intelligence.py::test_extract_perplexity_sources_supports_dict_payload tests/test_market_intelligence_provenance.py -q -ra` -> 10 passed; evidence-ref metadata must be an object when supplied, and only string metadata contributes support text for groundedness.
- M4 local guard bundle after support-text hardening on 2026-06-17: `pytest tests/test_market_intelligence_provenance.py tests/test_linkedin_continuous_verification.py tests/test_linkedin_empirical_register.py -q -ra` -> 31 passed.
- Post-market-evidence-support-text-hardening `make validate` on 2026-06-17 -> passed: backend default suite, `svelte-check` with 0 errors / 0 warnings, Vitest 85 files / 924 tests.
- Accessibility snapshot type hardening on 2026-06-17: `pytest tests/test_linkedin_continuous_verification.py -q -ra` -> 9 passed; accessibility-tree drift fixtures now reject non-object snapshots and non-string `role`/`name` fields instead of stringifying them into drift evidence.
- M4 local guard bundle after accessibility snapshot type hardening on 2026-06-17: `pytest tests/test_market_intelligence_provenance.py tests/test_linkedin_continuous_verification.py tests/test_linkedin_empirical_register.py -q -ra` -> 32 passed.
- Post-accessibility-snapshot-type-hardening `make validate` on 2026-06-17 -> passed: backend default suite, `svelte-check` with 0 errors / 0 warnings, Vitest 85 files / 924 tests.
- Evidence-artifact verifier coverage hardening on 2026-06-17: `pytest tests/test_sourcing_quality_kernel_evidence.py -q -ra` -> 5 passed; the guard now requires the artifact to track each named milestone verifier fragment plus every final LinkedIn live-bucket item from the spec.
- Final-bucket guard bundle after evidence-artifact verifier coverage hardening on 2026-06-17: `pytest tests/test_sourcing_quality_kernel_evidence.py tests/test_market_intelligence_provenance.py tests/test_linkedin_continuous_verification.py tests/test_linkedin_empirical_register.py tests/test_matching_contract.py -q -ra` -> 47 passed.
- Post-evidence-artifact-verifier-coverage-hardening `make validate` on 2026-06-17 -> passed: backend default suite, `svelte-check` with 0 errors / 0 warnings, Vitest 85 files / 924 tests.
- Final-bucket schema-version hardening on 2026-06-17: `pytest tests/test_linkedin_empirical_register.py -q -ra` -> 17 passed; supplied final-live-bucket `schema_version` values now fail closed unless they match `linkedin.final_live_bucket.v1`, without adding a new live-seat pending gate.
- Final-bucket guard bundle after schema-version hardening on 2026-06-17: `pytest tests/test_sourcing_quality_kernel_evidence.py tests/test_market_intelligence_provenance.py tests/test_linkedin_continuous_verification.py tests/test_linkedin_empirical_register.py tests/test_matching_contract.py -q -ra` -> 49 passed.
- Post-schema-version-hardening `make validate` on 2026-06-17 -> passed: backend default suite, `svelte-check` with 0 errors / 0 warnings, Vitest 85 files / 924 tests.
- Final-bucket closed-schema hardening on 2026-06-17: `pytest tests/test_linkedin_empirical_register.py -q -ra` -> 19 passed; the final live-bucket verifier now rejects unexpected top-level payload keys and unexpected cold-path result keys, excluding malformed cold-path rows from proof output.
- Final-bucket guard bundle after closed-schema hardening on 2026-06-17: `pytest tests/test_sourcing_quality_kernel_evidence.py tests/test_market_intelligence_provenance.py tests/test_linkedin_continuous_verification.py tests/test_linkedin_empirical_register.py tests/test_matching_contract.py -q -ra` -> 51 passed.
- Post-closed-schema-hardening `make validate` on 2026-06-17 -> passed: backend default suite, `svelte-check` with 0 errors / 0 warnings, Vitest 85 files / 924 tests.
- Final-bucket profile-probe schema hardening on 2026-06-17: `pytest tests/test_linkedin_empirical_register.py -q -ra` -> 20 passed; `profile_id_probe` inline evidence now rejects undeclared or non-string keys and keeps profile-ID availability absent when malformed.
- Final-bucket guard bundle after profile-probe schema hardening on 2026-06-17: `pytest tests/test_sourcing_quality_kernel_evidence.py tests/test_market_intelligence_provenance.py tests/test_linkedin_continuous_verification.py tests/test_linkedin_empirical_register.py tests/test_matching_contract.py -q -ra` -> 52 passed.
- Post-profile-probe-schema-hardening `make validate` on 2026-06-17 -> passed: backend default suite, `svelte-check` with 0 errors / 0 warnings, Vitest 85 files / 924 tests.
- Final-bucket envelope fail-empty hardening on 2026-06-17: `pytest tests/test_linkedin_empirical_register.py -q -ra` -> 20 passed; unsupported final-bucket schema versions or unexpected top-level payload keys now return invalid reports without derived matching/profile/cold-path proof fields.
- Final-bucket guard bundle after envelope fail-empty hardening on 2026-06-17: `pytest tests/test_sourcing_quality_kernel_evidence.py tests/test_market_intelligence_provenance.py tests/test_linkedin_continuous_verification.py tests/test_linkedin_empirical_register.py tests/test_matching_contract.py -q -ra` -> 52 passed.
- Post-envelope-fail-empty-hardening `make validate` on 2026-06-17 -> passed: backend default suite, `svelte-check` with 0 errors / 0 warnings, Vitest 85 files / 924 tests.
- M3 market-prior mapping type hardening on 2026-06-17: `pytest tests/test_adaptation_signal_state.py -q -ra` -> 13 passed; mapping-form typed market priors now reject non-string source/signal/evidence fields and non-numeric confidence instead of stringifying malformed adaptation input.
- M3/final-bucket guard bundle after market-prior mapping type hardening on 2026-06-17: `pytest tests/test_adaptation_signal_state.py tests/test_sourcing_quality_kernel_evidence.py tests/test_market_intelligence_provenance.py tests/test_linkedin_continuous_verification.py tests/test_linkedin_empirical_register.py tests/test_matching_contract.py -q -ra` -> 65 passed.
- Post-market-prior-mapping-type-hardening `make validate` on 2026-06-17 -> passed: backend default suite, `svelte-check` with 0 errors / 0 warnings, Vitest 85 files / 924 tests.
- Final LinkedIn live bucket remains deferred on 2026-06-17: `validate_final_linkedin_live_bucket({})` -> `pending`; pending gates are `cold_path_results`, `matching_counts`, `profile_id_probe`, and `stale_matching_contract_policy`.
- M2 parse-rate evidence hardening on 2026-06-17: `pytest tests/test_observability_monitors.py -q -ra` -> 5 passed; malformed raw candidate/attempt payload text containing `PARSE_FAILURE` no longer counts as parse-failure evidence unless the explicit decision columns or a valid parsed payload carry the typed decision.
- M2/final-bucket guard bundle after parse-rate evidence hardening on 2026-06-17: `pytest tests/test_observability_monitors.py tests/test_sourcing_quality_kernel_evidence.py tests/test_linkedin_empirical_register.py -q -ra` -> 30 passed.
- Post-parse-rate-evidence-hardening `make validate` on 2026-06-17 -> passed: backend default suite, `svelte-check` with 0 errors / 0 warnings, Vitest 85 files / 924 tests.
- M2 baseline aggregate type hardening on 2026-06-18: `pytest tests/test_observability_monitors.py -q -ra` -> 6 passed; baseline-rate aggregation now rejects malformed monitor rows, non-boolean `green_but_useless`, non-integer/negative judge counts, and parse-failure counts that exceed total judge decisions.
- M2/final-bucket guard bundle after baseline aggregate type hardening on 2026-06-18: `pytest tests/test_observability_monitors.py tests/test_sourcing_quality_kernel_evidence.py tests/test_linkedin_empirical_register.py -q -ra` -> 36 passed.
- Post-baseline-aggregate-type-hardening `make validate` on 2026-06-18 -> passed: backend default suite, `svelte-check` with 0 errors / 0 warnings, Vitest 85 files / 924 tests.
- M0 receipt-envelope type hardening on 2026-06-17: `pytest tests/test_receipts.py -q -ra` -> 13 passed; receipt core fields and version pins now reject non-string values instead of stringifying malformed caller-supplied evidence.
- M0/evidence/final-bucket guard bundle after receipt-envelope hardening on 2026-06-17: `pytest tests/test_receipts.py tests/test_runtime_state_*.py tests/test_sourcing_quality_kernel_evidence.py tests/test_linkedin_empirical_register.py -q -ra` -> 80 passed.
- Post-receipt-envelope-hardening `make validate` on 2026-06-17 -> passed: backend default suite, `svelte-check` with 0 errors / 0 warnings, Vitest 85 files / 924 tests.
- M1B empirical-value type hardening on 2026-06-17: `pytest tests/test_matching_contract.py -q -ra` -> 11 passed; lower-level verified matching facts now require `MatchingFact` plus the fact-specific enum value and object evidence instead of accepting stringified contract fields.
- M1B/final-bucket guard bundle after empirical-value type hardening on 2026-06-17: `pytest tests/test_matching_contract.py tests/test_linkedin_empirical_register.py tests/test_sourcing_quality_kernel_evidence.py -q -ra` -> 36 passed.
- Post-empirical-value-type-hardening `make validate` on 2026-06-17 -> passed: backend default suite, `svelte-check` with 0 errors / 0 warnings, Vitest 85 files / 924 tests.
- M1B matching-contract timestamp/envelope hardening on 2026-06-18: `pytest tests/test_matching_contract.py -q -ra` -> 13 passed; direct seat-count evidence now rejects non-object count envelopes, and verified matching-contract timestamps are normalized before becoming empirical evidence.
- M1B/final-bucket guard bundle after matching-contract timestamp/envelope hardening on 2026-06-18: `pytest tests/test_matching_contract.py tests/test_linkedin_empirical_register.py tests/test_sourcing_quality_kernel_evidence.py -q -ra` -> 43 passed.
- Post-matching-contract-timestamp-envelope-hardening `make validate` on 2026-06-18 -> passed: backend default suite, `svelte-check` with 0 errors / 0 warnings, Vitest 85 files / 924 tests.
- M1C normalizer explicit-input type hardening on 2026-06-17: `pytest tests/test_boolean_normalizer.py -q -ra` -> 9 passed; structured filter values, locale/morphology expansion maps, and explicit ubiquitous-term sets now reject non-string values instead of stringifying malformed local rule inputs.
- M1C seam bundle after normalizer explicit-input type hardening on 2026-06-17: `pytest tests/test_boolean_normalizer.py tests/test_seam_strategy_execution.py -q -ra` -> 75 passed.
- M1C/final-bucket guard bundle after normalizer explicit-input type hardening on 2026-06-17: `pytest tests/test_boolean_normalizer.py tests/test_seam_strategy_execution.py tests/test_sourcing_quality_kernel_evidence.py tests/test_linkedin_empirical_register.py -q -ra` -> 100 passed.
- Post-normalizer-explicit-input-hardening `make validate` on 2026-06-17 -> passed: backend default suite, `svelte-check` with 0 errors / 0 warnings, Vitest 85 files / 924 tests.
- M3 adapted-string firewall local-input hardening on 2026-06-17: `pytest tests/test_adaptation_signal_state.py -q -ra` -> 14 passed; adapted strings now reject malformed item-level `structured_filters` and wrap M1C normalizer input failures as `AdaptationValidationError`.
- M1C/M3 guard bundle after adapted-string firewall local-input hardening on 2026-06-17: `pytest tests/test_adaptation_signal_state.py tests/test_adaptation_*.py tests/test_boolean_normalizer.py tests/test_seam_strategy_execution.py -q -ra` -> 89 passed.
- M1C/M3/final-bucket guard bundle after adapted-string firewall local-input hardening on 2026-06-17: `pytest tests/test_adaptation_signal_state.py tests/test_adaptation_*.py tests/test_boolean_normalizer.py tests/test_seam_strategy_execution.py tests/test_sourcing_quality_kernel_evidence.py tests/test_linkedin_empirical_register.py -q -ra` -> 114 passed.
- Post-adapted-string-firewall-local-input-hardening `make validate` on 2026-06-17 -> passed: backend default suite, `svelte-check` with 0 errors / 0 warnings, Vitest 85 files / 924 tests.
- M3 adaptation-gate config type hardening on 2026-06-18: `pytest tests/test_adaptation_signal_state.py -q -ra` -> 15 passed; sufficiency, cooldown, reset, and SPRT threshold config now rejects malformed non-typed values or inverted bounds before adaptation-gate evaluation.
- M3/final-bucket guard bundle after adaptation-gate config type hardening on 2026-06-18: `pytest tests/test_adaptation_*.py tests/test_sourcing_quality_kernel_evidence.py tests/test_linkedin_empirical_register.py -q -ra` -> 45 passed.
- Post-adaptation-gate-config-type-hardening `make validate` on 2026-06-18 -> passed: backend default suite, `svelte-check` with 0 errors / 0 warnings, Vitest 85 files / 924 tests.
- Final artifact/adaptation/live-bucket guard after adaptation-gate config type hardening on 2026-06-18: `pytest tests/test_sourcing_quality_kernel_evidence.py tests/test_adaptation_signal_state.py tests/test_linkedin_empirical_register.py -q -ra` -> 45 passed.
- Final LinkedIn live bucket remains deferred on 2026-06-18: `validate_final_linkedin_live_bucket({})` -> `pending`; pending gates are `cold_path_results`, `matching_counts`, `profile_id_probe`, and `stale_matching_contract_policy`.
- Final-bucket schema-pin completion hardening on 2026-06-18: supplied live evidence cannot complete unless it carries `schema_version: linkedin.final_live_bucket.v1`; an empty payload still reports only the deferred live gates.
- Final-bucket schema-pin guard on 2026-06-18: `pytest tests/test_linkedin_empirical_register.py tests/test_sourcing_quality_kernel_evidence.py -q -ra` -> 31 passed.
- M4/final-bucket guard bundle after schema-pin completion hardening on 2026-06-18: `pytest tests/test_linkedin_empirical_register.py tests/test_sourcing_quality_kernel_evidence.py tests/test_matching_contract.py tests/test_linkedin_continuous_verification.py -q -ra` -> 59 passed.
- Post-schema-pin-completion-hardening `make validate` on 2026-06-18 -> passed: backend default suite, `svelte-check` with 0 errors / 0 warnings, Vitest 85 files / 924 tests.
- Final artifact/live-bucket guard after schema-pin completion hardening on 2026-06-18: `pytest tests/test_linkedin_empirical_register.py tests/test_sourcing_quality_kernel_evidence.py -q -ra` -> 31 passed; `validate_final_linkedin_live_bucket({})` -> `pending` with `cold_path_results`, `matching_counts`, `profile_id_probe`, and `stale_matching_contract_policy`.
- Final-bucket payload-template registry hardening on 2026-06-18: `pytest tests/test_linkedin_empirical_register.py tests/test_sourcing_quality_kernel_evidence.py -q -ra` -> 32 passed; the required live-evidence payload template is now generated from the same matching-count, cold-path, schema-version, and stale-policy registries as the verifier.
- M4/final-bucket guard bundle after payload-template registry hardening on 2026-06-18: `pytest tests/test_linkedin_empirical_register.py tests/test_sourcing_quality_kernel_evidence.py tests/test_matching_contract.py tests/test_linkedin_continuous_verification.py -q -ra` -> 60 passed.
- Post-payload-template-registry-hardening `make validate` on 2026-06-18 -> passed: backend default suite, `svelte-check` with 0 errors / 0 warnings, Vitest 85 files / 924 tests.
- Final artifact/live-bucket guard after payload-template registry hardening on 2026-06-18: `pytest tests/test_linkedin_empirical_register.py tests/test_sourcing_quality_kernel_evidence.py -q -ra` -> 32 passed; `validate_final_linkedin_live_bucket({})` -> `pending` with `cold_path_results`, `matching_counts`, `profile_id_probe`, and `stale_matching_contract_policy`.
- Final-bucket verifier CLI on 2026-06-18: `pytest tests/test_validate_linkedin_final_live_bucket_tool.py tests/test_linkedin_empirical_register.py tests/test_sourcing_quality_kernel_evidence.py -q -ra` -> 36 passed; `tools/validate_linkedin_final_live_bucket.py` now prints the canonical template and exits nonzero for pending or invalid live evidence.
- M4/final-bucket guard bundle after verifier CLI on 2026-06-18: `pytest tests/test_validate_linkedin_final_live_bucket_tool.py tests/test_linkedin_empirical_register.py tests/test_sourcing_quality_kernel_evidence.py tests/test_matching_contract.py tests/test_linkedin_continuous_verification.py -q -ra` -> 64 passed.
- Post-final-bucket-verifier-CLI `make validate` on 2026-06-18 -> passed: backend default suite, `svelte-check` with 0 errors / 0 warnings, Vitest 85 files / 924 tests.
- Final artifact/CLI guard after verifier CLI on 2026-06-18: `pytest tests/test_validate_linkedin_final_live_bucket_tool.py tests/test_linkedin_empirical_register.py tests/test_sourcing_quality_kernel_evidence.py -q -ra` -> 36 passed; `printf '{}' | .venv/bin/python tools/validate_linkedin_final_live_bucket.py -` -> exit 2 with `pending` gates `cold_path_results`, `matching_counts`, `profile_id_probe`, and `stale_matching_contract_policy`.
- Final-bucket Makefile verifier shortcuts on 2026-06-18: `tests/test_validate_linkedin_final_live_bucket_tool.py` pins `make sqk-live-bucket-template` and `make validate-sqk-live-bucket SQK_LIVE_BUCKET=/path.json` to the same local verifier CLI.
- Final-bucket Makefile shortcut guard on 2026-06-18: `pytest tests/test_validate_linkedin_final_live_bucket_tool.py tests/test_linkedin_empirical_register.py tests/test_sourcing_quality_kernel_evidence.py -q -ra` -> 37 passed; `make sqk-live-bucket-template` prints the canonical template and `make validate-sqk-live-bucket SQK_LIVE_BUCKET=/tmp/sqk-pending.json` exits 2 for pending evidence.
- M4/final-bucket guard bundle after Makefile shortcuts on 2026-06-18: `pytest tests/test_validate_linkedin_final_live_bucket_tool.py tests/test_linkedin_empirical_register.py tests/test_sourcing_quality_kernel_evidence.py tests/test_matching_contract.py tests/test_linkedin_continuous_verification.py -q -ra` -> 65 passed.
- Post-final-bucket-Makefile-shortcuts `make validate` on 2026-06-18 -> passed: backend default suite, `svelte-check` with 0 errors / 0 warnings, Vitest 85 files / 924 tests.
- Final artifact/live-bucket guard after Makefile shortcuts on 2026-06-18: `pytest tests/test_validate_linkedin_final_live_bucket_tool.py tests/test_linkedin_empirical_register.py tests/test_sourcing_quality_kernel_evidence.py -q -ra` -> 37 passed; `validate_final_linkedin_live_bucket({})` -> `pending` with `cold_path_results`, `matching_counts`, `profile_id_probe`, and `stale_matching_contract_policy`.
- M4 matching-contract freshness timestamp hardening on 2026-06-17: `pytest tests/test_linkedin_continuous_verification.py -q -ra` -> 10 passed; mapping-form `last_empirically_verified` must now be a string ISO timestamp before freshness is evaluated, so malformed contract mappings cannot become stale/fresh proof by stringification.
- M4/final-bucket guard bundle after freshness timestamp hardening on 2026-06-17: `pytest tests/test_market_intelligence_provenance.py tests/test_linkedin_continuous_verification.py tests/test_linkedin_empirical_register.py tests/test_sourcing_quality_kernel_evidence.py -q -ra` -> 43 passed.
- Post-freshness-timestamp-hardening `make validate` on 2026-06-17 -> passed: backend default suite, `svelte-check` with 0 errors / 0 warnings, Vitest 85 files / 924 tests.
- M4 cold-path probe metadata hardening on 2026-06-17: `pytest tests/test_linkedin_continuous_verification.py -q -ra` -> 12 passed; malformed probe names, silence bounds, live-seat flags, or last-run timestamps now produce typed `invalid_probe` results instead of crashes, coercion, or runner execution.
- M4/final-bucket guard bundle after cold-path probe metadata hardening on 2026-06-17: `pytest tests/test_market_intelligence_provenance.py tests/test_linkedin_continuous_verification.py tests/test_linkedin_empirical_register.py tests/test_sourcing_quality_kernel_evidence.py -q -ra` -> 45 passed.
- Post-cold-path-probe-metadata-hardening `make validate` on 2026-06-17 -> passed: backend default suite, `svelte-check` with 0 errors / 0 warnings, Vitest 85 files / 924 tests.
- M4 accessibility selector hardening on 2026-06-17: `pytest tests/test_linkedin_continuous_verification.py -q -ra` -> 12 passed; fixture accessibility-tree selector maps now reject non-object maps, non-string selector keys, and empty selectors before drift evidence is built.
- M4/final-bucket guard bundle after accessibility selector hardening on 2026-06-17: `pytest tests/test_market_intelligence_provenance.py tests/test_linkedin_continuous_verification.py tests/test_linkedin_empirical_register.py tests/test_sourcing_quality_kernel_evidence.py -q -ra` -> 45 passed.
- Post-accessibility-selector-hardening `make validate` on 2026-06-17 -> passed: backend default suite, `svelte-check` with 0 errors / 0 warnings, Vitest 85 files / 924 tests.
- M4 accessibility snapshot object hardening on 2026-06-17: `pytest tests/test_linkedin_continuous_verification.py -q -ra` -> 13 passed; fixture `AccessibilityNodeSnapshot` values are normalized to the validated map selector and their role/name fields are type-checked before drift evidence is built.
- M4/final-bucket guard bundle after accessibility snapshot object hardening on 2026-06-17: `pytest tests/test_market_intelligence_provenance.py tests/test_linkedin_continuous_verification.py tests/test_linkedin_empirical_register.py tests/test_sourcing_quality_kernel_evidence.py -q -ra` -> 46 passed.
- Post-accessibility-snapshot-object-hardening `make validate` on 2026-06-17 -> passed: backend default suite, `svelte-check` with 0 errors / 0 warnings, Vitest 85 files / 924 tests.
- M4 cold-path runner outcome schema hardening on 2026-06-17: `pytest tests/test_linkedin_continuous_verification.py -q -ra` -> 14 passed; fixture runner outcomes now reject unexpected or non-string keys instead of silently discarding side-channel proof fields.
- M4/final-bucket guard bundle after runner outcome schema hardening on 2026-06-17: `pytest tests/test_market_intelligence_provenance.py tests/test_linkedin_continuous_verification.py tests/test_linkedin_empirical_register.py tests/test_sourcing_quality_kernel_evidence.py -q -ra` -> 47 passed.
- Post-cold-path-runner-outcome-schema-hardening `make validate` on 2026-06-17 -> passed: backend default suite, `svelte-check` with 0 errors / 0 warnings, Vitest 85 files / 924 tests.
- M4 cold-path runner typed-status hardening on 2026-06-17: `pytest tests/test_linkedin_continuous_verification.py -q -ra` -> 15 passed; fixture runner outcomes now reject shorthand boolean/string/null returns plus missing, empty, or unknown statuses instead of deriving implicit typed status evidence.
- M4/final-bucket guard bundle after runner typed-status hardening on 2026-06-17: `pytest tests/test_market_intelligence_provenance.py tests/test_linkedin_continuous_verification.py tests/test_linkedin_empirical_register.py tests/test_sourcing_quality_kernel_evidence.py -q -ra` -> 48 passed.
- Post-cold-path-runner-typed-status-hardening `make validate` on 2026-06-17 -> passed: backend default suite, `svelte-check` with 0 errors / 0 warnings, Vitest 85 files / 924 tests.
- M4 continuous-verification report-config type hardening on 2026-06-17: `pytest tests/test_linkedin_continuous_verification.py -q -ra` -> 15 passed; accessibility cadence, cold-path alert-only mode, and matching-contract freshness age/alert configuration now reject malformed non-typed values before report emission.
- M4/final-bucket guard bundle after report-config type hardening on 2026-06-17: `pytest tests/test_market_intelligence_provenance.py tests/test_linkedin_continuous_verification.py tests/test_linkedin_empirical_register.py tests/test_sourcing_quality_kernel_evidence.py -q -ra` -> 48 passed.
- Post-report-config-type-hardening `make validate` on 2026-06-18 -> passed: backend default suite, `svelte-check` with 0 errors / 0 warnings, Vitest 85 files / 924 tests.
- Final-bucket profile-ID boolean hardening on 2026-06-18: `pytest tests/test_linkedin_empirical_register.py -q -ra` -> 21 passed; `profile_id_probe.stable_profile_id_seen` now distinguishes a real negative probe (`False`, pending) from malformed non-boolean evidence (`invalid`).
- M4/final-bucket guard bundle after profile-ID boolean hardening on 2026-06-18: `pytest tests/test_market_intelligence_provenance.py tests/test_linkedin_continuous_verification.py tests/test_linkedin_empirical_register.py tests/test_sourcing_quality_kernel_evidence.py -q -ra` -> 49 passed.
- Post-profile-ID-boolean-hardening `make validate` on 2026-06-18 -> passed: backend default suite, `svelte-check` with 0 errors / 0 warnings, Vitest 85 files / 924 tests.
- Final-bucket cold-path status hardening on 2026-06-18: `pytest tests/test_linkedin_empirical_register.py -q -ra` -> 23 passed; supplied cold-path rows now reject unknown status labels as invalid while keeping known failed probe statuses pending instead of complete.
- M4/final-bucket guard bundle after cold-path status hardening on 2026-06-18: `pytest tests/test_market_intelligence_provenance.py tests/test_linkedin_continuous_verification.py tests/test_linkedin_empirical_register.py tests/test_sourcing_quality_kernel_evidence.py -q -ra` -> 51 passed.
- Post-cold-path-status-hardening `make validate` on 2026-06-18 -> passed: backend default suite, `svelte-check` with 0 errors / 0 warnings, Vitest 85 files / 924 tests.
- Final-bucket cold-path proof normalization on 2026-06-18: `pytest tests/test_linkedin_empirical_register.py -q -ra` -> 24 passed; completed cold-path proof rows now emit canonical `name`, `status`, `ran_at`, and `evidence_ref` values instead of preserving whitespace-padded input.
- M4/final-bucket guard bundle after cold-path proof normalization on 2026-06-18: `pytest tests/test_market_intelligence_provenance.py tests/test_linkedin_continuous_verification.py tests/test_linkedin_empirical_register.py tests/test_sourcing_quality_kernel_evidence.py -q -ra` -> 52 passed.
- Post-cold-path-proof-normalization `make validate` on 2026-06-18 -> passed: backend default suite, `svelte-check` with 0 errors / 0 warnings, Vitest 85 files / 924 tests.
- Final-bucket profile-ID proof normalization on 2026-06-18: `pytest tests/test_linkedin_empirical_register.py -q -ra` -> 25 passed; completed `profile_id_probe` evidence now emits canonical `evidence_ref` and optional probe-level `verified_at` values inside `profile_id_availability`.
- M4/final-bucket guard bundle after profile-ID proof normalization on 2026-06-18: `pytest tests/test_market_intelligence_provenance.py tests/test_linkedin_continuous_verification.py tests/test_linkedin_empirical_register.py tests/test_sourcing_quality_kernel_evidence.py -q -ra` -> 53 passed.
- Post-profile-ID-proof-normalization `make validate` on 2026-06-18 -> passed: backend default suite, `svelte-check` with 0 errors / 0 warnings, Vitest 85 files / 924 tests.
- M4 report timestamp type hardening on 2026-06-18: `pytest tests/test_linkedin_continuous_verification.py -q -ra` -> 15 passed; accessibility drift, cold-path registry, cold-path runner, and matching-contract freshness report timestamps now reject malformed non-string/non-datetime values with typed `ValueError`s.
- M4/final-bucket guard bundle after report timestamp type hardening on 2026-06-18: `pytest tests/test_market_intelligence_provenance.py tests/test_linkedin_continuous_verification.py tests/test_linkedin_empirical_register.py tests/test_sourcing_quality_kernel_evidence.py -q -ra` -> 53 passed.
- Post-report-timestamp-type-hardening `make validate` on 2026-06-18 -> passed: backend default suite, `svelte-check` with 0 errors / 0 warnings, Vitest 85 files / 924 tests.
- Market-provenance threshold/shape hardening on 2026-06-18: `pytest tests/test_market_intelligence_provenance.py -q -ra` -> 9 passed; groundedness thresholds now must be finite numbers from 0 to 1, empty string evidence refs are rejected, and non-object claim entries fail closed instead of becoming incidental exceptions or false groundedness proof.
- M4/final-bucket guard bundle after market-provenance threshold/shape hardening on 2026-06-18: `pytest tests/test_market_intelligence_provenance.py tests/test_linkedin_continuous_verification.py tests/test_linkedin_empirical_register.py tests/test_sourcing_quality_kernel_evidence.py -q -ra` -> 54 passed.
- Post-market-provenance-threshold-shape-hardening `make validate` on 2026-06-18 -> passed: backend default suite, `svelte-check` with 0 errors / 0 warnings, Vitest 85 files / 924 tests.
- Final-bucket stdin verifier guard on 2026-06-18: `pytest tests/test_validate_linkedin_final_live_bucket_tool.py tests/test_linkedin_empirical_register.py tests/test_sourcing_quality_kernel_evidence.py -q -ra` -> 38 passed; the CLI path for `tools/validate_linkedin_final_live_bucket.py -` is now pinned with a complete payload instead of relying only on the shell evidence note.
- Post-final-bucket-stdin-verifier-guard `make validate` on 2026-06-18 -> passed: backend default suite, `svelte-check` with 0 errors / 0 warnings, Vitest 85 files / 924 tests.
- Global definition-of-done evidence guard on 2026-06-18: `pytest tests/test_search_string_lane_fields.py::test_work_unit_without_structured_filters_stays_boolean_byte_identical tests/test_seam_strategy_execution.py::test_all_keyword_lane_plan_keeps_legacy_queue_shape tests/test_phase0_contracts.py::test_run_log_event_vocabulary_matches_current_emitters tests/test_receipts.py -q -ra` -> 16 passed.
- Post-global-definition-of-done-evidence-guard `make validate` on 2026-06-18 -> passed: backend default suite, `svelte-check` with 0 errors / 0 warnings, Vitest 85 files / 924 tests.
- Invariant coverage evidence guard on 2026-06-18: `pytest tests/test_sourcing_quality_kernel_evidence.py tests/test_boolean_normalizer.py tests/test_designer_judging.py tests/test_matching_contract.py tests/test_observability_monitors.py tests/test_adaptation_signal_state.py -q -ra` -> 57 passed.
- Post-invariant-coverage-evidence-guard `make validate` on 2026-06-18 -> passed: backend default suite, `svelte-check` with 0 errors / 0 warnings, Vitest 85 files / 924 tests.
- Stop-rule/Empirical-Register evidence guard on 2026-06-18: `pytest tests/test_sourcing_quality_kernel_evidence.py tests/test_linkedin_empirical_register.py tests/test_validate_linkedin_final_live_bucket_tool.py -q -ra` -> 41 passed.
- Post-stop-rule-Empirical-Register-evidence-guard `make validate` on 2026-06-18 -> passed: backend default suite, `svelte-check` with 0 errors / 0 warnings, Vitest 85 files / 924 tests.
- Local SQK evidence Make gate on 2026-06-18: `make validate-sqk-evidence` -> 41 passed; the target runs the local evidence artifact guard, final live-bucket verifier tests, and Empirical Register tests without requiring live LinkedIn evidence.
- Post-local-SQK-evidence-Make-gate `make validate` on 2026-06-18 -> passed: backend default suite, `svelte-check` with 0 errors / 0 warnings, Vitest 85 files / 924 tests.
- Dependency-order evidence guard on 2026-06-18: `make validate-sqk-evidence` -> 42 passed; the artifact now explicitly tracks M0, M1A, M1B, M1C, M2, M3, and M4 dependency order alongside the final live-gate bucket.
- Post-dependency-order-evidence-guard `make validate` on 2026-06-18 -> passed: backend default suite, `svelte-check` with 0 errors / 0 warnings, Vitest 85 files / 924 tests.
- Final-bucket placeholder-evidence-ref hardening on 2026-06-18: `pytest tests/test_linkedin_empirical_register.py tests/test_validate_linkedin_final_live_bucket_tool.py -q -ra` -> 34 passed; literal `<live evidence artifact id>` placeholders now remain pending instead of satisfying matching-count, profile-ID, or cold-path proof.
- Post-placeholder-evidence-ref-hardening `make validate` on 2026-06-18 -> passed: backend default suite, `svelte-check` with 0 errors / 0 warnings, Vitest 85 files / 924 tests.
- Final-bucket matching-query self-description hardening on 2026-06-18: `make validate-sqk-evidence` -> 44 passed; the live payload must carry canonical `matching_queries` alongside `matching_counts`, so each supplied Recruiter count is tied to the exact query text from `linkedin.matching_contract`.
- Post-matching-query-self-description-hardening `make validate` on 2026-06-18 -> passed: backend default suite, `svelte-check` with 0 errors / 0 warnings, Vitest 85 files / 924 tests.
- Final-bucket matching-query report-preservation guard on 2026-06-18: `pytest tests/test_linkedin_empirical_register.py tests/test_validate_linkedin_final_live_bucket_tool.py tests/test_sourcing_quality_kernel_evidence.py -q -ra` -> 44 passed; completed verifier reports now preserve canonical `matching_queries` inside the derived matching-contract evidence.
- Post-matching-query-report-preservation `make validate` on 2026-06-18 -> passed: backend default suite, `svelte-check` with 0 errors / 0 warnings, Vitest 85 files / 924 tests.
- Final-bucket future-timestamp hardening on 2026-06-18: `pytest tests/test_linkedin_empirical_register.py tests/test_validate_linkedin_final_live_bucket_tool.py tests/test_sourcing_quality_kernel_evidence.py -q -ra` -> 45 passed; top-level verification, profile-ID probe, and cold-path timestamps must not be future-dated.
- Post-future-timestamp-hardening `make validate` on 2026-06-18 -> passed: backend default suite, `svelte-check` with 0 errors / 0 warnings, Vitest 85 files / 924 tests.
- Final-bucket supplied-evidence ordering hardening on 2026-06-18: `pytest tests/test_linkedin_empirical_register.py tests/test_validate_linkedin_final_live_bucket_tool.py tests/test_sourcing_quality_kernel_evidence.py -q -ra` -> 47 passed; malformed supplied `matching_counts` or `matching_queries` now report invalid before missing evidence refs or timestamps can mask them as pending.
- Post-supplied-evidence-ordering-hardening `make validate` on 2026-06-18 -> passed: backend default suite, `svelte-check` with 0 errors / 0 warnings, Vitest 85 files / 924 tests.
- Final-bucket prose-placeholder hardening on 2026-06-18: `pytest tests/test_linkedin_empirical_register.py tests/test_validate_linkedin_final_live_bucket_tool.py tests/test_sourcing_quality_kernel_evidence.py -q -ra` -> 48 passed; obvious placeholder evidence refs such as `TODO`, `TBD`, `placeholder`, or `live evidence artifact id` remain pending instead of satisfying final live proof fields.
- Post-prose-placeholder-hardening `make validate` on 2026-06-18 -> passed: backend default suite, `svelte-check` with 0 errors / 0 warnings, Vitest 85 files / 924 tests.
- Final-bucket proof-token/query-presence hardening on 2026-06-18: `pytest tests/test_linkedin_empirical_register.py tests/test_validate_linkedin_final_live_bucket_tool.py tests/test_sourcing_quality_kernel_evidence.py -q -ra` -> 50 passed; obvious non-evidence tokens such as `N/A`, `none`, `example`, `sample`, `dummy`, or `fake` remain pending as evidence refs, and malformed supplied `matching_queries` fail closed even when `matching_counts` is still missing.
- Post-proof-token/query-presence-hardening `make validate` on 2026-06-18 -> passed: backend default suite, `svelte-check` with 0 errors / 0 warnings, Vitest 85 files / 924 tests.
- Final-bucket placeholder-token separator hardening on 2026-06-18: `pytest tests/test_linkedin_empirical_register.py tests/test_validate_linkedin_final_live_bucket_tool.py tests/test_sourcing_quality_kernel_evidence.py -q -ra` -> 50 passed; separator variants and non-proof labels such as `example-proof`, `dummy_evidence`, `sample/ref`, `fake:evidence`, `missing evidence`, and `unknown` remain pending as evidence refs.
- Post-placeholder-token-separator-hardening `make validate` on 2026-06-18 -> passed: backend default suite, `svelte-check` with 0 errors / 0 warnings, Vitest 85 files / 924 tests.
- Final-bucket timezone-aware timestamp hardening on 2026-06-18: `pytest tests/test_linkedin_empirical_register.py tests/test_validate_linkedin_final_live_bucket_tool.py tests/test_sourcing_quality_kernel_evidence.py -q -ra` -> 51 passed; final-bucket `verified_at`, profile-probe `verified_at`, and cold-path `ran_at` values must include an explicit date, time, and timezone; date-only or timezone-less values cannot satisfy live proof timestamps.
- Post-timezone-aware-timestamp-hardening `make validate` on 2026-06-18 -> passed: backend default suite, `svelte-check` with 0 errors / 0 warnings, Vitest 85 files / 924 tests.
- Final-bucket supplied-count-ref ordering hardening on 2026-06-18: `pytest tests/test_linkedin_empirical_register.py tests/test_validate_linkedin_final_live_bucket_tool.py tests/test_sourcing_quality_kernel_evidence.py -q -ra` -> 52 passed; a supplied `matching_counts_evidence_ref` is validated even when `matching_counts` is still missing, so malformed refs fail closed and placeholder refs remain pending rather than hiding behind the broader missing-count gate.
- Post-supplied-count-ref-ordering-hardening `make validate` on 2026-06-18 -> passed: backend default suite, `svelte-check` with 0 errors / 0 warnings, Vitest 85 files / 924 tests.
- Final-bucket invalid-report fail-empty hardening on 2026-06-18: `pytest tests/test_linkedin_empirical_register.py tests/test_validate_linkedin_final_live_bucket_tool.py tests/test_sourcing_quality_kernel_evidence.py -q -ra` -> 53 passed; invalid final live-bucket reports now emit errors only, without derived matching-contract, profile-ID, cold-path, stale-policy, or pending-gate proof fields from incoherent payloads.
- Post-invalid-report-fail-empty-hardening `make validate` on 2026-06-18 -> passed: backend default suite, `svelte-check` with 0 errors / 0 warnings, Vitest 85 files / 924 tests.
- Final-bucket missing-schema fail-empty hardening on 2026-06-18: `pytest tests/test_linkedin_empirical_register.py tests/test_validate_linkedin_final_live_bucket_tool.py tests/test_sourcing_quality_kernel_evidence.py -q -ra` -> 53 passed; non-empty final live-bucket payloads without `schema_version` remain pending and emit no matching-contract, profile-ID, cold-path, or stale-policy proof fields.
- Post-missing-schema-fail-empty-hardening `make validate` on 2026-06-18 -> passed: backend default suite, `svelte-check` with 0 errors / 0 warnings, Vitest 85 files / 924 tests.
- Final LinkedIn live-bucket completion on 2026-06-18: `make validate-sqk-live-bucket SQK_LIVE_BUCKET=output/playwright/linkedin-live-evidence-2026-06-18T14-57-46-743Z/final-live-bucket.json` -> status complete, pending_gates []; evidence refs cover matching counts, profile-ID probe, save, drawer, pagination, fallback, and stale policy `alert_only`.

## Global Definition-of-Done Evidence

- Full-suite gate: `make validate` -> passed with the backend default suite, `svelte-check` with 0 errors / 0 warnings, and Vitest 85 files / 924 tests.
- Local SQK evidence gate: `make validate-sqk-evidence` validates the local SQK artifact and final live-bucket wiring without resolving deferred live-seat gates.
- Byte-identical-default guard: `tests/test_search_string_lane_fields.py::test_work_unit_without_structured_filters_stays_boolean_byte_identical` pins the plain generated-string fallback, and `tests/test_seam_strategy_execution.py::test_all_keyword_lane_plan_keeps_legacy_queue_shape` pins all-keyword queue shape.
- Run-log vocabulary guard: `tests/test_phase0_contracts.py::test_run_log_event_vocabulary_matches_current_emitters` scans production `log_event` emitters and requires exact equality with `RUN_LOG_EVENTS`.
- Receipt substrate guard: `tests/test_receipts.py` verifies missing receipt mirrors, hash tampering, and append-only triggers, and rejects invalid caller-supplied receipts.
- M2 baseline evidence remains explicit: `green_but_useless_rate` and `judge_parse_failure_rate` are reported in the M2 section below.
- Required artifact examples remain explicit below: the M0 sample typed receipt, M3 `Adapted-string firewall trace:`, M4 `Sample groundedness verdict:`, M4 `Sample accessibility drift report:`, and M4 `Sample cold-path fixture-runner report:`.

## Dependency Order Evidence

- M0 -> everything: the receipt/event-log substrate is proven first by `tests/test_receipts.py tests/test_runtime_state_*.py -q -ra` before downstream milestone evidence relies on typed receipts or run-log integrity.
- M1A -> M3: `pytest tests/test_designer_judging.py -q` proves fail-honest judging before `pytest tests/test_adaptation_*.py -q` evidence relies on judge-derived triage signals.
- M1B -> M1C: `pytest tests/test_matching_contract.py -q` proves the contract-owned matching model before `pytest tests/test_boolean_normalizer.py tests/test_seam_strategy_execution.py -q` evidence relies on lexical normalization and ubiquity enforcement.
- M2 depends on M0: `pytest tests/test_observability_monitors.py -q` reports run-level monitor evidence after typed receipt/run-log evidence is present.
- M4 remains last: `tests/test_market_intelligence_provenance.py tests/test_linkedin_continuous_verification.py` evidence is recorded only after M1A/M1B/M1C/M2/M3 local gates, and M4 still leaves live LinkedIn facts in the final bucket.
- M1B/M1C/M3 remain gated on the Empirical Register: matching counts, morphology/tokenization, profile-ID availability, and live-seat cold paths are not treated as complete until supplied through the final live-bucket verifier.

## Invariant Coverage Evidence

- INV-1: `linkedin/boolean_compiler.py`, `designer/judging.py`, `shared/receipts.py`, and `market_intelligence/provenance.py` enforce quality and status rules as code; `tests/test_boolean_normalizer.py`, `tests/test_designer_judging.py`, `tests/test_receipts.py`, and `tests/test_market_intelligence_provenance.py` pin those validators.
- INV-2: LinkedIn external behavior remains gated by `linkedin.empirical_register`; `validate_final_linkedin_live_bucket({})` still reports `pending`, and `tests/test_linkedin_empirical_register.py` verifies incomplete or malformed live evidence cannot complete the gate.
- INV-3: `linkedin/matching_contract.py` owns the matching-model facts, and `tests/test_matching_contract.py` scans production LinkedIn code for independent hardcoded matching claims.
- INV-4: Search method is reasoned then enforced at the execution seam, not authored as brief truth; `tests/test_seam_strategy_execution.py` and `tests/test_search_string_lane_fields.py` verify structured-filter propagation, queue-time normalization, and default keyword behavior.
- INV-5: `tests/test_receipts.py` verifies typed intended-vs-actual receipts and append-only event-log integrity, while `tests/test_observability_monitors.py` verifies green-but-useless and parse-failure monitors.
- INV-6: Byte-identical default behavior is pinned by `tests/test_search_string_lane_fields.py::test_work_unit_without_structured_filters_stays_boolean_byte_identical` and `tests/test_seam_strategy_execution.py::test_all_keyword_lane_plan_keeps_legacy_queue_shape`.
- INV-7: Guard coverage is itself guarded: `tests/test_sourcing_quality_kernel_evidence.py` requires milestone, global DoD, invariant, and final live-bucket evidence, while the milestone suites named above fail if their corresponding guard is removed.
- INV-8: `shared/receipts.py rejects boolean statuses`; `tests/test_receipts.py`, `tests/test_linkedin_judge_receipts.py`, and `tests/test_designer_judging.py` verify typed statuses for pipeline, LLM, and judge outcomes.
- INV-9: Adaptation uses the same firewall as initial execution; `tests/test_adaptation_signal_state.py::test_adapt_after_block_uses_typed_signal_state_and_validates_actions` verifies adapted strings pass the M1C normalizer before returning `adaptation.new_strings`.

## Stop-Rule and Empirical Gate Evidence

- product-judgment decision stop: no local artifact resolves pool breadth, filter-as-bound, who-gets-surfaced, ubiquity prevalence thresholds, autonomous reset policy, or stale-contract deploy policy by assumption.
- external-system fact not yet verified stop: LinkedIn behavior remains gated behind the final live bucket; `validate_final_linkedin_live_bucket({})` reports `pending`.
- high-risk files stop: high-risk files remain `linkedin/orchestrator.py`, `linkedin/browser.py`, and `shared/runtime_state/store.py`; changes outside a declared milestone scope remain pause conditions, not cleanup targets.
- test weakening or deletion stop: no evidence entry relies on weakened or deleted tests; guard coverage is enforced by `tests/test_sourcing_quality_kernel_evidence.py`.
- verification-command output stop: completion evidence is verifier output only, including focused pytest bundles, `make validate`, and the final live-bucket verifier.
- Keywords field stems / plural-collapses: final live bucket requires Recruiter counts for `benchmark`, `benchmarks`, and their union before this external fact can complete.
- Hyphen/space tokenization: final live bucket requires Recruiter counts for `fine-tuning`, `"fine tuning"`, `finetuning`, and their union before this external fact can complete.
- Skills-facet expansion fires on Keywords field: final live bucket requires Recruiter counts for `SaaS` versus `SaaS OR "Software as a Service"` before this external fact can complete.
- Recruiter API returns per-query profile IDs: final live bucket requires a positive stable-profile-ID probe before overlap/capture-recapture can use that signal.
- Current green-but-useless run rate: measured in M2 as `green_but_useless_rate` over the recent-run sample below.
- Current judge parse-failure rate: measured in M2 as `judge_parse_failure_rate` over the recent-run sample below.
- Operational M4 stop gates: stale matching contracts require an explicit `alert_only` or `deploy_blocking` policy decision, and live-seat cold paths require supplied live evidence before the final bucket can complete.

## M0 Evidence

Sample typed receipt:

```json
{"actual_detail":{"fell_back_to_keyword":false,"structured_applied":["companies"]},"actual_status":"ok","created_at":"2026-06-17T00:00:00+00:00","input_hash":"sha256:34a59f181334eea11812d103cb9a7de44c8992dff06e7cdef5aa496d8bae1eae","intended_postcondition":"structured filters land on the live Recruiter sidebar","producer":"evidence-sample","receipt_id":"sha256:beb6b7a098278f7fe27898896f07c189c286f3f764076d2724efebf0e10c052e","receipt_type":"pipeline_stage","schema_version":"receipt.v1","stage":"surface_applied","version_pins":{"sample":"2026-06-17"}}
```

Guard evidence:

- `shared/receipts.py` rejects boolean statuses.
- `shared/receipts.py` rejects non-string receipt-envelope fields and non-string/non-empty version-pin keys or values, so caller-supplied receipts cannot become well-formed by Python stringification.
- `shared/runtime_state/event_log.py` mirrors runtime events into append-only `run_event_log` rows with a hash chain and typed error/postcondition statuses.
- `shared/storage.py` attaches typed `pipeline_stage` receipts to JSONL `log_event` rows, including `error` status for error events and `postcondition_fail` for keyword-fallback surface events.
- `shared/storage.py` validates caller-supplied JSONL event receipts through `Receipt.from_dict` before writing, so an invalid supplied receipt cannot bypass INV-8.
- `shared/storage.py` rejects JSONL `log_event` writes whose event name is not registered in `RUN_LOG_EVENTS`, forcing new run-log vocabulary through the frozen contract before it can land in artifacts.
- `shared/llm_usage.py` attaches typed `llm_call` receipts to token-cost JSONL rows and returns `_llm_receipt` through mutable `usage_context` when no JSONL session is active, while preserving Langfuse disabled-vs-keys-absent JSONL byte equivalence.
- `shared/llm_clients.py` routes `cheap_llm` OpenAI, Google, and Anthropic provider branches through the same `record_llm_usage` receipt path.
- `shared/llm_clients.py` records typed `llm_call` receipts with `actual_status=error` on final provider failures before re-raising the original exception, including the streamed Opus wrapper.
- `cloris/anthropic_health.py` records typed `llm_call` receipts for the direct Anthropic launch-readiness probe on both healthy and error outcomes.
- `designer/vision_evaluation.py` records typed `llm_call` receipts for direct Gemini and Claude vision calls on both success and provider/import/client error outcomes, without changing the existing fallback or JSON parsing behavior.
- `market_intelligence/research_agent.py` records typed `llm_call` error receipts for direct Anthropic web-search and Perplexity external-research provider failures before preserving the original exception path.
- `shared/external_evidence/provider.py` records typed `llm_call` error receipts for candidate-level Perplexity provider failures before returning the existing typed `ExternalEvidenceFailure` classification.
- `tests/test_llm_cost_contracts.py` pins that all production `cheap_llm`, `facial_llm`, `opus_llm`, `opus_llm_cached`, and `opus_llm_cached_stream` call sites across `cloris/`, `designer/`, `exec_search/`, `github/`, `linkedin/`, `market_intelligence/`, `researcher/`, and `shared/` pass mutable `usage_context` so the LLM receipt hook is present.
- LinkedIn list/card/profile extraction, LinkedIn glance checks, GitHub strategy/adaptation/portfolio/outreach, Researcher strategy/adaptation, Designer and Exec Search default judging calls, LinkedIn run-report debriefing, intake slot extraction, and conversation query/narration now pass stage/source metadata through the LLM receipt hook without changing prompts or control flow.
- `tests/test_receipts.py` verifies missing receipt mirrors, hash tampering, and append-only triggers.
- `tests/test_receipts.py` verifies runtime-state mirror receipts preserve `error` and `postcondition_fail` statuses instead of collapsing every event to `ok`.
- `tests/test_receipts.py` verifies JSONL `log_event` rows carry typed receipts with stage, input hash, and status semantics.
- `tests/test_receipts.py` verifies caller-supplied JSONL event receipts are preserved when valid and rejected when they collapse status to a boolean.
- `tests/test_receipts.py` verifies unregistered JSONL run-log events raise before any row is written.
- `tests/test_receipts.py` also pins that receipt-backed `pipeline_*` and `surface_*` events stay registered in `RUN_LOG_EVENTS`.
- `tests/test_phase0_contracts.py::test_run_log_event_vocabulary_matches_current_emitters` now scans production `log_event` emitters plus fallback callback emitters across the repo and fails if any discovered run-log event is missing from `RUN_LOG_EVENTS` or any registered event is not emitted.
- `tests/test_llm_cost_contracts.py` verifies LLM usage receipts carry stage, status, input hash, and token details.
- `tests/test_llm_clients.py` verifies provider failure writes an `llm_call` error receipt without weakening the original exception path.
- `tests/test_cloris_anthropic_health.py` verifies the launch-readiness probe emits `ok` and `error` LLM usage receipts without changing readiness blocker behavior.
- `tests/test_designer_vision_evaluation.py` verifies direct Gemini and Claude vision wrappers emit `ok` and `error` receipts while preserving provider exceptions.
- `tests/test_market_intelligence.py` verifies direct Anthropic and Perplexity market-research wrappers emit `error` receipts while preserving provider exceptions.
- `tests/test_external_evidence.py` verifies candidate external-evidence provider errors emit `error` receipts while preserving the existing typed failure return contract.

Sample LLM receipt excerpt:

```json
{"actual_detail":{"input_tokens":1000,"output_tokens":200,"provider":"anthropic"},"actual_status":"ok","receipt_type":"llm_call","stage":"llm:facial"}
```

## M1A Evidence

- `designer/judging.py` emits judge receipts and separates parse/refusal states from negative decisions.
- `tests/test_designer_judging.py` asserts non-OK parse statuses do not map to `REJECT` and model refusals emit `actual_status=refused` while preserving `PARSE_FAILURE` as the candidate decision.
- `shared/judger.py` now attaches a typed `judge_receipt` to LinkedIn facial/full `prompt_capture` for OK, `PARSE_FAILURE`, and `JUDGMENT_FAILURE` returns without changing `OpusDecision.to_dict()` persistence shape.
- `linkedin/judgment_templates.py` classifies refusal-shaped raw judge outputs as fail-honest `PARSE_FAILURE` with refusal rationale.
- `tests/test_linkedin_judge_receipts.py` asserts LinkedIn facial parse failures emit `actual_status=parse_fail`, model refusals emit `actual_status=refused`, full saves emit `actual_status=ok`, and judgment exceptions emit `actual_status=error`.

Sample LinkedIn judge receipt excerpt:

```json
{"actual_detail":{"final_decision":"PARSE_FAILURE","parse_status":"parse_fail","render_route":"linkedin.facial.v2_structural"},"actual_status":"parse_fail","receipt_type":"judge","stage":"linkedin_facial_judge"}
```

## M1B Evidence

- `linkedin/matching_contract.py` owns the matching contract and keeps unverified facts labeled.
- `tests/test_matching_contract.py` scans the full `linkedin/` production tree, excluding the canonical `matching_contract.py`, for forbidden independent hardcoded matching claims.
- The Empirical Register rows for keyword stemming, hyphen/space tokenization, and Keywords-field skills-facet expansion are all typed contract fields.
- Seat-test-derived values are still gated. The unverified contract refuses `require_verified()`.
- Verified matching-contract values now require ISO-parseable `verified_at`, so `last_empirically_verified` cannot be malformed before M4 freshness checks consume it.
- Verified matching-contract values now also require typed `MatchingFact` keys, fact-specific enum values, and object evidence, so direct contract construction cannot bypass the seat-count builder with stringified external behavior.
- `linkedin/empirical_register.py` validates supplied final live evidence and refuses completion while any required count, matching-count evidence reference, profile-ID probe, cold-path result, or stale-contract policy is missing.

## M1C Evidence

Before/after fixture examples:

- Surface conflict stripping: `("Nubank" OR "Bancolombia" OR "fintech") AND ("ML Engineer" OR "platform")` -> `("fintech") AND ("platform")`; findings `surface_conflict_stripped`.
- Token-subset pruning: `("reward model" OR "reward model development")` -> `("reward model")`; findings `token_subset_superstring_pruned`.
- Ubiquitous gate: `("Python") AND ("PyTorch")` remains byte-identical but reports `ubiquitous_and_gate`.
- `shared.schemas.SearchString` now carries a default-empty `boolean_normalization` report, and `shared.sourcing_lanes.lane_fields_from_work_unit_item` propagates producer/adaptation normalizer reports into executable strings without changing legacy keyword-only strings.
- `linkedin/orchestrator.py` now runs generated compound strings, coverage-gap strings, and lane-only strings through the deterministic M1C normalizer before `SearchString` execution; structured-filter conflicts and token-subset redundancies are removed at the queue seam rather than merely represented if a producer already attached metadata.
- `linkedin/boolean_compiler.py` exposes `normalize_execution_work_item_boolean`, which honors only local explicit inputs (structured filters plus caller-supplied locale/morphology/ubiquity data), rejects malformed explicit rule inputs before they can be stringified, and fails closed on an explicit ubiquitous-term AND gate.
- `linkedin/surface_receipt.py` now includes `boolean_normalization` summaries on intended and applied surface receipts and aggregates explicit `normalization_guard_counts` for `ubiquitous_and_gate` and `token_subset_superstring_pruned`; zero remains explicit when no finding is present.
- The human-readable intended surface summary and the `surface_intended` run-log event now both expose `normalization_guard_counts`; the log event also records `normalization_strings_with_findings` and `normalization_finding_counts` for verifier-friendly receipt checks.
- `tests/test_seam_strategy_execution.py` pins queue-time normalization for generated strings and coverage gaps, and verifies explicit ubiquitous-term AND gates are rejected before execution.
- `tests/test_surface_receipt.py` pins that surface receipts report both guard counts and carry finding terms into `surface_applied` fields, including zero-count guard reporting.

Sample surface-normalization receipt excerpt:

```json
{"normalization_guard_counts":{"token_subset_superstring_pruned":1,"ubiquitous_and_gate":1},"normalization_strings_with_findings":2}
```

## M2 Evidence

Current baseline rates from `.venv/bin/python - <<'PY' ... current_baseline_rates("output", recent_limit=50)`:

```json
{"green_but_useless_rate":1.0,"green_but_useless_runs":50,"judge_decisions":0,"judge_parse_failure_rate":0.0,"judge_parse_failures":0,"runs_measured":50}
```

- Parse-failure monitor rates are evidence-derived from explicit runtime-state decision columns or valid parsed payload JSON; malformed raw payload text is ignored as non-proof, and baseline aggregates reject malformed monitor rows or incoherent judge counts before computing rates.

## M3 Evidence

Adapted-string firewall trace:

```json
{"firewall":{"passed":true,"reports":[{"changed":true,"findings":[{"code":"token_subset_superstring_pruned","message":"Superstring terms were pruned by explicit token-subset rule.","terms":["reward model development"]}],"normalized_boolean":"(\"reward model\")","original_boolean":"(\"reward model\" OR \"reward model development\")"}]},"new_strings":[{"boolean":"(\"reward model\")","boolean_normalization":{"changed":true,"findings":[{"code":"token_subset_superstring_pruned","message":"Superstring terms were pruned by explicit token-subset rule.","terms":["reward model development"]}],"normalized_boolean":"(\"reward model\")","original_boolean":"(\"reward model\" OR \"reward model development\")"}}]}
```

Local constraints and stop gates still held:

- SPRT thresholds and sufficiency calibration are explicit, type/range-validated parameters, not tuned as product facts.
- Profile-ID availability remains unverified by default; overlap is unavailable until a verified `ProfileIdAvailabilityContract` with evidence and an ISO-parseable `verified_at` is supplied, even if fixture rows contain profile IDs.
- Autonomous reset is blocked unless explicitly allowed.
- `_run_block_adaptation` now evaluates the typed `SearchSignalState` through an explicit `AdaptationGateConfig` before calling `adapt_after_block`; under-sufficient blocks emit an `adaptation_decision` event with `collect_more_signal` and clear the pending adaptation without spending an LLM call.
- `adapt_after_block` now applies the same M1C token-subset normalizer on the real adapted-string path before returning `adaptation.new_strings`, and `tests/test_adaptation_signal_state.py::test_adapt_after_block_uses_typed_signal_state_and_validates_actions` verifies the real path prunes an adapted redundant superstring.
- `apply_adapted_string_firewall` now rejects malformed adapted-string local rule inputs instead of silently ignoring invalid `structured_filters`, and reports M1C normalizer input failures through the adaptation validation contract.
- Market intelligence enters adaptation as a typed `MarketSignalPrior`, not a raw prose blob.
- Mapping-form `MarketSignalPrior` input fails closed on non-object signal rows, non-string source/signal/recommendation/evidence fields, and non-numeric confidence; free-text advisory context remains the only permissive parsing path.
- `linkedin/orchestrator.py` page-level adaptation also converts live advisory markdown to `## Typed MarketSignalPrior` before prompt insertion.

Typed market-prior fixture excerpt:

```json
{"context_hash":"sha256:c9c1385ffc0262fb9811265370bc2cd022e147e39b373d46505a77371876a24e","generated_at":null,"signals":[{"confidence":null,"evidence_ref_ids":[],"recommendation":"Lean into this lane.","signal_type":"exploit"}],"source":"market_intelligence.live_advisory"}
```

## M4 Evidence

- `market_intelligence/engine.py` now annotates artifact narrative records with `typed_evidence_refs` and `groundedness` verdicts during `_build_artifact`, and persists the full grounded/quarantined claim report under `evidence_index.groundedness`.
- `market_intelligence/provenance.py` rejects empty string evidence refs, mapping-form evidence refs whose typed fields are not strings, non-object evidence-ref metadata, malformed market-claim entries, and groundedness thresholds outside finite 0-to-1 numeric bounds; it ignores non-string metadata as support text and rejects mapping-form market claims with non-string claim IDs/text or non-object metadata, so invalid provenance cannot become groundedness evidence through stringification or threshold misuse.
- `market_intelligence/research_agent.py` preserves Perplexity source snippets in `evidence_index.external_sources` so groundedness checks can use provider-supplied support text rather than self-grounding from the claim.
- `linkedin/empirical_register.py` now requires supplied final-bucket timestamps (`verified_at`, optional profile-ID probe `verified_at`, and each cold-path `ran_at`) to include an explicit date, time, and timezone, parse as ISO timestamps, and not be future-dated; it also requires the profile-ID probe result flag to be a real boolean, rejects unknown cold-path status labels, normalizes completed profile-ID and cold-path proof rows, and owns the canonical placeholder payload template from the same registries as the verifier, so the final LinkedIn live bucket is machine-checkable rather than merely non-empty.
- `linkedin/empirical_register.py` treats literal placeholder evidence refs such as `<live evidence artifact id>` plus obvious prose placeholders and non-evidence tokens such as `TODO`, `TBD`, `placeholder`, `live evidence artifact id`, `N/A`, `none`, `example`, `sample`, `dummy`, `fake`, `missing evidence`, and `unknown` as missing proof, including separator forms such as `example-proof` or `dummy_evidence`; a partially filled template cannot accidentally complete the final live gate.
- `linkedin/empirical_register.py` requires canonical `matching_queries` when matching counts are supplied, tying each live Recruiter count to the exact query text before a matching contract can be derived.
- Completed final-bucket matching-contract reports preserve canonical `matching_queries`, so the verifier output remains self-describing rather than requiring the original payload to audit which query produced each count.
- Supplied malformed `matching_counts`, `matching_queries`, or `matching_counts_evidence_ref` fail closed before missing supporting fields are reported, so incoherent external evidence cannot be hidden as a simple pending gate.
- Invalid final live-bucket reports fail empty: `matching_contract`, `profile_id_availability`, `cold_path_results`, `stale_matching_contract_policy`, and `pending_gates` are withheld when any supplied evidence is incoherent.
- `linkedin/continuous_verification.py` now validates accessibility-tree fixture snapshot maps as objects with non-empty string selector keys, normalizes snapshot objects to the validated map selector, and requires string `role`/`name` fields; it also includes a fixture-only due-probe runner for the cold-path registry, executes stale/never-run non-live probes, reports malformed probe metadata as typed `invalid_probe`, requires closed-schema fixture runner outcomes with an explicit `ok` or `error` status and a string evidence reference for `ok`, rejects shorthand or malformed fixture outcome fields as `invalid_outcome`, validates report configuration timestamp and policy types before emission, reports fixture exceptions as typed `error`, treats malformed mapping-form matching-contract timestamps as unverified/invalid instead of stale/fresh proof, and keeps live-seat probes as `pending_live_seat`.
- `tests/test_market_intelligence.py::test_build_artifact_attaches_groundedness_without_optional_brief_fixture` verifies artifact-level typed evidence refs and groundedness across every groundable synthesized section (`lane_intelligence`, `talent_pool_intelligence`, `noise_patterns`, `employer_signal_intelligence`, `brief_recommendations`, `open_questions`, and `market_thesis.external_context`) and proves unsupported valid claims remain in `evidence_index.groundedness.quarantined_claims` without depending on optional local brief fixtures.
- `tests/test_linkedin_empirical_register.py::test_final_live_bucket_rejects_non_iso_timestamps` verifies malformed timestamp evidence cannot complete the final live gate.
- `tests/test_linkedin_continuous_verification.py` verifies the accessibility drift detector, default cold-path schedule, fixture cold-path runner, live-seat bucketing, and matching-contract freshness alerts.

Sample groundedness verdict:

```json
{"claim_id":"claim-asset-management","missing_terms":[],"rationale":"Evidence refs cover the claim's material terms.","status":"grounded","supported_ref_ids":["run:linkedin:123"]}
```

Sample accessibility drift report:

```json
{"alert_only":true,"cadence_days":7,"drifts":[{"baseline":{"name":"Save","role":"button","selector":"button[data-test-save]"},"current":{"name":"Save candidate","role":"button","selector":"button[data-test-save]"},"selector":"button[data-test-save]","status":"name_changed"}],"generated_at":"2026-06-17T00:00:00+00:00","status":"alert"}
```

Sample cold-path fixture-runner report:

```json
{"alert_only":true,"generated_at":"2026-06-17T00:00:00+00:00","results":[{"days_since_last_run":16,"error":null,"evidence_ref":"fixture:save:20260617","max_silence_days":7,"name":"save","ran_at":"2026-06-17T00:00:00+00:00","requires_live_seat":false,"status":"ok","was_due":true},{"days_since_last_run":null,"error":null,"evidence_ref":null,"max_silence_days":7,"name":"fallback","ran_at":null,"requires_live_seat":true,"status":"pending_live_seat","was_due":true}],"status":"alert"}
```

## Final LinkedIn Live Bucket

Do not mark the full goal complete until these are supplied or explicitly
resolved. The expected evidence payload is machine-checkable through
`linkedin.empirical_register.require_final_linkedin_live_bucket_complete`.

- Recruiter seat-test counts for `benchmark`, `benchmarks`, `benchmark OR benchmarks`.
- Recruiter seat-test counts for `fine-tuning`, `"fine tuning"`, `finetuning`, and their union.
- Recruiter seat-test counts for `SaaS` vs `SaaS OR "Software as a Service"`.
- Live evidence for stable per-result profile IDs before capture-recapture/overlap uses that signal.
- A policy decision on stale matching contracts: keep alert-only or make deploy-blocking.
- Any cold-path probe that requires a live LinkedIn seat.

Empty-payload verifier output with no supplied live-seat payload:

```text
status: pending
pending_gates: cold_path_results,matching_counts,profile_id_probe,stale_matching_contract_policy
```

Required payload shape (placeholders shown intentionally; replace with real live
evidence before running the verifier):

- Print the canonical template:
  `.venv/bin/python tools/validate_linkedin_final_live_bucket.py --template`
  or `make sqk-live-bucket-template`
- Validate a filled JSON payload:
  `.venv/bin/python tools/validate_linkedin_final_live_bucket.py <live-bucket.json>`
  or `make validate-sqk-live-bucket SQK_LIVE_BUCKET=<live-bucket.json>`

```text
schema_version: linkedin.final_live_bucket.v1
verified_at: <live verification timestamp>
matching_counts:
  benchmark: <Recruiter result count>
  benchmarks: <Recruiter result count>
  benchmark_or_benchmarks: <Recruiter result count>
  fine_tuning_hyphenated: <Recruiter result count>
  fine_tuning_spaced: <Recruiter result count>
  finetuning_closed: <Recruiter result count>
  fine_tuning_union: <Recruiter result count>
  saas_keyword: <Recruiter result count>
  saas_or_software_as_a_service: <Recruiter result count>
matching_queries:
  benchmark: benchmark
  benchmarks: benchmarks
  benchmark_or_benchmarks: benchmark OR benchmarks
  fine_tuning_hyphenated: fine-tuning
  fine_tuning_spaced: "fine tuning"
  finetuning_closed: finetuning
  fine_tuning_union: fine-tuning OR "fine tuning" OR finetuning
  saas_keyword: SaaS
  saas_or_software_as_a_service: SaaS OR "Software as a Service"
matching_counts_evidence_ref: <live evidence artifact id>
profile_id_probe:
  stable_profile_id_seen: true
  sample_size: <positive inspected-result count>
  evidence_ref: <live evidence artifact id>
cold_path_results:
  - name: save
    status: ok
    ran_at: <live probe timestamp>
    evidence_ref: <live evidence artifact id>
  - name: drawer
    status: ok
    ran_at: <live probe timestamp>
    evidence_ref: <live evidence artifact id>
  - name: pagination
    status: ok
    ran_at: <live probe timestamp>
    evidence_ref: <live evidence artifact id>
  - name: fallback
    status: ok
    ran_at: <live probe timestamp>
    evidence_ref: <live evidence artifact id>
stale_matching_contract_policy: alert_only | deploy_blocking
```

Supplied live-bucket verifier output:

```text
SQK_LIVE_BUCKET=output/playwright/linkedin-live-evidence-2026-06-18T14-57-46-743Z/final-live-bucket.json
status: complete
pending_gates:
matching_contract: keyword_stemming=no_stemming, hyphen_space_tokenization=distinct, keywords_skills_facet_expansion=no_keywords_expansion
profile_id_probe: stable_profile_id_seen=true, sample_size=5
cold_path_results: save=ok, drawer=ok, pagination=ok, fallback=ok
stale_matching_contract_policy: alert_only
```
