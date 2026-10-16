SHELL := /bin/bash

ROOT := $(abspath $(CURDIR))
PYTHON ?= $(if $(wildcard $(ROOT)/.venv/bin/python),$(ROOT)/.venv/bin/python,python3)

# Default-off feature flags carry dated decision gates enforced by
# tools/flag_ledger.py. In a pinned snapshot, evaluate gates as of the
# snapshot date unless overridden on the command line.
FLAG_LEDGER_AS_OF ?= 2026-10-15

.PHONY: help validate hygiene test-default test-full validate-sqk-evidence sqk-live-bucket-template validate-sqk-live-bucket

help: ## Show available targets
	@printf "\nAvailable targets:\n\n"
	@awk 'BEGIN {FS = ":.*## "}; /^[a-zA-Z0-9_-]+:.*## / {printf "  make %-26s %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@printf "\n"

validate: hygiene test-default ## Run repository hygiene, flag ledger, and credential-free test suite

hygiene: ## Run repository hygiene checks and feature-flag ledger
	$(PYTHON) tools/check_repo_hygiene.py
	$(PYTHON) tools/flag_ledger.py --as-of $(FLAG_LEDGER_AS_OF)

test-default: ## Run the default credential-free validation suite
	$(PYTHON) tools/run_validation.py default

test-full: ## Run the full pytest suite including extended replay tests
	$(PYTHON) tools/run_validation.py full

validate-sqk-evidence: ## Validate the sourcing-quality-kernel evidence artifact and final-gate wiring
	$(PYTHON) -m pytest \
		tests/test_sourcing_quality_kernel_evidence.py \
		tests/test_linkedin_empirical_register.py \
		tests/test_validate_linkedin_final_live_bucket_tool.py \
		-q -ra

sqk-live-bucket-template: ## Print the LinkedIn live-bucket JSON template
	$(PYTHON) tools/validate_linkedin_final_live_bucket.py --template

validate-sqk-live-bucket: ## Validate a LinkedIn live-bucket JSON (set SQK_LIVE_BUCKET=/path.json)
	@if [ -z "$(SQK_LIVE_BUCKET)" ]; then \
		echo "validate-sqk-live-bucket: set SQK_LIVE_BUCKET=/path/to/live-bucket.json" >&2; \
		exit 2; \
	fi
	$(PYTHON) tools/validate_linkedin_final_live_bucket.py "$(SQK_LIVE_BUCKET)"
