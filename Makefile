SHELL := /bin/zsh

ROOT := $(abspath $(CURDIR))
PYTHON ?= $(if $(wildcard $(ROOT)/.venv/bin/python),$(ROOT)/.venv/bin/python,python3)

HEAD_AI_BRIEF ?= $(ROOT)/config/brief-head-ai-lab-nyc-v2.json
HEAD_AI_RUN_DIR ?= $(ROOT)/output/runs/linkedin/3000000006/imported-2026-04-09T21-22-22-384264+00-00__legacy-2
HEAD_AI_SEARCH_MEMORY ?= $(HEAD_AI_RUN_DIR)/search_memory-3000000006.json
HEAD_AI_FINAL_JUDGMENTS ?= $(HEAD_AI_RUN_DIR)/final_judgments.jsonl

FDE_BRIEF ?= $(ROOT)/config/Forward-Deployed-Engineer-NYC/brief-forward-deployed-engineer-us-v1.4.json
FDE_RUN_DIR ?= $(ROOT)/output/runs/linkedin/3000000007/2026-04-12T12-26-33-178377+00-00__run-3
FDE_SEARCH_MEMORY ?= $(FDE_RUN_DIR)/search_memory-3000000007.json
FDE_FINAL_JUDGMENTS ?= $(FDE_RUN_DIR)/final_judgments.jsonl

.PHONY: help validate hygiene test-default frontend-validate frontend-build validate-intake validate-static-assets validate-package-smoke validate-package-modules validate-sqk-evidence sqk-live-bucket-template validate-sqk-live-bucket refresh-local-app qa-demo-app certify-tier0-fast certify-tier0-package certify-tier0-browser certify-tier0-browser-package certify-tier0-full certify-tier1 certify-tier2 certify-product dev test-full audit-ui audit-css-tokens head-ai-mi head-ai-brief fde-mi fde-brief

help: ## Show available shortcuts
	@printf "\nAvailable shortcuts:\n\n"
	@awk 'BEGIN {FS = ":.*## "}; /^[a-zA-Z0-9_-]+:.*## / {printf "  make %-16s %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@printf "\nOverride any path if needed, e.g.:\n"
	@printf "  make head-ai-mi HEAD_AI_RUN_DIR=/abs/path/to/other/run\n\n"

# 2026-08-02: the Svelte app is a RELIC. Cloris was once intended as a
# sellable product with a web UI; it is now strictly an internal tool run
# from the terminal (briefs arrive as config JSON to `load_brief` via
# tools/launch_*.sh, and the live run path imports zero cloris.* modules).
# `frontend-validate` therefore no longer gates day-to-day work — it was
# spending svelte-check + 937 Vitest tests + a Node/pnpm toolchain on every
# wave close for zero signal about the tool that actually ships. The target
# is KEPT and still runnable by hand; restoring the gate is appending
# `frontend-validate` back onto this line.
validate: hygiene test-default ## Run hygiene + the Python suite (frontend is a relic; see frontend-validate)

hygiene: ## Check repo hygiene and brief lifecycle inventory
	$(PYTHON) tools/check_repo_hygiene.py

test-default: ## Run the default green validation suite
	$(PYTHON) tools/run_validation.py default

# Frontend gate honors pnpm-lock.yaml — the same resolution dev and
# packaging (build-app.sh) use. Requires Node 20+ and pnpm on PATH.
frontend-validate: ## Run svelte-check + vitest against the parked copy in attic/frontend-2026-08
	@command -v node >/dev/null 2>&1 || { echo >&2 "frontend-validate: node not found (install Node 20+)"; exit 1; }
	@command -v pnpm >/dev/null 2>&1 || { echo >&2 "frontend-validate: pnpm not found (npm install -g pnpm)"; exit 1; }
	cd $(ROOT)/attic/frontend-2026-08 && \
		pnpm install --frozen-lockfile && \
		pnpm run check && pnpm run test

# Rebuild the frontend dist served by `python -m cloris start`. Use this
# whenever attic/frontend-2026-08/src/ changes — without it, the local Python
# server keeps serving the stale dist/ directory and the browser shows
# old chrome. The packaged Cloris.app bundles its own dist at packaging
# time, so this only affects the local-dev backend path.
frontend-build: ## Rebuild the parked frontend dist (the in-tree shell no longer serves it; see attic/README.md)
	@command -v node >/dev/null 2>&1 || { echo >&2 "frontend-build: node not found (install Node 20+)"; exit 1; }
	@command -v npm >/dev/null 2>&1 || { echo >&2 "frontend-build: npm not found"; exit 1; }
	cd $(ROOT)/attic/frontend-2026-08 && npm run build

validate-intake: ## Fast Tier-0 intake recovery/source strategy gate
	$(PYTHON) -m pytest \
		tests/test_source_capabilities.py \
		tests/test_source_packet.py \
		tests/test_intake_source_packet.py \
		tests/test_intake_conversation_extractor.py \
		tests/test_intake_conversation_orchestrator.py \
		tests/test_intake_conversation_voice.py \
		tests/test_intake_conversation_endpoint.py \
		-q
	cd $(ROOT)/attic/frontend-2026-08 && npm run test -- \
		src/components/__tests__/IntakeConversation.test.ts \
		src/components/__tests__/OnboardingFlow.test.ts \
		src/test/onboarding-api.test.ts \
		src/lib/onboarding/__tests__/brief_route_boot.test.ts
	cd $(ROOT)/attic/frontend-2026-08 && npm run check
	$(MAKE) frontend-build

validate-static-assets: ## Validate built frontend dist static asset contract
	$(PYTHON) tools/check_static_assets.py --dist $(ROOT)/attic/frontend-2026-08/dist
	$(PYTHON) -m pytest tests/test_static_asset_contract.py tests/test_dist_staleness_guard.py tests/test_bearer_auth.py -q

validate-package-smoke: ## Validate local dist/Cloris.app bundle shape and bundled assets
	$(PYTHON) tools/check_static_assets.py --app $(ROOT)/dist/Cloris.app

# Two-part packaged-app freshness pre-flight:
#   (1) cert-critical Python modules import cleanly inside the frozen
#       bundle (catches PyInstaller dropping a function-local import);
#   (2) the bundle's stamped content-hash fingerprint matches the
#       current source tree (catches "the .app is stale relative to
#       source," which the module check alone cannot — a stale bundle
#       that happens to retain the named modules would otherwise pass).
# See tools/check_packaged_modules.py for the contract.
validate-package-modules: ## Smoke-check dist/Cloris.app for missing modules AND source-tree freshness
	$(PYTHON) tools/check_packaged_modules.py --app $(ROOT)/dist/Cloris.app

validate-sqk-evidence: ## Validate local SQK evidence artifact and final-gate wiring
	$(PYTHON) -m pytest \
		tests/test_sourcing_quality_kernel_evidence.py \
		tests/test_linkedin_empirical_register.py \
		tests/test_validate_linkedin_final_live_bucket_tool.py \
		-q -ra

sqk-live-bucket-template: ## Print the final LinkedIn live-bucket JSON template
	$(PYTHON) tools/validate_linkedin_final_live_bucket.py --template

validate-sqk-live-bucket: ## Validate final LinkedIn live-bucket JSON (set SQK_LIVE_BUCKET=/path.json)
	@if [ -z "$(SQK_LIVE_BUCKET)" ]; then \
		echo "validate-sqk-live-bucket: set SQK_LIVE_BUCKET=/path/to/live-bucket.json" >&2; \
		exit 2; \
	fi
	$(PYTHON) tools/validate_linkedin_final_live_bucket.py "$(SQK_LIVE_BUCKET)"

refresh-local-app: ## Rebuild, sync, cache-clear, relaunch, and verify local Cloris.app
	CLORIS_PYINSTALLER_ARCH=$${CLORIS_PYINSTALLER_ARCH:-arm64} \
		PYTHON=$(PYTHON) \
		./cloris/packaging/scripts/refresh-local-app.sh

qa-demo-app: ## Run Head AI product data/layout stability QA against running Cloris.app
	@url="$${CLORIS_APP_URL:-}"; \
	if [ -z "$$url" ]; then \
		port="$$(lsof -nP -iTCP -sTCP:LISTEN 2>/dev/null | awk '/^Cloris/ && $$0 ~ /127[.]0[.]0[.]1:/ { sub(/^.*127[.]0[.]0[.]1:/, ""); sub(/ .*/, ""); print; exit }')"; \
		if [ -z "$$port" ]; then \
			echo "qa-demo-app: no running Cloris.app listener found; run make refresh-local-app first or set CLORIS_APP_URL=http://127.0.0.1:<port>" >&2; \
			exit 1; \
		fi; \
		url="http://127.0.0.1:$$port"; \
	fi; \
	port_label="$${url##*:}"; \
	port_label="$${port_label%%/*}"; \
	out="$(ROOT)/output/playwright/product-data-wiring-demo-$${port_label}"; \
	echo "Cloris demo QA -> $$url"; \
	$(PYTHON) tools/qa_product_data_wiring.py --base-url "$$url" --out "$$out"

certify-tier0-fast: ## Run deterministic Tier-0 product certification in source mode
	$(PYTHON) tools/check_certification_coverage.py
	$(PYTHON) tools/certify_product.py --tier tier0 --mode source \
		--json-report $(ROOT)/.certification/tier0-source.json \
		--markdown-report $(ROOT)/.certification/tier0-source.md

certify-tier0-package: frontend-build validate-static-assets validate-package-smoke validate-package-modules ## Run deterministic Tier-0 API certification against dist/Cloris.app
	$(PYTHON) tools/check_certification_coverage.py
	$(PYTHON) tools/certify_product.py --tier tier0 --mode package \
		--app $(ROOT)/dist/Cloris.app \
		--json-report $(ROOT)/.certification/tier0-package.json \
		--markdown-report $(ROOT)/.certification/tier0-package.md

# Source-mode browser-observed Tier-0 certification. Proves the async
# intake-upload split behaves correctly when running from source.
# Useful for local iteration; NOT sufficient for Tier-0 release sign-off
# because the user-observed bug happened inside the packaged .app —
# see certify-tier0-browser-package for the packaged proof.
certify-tier0-browser: frontend-build ## Run browser-observed Tier-0 product certification (source mode)
	$(PYTHON) tools/certify_product_browser.py --tier tier0 --mode source \
		--json-report $(ROOT)/.certification/tier0-browser.json \
		--markdown-report $(ROOT)/.certification/tier0-browser.md

# Packaged-app browser-observed Tier-0 certification. Drives a real
# Playwright Chromium session against dist/Cloris.app — the binary the
# trial recipient actually launches. This is the load-bearing gate for
# Tier-0 release sign-off because the user-observed "Cloris is offline"
# regression happened inside this binary, not in source mode. Runs in
# strict ``--fail-on-pending`` mode so unimplemented Tier-0 browser
# flows do NOT silently pass the gate.
certify-tier0-browser-package: frontend-build validate-static-assets validate-package-smoke validate-package-modules ## Run browser-observed Tier-0 certification against dist/Cloris.app
	$(PYTHON) tools/certify_product_browser.py --tier tier0 --mode package \
		--app $(ROOT)/dist/Cloris.app \
		--fail-on-pending \
		--json-report $(ROOT)/.certification/tier0-browser-package.json \
		--markdown-report $(ROOT)/.certification/tier0-browser-package.md

# Tier-0 release-readiness gate. Source-mode browser cert remains
# useful for iteration but is NOT sufficient on its own; the packaged
# layers must also be green because that is the binary the recipient
# launches. The packaged-browser leg above runs in strict mode, so
# this target genuinely reflects "Tier-0 release sign-off": it goes
# RED while any Tier-0 browser flow is still ``pending``.
certify-tier0-full: certify-tier0-package certify-tier0-browser certify-tier0-browser-package ## API + browser Tier-0 release sign-off (strict on pending browser flows)

certify-tier1: ## Run deterministic Tier-1 product certification in source mode
	$(PYTHON) tools/certify_product.py --tier tier1 --mode source \
		--json-report $(ROOT)/.certification/tier1-source.json \
		--markdown-report $(ROOT)/.certification/tier1-source.md

certify-tier2: ## Run Tier-2 certification coverage checks in source mode
	$(PYTHON) tools/check_certification_coverage.py
	$(PYTHON) tools/certify_product.py --tier tier2 --mode source \
		--json-report $(ROOT)/.certification/tier2-source.json \
		--markdown-report $(ROOT)/.certification/tier2-source.md

certify-product: certify-tier0-fast certify-tier1 certify-tier2 ## Run deterministic product certification in source mode
	$(PYTHON) tools/certify_product.py --mode source \
		--json-report $(ROOT)/.certification/product-source.json \
		--markdown-report $(ROOT)/.certification/product-source.md

DEV_PORT ?= 8080

# Dev one-liner: kills any stale packaged Cloris.app, rebuilds the
# frontend, starts the Python backend at a fixed port, then opens
# Chrome to it. Ctrl-C to stop; re-run to pick up frontend changes.
dev: frontend-build ## Build frontend dist + serve at localhost:8080
	@pkill -f "dist/Cloris.app" 2>/dev/null || true
	@echo "Cloris dev server → http://127.0.0.1:$(DEV_PORT)"
	$(PYTHON) -m cloris start --port $(DEV_PORT)

test-full: ## Run the full pytest suite, including heavier replay coverage
	$(PYTHON) tools/run_validation.py full

audit-css-tokens: ## Validate var(--name) CSS references against tokens.css definitions
	@$(PYTHON) tools/audit_css_static.py

audit-ui: audit-css-tokens ## Walk every Cloris surface, score against design rules, emit Markdown report
	@$(PYTHON) tools/audit_surfaces.py --viewports 1024 1280 1440
	@$(PYTHON) tools/audit_rules.py
	@$(PYTHON) tools/audit_report.py

head-ai-mi: ## Run Head of Applied AI market intel with external research
	$(PYTHON) tools/update_market_intel.py \
		--brief $(HEAD_AI_BRIEF) \
		--run-dir $(HEAD_AI_RUN_DIR) \
		--mode post_run \
		--with-external-research \
		--force-external-research \
		--force-edge-case-research \
		--heuristic-planner

head-ai-brief: ## Update the Head of Applied AI brief using the latest market intel
	$(PYTHON) -m tools.iterate_brief \
		--brief $(HEAD_AI_BRIEF) \
		--report $(HEAD_AI_RUN_DIR)/run-report.json \
		--search-memory $(HEAD_AI_SEARCH_MEMORY) \
		--final-judgments $(HEAD_AI_FINAL_JUDGMENTS) \
		--output-dir $(ROOT)/output

fde-mi: ## Run FDE market intel with external research on the latest known run
	$(PYTHON) tools/update_market_intel.py \
		--brief $(FDE_BRIEF) \
		--run-dir $(FDE_RUN_DIR) \
		--mode post_run \
		--with-external-research \
		--force-external-research \
		--force-edge-case-research \
		--heuristic-planner

fde-brief: ## Update the FDE brief using the latest market intel
	$(PYTHON) -m tools.iterate_brief \
		--brief $(FDE_BRIEF) \
		--report $(FDE_RUN_DIR)/run-report.json \
		--search-memory $(FDE_SEARCH_MEMORY) \
		--final-judgments $(FDE_FINAL_JUDGMENTS) \
		--output-dir $(ROOT)/output
