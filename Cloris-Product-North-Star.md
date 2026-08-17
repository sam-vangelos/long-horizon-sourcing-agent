# Cloris Product North Star


This document defines what Cloris is now. It replaces the startup-era North
Star, which is preserved verbatim at
Cloris-Product-North-Star-startup-era
— that document described a sellable desktop product with buyer personas and
pricing tiers, and none of that is the operating reality anymore.

## 1. What Cloris is

Cloris is an internal autonomous sourcing tool at Acme, built and operated
by Sam, run from the terminal. It exists to source better than the tools and
teams around it, on real requisitions, using Sam's recruiting judgment encoded
as calibrated evaluation.

The operating loop: a brief arrives as config JSON (`config/<role>/brief-*.json`)
and enters through `load_brief` (`linkedin/orchestrator.py:499`) via
`tools/launch_*.sh`; preflight and formation turn it into an execution
strategy; discovery runs against LinkedIn (first-class) and OSS/GitHub
(renewed 2026-07-31, multi-hub strategy gated); a calibrated judgment pipeline
evaluates candidates (strategy on Fable, judgment on GLM-5.2, cheap extraction
on Haiku — per the 2026-07-06/07 promotion); saves, run reports, and market
intelligence land as files and runtime state. The operating surfaces are the
live run console, the run report, `progress.json` / the runtime DB, and
`tools/` diagnostics. The live run path imports zero `cloris.*` modules.

## 2. The pivot, recorded

Cloris began as Sam's attempt at a company: a local desktop product
(pywebview + FastAPI + Svelte) for recruiter customers. On 2026-08-01 Sam
declared that era over: "It is now an internal tool." The web surface is a
relic (spec Addendum I, CLAUDE.md operator-facing section). The Svelte app is
parked at `attic/frontend-2026-08/` and the last fully-green desktop shell is
tagged `desktop-shell-last-green`; a future UI, if one is ever wanted, gets
built fresh against the then-current pipeline with the attic as reference
material, not resurrected in place.

## 3. What survives from the startup era

These commitments predate the pivot and still bind, because they are about
sourcing quality, not about product form:

- **The calibrated evaluation pipeline is the durable asset.** The four-step
  judgment procedure (capability mapping → depth test → transferability →
  decision) and the brief-carried calibration vocabulary
  (`shared/brief_schema.py`) are the thing that beats generic search. Data
  acquisition is commodity; evaluation calibrated to the role is not.
- **High-bar, not hedged.** An uncertain candidate is a DROP. A "maybe"
  poisons trust in every "yes."
- **The brief is the execution contract.** Role truth lives in the approved
  brief config, not in code. Swapping roles means swapping briefs.
- **Editorial over database.** Operator output leads with the answer;
  mechanism second, raw values last — in the console, the report, and every
  sweep summary.

## 4. What Cloris is not

- Not a product for sale. No buyers, no pricing, no onboarding, no
  multi-tenancy. Anything reintroducing customer framing should be challenged
  against this document.
- Not a web app. The shell is a relic; do not audit, extend, or gate on it.
- Not an outreach-automation or candidate-database system.
- Not a generalist high-volume screening tool. Depth over throughput, still.

## 5. Module reality (2026-08)

LinkedIn is the first-class discovery module. OSS Maintainers is renewed with
the multi-hub expansion gated (oss-hub-expansion-strategy). Market
intelligence runs as a live support surface. Researcher is deprioritized;
Designer and Exec Search are sunset (their specs are archived). The
GitHub↔LinkedIn reconciliation doctrine
(`GitHub-LinkedIn-Reconciliation-Source-of-Truth.md`) remains the standard for
how a second discovery surface defers to the LinkedIn brief.

## 6. Where the rest of the truth lives

`AGENTS.md` for engineering norms; CLAUDE for posture and fleet;
INDEX for the canonical documentation set; INDEX for
active work. Provenance stamps on every doc say what to trust; location no
longer does.
