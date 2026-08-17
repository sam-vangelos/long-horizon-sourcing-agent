# GitHub → LinkedIn Recruiter Reconciliation Source of Truth

## Summary

This document defines the canonical post-GitHub-run reconciliation workflow.

Its purpose is to turn the output of a completed GitHub sourcing run into a recruiter-actionable LinkedIn Recruiter evaluation workflow that:

- identifies the correct LinkedIn profile for each GitHub-sourced lead
- verifies holistic fit against the role brief
- verifies recruiter engagement / novelty status
- saves only candidates that clear all required gates
- preserves a full evaluation record for every processed lead, not just the saved subset

This document is the behavioral source of truth for the reconciliation tool. It is not a brainstorming note, a roadmap fragment, or an experiment log.

## Two workflow modes

`tools/run_recruiter_identity_resolver.py` defaults to
`--workflow-mode identity_collect` (per
recruiter-identity-collection-first, shipped). In that default
mode, no LinkedIn brief or judger is loaded, holistic fit is skipped, and the
action taxonomy is `COLLECT` / `MANUAL_REVIEW` / `REJECT` via
`collection_action` / `project_save_state`. The `SAVE` / `MANUAL_REVIEW` /
`REJECT` + subreason contract documented in this file applies only to the
explicit legacy mode `fit_gated_save`.

The source of record for `identity_collect` is the
`plans/recruiter-identity-collection-*.md` series:
recruiter-identity-collection-first,
recruiter-identity-collection-followups,
recruiter-identity-collection-review-fixes, and
recruiter-identity-collection-summary-alignment.

## Core Product Position

GitHub sourcing is an upstream screening stage, not the full hiring loop.

The GitHub run produces a set of leads that are promising but not yet operationally actionable. A recruiter still needs a secondary-source review, primarily through LinkedIn Recruiter, to answer three distinct questions:

1. Is this the right person?
2. Does this person actually fit the role holistically?
3. Is this person still worth working, given prior recruiter engagement and novelty status?

The reconciliation tool exists to answer those questions in one flow.

## Canonical Inputs

The reconciliation stage begins only after a GitHub sourcing run has completed.

The canonical input is the completed GitHub run output directory, including:

- `candidates.jsonl`
- `saves.jsonl`
- `final_judgments.jsonl` when present
- `outreach.jsonl` when present
- `run-manifest.json`

The reconciliation cohort is the GitHub run’s SAVE-family survivors.

That means:

- include `SAVE`
- include `INFERENTIAL_SAVE`
- include `TRANSFERABLE_SAVE`
- include `SIGNAL_SAVE`

If `final_judgments.jsonl` is unavailable or empty, reconciliation may fall back to `saves.jsonl`.

## Canonical Fit Standard

The canonical fit standard for reconciliation is the **LinkedIn brief**, not the narrower GitHub brief.

Rules:

- The GitHub brief remains the upstream screening brief and provenance context.
- The LinkedIn brief is the final role-definition authority for reconciliation.
- If a matching LinkedIn brief exists for the same role/market, reconciliation must evaluate fit against that LinkedIn brief.

Reason:

- the GitHub brief is intentionally narrower and builder-evidence oriented
- the LinkedIn brief is the fuller hiring and recruiter-evaluation surface
- reconciliation is the bridge from screening to recruiter action, so it must use the richer role standard

## Canonical Surface

The canonical v1 surface is **LinkedIn Recruiter**.

Normal LinkedIn and web search are not the canonical product path for reconciliation.

The required workflow is:

1. Human opens LinkedIn Recruiter.
2. Human creates or selects the project.
3. Human locks the location filter.
4. Agent attaches to the already-prepared Recruiter search page.
5. Agent performs name-based reconciliation inside that prepared surface.

In v1, the agent does **not** own project creation or location-filter clicking.

## Canonical Workflow

### 1. Run Handoff

The reconciliation tool receives:

- the GitHub run output directory
- the canonical LinkedIn brief path
- the manually prepared LinkedIn Recruiter search page
- run metadata such as project identity and locked location

### 2. Per-Lead Search

For each GitHub lead:

- build a cleaned lookup name from available GitHub identity fields
- use recruiter-prepared geography as the search context
- search by cleaned person name inside Recruiter

Retrieval should not depend on brittle over-constraint.

In v1:

- use location as the human-prepared context
- use cleaned name as the search keyword
- treat employer/title as verification evidence, not retrieval hard filters

### 3. Humanized Review Behavior

The reconciliation agent must use the same humanized interaction model as the main sourcing agent for:

- typing
- cursor movement
- scrolling
- dwell/read behavior
- profile opening and return behavior

It should not operate like a zero-dwell scraping bot.

It does **not** need to be as slow as the full sourcing agent, but it must behave as a controlled humanized browser workflow, not DOM-speed automation.

### 4. Identity Resolution

The tool must not assume the first visible card is the right person.

Required behavior:

- review the first N Recruiter result cards
- identify plausible candidates
- reject clearly wrong people
- if no plausible candidate exists, mark the lead `REJECT`
- if exactly one candidate clearly dominates, open that profile
- if multiple candidates remain plausible, open up to the top K plausible profiles in order until:
  - one is confirmed as the right person, or
  - ambiguity remains, in which case mark `MANUAL_REVIEW`

Identity resolution is not complete at facial review alone.

### 5. Holistic Fit Evaluation

Once a plausible matched profile is opened, the tool must evaluate fit **holistically**, not just identity and engagement.

Required behavior:

- extract full Recruiter profile content
- parse the profile into the repo’s canonical LinkedIn profile summary representation
- run the existing full-evaluation judgment path against the canonical LinkedIn brief

Fit evaluation must answer the real hiring question:

- does this person clear the actual role bar?

It must not stop at:

- card-level facial review
- company/title vibes
- engagement status alone

### 6. Engagement / Novelty Evaluation

For a profile that has passed identity resolution and has been opened:

- read Recruiter-side activity/status
- capture:
  - already saved
  - messages
  - projects
  - views
  - saved by
  - last outbound contact
  - inferred reachout status
  - novelty pressure

Engagement is a second gating dimension.

It does not replace fit.

### 7. Final Decision

The canonical top-level outcomes are:

- `SAVE`
- `MANUAL_REVIEW`
- `REJECT`

#### SAVE

Only assign `SAVE` when all three are true:

- identity is sufficiently confirmed
- holistic fit clears the LinkedIn brief’s save bar
- engagement status does not disqualify the lead from active recruiter use

In v1, `SAVE` triggers actual Recruiter save behavior.

#### MANUAL_REVIEW

Use `MANUAL_REVIEW` when:

- identity remains ambiguous
- multiple plausible same-name candidates persist
- fit is borderline / inferential after full review
- the tool cannot cleanly resolve a conflict between identity, fit, and engagement signals

#### REJECT

Use `REJECT` when:

- no plausible profile exists
- the surfaced profile is the wrong person
- holistic fit fails
- engagement status makes the candidate unsuitable to work

## Required Subreasons

The tool should keep the top-level action taxonomy simple, but every non-save outcome must carry a structured subreason.

Required subreason family:

- `no_plausible_profile`
- `identity_ambiguous`
- `fit_reject`
- `already_worked`
- `already_saved_elsewhere`
- `low_novelty`
- `borderline_fit`
- `tool_failure`

A wrong-person case collapses into `identity_ambiguous` (MANUAL_REVIEW) — the
canonical gate has no distinct `wrong_person` branch; `drop_wrong_person`
exists only in the retired path's different taxonomy.

These are artifact/reporting dimensions, not necessarily top-level actions.

## Required Artifacts

The reconciliation stage must always produce two persistent outputs.

### 1. Full Evaluation Artifact

This is the canonical permanent record.

It must include **every processed lead**, including:

- saved
- rejected
- manual review

Required formats:

- JSONL as source-of-truth machine artifact
- CSV as recruiter-readable companion

Each row must include at minimum:

- GitHub identity/context
- cleaned search name
- search location context
- reviewed Recruiter candidates
- selected/opened profile
- identity evidence and ambiguity notes
- holistic fit decision and rationale
- engagement/novelty fields
- final action
- structured subreason

### 2. Saved-Only Artifact

This is the operational handoff subset.

It contains only `SAVE` rows and should be easy for a recruiter to work from directly.

### 3. Run Summary Artifact

The summary must include:

- total processed leads
- counts by final action
- counts by subreason
- counts for already-saved / already-worked / low-novelty cases
- counts for no plausible profile / ambiguous identity cases

## Non-Negotiable Behavioral Rules

- No processed lead may disappear from the full evaluation artifact.
- The tool must not stop at facial review alone.
- The tool must not stop at engagement review alone.
- The tool must not save candidates on identity vibes without holistic fit.
- The tool must not save candidates that fail identity, fit, or engagement gates.
- The tool must use Recruiter-first manual setup as the canonical v1 operating model.
- Public LinkedIn / web-search experimentation is not the canonical product path.

## Acceptance Criteria

The system is behaving correctly only if all of the following are true:

- A completed GitHub run can be handed to reconciliation without manual file surgery.
- Reconciliation resolves the canonical LinkedIn brief and uses it for fit evaluation.
- In manual-setup mode, the tool does not attempt to create the project or click location filters.
- The agent operates with the same humanized browser interaction model as the main sourcing agent.
- The agent searches by cleaned name and evaluates plausible Recruiter matches.
- The agent opens profiles when needed to resolve identity and evaluate fit.
- The agent runs holistic fit evaluation on the opened matched profile.
- The agent reads Recruiter engagement/novelty status for matched profiles.
- Only candidates that pass identity + fit + engagement are saved.
- Every processed lead is recorded in the full evaluation artifact.

## Canonical Example Scenarios

### Exact match, fit pass, workable engagement

- candidate found cleanly
- identity confirmed
- fit passes full evaluation
- engagement is not disqualifying
- result: `SAVE`

### Exact match, fit pass, already worked

- identity confirmed
- fit passes
- messages / last outbound contact indicate active recent recruiter engagement
- result: `REJECT`
- subreason: `already_worked`

### Same-name ambiguity

- multiple plausible profiles remain
- tool cannot confidently select one
- result: `MANUAL_REVIEW`
- subreason: `identity_ambiguous`

### Identity confirmed, fit fails

- correct person found
- profile opened
- full LinkedIn evaluation fails
- result: `REJECT`
- subreason: `fit_reject`

### No plausible Recruiter profile

- cleaned name search returns no believable candidate in the prepared geography
- result: `REJECT`
- subreason: `no_plausible_profile`

## Canonical Implementation

The canonical implementation of this contract is a single code path. No
other code path in this repo is permitted to produce recruiter-facing
reconciliation artifacts.

Canonical runtime entry point:

- `tools/run_recruiter_identity_resolver.py`

Canonical service and decision surface:

- `linkedin/recruiter_identity_resolver.py`
  (`RecruiterIdentityResolver`)
- `shared/recruiter_reconciliation_decision.py`
  (`decide_final_reconciliation_action`)
- `shared/recruiter_identity_schemas.py`
  (`RecruiterIdentityResolution`, `RecruiterIdentityCandidate`,
  `PlausibleProfileReview`)
- `shared/recruiter_ambiguity_resolution.py`
  (multi-profile consolidation)
- `shared/recruiter_brief_resolution.py`
  (LinkedIn brief resolution for a GitHub run)

Canonical artifact writers:

- `github/recruiter_identity_report.py`

Retired path:

- `linkedin/reconciliation.py`
  (`LinkedInReconciliationService` — emits a deprecation warning; do
  not introduce new callers)
- `tools/reconcile_github_to_linkedin.py`
  (retired CLI; exits non-zero with a redirection message)

The retired path emits a different action taxonomy (`promote` /
`drop_wrong_person` / `drop_already_worked` / `promote_low_novelty` /
`manual_review`), does not open matched profiles, and does not run the
LinkedIn brief's holistic fit judge. It is therefore incompatible with
this document's non-negotiable behavioral rules.

Future extensions to reconciliation (scoring changes, new subreasons,
new artifact fields) must go through the canonical implementation only.
Removal of the retired path is a follow-on cleanup; see the "Guidance
For Future Work" in `docs/shared-candidate-execution-engine.md` for the
general boundary-first pattern to follow.

## Defaults and Assumptions

- Reconciliation is a **post-GitHub-run stage**, not part of the primary GitHub search loop.
- The LinkedIn brief is the final fit authority for reconciliation.
- The GitHub brief remains upstream screening provenance only.
- The canonical v1 execution surface is LinkedIn Recruiter.
- The canonical v1 workflow uses **human-prepared project and location setup**.
- Auto-save is enabled only for candidates that clear all required gates.
- The full evaluation artifact is the permanent source of truth.
- The saved-only artifact is the recruiter-action subset.
