# Cloris — Public-LinkedIn Playwright fixture stubs (fallback path)

**Source of truth:** `../../manifests/linkedin-public-selectors.yaml` (v0.1.0).

**Companion docs:**
- `../../docs/linkedin-public-dom-map.md` — sections A–J DOM map
- `../../docs/pass-2a-live-observations.md` — unauthenticated cloud captures
- `../../docs/pass-2b-followup-prompt.md` — authenticated capture prompt you run on your seat
- `../../docs/research-public-linkedin.md` — URL params, URNs, anti-automation hygiene
- `../../capture-snippets/` — DevTools console scripts for Pass 2b

## What this is

A starter Playwright + TypeScript layer that the Cloris fallback worker
can import to drive **normal logged-in linkedin.com** (not Recruiter).
It encodes the selector strategy and the safety envelope from the YAML
manifest as page objects, so the planner never writes raw selectors and
never forgets which actions are mutating.

Four components:

1. **Page objects** (`src/pages/`) — typed locators per surface,
   grouped by what Pass 2a verified vs what is still `unknown`.
2. **SafetyGuard** (`src/safety/`) — mutation gate + guest-view budget
   counter. Every mutating call requires an `Intent` envelope scoped to
   a `targetVanity` (or `targetContentPath`); default mode is read-only.
3. **RecoverySignals** (`src/pages/recovery/`) — pure URL classifier
   that throws `AuthWallJoinError` / `AuthWallSignInError` /
   `BlockedError` / `GuestViewLimitError` on the canonical recovery
   routes (`/authwall`, `/uas/login`, `/checkpoint/challenge`, ...).
4. **Fixtures** (`tests/fixtures/`) — Playwright test scaffolds plus
   recorded-DOM placeholders for offline replay (`PLAYWRIGHT_MODE=offline`).

## What this is NOT

- **Not a runtime.** Cloris attaches its workers via CDP, not
  `playwright.launch()`. These page objects accept any `Page` instance.
  Use `chromium.connectOverCDP()` in the worker bridge.
- **Not a mutation harness.** Every mutating method is wrapped in
  `SafetyGuard.requireIntent()` and throws by default. Promotion to a
  live click requires an `Intent` carrying `targetVanity`,
  `idempotencyToken`, `humanConfirmed=true`, and an explicit
  `ActionName`.
- **Not Recruiter-equivalent.** The fallback path has a 3-view-per-session
  guest budget, no `data-test-*` attributes, and a much smaller verified
  surface area (only `/in/{vanity}` + auth-wall templates at v0.1.0).
- **Not seat-specific in v0.1.0.** Pass 2a was unauthenticated. After
  Pass 2b, an authenticated-only surface (`/search/results/people/`,
  `/company/{slug}/people/`, `/details/experience/`) gets promoted from
  `unknown` to `stable_now` in the manifest. Until then, those page
  objects throw `EnvelopeError` unless instantiated with
  `acceptUnverified: true`.

## Layout

```
fixtures/playwright-public/
  README.md                          ← this file
  package.json                       ← minimal Playwright + TS deps
  playwright.config.ts               ← CDP-attach default; offline replay opt-in
  tsconfig.json
  src/
    types.ts                         ← Intent, EnvelopeError, recovery classes,
                                       ActionName, Surface, JSON-LD types
    safety/
      SafetyGuard.ts                 ← mutation gate + guest-view budget counter
      forbidden.ts                   ← FORBIDDEN_IN_DEFAULT_MODE / READ_ONLY_ALWAYS_ALLOWED
    selectors/
      manifest.ts                    ← typed routes + URN maps + selector tables
    pages/
      PublicProfilePage.ts           ← /in/{vanity} (Pass 2a verified)
      AuthWallPage.ts                ← /authwall + /uas/login (Pass 2a verified)
      SearchResultsPeoplePage.ts     ← /search/results/people/ (unknown until 2b)
      CompanyPeoplePage.ts           ← /company/{slug}/people/ (unknown until 2b)
      recovery/
        RecoverySignals.ts           ← detectFromUrl + assertOnSurface
  tests/
    fixtures/
      dom-snapshots/
        README.md                    ← what to capture, sanitization rules
        public_profile_jordanrivera.html    ← placeholder
        authwall_join_company_people.html    ← placeholder
        authwall_signin_people_search.html   ← placeholder
    public-profile.spec.ts           ← offline tests for top-card + JSON-LD + experience
    authwall.spec.ts                 ← offline tests for both auth-wall templates
    safety-guard.spec.ts             ← envelope contract + guest-view budget
    recovery-signals.spec.ts         ← detectFromUrl URL classification
    search-url-builder.spec.ts       ← buildPublicPeopleSearchUrl params + URN maps
```

## Modes

`playwright.config.ts` reads `PLAYWRIGHT_MODE`:

- `offline` (default) — Tests load HTML from
  `tests/fixtures/dom-snapshots/*.html` via `page.setContent()`. Safe in
  CI; no network and no LinkedIn contact.
- `cdp` — Tests attach to a running Chromium via
  `chromium.connectOverCDP(WS_ENDPOINT)`. Used for live verification on
  your authenticated seat. **Never run mutating-mode tests against a
  real seat without a human-confirmed Intent.**

## Quick start

```bash
cd fixtures/playwright-public
npm install
PLAYWRIGHT_MODE=offline npx playwright test
```

All five test specs run against the recorded DOM snapshots and exercise
the safety envelope, recovery classifier, URL builder, and the
public-profile / auth-wall selectors verified in Pass 2a.

For live CDP attach (Pass 2b promotion work):

```bash
# In one terminal: start Chromium with debugging port and log in to LinkedIn.
google-chrome --remote-debugging-port=9222 --user-data-dir=/tmp/li-profile

# In another terminal:
PLAYWRIGHT_MODE=cdp CDP_WS=http://localhost:9222 npx playwright test \
  tests/public-profile.spec.ts -g "live"
```

## Worker integration

```ts
import { PublicProfilePage } from './src/pages/PublicProfilePage.js';
import { SafetyGuard } from './src/safety/SafetyGuard.js';

const guard = new SafetyGuard({ mode: 'read_only', guestViewBudget: 3 });
const profile = new PublicProfilePage(page, guard);

// Read-only — counts toward the guest-view budget when on the guest path.
await profile.goto('jordanrivera');
const top = await profile.readTopCard();
const ld = await profile.readJsonLd();

// Mutating — explicitly opted in by the planner with a fresh idempotency token.
const mutatingGuard = new SafetyGuard({
  mode: 'mutating',
  intent: {
    action: 'connect',
    targetVanity: 'jordanrivera',
    idempotencyToken: crypto.randomUUID(),
    humanConfirmed: true,
  },
});
// const connecting = new PublicProfilePage(page, mutatingGuard);
// await connecting.clickConnect(); // NOT IMPLEMENTED in v0.1.0
```

## Promotion rules (v0.1.0 → v0.2.0)

Before promoting any `unknown` selector to `stable_now`:

1. Run Pass 2b (`../../docs/pass-2b-followup-prompt.md`) on your
   authenticated seat. Save the sanitized JSON + HTML into
   `tests/fixtures/dom-snapshots/`.
2. Add an offline spec mirroring `tests/public-profile.spec.ts` against
   the new snapshot.
3. Promote the manifest row from `unknown` → `stable_now`, bump
   `version: 0.2.0`, and remove `acceptUnverified: true` from the
   relevant worker config.
4. Re-run `npx playwright test` — every previously failing assertion
   should now pass against the recorded DOM.

## Safety summary (re-read before every PR)

- **Default mode is `read_only`.** `mode: 'mutating'` requires an
  `Intent` with `humanConfirmed: true`, a matching `targetVanity`, an
  unconsumed `idempotencyToken`, and an `action` that matches the
  method being called.
- **Guest-view budget: 3 profiles per session.** SafetyGuard logs a
  deny event when the budget is exhausted; the page object then throws
  `GuestViewLimitError`.
- **Forbidden actions in default mode** are enumerated in
  `src/safety/forbidden.ts`. The set is the union of (a) classic mutating
  actions (connect, follow, message, post_react, etc.) and (b) auth-wall
  tripwires (any click inside `/authwall`, `/uas/login`, `/signup/...`).
- **Unverified selectors throw** unless the guard is constructed with
  `acceptUnverified: true`. Promotion to a verified row requires Pass 2b
  evidence.
