# Cloris — Public LinkedIn (linkedin.com) DOM / Interface Map

**Implementation status (2026-08-02):** This document specifies the DOM contract for a browser-driven public-profile worker that has NOT been built. The live `linkedin_public` acquisition mode is an x-ray search-engine query identical to `xray` mode (linkedin/fallback_search.py:131, linkedin/fallback_acquisition.py:76-83) — none of the DOM/JSON-LD/auth-wall/cookie-budget logic below is wired into a live worker, and the YAML manifest is read by no Python module. A real, unwired TypeScript/Playwright scaffold exists at linkedin-public-map/fixtures/playwright-public/ and is the intended eventual consumer. "ALIVE" in the provenance stamp records Sam's 2026-08-02 decision to keep this path rather than archive it — a strategic keep, not an implemented worker.

**Status:** Pass 1 base map (this document) + Pass 2a unauthenticated live verification. The YAML manifest at `../manifests/linkedin-public-selectors.yaml` (currently v1) is the **single source of truth** for the fallback-sourcing worker. This document is the human-readable reference; if it disagrees with the manifest, the manifest wins.

**Scope:** Public, signed-out-accessible surfaces of `www.linkedin.com` — i.e. the fallback path Cloris uses when the Recruiter seat is unavailable, throttled, or politically inappropriate to touch. **Not** Recruiter (`/talent/...`), **not** Sales Navigator (`/sales/...`), **not** Recruiter Lite.

## Read this in order

| Pass | Doc | What it covers |
| --- | --- | --- |
| 1 | this file | Scaffolded map across all public surfaces — public profile `/in/{vanity}`, company `/company/{slug}/people/`, guest people directory `/pub/dir/`, public search `/search/results/people/`, plus the two distinct auth-wall templates. |
| 2a | [`pass-2a-live-observations.md`](./pass-2a-live-observations.md) | Live cloud-browser captures of all four surfaces **without auth**. Confirms `/in/{vanity}` renders full content with a dismissible modal; the other three hard-redirect to either `/authwall` (Join LinkedIn) or `/uas/login` (Sign in). |
| 2b | `pass-2b-followup-prompt.md` | The next safe capture pass — **runs on your authenticated seat**. Targets: authenticated `/in/{vanity}/` full sections, `/details/experience/`, `/details/contact-info/`, `/search/results/people/?keywords=…` results structure, Connect/Message dialogs, company `/people/` employee cards. |
| n/a | [`research-public-linkedin.md`](./research-public-linkedin.md) | Background research: URL params, URN structures, Boolean rules, JSON-LD fallback, guest-view rate limit, `/pub/dir/` deprecation, cookie inventory. |

**Where Pass 2a contradicts Pass 1**, Pass 2a wins. Notable Pass-2a findings:

- `/in/{vanity}` does **not** redirect for the first few guest views; it renders the full top card + Experience + Education + About + Activity + Articles + Similar profiles inside `<main>`, with a `role=dialog` "View {firstName}'s full profile" modal overlaid. The modal is ESC-dismissible and the underlying content remains fully readable.
- `/company/{slug}/people/` is a **hard server-side redirect** to `/authwall?trk=bf`. No company-page DOM is rendered. Zero employee cards available without auth.
- `/pub/dir/+/+` is also a **hard redirect** to `/authwall?trk=gf`. The historical guest directory is effectively deprecated.
- `/search/results/people/?keywords=…` redirects to **`/uas/login`** (a *different* template from `/authwall`), with `<h1>Sign in</h1>`. The redirect carries `session_redirect=` so post-login lands on the search.
- **Zero `data-test-id` attributes** observed across any public surface. Selector strategy is forced to: accessible role + accessible name + `href` `trk=` substring + visible text + `og:` meta + JSON-LD `Person` schema.

The Pass-1 content below is the structural reference; any row not yet promoted to `verified: true` in the manifest is **provisional**.

---

**Hard rules for the Cloris public-LinkedIn fallback worker:**

1. Never click *any* of: Connect, Follow, Message, InMail upsell, "Send InMail", endorse, recommend, report, block, "I'm interested" on jobs, any reaction (Like/Celebrate/Insightful/Support), Save (post/job), Repost, Share, "Send connection request", "Subscribe to newsletter" — unless an explicit mutating intent is set on the task envelope. The public surface has fewer mutating affordances than Recruiter, but the ones that exist are higher-trust signals to LinkedIn's anti-automation systems.
2. **Never click "Sign in", "Join now", "Continue with Google/Apple/Microsoft", "Agree & Join", or any modal "Sign in with Email" button.** These are auth-wall traps; clicking them surfaces a credentialing flow the worker is forbidden from completing.
3. Treat every URL change as ground truth for surface verification — read the `pathname` and the `trk=` query param before treating a page as on-surface. A `/authwall` or `/uas/login` URL means **we are off-surface**; emit the appropriate recovery signal and halt.
4. LinkedIn public web has **no useful `data-test-*` attributes** and **rotated, obfuscated CSS classes**. The selector strategy is, in priority order: **accessible role + accessible name** → **`href` substring match on `trk=` token** → **visible text** → **`og:` / `<meta>` / `<link rel="canonical">` / JSON-LD `Person` extraction**. Never fall back to class names.
5. The guest profile-view counter is hard: after **3–5 unauthenticated profile loads from the same IP/cookie**, LinkedIn escalates the modal into an unskippable wall and (eventually) IP-throttles. Cloris MUST budget guest views per session and emit `recovery.guest_view_limit` proactively before hitting the wall.
6. Cookie `li_g_view` tracks guest view count. Cookies `li_at`, `JSESSIONID`, `lidc`, `bcookie`, `bscookie` MUST be preserved across CDP-attach restarts; clearing them resets the guest-view budget but flags the session as cookie-bouncing (worse than just being capped).

---

## A. Page / session / route state

### A.1 Public LinkedIn URL patterns

| Surface | URL pattern | Auth behavior | Notes |
| --- | --- | --- | --- |
| Public profile (vanity) | `https://www.linkedin.com/in/<vanity>` | **No redirect** for first ~3–5 guest views, then escalates. Modal `role=dialog` overlays the content. | Vanity slug is user-chosen (e.g. `jordanrivera`). Confirmed live 2026-05-19. |
| Public profile (encoded id) | `https://www.linkedin.com/in/<base64-like-id>` | Same as vanity. | Some profiles use opaque ids (e.g. `ACoAAB...`) when no vanity is set. Treat both as the same surface. |
| Profile sub-page — experience | `https://www.linkedin.com/in/<vanity>/details/experience/` | **Auth required.** Provisional. | Unverified; Pass 2b. |
| Profile sub-page — education | `https://www.linkedin.com/in/<vanity>/details/education/` | **Auth required.** Provisional. | Unverified; Pass 2b. |
| Profile sub-page — skills | `https://www.linkedin.com/in/<vanity>/details/skills/` | **Auth required.** Provisional. | Skills NOT rendered at all on the unauthenticated `/in/` view. |
| Profile sub-page — contact info | `https://www.linkedin.com/in/<vanity>/overlay/contact-info/` | **Auth required, 1st-degree only.** | Modal overlay route; verify in Pass 2b. |
| Profile sub-page — recent activity | `https://www.linkedin.com/in/<vanity>/recent-activity/all/` | **Auth required** for full view. | Unauth `/in/` shows truncated activity inline. |
| Company landing | `https://www.linkedin.com/company/<slug>` | Mostly readable signed-out (About, Overview). | Worth confirming in Pass 2b. |
| Company People tab | `https://www.linkedin.com/company/<slug>/people/` | **Hard 302 → `/authwall?trk=bf`.** | Confirmed live 2026-05-19. Zero content behind it. |
| Company Jobs tab | `https://www.linkedin.com/company/<slug>/jobs/` | Partially readable signed-out. | Out of fallback-sourcing scope. |
| Guest people directory | `https://www.linkedin.com/pub/dir/+/+` (legacy `pub/dir/<First>/<Last>` also) | **Hard 302 → `/authwall?trk=gf`.** | Effectively deprecated. Confirmed live 2026-05-19. |
| Public people search | `https://www.linkedin.com/search/results/people/?keywords=<...>` | **Hard 302 → `/uas/login?session_redirect=…&skipRedirect=true`.** | Confirmed live 2026-05-19. No guest results page. |
| Public posts / activity URL | `https://www.linkedin.com/posts/<vanity>_<slug>-activity-<numericId>-<random>` | Partially readable signed-out (post body, reactions count). | Linkable from `/in/{vanity}` activity surface. |
| Pulse article URL | `https://www.linkedin.com/pulse/<slug>-<author-firstname>-<author-lastname>-<short-id>` | Readable signed-out. | Stable. |
| Auth wall (template 1) | `https://www.linkedin.com/authwall?trk=<bf\|gf>&trkInfo=<base64>&sessionRedirect=<encoded-url>` | Terminal. | `bf` = browse/follow (company), `gf` = guest flow (pub/dir). |
| Login wall (template 2) | `https://www.linkedin.com/uas/login?session_redirect=<encoded-url>&skipRedirect=true` | Terminal. | Different template; `<h1>Sign in</h1>` vs authwall's `<h2>Join LinkedIn</h2>`. |
| Signup cold-join | `https://www.linkedin.com/signup/cold-join?session_redirect=<encoded-url>` | Terminal. | Reached from login wall's "Join now" link. |
| Signup public-profile-join | `https://www.linkedin.com/signup/public-profile-join?vieweeVanityName=<vanity>&trk=<...>` | Terminal. | Reached from `/in/{vanity}` modal "Join to view profile" CTA. |

### A.2 Surface detection — the only durable signal

Public LinkedIn has no project context, no breadcrumb, no `data-test-*` surface hint. The worker MUST identify which surface it is on by combining:

| Signal | Source | Classification | Notes |
| --- | --- | --- | --- |
| `window.location.pathname` regex | client | `stable_now` | First gate. Match against the patterns in §A.1. |
| `document.title` | `<title>` | `stable_now` | Profile: `"<Name> - <Headline> \| LinkedIn"`. Authwall: `"Sign Up \| LinkedIn"`. Login wall: `"LinkedIn Login, Sign in \| LinkedIn"`. Verified Pass 2a. |
| `<meta property="og:type">` | `<head>` | `stable_now` | `profile` on `/in/{vanity}`; absent or `website` on auth walls. Verified Pass 2a. |
| `<link rel="canonical">` | `<head>` | `stable_now` candidate | Should always be `https://www.linkedin.com/in/<vanity>` on public profile pages, irrespective of redirect-from URL. Confirm in Pass 2b. |
| JSON-LD `Person` block | `<script type="application/ld+json">` in `<head>` | `stable_now` candidate | Per research, public profiles embed a JSON-LD `Person`/`ProfilePage` schema. Worker should prefer this as the primary extraction target. Confirm exact shape Pass 2b. |
| `trk` query param on a hostname-rewritten URL | URL search params | `stable_now` | If `trk=bf` or `trk=gf` appears and pathname is `/authwall`, this is an auth wall. |

### A.3 Wrong-surface / failure states

| State | Visible signal | Recovery signal |
| --- | --- | --- |
| Logged out (worker had session, lost it) | URL becomes `/uas/login`; `<h1>Sign in</h1>` present | `recovery.logged_out` — halt all worker steps; emit re-auth task. |
| Auth-wall — join template | URL contains `/authwall?trk=`; `<h2>Join LinkedIn</h2>`; `button "Agree & Join"` present | `recovery.authwall_join` — halt; this URL family blocks worker regardless of session state. |
| Auth-wall — sign-in template | URL contains `/uas/login`; `<h1>Sign in</h1>`; Apple/Google/Microsoft SSO iframes present | `recovery.authwall_signin` — halt; emit re-auth task (same recovery channel as `logged_out`). |
| Guest-view limit hit | Modal becomes unskippable (no ESC dismiss); profile content removed from DOM | `recovery.guest_view_limit` — halt; cool down ≥30 min and rotate identity (only if policy allows; default: stop). |
| Rate-limited / challenge | URL `/checkpoint/challenge/*`; captcha iframe; "Unusual activity" banner | `recovery.blocked` — back off (≥24h), do not retry. |
| Hard-redirected company People | Intended `/company/<slug>/people/` resolved to `/authwall?trk=bf` | `recovery.surface_unavailable_authwall` — fallback to alternate surface (Pulse, posts) or escalate to Recruiter path. |
| Hard-redirected `/pub/dir/` | Intended `/pub/dir/+/+` resolved to `/authwall?trk=gf` | `recovery.surface_unavailable_authwall` — same as above; this path is effectively deprecated. |
| Hard-redirected public search | Intended `/search/results/people/?keywords=…` resolved to `/uas/login` | `recovery.surface_unavailable_login` — public search is not a guest surface; pivot to Voyager or Recruiter path. |
| Profile not found | `<h1>This page doesn't exist</h1>` or 404 template | `recovery.profile_404` — drop the input; no retry. |
| Browser crash / blank | `document.body.children.length < 2` or no `<main>` | `recovery.browser_crash` — restart worker session. |

### A.4 Search state encoding — public search is URL-encoded (unlike Recruiter)

Unlike Recruiter (opaque POST bodies), the public `/search/results/people/` route encodes essentially all filter state in URL query params. From the research doc:

| Param | Type | Semantics | Notes |
| --- | --- | --- | --- |
| `keywords` | string | Boolean-syntax keyword query | AND/OR/NOT must be UPPERCASE; parens supported; quoted phrases supported; **no wildcards**; ~2000 char practical limit. |
| `origin` | enum | `FACETED_SEARCH`, `GLOBAL_SEARCH_HEADER`, `SWITCH_SEARCH_VERTICAL` | Required-ish for the search to render correctly. `FACETED_SEARCH` is the safe default. |
| `geoUrn` | JSON array (URL-encoded) | List of geographic URN ids | e.g. `["103644278"]` for US, `["101165590"]` for UK. Worker maps human geo names → URN ids via the static map in `manifests/linkedin-public-selectors.yaml#geo_urns`. |
| `currentCompany` | JSON array (URL-encoded) | Mini-company URN ids | e.g. `["1035"]` for Microsoft. |
| `pastCompany` | JSON array (URL-encoded) | Mini-company URN ids | Same encoding as `currentCompany`. |
| `schoolFilter` | JSON array (URL-encoded) | School URN ids | n/a public — usually a school URN. |
| `industry` | JSON array (URL-encoded) | Industry URN ids | e.g. `["4"]` for Computer Software (legacy id). |
| `network` | string | Connection-degree filter | `F` = 1st, `S` = 2nd, `O` = 3rd+, comma-joined. |
| `profileLanguage` | string | ISO language code | e.g. `en`, `es`, `nl`. |
| `serviceCategory` | string | Services-marketplace category | Out of scope for sourcing. |
| `firstName` / `lastName` / `title` | string | Direct field filters | Less powerful than `keywords` Boolean. |

**Result cap:** 1,000 total results regardless of page size. Worker must shard searches (e.g. by geo, by title, by company) to extract > 1,000 candidates.

**URN format reference:**
- Geo: `urn:li:fs_geo:103644278` (US)
- Mini-company: `urn:li:fs_miniCompany:1035` (Microsoft)
- Industry: `urn:li:fs_industry:4` (Computer Software, legacy)
- Profile: `urn:li:fs_profile:ACoAAB...`
- Member (numeric, internal): `urn:li:member:<id>` — not in URL but appears in JSON-LD and API responses.

Worker MUST treat URN ids as opaque tokens — never construct them, only look them up from the static map or extract them from a previous response.

---

## B. Public profile `/in/{vanity}` — DOM structure (unauthenticated, Pass 2a verified)

The unauthenticated public profile is Cloris's primary read surface. Pass 2a confirmed all of the rows below against `https://www.linkedin.com/in/jordanrivera`.

| Region | Selector strategy | Classification | Notes |
| --- | --- | --- | --- |
| Page heading — name | `h1` (the first and only `<h1>` on the page) | `stable_now` | Verified Pass 2a: `<h1>Jordan Rivera</h1>`. Accessible name = the person's display name. |
| Headline | First `h2` after the name `h1`, inside the top-card region | `stable_now` | Verified Pass 2a: `<h2>Chair, Rivera Foundation and Founder, Bright Energy</h2>`. |
| Combined location / followers / connections line | Second `h2` after the name `h1` | `stable_now` | Verified Pass 2a: `Austin, Texas, United States · Contact Info 12K followers · 8 connections`. **Location + follower count + connection count + "Contact Info" link are flattened into a single `h2` text node** — worker must regex-split (e.g. `^(?<location>.+?) · Contact Info (?<followers>[\d.KM]+) followers · (?<connections>\d+) connections?$`). |
| Profile photo | `button[name="<Person Name>"] > img[alt="<Person Name>"]` inside the top-card | `stable_now` | Photo is wrapped in a clickable `<button>` (opens a lightbox in the authenticated view; in guest view the click triggers the modal). `alt` text mirrors the accessible name. CDN URL pattern `https://media.licdn.com/dms/image/v2/...`. |
| Background banner | `figure > image` in top-card | `stable_now` | Also `alt="<Person Name>"`. Useful as a "the top card rendered" presence check. |
| Top-card current company link | `a[href*="trk=public_profile_topcard-current-company"]` | `stable_now` | Resolves to `/company/<slug>?trk=public_profile_topcard-current-company`. Worker can extract company slug from the href. |
| Top-card school link | `a[href*="trk=public_profile_topcard-school"]` | `stable_now` | Resolves to `/school/<slug>/?trk=public_profile_topcard-school`. |
| Top-card external website link | `a[rel="noopener"]` outside the company/school anchors, in the top-card right column | `mock_only` | Pass 2a saw `Blog` link for Gates; pattern not yet confirmed across non-influencer profiles. |
| About section heading | `h2:has-text("About")` inside `<section>` in `<main>` | `stable_now` | Verified Pass 2a. |
| About section text | The text node(s) inside the About `<section>` after the `h2` | `mock_only` | Visible text is **truncated to ~69 characters** for unauthenticated viewers since March 2024 (per research). Full text is in the page's JSON-LD `Person.description` block. **Prefer JSON-LD over DOM scrape for About.** |
| Experience section heading | `h2:has-text("Experience")` inside a `<section>` | `stable_now` | Verified Pass 2a. |
| Experience items | All `a[href*="trk=public_profile_experience-item_profile-section-card_image-click"]` inside the Experience section | `stable_now` | Each anchor wraps a company-logo image and links to `/company/<slug>`. The associated `<h3>` (job title), `<h4>` (company name), and following `StaticText` (date range + duration) live in the same item subtree. |
| Education section heading | `h2:has-text("Education")` inside a `<section>` | `stable_now` | Verified Pass 2a. |
| Education items | All `a[href*="trk=public_profile_school_profile-section-card_image-click"]` inside the Education section | `stable_now` | Each anchor wraps a school-logo image; associated `<h3>` (school name), `<h4>` (degree/field if present), and date range live in the same item subtree. |
| Skills section | **NOT PRESENT in unauthenticated view.** | `unknown` | Verified absent Pass 2a. Skills require auth; selector deferred to Pass 2b. |
| Recommendations / Endorsements / Honors | **NOT PRESENT in unauthenticated view.** | `unknown` | Pass 2b. |
| Activity section heading | `h2:has-text("Activity")` | `stable_now` | Verified Pass 2a. |
| Activity sub-heading (follower count) | `h2:has-text("followers")` adjacent to the Activity h2 | `stable_now` | e.g. `12K followers`. |
| Articles section heading | `h2:text-matches(/^Articles by /)` | `stable_now` | Verified Pass 2a: `Articles by Bill`. |
| Recent post links | `a[href*="/posts/"]` inside the Activity region | `stable_now` | Hrefs point to `/posts/<vanity>_<slug>-activity-<numericId>-<random>`. |
| "See all activities" CTA | `a[href*="/signup/cold-join"][href*="trk=public_profile"]` | `mock_only` | This is a join-wall CTA — worker MUST NOT click. Treat as a presence marker only. |
| Similar profiles ("People you may know") region | `section` containing `h2 "Similar profiles"` or `h2 "People also viewed"` | `defer` | Cards exist; worker should NOT auto-follow because every interaction routes through `cold-join`. |
| Sign-in modal (overlay) | `role=dialog` with heading `h2 "View <FirstName>'s full profile"` | `stable_now` | Verified Pass 2a. Modal has NO `aria-label`. Heading regex: `/^View .+'s full profile$/`. |
| Modal — "Sign in with Email" button | `dialog >> role=button[name="Sign in with Email"]` | **forbidden_in_default_mode** | Worker MUST NOT click. |
| Modal — "Join now" link inside | `dialog >> role=link[name="Join now"]` | **forbidden_in_default_mode** | Worker MUST NOT click. |
| Modal dismiss | Keyboard `Escape` | `stable_now` | Verified Pass 2a — ESC closes the modal and underlying content remains in DOM. Worker should always dismiss the modal once before scraping. |
| Header — "Sign in" link | `a[href*="trk=public_profile_nav-header-signin"]` | **forbidden_in_default_mode** | Presence marker only. |
| Header — "Join now" link | `a[href*="trk=public_profile_nav-header-join"]` | **forbidden_in_default_mode** | Presence marker only. |
| Top-card "Join to view profile" button | `a[href*="trk=public_profile_top-card-primary-button-join-to-view-profile"]` | **forbidden_in_default_mode** | Presence marker only. |
| Bottom CTA banner "Join to view full profile" | `a[href*="trk=public_profile_bottom-cta-banner"]` | **forbidden_in_default_mode** | Presence marker only. |

### B.1 Recommended extraction pipeline for `/in/{vanity}`

The worker should run the following order against the unauthenticated profile DOM. Earlier steps are more durable; later steps are fallbacks.

1. **JSON-LD `Person` block** — `document.head.querySelectorAll('script[type="application/ld+json"]')`, parse each, find the one with `@type === "Person"` or `mainEntityOfPage["@type"] === "ProfilePage"`. Extract `name`, `jobTitle`, `description`, `image`, `worksFor`, `alumniOf`, `address`, `url`, `sameAs`. This is the **truthiest** read because LinkedIn's own SEO consumes it.
2. **`og:` meta tags** — `og:title`, `og:description`, `og:image`, `og:type=profile`. Useful for verification that we're on a profile surface at all.
3. **Top-card DOM** — `h1`, first `h2`, second `h2`, top-card company/school links, photo `button[name]`. Use to corroborate JSON-LD and to extract `currentCompanySlug` / `currentSchoolSlug` from the `trk=` anchors.
4. **Experience section DOM** — for each `a[href*="trk=public_profile_experience-item_profile-section-card_image-click"]`, walk to its enclosing item and extract `h3` (title), `h4` (company), date-range text, company slug from the anchor href.
5. **Education section DOM** — same pattern with `school_profile-section-card_image-click`.
6. **About section DOM** — only as a fallback if JSON-LD `description` is missing; remember the public-DOM About is truncated to ~69 chars.

### B.2 Recruiter vs public profile — what's different

| Field | Recruiter (`/talent/profile/<recruiterMemberId>`) | Public (`/in/<vanity>`) | Notes |
| --- | --- | --- | --- |
| Identifier | `recruiterMemberId` (opaque, Recruiter-scoped) | `vanity` (user-chosen) OR opaque `ACoA...` slug | Worker must resolve `recruiterMemberId` → `vanity` via the Recruiter drawer's "Public profile" → "Open link in new tab" button (already mapped in Recruiter Pass 3). |
| Skills | Full list, endorsements, ratings | **Not present** | Largest gap — public is unusable for skills-based sourcing. |
| Contact info | Email, phone, websites (when shared, 1st-degree only in public auth path) | Not present without auth | `/details/contact-info/` overlay exists but requires auth + 1st-degree. |
| Recommendations | Full list with text | Not present | |
| "In N projects" save state | Visible inside Recruiter drawer | Not applicable | Public has no Recruiter pipeline concept. |
| Activity feed | Full | Truncated to 2 recent posts + 3 recent articles | Worker can paginate via authenticated `/recent-activity/all/`. |
| About text | Full | Truncated ~69 chars in DOM; full in JSON-LD | Prefer JSON-LD for public. |
| Open-to-work badge | Surfaced via Recruiter spotlight | Not visible publicly | Public profiles don't expose Open-to-Work to non-logged-out viewers. |
| Save / Connect / Message | Mutating buttons in Recruiter drawer | Auth-wall CTAs only | Worker forbidden from clicking either in default mode. |
| Mutual-connections preview | Yes (1st/2nd) | No | |

---

## C. Public people search `/search/results/people/` — auth-walled

Pass 2a confirmed the unauthenticated `/search/results/people/?keywords=…` hard-redirects to `/uas/login`. **There is no guest version of this surface.** All selectors below are `unknown` until Pass 2b runs on the authenticated seat.

| Control | Visible label (expected) | Classification | Notes |
| --- | --- | --- | --- |
| Top search keyword input (header) | "Search" with people-tab pre-selected | `unknown` | Probable role `combobox`. Pass 2b. |
| People-vertical tab | "People" | `unknown` | Probable role `link` or `tab`. |
| Active filter chips row | Above results | `unknown` | Removable via `×` button per chip. Cloris worker should NOT auto-remove without intent. |
| Filter — Connections | "Connections" dropdown | `unknown` | Maps to URL `network=F\|S\|O`. |
| Filter — Connections of | "Connections of" dropdown | `unknown` | Pass 2b. |
| Filter — Locations | "Locations" dropdown | `unknown` | Maps to URL `geoUrn=[...]`. |
| Filter — Current company | "Current company" dropdown | `unknown` | Maps to URL `currentCompany=[...]`. |
| Filter — Past company | "Past company" dropdown | `unknown` | Maps to URL `pastCompany=[...]`. |
| Filter — School | "School" dropdown | `unknown` | Maps to URL `schoolFilter=[...]`. |
| Filter — Industry | "Industry" dropdown | `unknown` | Maps to URL `industry=[...]`. |
| Filter — Profile language | "Profile language" dropdown | `unknown` | Maps to URL `profileLanguage=...`. |
| Filter — Service categories | "Service categories" dropdown | `unknown` | Out of fallback-sourcing scope. |
| All filters modal | "All filters" button → modal with all facets | `unknown` | Pass 2b. |
| Result card container | `<li>` or `role=listitem` per result | `unknown` | Pass 2b. |
| Result card — name link | First link inside card, text = display name, href = `/in/<vanity>` | `unknown` | This is the canonical extraction handle. |
| Result card — headline | Below name | `unknown` | |
| Result card — location | Below headline | `unknown` | |
| Result card — Connect / Follow / Message button | Primary action button on the card | **forbidden_in_default_mode** | Mutating; worker MUST NOT click. |
| Result count text | "About N results" at top of results | `unknown` | Caps at 1,000. |
| Pagination — Next | `button[aria-label="Next"]` | `unknown` | Worker may click for pagination. |
| "Out of network" overlay | Some results show "Out of network — click to view" | `unknown` | Worker should treat as read-only. |
| LinkedIn Member redacted card | Name = "LinkedIn Member" | `unknown` | Cards beyond network visibility are redacted; worker drops them. |

**Worker policy:** because all of §C is `unknown` until Pass 2b, the worker SHOULD NOT use the search surface as a primary path. Fallback order:

1. Voyager API (out of scope of this doc — separate auth path)
2. Authenticated `/search/results/people/?keywords=…` *with verified Pass-2b selectors and `acceptUnverified: false`*
3. Direct `/in/{vanity}` extraction when a candidate list is already known
4. Recruiter `/talent/...` (already mapped — primary path)

---

## D. Company `/company/{slug}/people/` — auth-walled

Pass 2a confirmed hard-redirect to `/authwall?trk=bf`. **No guest version exists.** All selectors below are `unknown` until Pass 2b.

| Control | Visible label (expected) | Classification | Notes |
| --- | --- | --- | --- |
| Company name heading | `<h1>` with company name | `unknown` | Likely shared with the `/company/<slug>` landing page (which may be readable signed-out — confirm in Pass 2b). |
| People tab nav | "People" tab in company-page tabs | `unknown` | |
| Headcount line | "<N> employees" | `unknown` | Useful sourcing signal. |
| Employee filter — "Where they live" | facet | `unknown` | |
| Employee filter — "Where they studied" | facet | `unknown` | |
| Employee filter — "What they do" | facet | `unknown` | |
| Employee filter — "What they studied" | facet | `unknown` | |
| Employee filter — "What they're skilled at" | facet | `unknown` | |
| Employee card — name link | First link in card, href `/in/<vanity>` | `unknown` | |
| Employee card — title | Below name | `unknown` | |
| Connect button on employee card | mutating | **forbidden_in_default_mode** | |
| "Show more" pagination | Button at bottom of list | `unknown` | |

---

## E. Guest people directory `/pub/dir/` — effectively deprecated

Pass 2a confirmed hard-redirect to `/authwall?trk=gf`. Historical functionality (alphabetical guest directory) is **no longer accessible without auth as of 2025–2026**. Treat `/pub/dir/` URLs as a fallback only for confirming `recovery.surface_unavailable_authwall` and never as a primary path.

| Control | Status | Notes |
| --- | --- | --- |
| Letter index navigation | **NOT RENDERED** | Used to be an alphabetic `A B C …` grid. Now redirects. |
| First-letter / last-letter pages (e.g. `/pub/dir/S/V`) | **NOT RENDERED** | Same redirect. |
| Per-name listings (e.g. `/pub/dir/Sam/Vangelos`) | **NOT RENDERED** | Same redirect. |

Worker behavior: if the input pipeline ever produces a `/pub/dir/` URL, immediately emit `recovery.surface_unavailable_authwall` and drop the input.

---

## F. Auth-wall templates — two distinct surfaces

### F.1 `/authwall` (Join LinkedIn) — Pass 2a verified

Reached via hard redirect from `/company/<slug>/people/` (`trk=bf`) or `/pub/dir/*` (`trk=gf`). Targets new users.

| Element | Selector | Classification | Notes |
| --- | --- | --- | --- |
| Page title | `<title>Sign Up \| LinkedIn</title>` | `stable_now` | |
| Page heading | `h2:has-text("Join LinkedIn")` (rendered at heading level; AX tree shows as `heading`) | `stable_now` | |
| Email input | `input[type="text"]` with label "Email" | **forbidden_in_default_mode** | Never fill. |
| Password input | `input[type="password"]` with label "Password (6+ characters)" | **forbidden_in_default_mode** | Never fill. |
| "Agree & Join" button | `button:has-text("Agree & Join")` | **forbidden_in_default_mode** | Never click. |
| "Continue with google" button (Google OAuth iframe) | `iframe[title="Sign in with Google Button"]`; OAuth client_id `990339570472-k6nqn1tpmitg8pui82bfaun3jrpmiuhs.apps.googleusercontent.com` | **forbidden_in_default_mode** | Never click. |
| "Sign in" link (existing-user pivot) | `button:has-text("Sign in")` | **forbidden_in_default_mode** | Pivots to `/uas/login`. |
| User Agreement / Privacy / Cookie Policy links | `trk=seo-authwall-base_join-form-*` | `stable_now` | Read-only. |
| App-upsell toast | `dialog` at bottom-right with "LinkedIn is better on the app" | `defer` | Dismissible via `button:has-text("Dismiss")`. Worker may dismiss. |
| Footer legal links | `trk=seo-authwall-base_footer-*` | `stable_now` | Read-only. |

### F.2 `/uas/login` (Sign in) — Pass 2a verified

Reached via hard redirect from `/search/results/people/?keywords=…`. Different template; targets existing users.

| Element | Selector | Classification | Notes |
| --- | --- | --- | --- |
| Page title | `<title>LinkedIn Login, Sign in \| LinkedIn</title>` | `stable_now` | |
| Page heading | `h1:has-text("Sign in")` | `stable_now` | |
| "Sign in with Apple" button | `button:has-text("Sign in with Apple")` | **forbidden_in_default_mode** | |
| "Continue with Google" iframe | `iframe[title="Sign in with Google Button"]`; OAuth client_id `990339570472-...` (same as authwall) | **forbidden_in_default_mode** | |
| "Sign in with Microsoft" iframe | `iframe[title="Sign in with Microsoft button"]`; client_id `3fa91358-6f74-4525-b5df-da149652be36`; session-detect iframe uuid `cc38a71c-1f54-4236-85d2-6dd283cc6b54` | **forbidden_in_default_mode** | |
| Email/phone input | `input[type="email"]` with label "Email or phone" | **forbidden_in_default_mode** | |
| Password input | `input[type="password"]` with label "Password" | **forbidden_in_default_mode** | |
| "Show" password reveal | `button:has-text("Show")` | **forbidden_in_default_mode** | |
| "Keep me logged in" checkbox | `input[type="checkbox"]`, checked by default | **forbidden_in_default_mode** | |
| "Sign in" form submit button | `button:has-text("Sign in")` (NOT inside an iframe) | **forbidden_in_default_mode** | |
| "Forgot password?" link | `a[href*="/checkpoint/rp/request-password-reset"]` | `stable_now` | Read-only. |
| "Join now" link | `a[href*="/signup/cold-join"]` | **forbidden_in_default_mode** | |
| Footer legal links | `trk=d_checkpoint_lg_consumer_login_ft_*` | `stable_now` | Read-only. |

### F.3 Telling the two templates apart

```
function detectAuthSurface() {
  const url = new URL(location.href);
  if (url.pathname === '/authwall') {
    return { kind: 'authwall_join', trk: url.searchParams.get('trk') };  // 'bf' or 'gf'
  }
  if (url.pathname === '/uas/login' || url.pathname.startsWith('/checkpoint/')) {
    return { kind: 'authwall_signin', trk: null };
  }
  return null;
}
```

If `detectAuthSurface()` returns non-null, the worker is off-surface and MUST emit the matching `recovery.*` signal and halt.

---

## G. TRK parameter taxonomy

Every navigational element on public LinkedIn carries a `trk=` query parameter encoding its provenance. The worker uses these as durable selector fragments (since CSS classes are obfuscated and `data-test-*` is absent). Pass 2a captured these:

| `trk=` value | Context | Use as selector? |
| --- | --- | --- |
| `public_profile_nav-header-signin` | Top nav "Sign in" link on `/in/` | Presence-only (forbidden) |
| `public_profile_nav-header-join` | Top nav "Join now" link on `/in/` | Presence-only (forbidden) |
| `public_profile_guest_nav_menu_people` | Top nav → People directory link | Presence-only |
| `public_profile_top-card-primary-button-join-to-view-profile` | Top-card primary CTA | Presence-only (forbidden) |
| `public_profile_bottom-cta-banner` | Bottom-of-profile join CTA | Presence-only (forbidden) |
| `public_profile_topcard-current-company` | Top-card current-company link | **Extract company slug** |
| `public_profile_topcard-school` | Top-card current-school link | **Extract school slug** |
| `public_profile_experience-item_profile-section-card_image-click` | Experience item logo anchor | **Iterate experience items** |
| `public_profile_school_profile-section-card_image-click` | Education item logo anchor | **Iterate education items** |
| `public_profile__posts_social-actions-reactions` | Post reaction CTA | Forbidden — never click |
| `public_profile__posts_social-actions-comments` | Post comment CTA | Forbidden — never click |
| `public_profile__posts_comment-cta` | Post comment button | Forbidden — never click |
| `public_profile` | "See all activities" CTA | Presence-only (forbidden) |
| `public_profile_browsemap_browse-map_connect-button` | Similar-profile View button | Forbidden — never click |
| `seo-authwall-base_nav-header-logo` | Logo on authwall page | Surface marker |
| `seo-authwall-base_join-form-user-agreement` | Legal link on authwall | Surface marker |
| `seo-authwall-base_footer-*` | Footer on authwall page | Surface marker |
| `d_checkpoint_lg_consumer_login_ft_*` | Footer on `/uas/login` | Surface marker |
| `bf` (in `/authwall?trk=bf`) | Surface origin = company browse/follow | Surface marker |
| `gf` (in `/authwall?trk=gf`) | Surface origin = guest flow / pub-dir | Surface marker |

---

## H. Anti-automation signals & worker hygiene

LinkedIn's public surface is the most heavily monitored sourcing surface in the industry. Cloris worker MUST conform to the following hygiene:

### H.1 Per-session budgets

| Budget | Limit | Source |
| --- | --- | --- |
| Unauthenticated profile views | 3–5 before modal escalates to unskippable | Research doc + Pass 2a behavior |
| Search queries from one session | <100 per hour (informal) | Research doc |
| Page loads per minute | <30 (informal heuristic) | Research doc |
| Cookies that MUST persist | `li_at`, `JSESSIONID`, `lidc`, `bcookie`, `bscookie`, `li_g_view` | Research doc |
| Cookies that signal a fresh session | absence of any of the above + new `bcookie` | Research doc |

### H.2 Forbidden actions (default mode)

The worker MUST NOT, without an explicit Intent envelope:

- Click any auth-wall CTA on any surface (Sign in, Join now, Continue with X, Sign in with Email, Agree & Join, Forgot password, modal CTAs)
- Click Connect, Follow, Message, "Send InMail", endorse, recommend, report, block on any surface
- React to (Like / Celebrate / Insightful / Support) any post, article, or comment
- Comment on any post or article
- Save / Repost / Share any post
- Click "I'm interested" on any job posting
- Subscribe / Unsubscribe from newsletters or hashtag follows
- Click "See all activities" / "Show more" join-wall CTAs
- Type into any input on any auth-wall surface
- Click "Sign in with Apple / Google / Microsoft" iframes or buttons
- Click the "Open the app" deep-link toast on auth-wall surfaces
- Click anything inside the `cold-join` signup flow

### H.3 Allowed read-only actions

The worker MAY, in default mode:

- `read_public_profile_dom` — parse `/in/{vanity}` JSON-LD + DOM
- `dismiss_signin_modal` — press `Escape` to clear the `role=dialog` overlay on `/in/{vanity}`
- `dismiss_app_toast` — click `button:has-text("Dismiss")` on the LinkedIn-app upsell toast (when present and dismissible)
- `read_meta_tags` — extract `og:*` and `<link rel="canonical">`
- `read_jsonld` — parse `<script type="application/ld+json">`
- `navigate_to_public_profile` — `goto('/in/{vanity}')` within the per-session budget
- `navigate_to_company_landing` — `goto('/company/{slug}')` (landing is usually readable signed-out — confirm Pass 2b)
- `paginate_authenticated_search_next` — only after Pass 2b verifies the Next button selector AND only inside an authenticated session

### H.4 Recovery signals (canonical names)

| Signal | When to emit | Recommended action |
| --- | --- | --- |
| `recovery.authwall_join` | URL becomes `/authwall?trk=(bf\|gf)` | Halt, emit re-auth task |
| `recovery.authwall_signin` | URL becomes `/uas/login` or `/checkpoint/*` | Halt, emit re-auth task |
| `recovery.guest_view_limit` | Modal becomes unskippable; `Escape` no longer dismisses; underlying content removed from DOM | Halt; cool down ≥30 min OR drop session and rotate identity (only if policy allows) |
| `recovery.surface_unavailable_authwall` | Attempted surface `/company/<>/people/` or `/pub/dir/*` and got redirected to `/authwall` | Drop input; pivot to alternate surface |
| `recovery.surface_unavailable_login` | Attempted `/search/results/people/` unauthenticated and got `/uas/login` | Drop input; pivot to Voyager or Recruiter path |
| `recovery.profile_404` | Profile DOM has `h1:has-text("This page doesn't exist")` or HTTP 404 | Drop input; no retry |
| `recovery.blocked` | `/checkpoint/challenge/`, captcha iframe, "Unusual activity" banner | Halt 24h+, do not retry, escalate |
| `recovery.logged_out` | Session worker had auth, URL is now `/uas/login` | Re-auth task |
| `recovery.browser_crash` | `document.body.children.length < 2` or `<main>` missing | Restart worker |
| `recovery.stale_search` | URL params reflect filter set but result count is 0 unexpectedly | Rebuild search from scratch |
| `recovery.unknown_surface` | Pathname does not match any A.1 pattern AND is not an auth-wall surface | Halt; surface for human review |

---

## I. Reading order & promotion checklist

### I.1 Promoting a selector from `unknown` → `stable_now`

A row in the YAML manifest may be promoted to `stable_now` only when:

1. It was observed in a Pass 2a (unauthenticated) or Pass 2b (authenticated) live capture, OR a DevTools snippet from `capture-snippets/` ran against the live DOM and emitted the matching role/name/href.
2. The capture is committed in `linkedin-public-map/fixtures/playwright-public/tests/fixtures/dom-snapshots/` and referenced in the manifest row's `verified` / `evidence` field.
3. The selector resolves uniquely on the captured DOM (no false-positive siblings).
4. The selector survives a refresh of the same URL (idempotency check).
5. For any `mutating: true` row, it is also added to `worker_policy.forbidden_in_default_mode` and the matching `ActionName` type is added to `src/types.ts`.

### I.2 First-wave automation — smallest safe `stable_now` set

The fallback worker should ship with **only these surfaces** wired live:

1. `/in/{vanity}` JSON-LD extraction (with DOM fallback for `h1`, `h2` headline, `h2` location, experience anchors, education anchors).
2. Auth-wall surface detection (`/authwall`, `/uas/login`).
3. `Escape`-to-dismiss the sign-in modal.
4. Guest-view-budget tracker reading `li_g_view` cookie.

Everything else stays `mock_only` / `unknown` until Pass 2b.

### I.3 Pass 2b targets (authenticated capture — runs on user's seat)

See `pass-2b-followup-prompt.md` for the verbatim prompt. The minimal Pass 2b deliverable is captures of:

- Authenticated `/in/{vanity}/` (the full version, not the modal-blocked version)
- `/in/{vanity}/details/experience/`
- `/in/{vanity}/details/education/`
- `/in/{vanity}/details/skills/`
- `/in/{vanity}/overlay/contact-info/` (1st-degree only)
- `/search/results/people/?keywords=software+engineer&origin=FACETED_SEARCH`
- `/company/acme-software/people/` (any large company)
- Connect button + the resulting modal (read-only — no submission)
- Message dialog (read-only — no submission)

---

## J. Source citations

- Pass 2a live captures: [`pass-2a-live-observations.md`](./pass-2a-live-observations.md)
- Background research: [`research-public-linkedin.md`](./research-public-linkedin.md)
- External — search syntax & URL params: [Lobstr LinkedIn search guide](https://www.lobstr.io/blog/linkedin-search-ultimate-guide), [OSINT Combine corporate-profiling](https://www.osintcombine.com/post/corporate-profiling-advanced-linkedin-searching-more), [Apify LinkedIn people-search scraper](https://apify.com/logical_scrapers/linkedin-people-search-scraper)
- External — URN reference: [Microsoft Learn LinkedIn URNs](https://learn.acme-software.com/en-us/linkedin/shared/api-guide/concepts/urns), [Captain Data geocodeUrn list](https://support.captaindata.com/en/articles/10725212-list-of-geocodeurn-to-use-in-your-geography-parameter)
- External — cookies: [LinkedIn legal cookie table](https://www.linkedin.com/legal/l/cookie-table)
- External — Voyager API sample (citation only; no automation against this surface): [yangchenyun gist](https://gist.github.com/yangchenyun/74cb2bb5b6faaab1e12a7f6862cd1e2f)
