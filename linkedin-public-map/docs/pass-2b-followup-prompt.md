# Pass 2b — Follow-up prompt for the authenticated public-LinkedIn capture

Pass 2a (already run, unauthenticated cloud browser) verified the public
`/in/{vanity}` profile and both auth-wall templates. **Pass 2b is the
authenticated sweep that you (Sam) run on your own LinkedIn seat** — either
via Claude in Chrome, or by pasting the scripts in `capture-snippets/`
into DevTools yourself.

This pass is the gate between v0.1.0 (current — search + company surfaces
are `unknown`) and v0.2.0 of `manifests/linkedin-public-selectors.yaml`,
where those surfaces get promoted to `stable_now`.

## Hard safety rules (re-state to the inspector before running)

- **Do not click** Connect, Follow, Message, Save, Like, React, Comment,
  Share, Repost, Subscribe, Apply, "I'm interested", "Send", "Send now",
  "Send without a note", or any inline candidate-row action button.
- **Do not click** mailto:/tel:/external URLs inside the contact-info
  overlay. Hovering is fine; clicking is not.
- **Do not type** into the global search bar, the in-page search, or any
  composer/note textarea. If you open the Connect dialog by keyboard,
  capture state, then **press Escape**.
- **Stay logged in to the same seat** for the entire pass — do not log
  out and back in mid-sweep. If you hit a checkpoint or "Unusual
  activity" banner, **stop immediately** and report.
- **Read-only is the contract.** If a click would change route to
  `/messaging/`, `/feed/`, `/jobs/`, `/learning/`, or open a write
  dialog, do not click.

## Surfaces to capture (in order)

| # | Surface URL | Capture script | Outcome |
| - | --- | --- | --- |
| 1 | `https://www.linkedin.com/in/jordanrivera/` (or your test vanity) | `01_authenticated_profile.js` | Confirm guest selectors still pin the same elements when logged in; map the authenticated-only Connect/Message/More buttons and Open To / Verifications cards. |
| 2 | `https://www.linkedin.com/in/{vanity}/details/experience/` | `02_profile_details_experience.js` | Promote `profile_details_experience.*` rows. |
| 3 | `https://www.linkedin.com/in/{vanity}/overlay/contact-info/` | `03_profile_overlay_contact_info.js` | Run as 1st-degree if possible; capture redacted vs full state. |
| 4 | `https://www.linkedin.com/search/results/people/?keywords=staff+engineer&geoUrn=%5B%22103644278%22%5D&network=%5B%22S%22%5D&origin=FACETED_SEARCH` (or equivalent — see §III) | `04_search_results_people.js` | Promote `search_results_people.*` rows. |
| 5 | `https://www.linkedin.com/company/acme-software/people/` (or any company you have access to) | `05_company_people_tab.js` | Promote `company_people.*` rows. |
| 6 | any profile with a visible Connect button | `06_connect_dialog_readonly.js` | Map the Connect dialog **using the keyboard-only opening procedure** in the script header. |
| 7 | run on every above page once it has settled | `99_outer_html.js` | Sanitize cookies + URNs, then save under `tests/fixtures/dom-snapshots/`. |

## What to do with each capture

For each script:

1. Note the URL bar at capture time. Confirm it matches the route in the
   table above; if LinkedIn silently redirected to `/authwall` or
   `/uas/login`, **stop** and instead capture that route as additional
   evidence for §F of the DOM map.
2. Let the script copy its JSON payload to the clipboard (it will log
   `[cloris-public] copied to clipboard`). If clipboard write is
   blocked, copy from the console.
3. Save the JSON next to the matching `.html` snapshot in
   `fixtures/playwright-public/tests/fixtures/dom-snapshots/`:
   - `public_profile_authenticated_jordanrivera.json` + `.html`
   - `profile_details_experience_jordanrivera.json` + `.html`
   - `profile_overlay_contact_info_jordanrivera.json` + `.html`
   - `search_results_people_staff_engineer_us_2nd.json` + `.html`
   - `company_people_acme-software.json` + `.html`
   - `connect_dialog_readonly_jordanrivera.json` + `.html`
4. Run the sanitization checklist in `capture-snippets/README.md`
   on every file before committing.

## Task block (paste this verbatim into Claude in Chrome, if using that path)

> You are continuing the Cloris fallback-path LinkedIn DOM-mapping work as a
> read-only inspector on an authenticated LinkedIn seat. Stay within the
> safety rules above. For each capture, dump (a) the AX-tree subtree rooted
> at the relevant container, (b) the role / accessible name / aria-label /
> first `trk=` substring triple for each interesting node, and (c) any
> structural notes a Playwright author would need (parent role, sibling
> order, whether the node is inside `role="dialog"`).
>
> **Capture 1 — Authenticated profile top card**
>
> 1. Load `https://www.linkedin.com/in/{vanity}/`.
> 2. Capture: `h1` (name), the first `h2` (headline), the second `h2`
>    (location · Contact Info · followers · connections — **note whether
>    the string layout is identical to the guest view**).
> 3. Capture every visible button inside `main section` — record role,
>    accessible name, `aria-label`, and whether it is followed by an
>    overflow menu (`aria-haspopup="menu"`).
> 4. Capture the photo: `main button[aria-label] > img[alt]`.
> 5. Capture the section headings list (`main section > h2`). Confirm
>    the order: About, Activity, Featured, Experience, Education, Skills,
>    Volunteer, Recommendations, Honors, Languages, Interests.
> 6. **Do not** click any button.
>
> **Capture 2 — Experience overlay (`/details/experience/`)**
>
> 1. Note the URL.
> 2. Identify the list root (likely `main section ul[role="list"]` or
>    `main ul`). Capture the first 5 list items.
> 3. For each `<li>`: capture every `<a href>`, every visible
>    `<span aria-hidden="true">`, every `<span class="visually-hidden">`.
>    Record the text content order — title, dates, location, bullets.
> 4. **Do not** click "Show more".
>
> **Capture 3 — Contact info overlay**
>
> 1. Load `/overlay/contact-info/` for a 1st-degree connection. If you
>    don't have a public-vanity 1st-degree contact, skip — report and
>    move on.
> 2. Capture `role="dialog"`. List every heading and section. Note which
>    fields render (email, phone, websites, IM, address, birthday).
> 3. **Do not** click any of the contact links. Press Escape to close.
>
> **Capture 4 — People search results**
>
> 1. Load the URL from row 4 of the surfaces table (Boolean keywords
>    `("staff engineer" OR "principal engineer")` works well; combine
>    with `geoUrn=["103644278"]` and `network=["S"]`).
> 2. Capture the filter chip bar, the result count text (often inside
>    an `aria-live` region), and the pagination nav.
> 3. Capture the first 5 result cards. For each: the `/in/{vanity}`
>    anchor, the degree pill text, the headline lines, the action
>    buttons (Connect / Message / Follow / More). **Do not click any
>    action button.**
>
> **Capture 5 — Company /people/ tab**
>
> 1. Load `/company/{slug}/people/` for any company you have access to.
> 2. Capture the breadcrumb, the search-within-company input, and the
>    `Browse by X` facet section roots.
> 3. Capture the first 8 employee tiles. Each tile's `/in/{vanity}`
>    anchor, role text, action buttons.
>
> **Capture 6 — Connect dialog (keyboard-only)**
>
> 1. With a profile loaded that shows a `Connect` button in the top card,
>    place keyboard focus on the Connect button using Tab. **Do not
>    click it.**
> 2. Press Enter to open the dialog (this is the same code-path as a
>    click but lets you abort if anything looks wrong).
> 3. If a "Send" or "Send without a note" button has keyboard focus
>    after opening, **press Escape immediately** and abort the capture.
> 4. Otherwise, capture the dialog: headings, all buttons (incl. their
>    accessible names), Add-a-note button, send button, dismiss button,
>    textarea, char counter.
> 5. **Press Escape** to dismiss. **Do not click Send.**

## After Pass 2b

When all six captures are saved + sanitized:

1. Promote `unknown` rows in `manifests/linkedin-public-selectors.yaml`
   to `stable_now` (or `mock_only` if behaviour is ambiguous).
2. Bump the manifest `version` to `0.2.0`.
3. Rewrite the placeholder DOM snapshots in
   `tests/fixtures/dom-snapshots/` with the sanitized real captures.
4. Remove `acceptUnverified: true` from any worker config that uses
   `SearchResultsPeoplePage` or `CompanyPeoplePage`.
5. Add a new `tests/search-results-people.spec.ts` and
   `tests/company-people.spec.ts` mirroring `tests/public-profile.spec.ts`,
   driving the saved snapshots in offline mode.
