# Public-LinkedIn DOM Capture Snippets — Pass 2b

These scripts collect the DOM evidence we need to promote
`search_results_people.*`, `company_people.*`, and authenticated-profile
selectors from `classification: unknown` to `stable_now` in
`manifests/linkedin-public-selectors.yaml`.

**You run these on your own authenticated LinkedIn seat in a real Chromium
DevTools console.** Nothing here calls a server tool, sends a message,
or mutates state — every script is read-only and explicit about it.

## Procedure

1. Open `chrome://flags` and confirm Chrome DevTools is not running in
   a sandboxed extension context. Use a normal Chromium window (Chrome,
   Edge, Arc) signed in to LinkedIn.
2. Navigate to the target surface listed in the script header.
3. Open DevTools → Console.
4. Paste `_lib.js` first. You should see `[cloris-public] _lib loaded.`
5. Paste the `0X_*.js` script that matches the page.
6. Either let the script copy the JSON to the clipboard, or copy from
   the console. Paste into:
   `fixtures/playwright-public/tests/fixtures/dom-snapshots/{label}.json`
7. For complete-page archival, run `99_outer_html.js` and paste the
   sanitized HTML (see the sanitization checklist inside that file) into
   the matching `.html` file in the same directory.

## Scripts

| Script | Surface | Notes |
| --- | --- | --- |
| `_lib.js` | _shared_ | Paste FIRST. Idempotent. |
| `01_authenticated_profile.js` | `/in/{vanity}` (logged in) | Diffs against the Pass 2a guest capture. |
| `02_profile_details_experience.js` | `/in/{vanity}/details/experience/` | Authenticated-only overlay. |
| `03_profile_overlay_contact_info.js` | `/in/{vanity}/overlay/contact-info/` | Run as a 1st-degree connection if possible. |
| `04_search_results_people.js` | `/search/results/people/?...` | Apply ≥1 filter before capturing. |
| `05_company_people_tab.js` | `/company/{slug}/people/` | Authenticated-only. |
| `06_connect_dialog_readonly.js` | any profile w/ Connect button | **Use keyboard-only path described in script.** |
| `99_outer_html.js` | any | Sanitize cookies + URNs before saving. |

## Safety rules

- **Do not** click Connect, Message, Follow, Save, Like, Comment, or Send
  during capture. The Connect dialog script uses keyboard-only opening so
  you never accidentally trigger a one-click invite.
- **Do not** click anything inside the contact-info overlay (avoid mailto:
  links that may launch your client and reveal your own email in a beacon).
- **Do not** screen-share or stream while capturing — sanitize cookies and
  URNs before saving anything.
- Stay under the 3-view-per-session guest budget when capturing the
  unauthenticated path. The authenticated path has no documented budget
  but applies bot-detection if you load > ~30 profiles/min.

## Sanitization checklist for committed snapshots

Before saving any JSON or HTML payload to the repo, search-and-replace:

- Cookie values: `li_at`, `JSESSIONID`, `lidc`, `bcookie`, `bscookie`,
  `li_g_view` → `REDACTED`
- Auth-related: `csrfToken`, `clientApplicationInstance`, `serviceTrace`
  → `REDACTED`
- Member URNs containing your own id:
  `urn:li:fs_member:NNNNN` / `urn:li:fs_profile:ACoA…` → `urn:li:fs_member:0`
- Beacon endpoints:
  `https://www.linkedin.com/li/track`, `https://px.ads.linkedin.com/…`
  → `https://example.test/REDACTED`
- Personal details on snapshots of your own profile: name, photo URL,
  contact info, vanity → placeholder values.
