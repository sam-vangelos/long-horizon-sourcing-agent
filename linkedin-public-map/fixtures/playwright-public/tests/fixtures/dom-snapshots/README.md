# DOM snapshots — recorded fixtures for offline tests

This directory holds HTML snapshots captured from live LinkedIn surfaces. Tests in
`tests/*.spec.ts` use `page.setContent(html)` when `PLAYWRIGHT_MODE=offline` to run
without any network or auth.

## Required snapshots

The placeholders below MUST be filled before the offline test suite is meaningful:

| File | Source surface | Provenance |
| --- | --- | --- |
| `public_profile_jordanrivera.html` | `https://www.linkedin.com/in/jordanrivera` | Pass 2a unauthenticated capture, 2026-05-19 |
| `authwall_join_company_people.html` | `/authwall?trk=bf` (from `/company/acme-software/people/`) | Pass 2a unauthenticated capture |
| `authwall_join_pub_dir.html` | `/authwall?trk=gf` (from `/pub/dir/+/+`) | Pass 2a unauthenticated capture |
| `authwall_signin_people_search.html` | `/uas/login` (from `/search/results/people/?keywords=software%20engineer`) | Pass 2a unauthenticated capture |
| `public_profile_authenticated_jordanrivera.html` | `/in/jordanrivera` (authenticated) | **Pass 2b — runs on user's seat.** See `../../../../docs/pass-2b-followup-prompt.md`. |
| `search_results_people_authenticated_swe.html` | `/search/results/people/?keywords=software%20engineer` (authenticated) | Pass 2b |
| `company_people_authenticated_acme-software.html` | `/company/acme-software/people/` (authenticated) | Pass 2b |
| `profile_details_experience_authenticated.html` | `/in/{vanity}/details/experience/` (authenticated) | Pass 2b |

## How to capture

For unauthenticated surfaces (Pass 2a) the captures already exist in
`../../../../docs/pass-2a-live-observations.md` as AX-tree dumps. Convert each to
raw HTML via the DevTools snippet:

```js
copy(document.documentElement.outerHTML);
```

…then paste into the matching `.html` file here. Strip personally identifying
beacon URLs from the captured HTML (regex: `https://www\.linkedin\.com/li/track`,
`/beacon/`, image CDN tokens) before committing — recorded fixtures should not
exfiltrate session-identifying telemetry.

For authenticated surfaces (Pass 2b), see the captured DOM workflow in
`capture-snippets/` at the project root.
