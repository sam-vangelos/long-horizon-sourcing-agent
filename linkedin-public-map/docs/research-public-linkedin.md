# LinkedIn Public Surface DOM Research Report

**Purpose:** Support for building a DOM map for a recruiting-automation tool targeting LinkedIn's public (non-Recruiter) web surface.
**Scope:** URL taxonomy, profile sub-pages, company people pages, guest visibility, anti-automation signals, Boolean search, structured metadata, legacy directories, and ID mapping.
**Confidence legend:** `[HIGH]` = authoritatively documented; `[MED]` = community-reverse-engineered with wide corroboration; `[LOW]` = limited or stale evidence.

---

## 1. URL Parameter Taxonomy: `/search/results/people/`

Base URL: `https://www.linkedin.com/search/results/people/`

All parameters are query-string key-value pairs appended after `?`. Array-type values are JSON-encoded and URL-encoded (e.g., `["1035"]` becomes `%5B%221035%22%5D`).

### Parameter Reference Table

| Parameter | Accepts | Format / Example | Source Type | Confidence |
|-----------|---------|-------------------|-------------|------------|
| `keywords` | Free text, Boolean string | `keywords=software+engineer` | Documented (LinkedIn Help) | HIGH |
| `origin` | Enum string | `FACETED_SEARCH`, `GLOBAL_SEARCH_HEADER`, `SWITCH_SEARCH_VERTICAL` | Reverse-engineered | MED |
| `sid` | Opaque session token | Auto-generated hexadecimal string | Reverse-engineered | MED |
| `geoUrn` | JSON array of numeric geo IDs | `%5B%22103644278%22%5D` (United States) | Reverse-engineered | HIGH |
| `currentCompany` | JSON array of numeric company IDs | `%5B%221035%22%5D` (Microsoft) | Reverse-engineered | HIGH |
| `pastCompany` | JSON array of numeric company IDs | `%5B%222382910%22%5D` | Reverse-engineered | HIGH |
| `schoolFilter` | JSON array of numeric school IDs | `%5B%2218166%22%5D` (Stanford) | Reverse-engineered | HIGH |
| `industry` | JSON array of numeric industry codes | `%5B%224%22%5D` (Computer Software) | Reverse-engineered | HIGH |
| `network` | JSON array of connection-degree codes | `%5B%22F%22%5D` (1st), `%5B%22S%22%5D` (2nd), `%5B%22O%22%5D` (3rd+) | Reverse-engineered | MED |
| `profileLanguage` | JSON array of BCP-47 language codes | `%5B%22en%22%5D` (English), `%5B%22fr%22%5D` (French) | Reverse-engineered | MED |
| `serviceCategory` | JSON array of numeric service codes | `%5B%223387%22%5D` (Consulting) | Reverse-engineered | MED |
| `connectionOf` | Member URN or member ID | Numeric LinkedIn member ID | Reverse-engineered | LOW |
| `firstName` | Free text | `firstName=John` | Reverse-engineered | HIGH |
| `lastName` | Free text | `lastName=Smith` | Reverse-engineered | HIGH |
| `title` | Free text keyword | `title=Director` | Reverse-engineered | HIGH |
| `company` | Free text (non-ID) | `company=Microsoft` | Reverse-engineered | HIGH |
| `school` | Free text keyword | `school=Harvard` | Reverse-engineered | HIGH |
| `titleFreeText` | Free text job title | `titleFreeText=VP+Engineering` | Reverse-engineered | MED |
| `countryCode` | ISO 3166-1 alpha-2 | `countryCode=au` | Reverse-engineered | MED |
| `postalCode` | Local postal/zip code | `postalCode=2000` | Reverse-engineered | MED |
| `distance` | Integer (miles) | `distance=30` | Reverse-engineered | MED |
| `experience` | JSON array of experience level codes | `%5B%222%22%5D` (2–5 years) | Reverse-engineered | LOW |
| `page` | Integer | `page=2` | Reverse-engineered | MED |

**Notes:**
- LinkedIn caps public search results at **1,000 total results**, regardless of page/offset ([lobstr.io](https://www.lobstr.io/blog/linkedin-search-ultimate-guide), 2024).
- `origin=FACETED_SEARCH` is the standard value when filters are applied via the UI.
- The older `facetCurrentCompany`, `facetPastCompany`, `facetGeoRegion` parameter names (prefixed with `facet`) appear in older scraper code and some community sources but have largely been superseded by the shorter names (`currentCompany`, `geoUrn`). Both variants may work intermittently ([OSINT Combine](https://www.osintcombine.com/post/corporate-profiling-advanced-linkedin-searching-more)).

### URN Structure Reference

LinkedIn's internal Voyager API (and public-facing URL parameters) use the following URN formats:

| Entity | Internal URN (Voyager API) | URL Parameter Format |
|--------|---------------------------|---------------------|
| Geo / Location | `urn:li:fs_geo:103644278` | Numeric ID only: `103644278` |
| Country (API v2) | `urn:li:geo:103644278` | Same numeric suffix |
| Company | `urn:li:fs_miniCompany:1035` | Numeric ID only: `1035` |
| Industry | `urn:li:fs_industry:4` | Numeric ID only: `4` |
| Profile (member) | `urn:li:fs_profile:ACoAABAxjP4B...` | ACoA… alphanumeric string |
| Mini-profile | `urn:li:fs_miniProfile:ACoAABAxjP4B...` | Same |
| Member (numeric) | `urn:li:member:271682814` | Integer |
| School | `urn:li:fs_miniSchool:17954` | Numeric |
| Field of study | `urn:li:fs_fieldOfStudy:100892` | Numeric |

Source: [Voyager API response sample](https://gist.github.com/yangchenyun/74cb2bb5b6faaab1e12a7f6862cd1e2f); [Microsoft Learn Geo Typeahead API](https://learn.acme-software.com/en-us/linkedin/shared/references/v2/standardized-data/locations/geo-typeahead); [LinkedIn API URNs](https://learn.acme-software.com/en-us/linkedin/shared/api-guide/concepts/urns).

**Key geo IDs (confirmed):** US = `103644278`, UK = `101165590`, Canada = `101174742`, Australia = `101452733`, India = `102713980`, Germany = `101282230`, France = `105015875` ([Captain Data](https://support.captaindata.com/en/articles/10725212-list-of-geocodeurn-to-use-in-your-geography-parameter)).

---

## 2. Profile Sub-Page URL Structure

Base: `https://www.linkedin.com/in/{vanitySlug}/`

### Confirmed Sub-Pages (as of 2025–2026)

| Sub-Page | URL Pattern | Auth Required | Confidence |
|----------|-------------|---------------|------------|
| Main profile | `/in/{vanity}/` | Guest (limited) | HIGH |
| Experience details | `/in/{vanity}/details/experience/` | Logged-in | MED |
| Education details | `/in/{vanity}/details/education/` | Logged-in | MED |
| Skills details | `/in/{vanity}/details/skills/` | Logged-in | MED |
| Certifications | `/in/{vanity}/details/certifications/` | Logged-in | MED |
| Projects | `/in/{vanity}/details/projects/` | Logged-in | MED |
| Publications | `/in/{vanity}/details/publications/` | Logged-in | MED |
| Courses | `/in/{vanity}/details/courses/` | Logged-in | MED |
| Honors & awards | `/in/{vanity}/details/honors/` | Logged-in | MED |
| Languages | `/in/{vanity}/details/languages/` | Logged-in | MED |
| Contact info | `/in/{vanity}/details/contact-info/` | 1st-degree only | HIGH |
| Recommendations given/received | `/in/{vanity}/details/recommendations/` | Logged-in | MED |
| Recent activity (all) | `/in/{vanity}/recent-activity/all/` | Logged-in | MED |
| Recent activity: posts/shares | `/in/{vanity}/recent-activity/shares/` | Logged-in | MED |
| Recent activity: comments | `/in/{vanity}/recent-activity/comments/` | Logged-in | MED |
| Recent activity: reactions | `/in/{vanity}/recent-activity/reactions/` | Logged-in | MED |

**Notes:**
- The `/details/` sub-pages are not separately crawlable by search engines; they require authentication in almost all user configurations.
- Contact info (email, phone, Twitter/X handle) is only visible to **1st-degree connections**, per [LinkedIn Help](https://www.linkedin.com/help/linkedin/answer/a545600/what-people-can-see-on-your-profile).
- The `/recent-activity/` sub-pages are only accessible when logged in; unauthenticated visitors are redirected to the authwall.
- LinkedIn has no officially published list of these sub-page URLs; all are reverse-engineered from UI observation. `[MED]`

---

## 3. Company `/people/` Subpage

Base pattern: `https://www.linkedin.com/company/{slug}/people/`

Also accessible by numeric ID: `https://www.linkedin.com/company/{numericId}/people/`

### URL Filter Parameters (Voyager/Internal API, surfaced in URL)

| Parameter | Description | Example |
|-----------|-------------|---------|
| `facetGeoRegion` | Filter by geographic region (numeric geo ID) | `facetGeoRegion=101452733` (Australia) |
| `facetCurrentFunction` | Filter by job function (numeric code) | `facetCurrentFunction=8` (Engineering) |
| `facetSchool` | Filter by school attended (numeric school ID) | `facetSchool=18166` (Stanford) |
| `facetCurrentRole` | Filter by job title keywords (free text) | Observed in Voyager API requests |
| `facetFieldOfStudy` | Filter by field of study | Voyager API parameter |

**Example URL:** `https://www.linkedin.com/company/google/people/?facetCurrentFunction=8&facetGeoRegion=101452733` (Google engineers in Australia) — observed in practice ([Reddit r/cscareerquestionsOCE](https://www.reddit.com/r/cscareerquestionsOCE/comments/1gwbhoc/yes_i_fucked_up_yes_i_need_help/)).

The underlying Voyager API request to `api.linkedin.com/voyager/api/search/hits` uses a richer parameter set including `facetSkillExplicit`, `facetNetwork`, and `q=people` — visible via browser DevTools Network tab on the company people page ([r/scrapinghub](https://www.reddit.com/r/scrapinghub/comments/l6v2wg/linkedin_scraper_dynamically_loading_webpage/)).

**Confidence:** `[MED]` — parameters are reverse-engineered from network inspection; LinkedIn does not document them publicly.

---

## 4. Guest Views vs. Auth-Walled Views

### `/in/{vanity}/` — Public Profile

LinkedIn allows members to control which sections appear publicly. Visibility is set per-section at *Settings & Privacy > Visibility > Edit your public profile* ([LinkedIn Help](https://www.linkedin.com/help/linkedin/answer/a518980)).

| Section | Default Guest Visibility | Notes |
|---------|--------------------------|-------|
| Name | ✅ Visible | Always shown |
| Headline | ✅ Visible | Reduced to ~69 chars for guests since early 2024 |
| Profile photo | ⚠️ Depends | User-controlled; blurred or hidden if restricted |
| About / Summary | ⚠️ Partial | First ~69 characters shown to guests since March 2024 ([LinkedIn pulse](https://www.linkedin.com/pulse/linkedin-new-features-updates-march-24-edition-link-ability-nz-2cwtc)) |
| Current position/company | ✅ Visible | Title and employer shown |
| Full work experience | ⚠️ Limited | Job titles/company names visible; descriptions blocked |
| Education | ⚠️ Limited | Institution names visible; details may be blocked |
| Skills | ⚠️ Partial | Some skills visible on public profile |
| Contact info | ❌ Blocked | 1st-degree only |
| Recommendations | ❌ Blocked | Requires login |
| Activity / Posts | ❌ Blocked | Authwall after login prompt |
| JSON-LD / structured data | ✅ Visible | Embedded in HTML `<head>` — see Section 7 |

**Authwall trigger:** LinkedIn typically triggers a sign-in prompt after **3–5 anonymous profile views** within a session ([Scrapfly](https://scrapfly.io/blog/posts/how-to-scrape-linkedin)). Mechanism: the `li_g_view` cookie (30-day expiry) tracks guest view count; `recent_history_status` (10-year) stores the guest history setting.

### `/company/{slug}/` — Company Page

- **About tab:** Publicly visible — company description, website, industry, size, founded date.
- **Posts/Updates tab:** Requires login to see full feed.
- **Jobs tab:** Partially public — job listings at `linkedin.com/jobs/search/?f_C={companyId}` are accessible without login via a separate guest-facing API.
- **People tab (`/company/{slug}/people/`):** Requires login; full employee directory is auth-walled.
- **Insights tab:** Fully auth-walled (Premium/Recruiter only).

### `/jobs/view/{jobId}/` — Job Posting

**Public (no login required).** LinkedIn exposes a guest-facing API at:
`https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/...`
Individual job pages at `/jobs/view/{jobId}` are indexable and partially visible without authentication ([Scrapfly](https://scrapfly.io/blog/posts/how-to-scrape-linkedin)). `[MED]`

### `/pub/dir/` — Legacy People Directory

**Status: Functionally deprecated for SEO/public use as of 2024–2026.** `[MED]`

The `/pub/dir/` path once hosted alphabetical public profile directories. Evidence from OSINT researchers and recruiters shows these patterns were referenced in Google X-ray queries using `-inurl:pub/dir` exclusions to avoid directory listing pages rather than individual profiles ([ERE.net](https://www.ere.net/articles/back-to-the-basics-how-to-x-ray-linkedin-for-profiles-using-google), 2013; [LinkedIn recruiter post](https://www.linkedin.com/posts/t-brad-kielinski-philadelphia-tech-recruiter_jobsearch-recruiterlife-activity-7425519892214235136-Fq5a), 2026). The directory pages are still referenced in Google indexing but LinkedIn now typically redirects or auth-walls the `/pub/` namespace. The `/pub/{vanity}/` format (older profile URL format before `/in/` became standard) also redirects to `/in/` equivalents.

---

## 5. Anti-Automation Signals

### Cookie Arsenal (from [LinkedIn Cookie Table](https://www.linkedin.com/legal/l/cookie-table))

| Cookie | Domain | Purpose | Expiry | Category |
|--------|--------|---------|--------|----------|
| `li_at` | `.www.linkedin.com` | **Primary auth token** — authenticates members and API clients | 1 year | Auth |
| `JSESSIONID` | `.www.linkedin.com` | **CSRF protection** — also used as `csrf-token` in Voyager API headers (strip the `ajax:` prefix) | Session | Security |
| `lidc` | `.linkedin.com` | Data center routing/selection | 24 hours | Functional |
| `bcookie` | `.linkedin.com` | Browser identifier — uniquely identifies devices, detects abuse | 1 year | Security |
| `bscookie` | `.www.linkedin.com` | Remembers 2FA verification, prevents repeat auth challenges | 1 year | Security |
| `trkInfo` | `www.linkedin.com` | **Anti-abuse / threat analysis** — short-lived tracking | 5 seconds | Security |
| `trkCode` | `www.linkedin.com` | Anti-abuse process tracking | 5 seconds | Security |
| `denial-client-ip` | `www.linkedin.com` | Stores visitor IP for anti-scraping / DOS prevention | 5 seconds | Security |
| `denial-reason-code` | `www.linkedin.com` | Anti-scraping / DOS reason code | 5 seconds | Security |
| `f_token` | `.linkedin.com` | Bot detection for anti-scraping | 3 minutes | Security |
| `li_referer` | `.linkedin.com` | Detects bots by remembering referring page before CAPTCHA redirect | 15 minutes | Security |
| `rtc` | `www.linkedin.com` | Anti-abuse processes | 120 seconds | Security |
| `fcookie` | `linkedin.com` | Bot detection | 7 days | Security |
| `ccookie` | `linkedin.com` | Remembers if user received a CAPTCHA challenge | 20 minutes | Security |
| `li_g_view` | `.linkedin.com` | Counts guest profile views before triggering login prompt | 30 days | Functional |
| `spectroscopyId` | `www.linkedin.com` | Catches malicious activity through browser extensions | Session | Security |
| `__cf_bm` | Cloudflare domains | Cloudflare bot detection | 30 minutes | Security |
| `_px*` | `protechts.net` | HUMAN Security bot detection suite | Various | Security |

### Voyager API Headers

LinkedIn's internal Voyager API (`https://www.linkedin.com/voyager/api/...`) requires the following headers to mimic authenticated browser requests:

```
Cookie: li_at={session_token}; JSESSIONID="ajax:{value}"
csrf-token: ajax:{JSESSIONID value without quotes}
x-restli-protocol-version: 2.0.0
x-li-lang: en_US
x-li-page-instance: urn:li:page:...
```

The `csrf-token` value equals the `JSESSIONID` cookie value with surrounding quotes stripped ([Medium / Voyager API write-up](https://medium.com/data-science/using-browser-cookies-and-voyager-api-to-scrape-linkedin-via-python-25e4ae98d2a8); [Reddit r/SaaS](https://www.reddit.com/r/SaaS/comments/1o5exod/how_to_access_linkedin_data_profiles_messages_and/)).

### trkInfo / trk URL Parameters

`trkInfo` appears in **cookies** (5-second TTL, anti-abuse tracking) and also as a URL query parameter on some internal navigation links. It is not a user-facing parameter — it is a short-lived signed token that LinkedIn uses for click-stream integrity checking and abuse detection. Do not rely on it for automation; its format is opaque and rotates frequently. `[MED]`

### Rate Limits

LinkedIn does not publish official rate limits for public web access. Community-reported limits:

| Action | Approximate Limit | Source |
|--------|------------------|--------|
| Guest profile views before authwall | 3–5 per session | Scrapfly (2026) |
| Public search (authenticated) | ~1,000 results max per query | lobstr.io (2024) |
| Commercial search limit | Resets monthly (varies by account tier) | YouTube tutorial (2022) |
| Profile views (API, standard) | ~80/day | LiSeller (community report) |
| Profile views (API, premium) | ~1,000/day | LiSeller (community report) |
| Connection requests | ~100/week | Community-reported |
| Messages | ~150/day | Community-reported |
| Safe automation actions per account | ≤100/day, ≤10/hour | [Unipile provider limits](https://developer.unipile.com/docs/provider-limits-and-restrictions) |

Official LinkedIn API rate limits (for partners) are not published in documentation and must be viewed per-endpoint in the [Developer Portal Analytics tab](https://learn.acme-software.com/en-us/linkedin/shared/api-guide/concepts/rate-limits).

### `/authwall` Redirect Mechanism

When LinkedIn detects an unauthenticated or over-quota request, it redirects to `/authwall?trk=...&trkInfo=...&original_referer={encodedURL}&sessionRedirect={encodedURL}`. The paths `/signup/cold-join`, `/signup`, `/login`, and `/authwall` are recognized as "not logged in" states by automation tools. `[HIGH]`

---

## 6. Boolean Operators in `&keywords=`

LinkedIn **officially documents** Boolean search support in the keywords field ([LinkedIn Help — Use Boolean Search](https://www.linkedin.com/help/linkedin/answer/a524335)):

| Operator | Format | Effect |
|----------|--------|--------|
| Exact phrase | `"product manager"` | Matches that exact string |
| AND | `accountant AND finance AND CPA` | All terms required (uppercase mandatory) |
| OR | `sales OR marketing OR advertising` | Any term matches (uppercase mandatory) |
| NOT | `programmer NOT manager` | Excludes the following term (uppercase mandatory) |
| Parentheses | `VP NOT (assistant OR SVP)` | Groups logic; only `()` recognized |

**Operator precedence (documented):** Quotes → Parentheses → NOT → AND → OR.

**Limitations (documented):**
- `AND`, `OR`, `NOT` **must be uppercase** — lowercase is treated as a keyword.
- Wildcards (`*`) are **not supported**.
- `+` and `-` are not officially supported.
- Stop words (`by`, `in`, `with`) are excluded from quoted phrase searches.
- Boolean strings are capped at approximately **2,000 characters**.
- Boolean logic works **only in the keywords field** — not in title, company, or school filter fields.

**URL encoding example:**
```
https://www.linkedin.com/search/results/people/?currentCompany=%5B%221035%22%5D&geoUrn=%5B%22103644278%22%5D&keywords=(%22harvard%22%20OR%20%22mit%22)%20AND%20%22general%20manager%22&profileLanguage=%5B%22en%22%5D
```
Decoded: `keywords=("harvard" OR "mit") AND "general manager"` — sourced from [lobstr.io](https://www.lobstr.io/blog/linkedin-search-ultimate-guide).

**Confidence:** `[HIGH]` — officially documented and widely corroborated.

---

## 7. Public Profile JSON-LD / OpenGraph Metadata

LinkedIn public profile pages (`/in/{vanity}/`) embed structured metadata in `<head>` that remains accessible to unauthenticated clients and search engine crawlers.

### OpenGraph Tags (confirmed via scraper writeups)

```html
<meta property="og:title" content="{Name} - {Headline} - LinkedIn" />
<meta property="og:description" content="{Truncated About/Summary}" />
<meta property="og:image" content="https://media.licdn.com/dms/image/..." />
<meta property="og:url" content="https://www.linkedin.com/in/{vanity}/" />
<meta property="og:type" content="profile" />
<meta property="profile:first_name" content="{firstName}" />
<meta property="profile:last_name" content="{lastName}" />
<meta property="profile:username" content="{vanityName}" />
```

The OpenGraph `profile` namespace (`http://ogp.me/ns/profile#`) supports `first_name`, `last_name`, `username`, and `gender` properties. LinkedIn includes at minimum `og:title`, `og:description`, `og:image`, and `og:url`. `[MED]`

### JSON-LD (confirmed present in `<script type="application/ld+json">`)

LinkedIn embeds a `ProfilePage` schema containing a `Person` or `Organization` entity. The [Scrapfly](https://scrapfly.io/blog/posts/how-to-scrape-linkedin) analysis confirms that "core details [are] available in `application/ld+json` script tags" including `articleBody` for articles. A typical structure:

```json
{
  "@context": "https://schema.org",
  "@type": "ProfilePage",
  "mainEntity": {
    "@type": "Person",
    "name": "Full Name",
    "url": "https://www.linkedin.com/in/vanity/",
    "image": "https://media.licdn.com/dms/image/...",
    "jobTitle": "Current title",
    "worksFor": { "@type": "Organization", "name": "Company Name" },
    "description": "Headline text",
    "sameAs": ["https://www.linkedin.com/in/vanity/"]
  }
}
```

**Key limitation:** As of March 2024, LinkedIn **reduced what non-authenticated visitors see** in the rendered HTML — the About section is truncated to ~69 characters and the Experience section descriptions may be blurred ([LinkedIn March 2024 update post](https://www.linkedin.com/pulse/linkedin-new-features-updates-march-24-edition-link-ability-nz-2cwtc)). However, the JSON-LD in `<head>` may still contain richer data than the visible DOM, making it a reliable fallback for automation. `[MED]`

**Practical implication for DOM mapping:** Always parse `<script type="application/ld+json">` before attempting DOM extraction. LinkedIn's approach to limiting guest-visible DOM content while preserving head metadata for SEO means the JSON-LD/OG layer is the most stable extraction surface for unauthenticated requests.

---

## 8. The `/pub/dir/{firstName}/{lastName}` Legacy Directory

**Pattern:** `https://www.linkedin.com/pub/dir/{firstName}/{lastName}`

**Example:** `https://www.linkedin.com/pub/dir/john/smith` — once returned a paginated list of all public LinkedIn profiles with that name.

**Status as of 2025–2026: Effectively non-functional / deprecated.** `[MED]`

Evidence:
- The pattern is referenced in OSINT search guides from 2013–2019 as an exclusion target in Google X-ray queries (using `-inurl:pub/dir` to avoid directory pages and target individual `/in/` profiles) ([ERE.net](https://www.ere.net/articles/back-to-the-basics-how-to-x-ray-linkedin-for-profiles-using-google)).
- A 2026 LinkedIn recruiter post still references `site:www.linkedin.com/pub/` alongside `/in/` in Google Boolean strings ([LinkedIn post by T. Brad Kielinski](https://www.linkedin.com/posts/t-brad-kielinski-philadelphia-tech-recruiter_jobsearch-recruiterlife-activity-7425519892214235136-Fq5a)), suggesting these paths still produce some Google-indexed content.
- No recent (2024–2026) primary source confirms that `/pub/dir/` returns live, useful data when accessed directly. Most evidence suggests it now redirects to the authwall or returns empty/404 responses.

**The `/pub/{vanity}/` format** (older profile URL format predating `/in/`) still functions as a redirect to the corresponding `/in/` URL. `[HIGH]`

---

## 9. Profile Vanity URL vs. Numeric/Obfuscated ID

### Two URL Namespaces

LinkedIn uses two functionally equivalent ways to identify a profile via URL:

| Format | Example | Description |
|--------|---------|-------------|
| Vanity URL | `/in/jordanrivera` | Human-readable custom slug; 3–100 chars; case-insensitive |
| Obfuscated ID | `/in/ACoAAAfooo...` | Base64-encoded internal member ID (the `ACoA...` format) |

Both resolve to the same profile page. `[HIGH]` (confirmed by LinkedIn Help, multiple developer sources)

### Vanity URL Rules ([LinkedIn Help](https://www.linkedin.com/help/linkedin/answer/a542685/manage-your-public-profile-url))

- 3–100 characters; letters, numbers, hyphens only; no spaces or special characters.
- Case-insensitive (`/in/JohnSmith` = `/in/johnsmith`).
- Can be changed up to **5 times per 180 days**.
- When a user changes their vanity URL, **old URLs continue to redirect** to the new URL.
- Auto-generated on account creation; can be customized by the member.
- Country-specific subdomains exist (e.g., `ca.linkedin.com/in/name`) and redirect to `www.linkedin.com/in/name`.

### Obfuscated ID (`ACoA...` format)

The `ACoA...` string is a Base64-encoded representation of the internal numeric member ID. The internal numeric ID can also be accessed directly via the Voyager API (`urn:li:member:271682814`) or via the LinkedIn API v2 `id` field. The `entityUrn` format in Voyager responses is `urn:li:fs_profile:ACoAABAxjP4B...` where the alphanumeric suffix is the same `ACoA...` string used in `/in/` URLs. `[MED]`

**API note:** The [LinkedIn Profile API](https://learn.acme-software.com/en-us/linkedin/shared/integrations/people/profile-api) uses `vanityName` (the slug) to construct the public URL, and a separate application-scoped `id` (not the `ACoA...` format) that is context-specific per OAuth application:

> "Each member `id` is unique to the context of your application only. Sharing a `person ID` across applications will not work and result in a 404 error."

This means the numeric member IDs in `urn:li:member:NNNN` (from Voyager) and the OAuth API `id` field are **not the same namespace**.

### Practical Mapping

For recruiting-automation purposes:
1. Vanity URL is the stable, canonical identifier to store.
2. The `/in/ACoA.../` format is equivalent and can be used interchangeably for direct navigation.
3. The `urn:li:fs_profile:ACoA...` URN, used in Voyager API internals, shares the same `ACoA...` identifier as the URL form.
4. The OAuth API's `id` (e.g., `"yrZCpj2Z12"`) is application-scoped and should not be stored as a cross-system identifier.

---

## Appendix: Quick Reference — Known LinkedIn Numeric IDs

| Entity | Name | Numeric ID |
|--------|------|------------|
| Company | Microsoft | 1035 |
| Company | Google | 1441 |
| Geo | United States | 103644278 |
| Geo | United Kingdom | 101165590 |
| Geo | San Francisco Bay Area | 90000084 |
| Geo | Greater New York | 90000070 |
| Geo | Canada | 101174742 |
| Geo | Australia | 101452733 |
| Geo | India | 102713980 |
| Geo | Germany | 101282230 |
| Geo | France | 105015875 |
| Geo | Singapore | 102454443 |
| Industry | Computer Software | 4 |

Sources: [Captain Data geo URN list](https://support.captaindata.com/en/articles/10725212-list-of-geocodeurn-to-use-in-your-geography-parameter); [Apify LinkedIn scraper docs](https://apify.com/logical_scrapers/linkedin-people-search-scraper); [OSINT Combine](https://www.osintcombine.com/post/corporate-profiling-advanced-linkedin-searching-more).

---

## Key Findings Summary

1. **URL parameter taxonomy** is reverse-engineered; LinkedIn publishes only Boolean search operators officially. The `geoUrn`, `currentCompany`, `industry`, `network` parameters are stable across community tools but not documented.

2. **Profile sub-pages** (`/details/experience/`, etc.) exist as rendered routes but require authentication; they are not independently crawlable without a session.

3. **Company `/people/`** supports `facetGeoRegion` and `facetCurrentFunction` URL filters, reverse-engineered from Voyager API network requests.

4. **Guest visibility** was materially reduced in early 2024: About sections truncated to ~69 chars, Experience descriptions blurred. The JSON-LD in `<head>` remains the most reliable unauthenticated data source.

5. **Anti-automation stack** is multi-layered: LinkedIn-native cookies (`li_at`, `JSESSIONID` as CSRF, `bcookie`, `trkInfo`), Cloudflare (`__cf_bm`), HUMAN Security (`_px*`), plus 5-second tracking cookies (`denial-client-ip`, `denial-reason-code`, `trkCode`).

6. **Boolean search** in `&keywords=` is authoritatively documented. Operators AND/OR/NOT must be uppercase. No wildcards. Parens supported. ~2,000-character cap.

7. **JSON-LD / OG metadata** is present in public profile HTML `<head>` and is the most durable DOM extraction surface for unauthenticated requests.

8. **`/pub/dir/`** is effectively deprecated; no confirmed live functionality in 2025–2026. Use Google X-ray (`site:linkedin.com/in/`) instead.

9. **Vanity URL vs. ACoA ID**: Both work interchangeably in `/in/` URLs. Vanity URL is the preferred stable identifier. `ACoA...` strings are Base64-encoded internal IDs. OAuth API `id` values are application-scoped and non-portable.
