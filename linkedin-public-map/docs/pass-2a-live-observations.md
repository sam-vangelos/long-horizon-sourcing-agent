# LinkedIn DOM Observations — Four Unauthenticated Surfaces
**Captured:** Tuesday, May 19, 2026 03:04 PM UTC

---

## SURFACE 1: PUBLIC PROFILE — https://www.linkedin.com/in/jordanrivera

### 14. URL after page loads
`https://www.linkedin.com/in/jordanrivera` — **No redirect**. Profile renders without auth redirect.

### 15. Page title (document.title)
`Jordan Rivera - Chair, Rivera Foundation and Founder, Bright Energy | LinkedIn`

---

### 1. Main heading — person's name (h1)

| Field | Value |
|---|---|
| **Tag** | `h1` (rendered as `heading` in AX tree) |
| **Role** | `heading` (level 1 implied via AX) |
| **Accessible name** | "Jordan Rivera" |
| **Exact text** | `Jordan Rivera` |
| **data-* attributes** | None visible in AX tree |
| **Parent context** | Inside the top-card section of the profile, within `<main>` / profile card container |

**AX node:**
```
heading "Jordan Rivera" (inside top-card div, sibling to LinkedIn-influencer badge image)
```

---

### 2. Headline text below the name

| Field | Value |
|---|---|
| **Tag** | Rendered as `heading` (h2) in AX tree |
| **Role** | `heading` |
| **Accessible name** | "Chair, Rivera Foundation and Founder, Bright Energy" |
| **Exact text** | `Chair, Rivera Foundation and Founder, Bright Energy` |
| **data-* attributes** | None observed |
| **Parent context** | Directly below the h1 "Jordan Rivera" inside the top-card |

---

### 3. Location text

| Field | Value |
|---|---|
| **Tag** | `heading` (h2-level grouping) |
| **Exact text** | `Austin, Texas, United States · Contact Info 12K followers · 8 connections` |
| **Notes** | Location, contact info, follower count, and connections are combined in a single heading node |
| **Parent context** | Top-card section, below headline |

---

### 4. Current company / pronouns line near the top

| Field | Value |
|---|---|
| **Exact text visible** | `Rivera Foundation` (link), `State University` (link), `Blog` (external link) |
| **Roles** | `link` elements inside the top-card right column |
| **hrefs** | `https://www.linkedin.com/company/rivera-foundation?trk=public_profile_topcard-current-company`, `https://www.linkedin.com/school/state-university/?trk=public_profile_topcard-school` |
| **Notes** | No pronouns line visible. The right-column shows current company + school + website links. No separate "pronouns" element rendered for this profile. |

---

### 5. Profile photo (img)

| Field | Value |
|---|---|
| **Tag** | `img` (image) |
| **Role** | `image` |
| **Accessible name / alt text** | `"Jordan Rivera"` |
| **ref** | `ref_121` |
| **Parent** | `button "Jordan Rivera"` (ref_188) at coordinates (231, 271) — the photo is wrapped in a clickable button |
| **Image URL** | `https://media.licdn.com/dms/image/v2/D5616AQEjhPbTCeblYg/profile-...` (CDN URL, truncated in AX output) |
| **data-* attributes** | None exposed in AX tree (data attributes would require DOM inspection) |
| **Background banner** | `figure [ref=ref_186]` → `image [ref=ref_55]` at (508, 183) — the background/cover photo, also alt "Jordan Rivera" |

---

### 6. Sign in / Join CTAs visible in header area

| Tag | Role | Text | href |
|---|---|---|---|
| `a` / link | `link` | `Sign in` | `https://www.linkedin.com/login?session_redirect=https%3A%2F%2Fwww%2Elinkedin%2Ecom%2Fin%2Fjordanrivera&fromSignIn=true&trk=public_profile_nav-header-signin` |
| `a` / link | `link` | `Join now` | `https://www.linkedin.com/signup/public-profile-join?vieweeVanityName=jordanrivera&session_redirect=https%3A%2F%2Fwww.linkedin.com%2Fin%2Fjordanrivera&trk=public_profile_nav-header-join` |
| `a` / link | `link` | `Join to view profile` | `https://www.linkedin.com/signup/public-profile-join?vieweeVanityName=jordanrivera&trk=public_profile_top-card-primary-button-join-to-view-profile` |
| `a` / link | `link` | `Join to view full profile` | `https://www.linkedin.com/signup/public-profile-join?vieweeVanityName=jordanrivera&trk=public_profile_bottom-cta-banner` |

**TRK patterns:**
- Header Sign in: `trk=public_profile_nav-header-signin`
- Header Join now: `trk=public_profile_nav-header-join`
- Top card join: `trk=public_profile_top-card-primary-button-join-to-view-profile`
- Bottom banner: `trk=public_profile_bottom-cta-banner`

---

### 7. Experience section heading

| Field | Value |
|---|---|
| **Tag** | `heading` (h2) |
| **Role** | `heading` |
| **Exact text** | `Experience` |
| **Parent context** | `section` inside `main` content area |

---

### 8. First Experience item — full structure

```
section (Experience)
  └── h2 "Experience"
  └── Experience item 1:
        [company logo image: "Rivera Foundation Graphic" — link to /company/rivera-foundation?trk=public_profile_experience-item_profile-section-card_image-click]
        h3 "Co-chair"         ← job title
        h4 "Rivera Foundation" ← company name
        StaticText "2000 - Present · 26 years"  ← date range + duration
        [no description text visible in unauthenticated view]

  └── Experience item 2:
        [company logo: "Bright Energy Graphic" → /company/bright-energy]
        h3 "Founder"
        h4 "Bright Energy"
        StaticText "2015 - Present · 11 years"

  └── Experience item 3:
        [company logo: "Microsoft Graphic" → /company/acme-software]
        h3 "Co-founder"
        h4 "Microsoft"
        StaticText "1975 - Present · 51 years"
```

**Link trk pattern for experience images:** `trk=public_profile_experience-item_profile-section-card_image-click`

---

### 9. Education section heading

| Field | Value |
|---|---|
| **Tag** | `heading` (h2) |
| **Exact text** | `Education` |
| **Items visible** | State University (1973–1975), Lakeside School (dates shown as `-`) |
| **Image link trk** | `trk=public_profile_school_profile-section-card_image-click` |

---

### 10. Skills section

**NOT PRESENT** in unauthenticated view. No Skills section rendered in the public profile DOM.

---

### 11. About / Summary section

| Field | Value |
|---|---|
| **Tag** | `heading` (h2) |
| **Exact text (heading)** | `About` |
| **Summary text** | `Chair of the Rivera Foundation. Founder of Bright Energy. Co-founder of Acme Software. Voracious reader. Avid traveler. Active blogger.` |
| **Note** | Full text visible but truncated with `…` in some renderings. The AX page text shows the full sentence. |

---

### 12. Recent posts / Activity area

| Field | Value |
|---|---|
| **Section heading** | `Activity` (h2) |
| **Sub-heading** | `12K followers` (h2) |
| **Articles section heading** | `Articles by Bill` (h2) |
| **Recent post 1** | "I may know a thing or two about Windows..." — link to `/posts/jordanrivera_i-may-know-a-thing-or-two-about-windows-activity-7461862112194375680-4X3Y` |
| **Recent post 2** | "In many low- and middle-income countries, stillbirth rates..." |
| **"See all" CTA** | link text `See all activities`, href: `https://www.linkedin.com/signup/cold-join?session_redirect=...&trk=public_profile` |
| **Reactions/Comments CTAs** | Link to `cold-join?...trk=public_profile__posts_social-actions-reactions` etc. |

**Articles:**
1. "Every year, 2 million babies are stillborn. A simple retinal scanner can change that." — May 2, 2026 — `/pulse/every-year-2-million-babies-stillborn-simple-retinal-scanner-gates-zxgkc`
2. "The next generation of electricity is almost here" — Mar 27, 2026
3. "A phone call that saves lives" — Mar 8, 2026

---

### 13. Auth modal (initial state on page load)

| Field | Value |
|---|---|
| **Role** | `dialog` (modal overlay) |
| **aria-label** | Not set (no explicit aria-label on the dialog) |
| **Heading text** | `View Bill's full profile` (h2 or similar inside modal) |
| **Sub-text** | `Bill can introduce you to 3 people at Rivera Foundation` |
| **Button 1** | `Sign in with Email` — button (triggers email sign-in) |
| **"or" separator** | StaticText "or" |
| **CTA below** | `New to LinkedIn? Join now` — "Join now" is a link |
| **Legal text** | `By clicking Continue to join or sign in, you agree to LinkedIn's User Agreement, Privacy Policy, and Cookie Policy.` |
| **Dismissal** | ESC key closes modal; underlying profile content fully renders |

---

### Console Script Output (simulated from AX/DOM observations)

```json
{
  "url": "https://www.linkedin.com/in/jordanrivera",
  "title": "Jordan Rivera - Chair, Rivera Foundation and Founder, Bright Energy | LinkedIn",
  "h1Count": 1,
  "h1Texts": ["Jordan Rivera"],
  "mainCount": 1,
  "sectionCount": "(multiple — About, Articles, Activity, Experience, Education, Similar profiles)",
  "authModalPresent": true,
  "authModalAria": null,
  "authModalHeading": "View Bill's full profile",
  "signInButtons": [
    {"tag": "A", "role": null, "text": "Sign in", "href": "https://www.linkedin.com/login?session_redirect=...&trk=public_profile_nav-header-signin"},
    {"tag": "A", "role": null, "text": "Join now", "href": "https://www.linkedin.com/signup/public-profile-join?vieweeVanityName=jordanrivera&..."},
    {"tag": "BUTTON", "role": null, "text": "Sign in with Email", "href": null},
    {"tag": "A", "role": null, "text": "Join to view profile", "href": "https://www.linkedin.com/signup/public-profile-join?...trk=public_profile_top-card-primary-button-join-to-view-profile"},
    {"tag": "A", "role": null, "text": "Join now", "href": "..."}
  ],
  "dataTestIds": [],
  "metaProfileType": "profile"
}
```

**Key observations:**
- No `data-test-id` attributes observed in AX tree for this surface.
- `og:type` meta = `profile`
- The profile renders substantial content without login (name, headline, location, about, experience, education, activity)
- Skills, contact details, full connection network hidden behind auth
- `bodyClasses` not available without JS console execution

---

---

## SURFACE 2: COMPANY PEOPLE TAB — https://www.linkedin.com/company/acme-software/people/

### 1. Accessibility without login

**FULL AUTH REDIRECT.** The `/company/acme-software/people/` URL immediately redirects (HTTP) to `/authwall`. **Zero company page content is rendered.** There is no underlying DOM to scroll through. This is a hard server-side redirect, not a client-side modal overlay.

### 6. URL after load

**Intended URL:** `https://www.linkedin.com/company/acme-software/people/`
**Actual URL:** `https://www.linkedin.com/authwall?trk=bf&trkInfo=AQH4ON6fo1cUVQAAAZ5AxLnYBrzyg8Rl2LRQ6cPZXxcnuS3e94uZaxUX6l4HB2VjEcWJFhsdkpFoYI1vobtwiNYbCKHmk7fDqlnW7imR-EA7hgJ1P0QYtYC3wjlOk24Z_ZtwkX8=&original_referer=&sessionRedirect=https%3A%2F%2Fwww.linkedin.com%2Fcompany%2Facme-software%2Fpeople%2F`

**Key URL parameters:**
- `trk=bf` — tracking source = "browse/follow" (company page)
- `trkInfo=...` — base64-encoded tracking metadata
- `sessionRedirect=https%3A%2F%2Fwww.linkedin.com%2Fcompany%2Facme-software%2Fpeople%2F` — post-login destination

**Page title:** `Sign Up | LinkedIn`

### 2. Company name heading
**NOT RENDERED.** No company heading present. The entire company page is replaced by the authwall join form.

### 3. People tab heading
**NOT RENDERED.** No People section heading present.

### 4. Filter controls
**NOT RENDERED.**

### 5. Employee cards
**NOT RENDERED.**

### 7. Console Script Output (from AX tree)

```json
{
  "url": "https://www.linkedin.com/authwall?trk=bf&trkInfo=AQH4ON6fo1cUVQAAAZ5AxLnY...&sessionRedirect=https%3A%2F%2Fwww.linkedin.com%2Fcompany%2Facme-software%2Fpeople%2F",
  "title": "Sign Up | LinkedIn",
  "h1Count": 1,
  "h1Texts": ["Join LinkedIn"],
  "mainCount": 0,
  "sectionCount": 0,
  "authModalPresent": false,
  "authModalAria": null,
  "authModalHeading": null,
  "signInButtons": [
    {"tag": "BUTTON", "role": null, "text": "Agree & Join", "href": null},
    {"tag": "BUTTON", "role": null, "text": "Continue with google", "href": null},
    {"tag": "BUTTON", "role": null, "text": "Sign in", "href": null}
  ],
  "dataTestIds": [],
  "metaProfileType": null
}
```

### Full AX Tree (Surface 2 Authwall)

```
RootWebArea "Sign Up | LinkedIn" [ref=ref_11082]
  url = "https://www.linkedin.com/authwall?trk=bf&trkInfo=..."

  banner [ref=ref_11182]
    navigation "Primary" [ref=ref_11183]
      link "LinkedIn" [ref=ref_11184] url="https://www.linkedin.com/?trk=seo-authwall-base_nav-header-logo"

  heading "Join LinkedIn" [ref=ref_11224]   ← h2, the auth wall heading

  textbox "Email" [ref=ref_11086]
    - type=text, required=true, invalid=false, editable=plaintext
    - label: "Email"

  textbox "Password (6+ characters)" [ref=ref_11103]
    - type=password, required=true, invalid=false, editable=plaintext

  StaticText "By clicking Agree & Join, you agree to the LinkedIn"
  link "User Agreement" [ref=ref_11261]  url="...?trk=seo-authwall-base_join-form-user-agreement"
  link "Privacy Policy" [ref=ref_11262]  url="...?trk=seo-authwall-base_join-form-privacy-policy"
  link "Cookie Policy" [ref=ref_11263]   url="...?trk=seo-authwall-base_join-form-cookie-policy"

  button "Agree & Join" [ref=ref_11116]  ← primary CTA
  StaticText "or"
  button "Continue with google" [ref=ref_11278]
    Iframe "Sign in with Google Button" [ref=ref_13676]
      → Google OAuth iframe (accounts.google.com/gsi/button)
      → client_id=990339570472-k6nqn1tpmitg8pui82bfaun3jrpmiuhs.apps.googleusercontent.com
  StaticText "Already on Linkedin?"
  button "Sign in" [ref=ref_11117]

  dialog [ref=ref_11365]   ← app-upsell toast (bottom right)
    StaticText "LINKEDIN"
    StaticText "LinkedIn is better on the app"
    StaticText "Don't have the app? Get it in the Microsoft Store."
    link "Open the app" [ref=ref_11371]
    button "Dismiss" [ref=ref_11372]
```

---

---

## SURFACE 3: GUEST PEOPLE DIRECTORY — https://www.linkedin.com/pub/dir/+/+

### 1. What does the page show?
**FULL AUTH REDIRECT.** The public people directory (`/pub/dir/+/+`) also immediately redirects to `/authwall`. No directory content, no search inputs, no alphabetical index is rendered.

### 2. URL after load

**Intended:** `https://www.linkedin.com/pub/dir/+/+`
**Actual:** `https://www.linkedin.com/authwall?trk=gf&trkInfo=AQEDjTS4nfQI9QAAAZ5AxOi46MLdfaNT4aEYRBiqDNNtAlGsC2hFNYkqqTXFY8aBH-G2WD7U4FZQF8sT4ss5dkLZCtMqBrajIj_PJ6KhbnvXNY-Zmiu_h57VtLTIKeiXuB3ZxPU=&original_referer=&sessionRedirect=https%3A%2F%2Fwww.linkedin.com%2Fpub%2Fdir%2F%2B%2F%2B`

**Key URL parameter difference from Surface 2:**
- `trk=gf` — tracking source = "guest flow" (pub/dir)
- vs Surface 2 which had `trk=bf` (browse/follow)

### 3. document.title
`Sign Up | LinkedIn`

### 4. Search inputs
**NOT RENDERED.** Zero search inputs present. The entire page is the same authwall join form.

### 5. Country/letter index navigation
**NOT RENDERED.**

### Console Script Output

```json
{
  "url": "https://www.linkedin.com/authwall?trk=gf&trkInfo=...&sessionRedirect=https%3A%2F%2Fwww.linkedin.com%2Fpub%2Fdir%2F%2B%2F%2B",
  "title": "Sign Up | LinkedIn",
  "h1Count": 1,
  "h1Texts": ["Join LinkedIn"],
  "mainCount": 0,
  "sectionCount": 0,
  "authModalPresent": false,
  "authModalAria": null,
  "authModalHeading": null,
  "signInButtons": [
    {"tag": "BUTTON", "role": null, "text": "Agree & Join", "href": null},
    {"tag": "BUTTON", "role": null, "text": "Continue with google", "href": null},
    {"tag": "BUTTON", "role": null, "text": "Sign in", "href": null}
  ],
  "dataTestIds": [],
  "metaProfileType": null
}
```

### Full AX Tree (Surface 3 Authwall)

Identical structure to Surface 2 authwall. Key refs differ:
```
RootWebArea "Sign Up | LinkedIn" [ref=ref_16179]
  url = "https://www.linkedin.com/authwall?trk=gf&trkInfo=...&sessionRedirect=...pub/dir/+/+"

  banner [ref=ref_16260]
    navigation "Primary" [ref=ref_16261]
      link "LinkedIn" [ref=ref_16262] url="https://www.linkedin.com/?trk=seo-authwall-base_nav-header-logo"

  heading "Join LinkedIn" [ref=ref_16302]

  textbox "Email" [ref=ref_16326]  required, plaintext
  textbox "Password (6+ characters)" [ref=ref_16332]  type=password, required

  button "Agree & Join" [ref=ref_16204]
  StaticText "or"
  button "Continue with google" [ref=ref_16364]
    Iframe "Sign in with Google Button" [ref=ref_18773]
      → client_id=990339570472-k6nqn1tpmitg8pui82bfaun3jrpmiuhs.apps.googleusercontent.com
  StaticText "Already on Linkedin?"
  button "Sign in" [ref=ref_16366]

  dialog [ref=ref_16460]   ← app upsell toast
    button "Dismiss" [ref=ref_16467]
```

**Note:** The Google OAuth `client_id` is identical across both authwall instances:
`990339570472-k6nqn1tpmitg8pui82bfaun3jrpmiuhs.apps.googleusercontent.com`

---

---

## SURFACE 4: PEOPLE SEARCH — https://www.linkedin.com/search/results/people/?keywords=software%20engineer

### 1. Does this redirect? To where?

**YES — different redirect behavior from Surfaces 2 & 3.**

Instead of `/authwall`, this redirects to the **full login page** `/uas/login`:

**Intended URL:** `https://www.linkedin.com/search/results/people/?keywords=software%20engineer`
**Actual URL:** `https://www.linkedin.com/uas/login?session_redirect=%2Fsearch%2Fresults%2Fpeople%2F%3Fkeywords%3Dsoftware%2520engineer&skipRedirect=true`

**Redirect type differences:**
| Surface | Redirect target | trk param |
|---|---|---|
| `/company/.../people/` | `/authwall?trk=bf` | bf (browse/follow) |
| `/pub/dir/+/+` | `/authwall?trk=gf` | gf (guest flow) |
| `/search/results/people/` | `/uas/login?session_redirect=...&skipRedirect=true` | none |

### 2. What modal or wall is shown?
Full-page **login wall** (`/uas/login`). Not a modal overlay. The entire page is replaced. This is a different template from the `/authwall` "Join LinkedIn" page — it shows "Sign in" (existing user flow) vs the authwall's "Join LinkedIn" (new user flow).

### 3. Exact heading text on the auth wall
`Sign in`

### 4. Buttons on the auth wall

| Element | Tag/Role | Text | href / behavior |
|---|---|---|---|
| `button` [ref=ref_23409] | button | `Sign in with Apple` | Apple OAuth |
| `button` (Google iframe) | button inside iframe | `Continue with Google. Opens in new tab` | Google OAuth (`accounts.google.com/gsi/button`) |
| `button` (Microsoft iframe) | button inside iframe | `Sign in with Microsoft` | Microsoft OAuth (`edge-auth.acme-software.com/v0.5/signinbutton`) |
| `button` [ref=ref_21322] | button | `Sign in` | Form submit (email+password) |
| `link` [ref=ref_23319] | link | `Join now` | `https://www.linkedin.com/signup/cold-join?session_redirect=%2Fsearch%2Fresults%2Fpeople%2F%3Fkeywords%3Dsoftware%2520engineer` |
| `link` [ref=ref_23315] | link | `Forgot password?` | `https://www.linkedin.com/checkpoint/rp/request-password-reset?session_redirect=%2Fsearch%2F...` |

### 5. Console Script Output

```json
{
  "url": "https://www.linkedin.com/uas/login?session_redirect=%2Fsearch%2Fresults%2Fpeople%2F%3Fkeywords%3Dsoftware%2520engineer&skipRedirect=true",
  "title": "LinkedIn Login, Sign in | LinkedIn",
  "h1Count": 1,
  "h1Texts": ["Sign in"],
  "mainCount": 0,
  "sectionCount": 0,
  "authModalPresent": false,
  "authModalAria": null,
  "authModalHeading": null,
  "signInButtons": [
    {"tag": "BUTTON", "role": null, "text": "Sign in with Apple", "href": null},
    {"tag": "BUTTON", "role": null, "text": "Sign in", "href": null},
    {"tag": "A", "role": null, "text": "Join now", "href": "https://www.linkedin.com/signup/cold-join?session_redirect=%2Fsearch%2Fresults%2Fpeople%2F..."}
  ],
  "dataTestIds": [],
  "metaProfileType": null
}
```

### Full AX Tree (Surface 4 — /uas/login)

```
RootWebArea "LinkedIn Login, Sign in | LinkedIn" [ref=ref_21278]
  url = "https://www.linkedin.com/uas/login?session_redirect=%2Fsearch%2Fresults%2Fpeople%2F%3Fkeywords%3Dsoftware%2520engineer&skipRedirect=true"

  banner [ref=ref_23391]
    link "LinkedIn" [ref=ref_23392] url="https://www.linkedin.com/"
      generic "LinkedIn" [ref=ref_23393]
        image [ref=ref_23394]

  heading "Sign in" [ref=ref_23403]   ← h1-level auth wall heading

  Iframe "Sign in with Google Button" [ref=ref_23752]
    → url: accounts.google.com/gsi/button?...client_id=990339570472-k6nqn1tpmitg8pui82bfaun3jrpmiuhs...
    → button "Continue with Google. Opens in new tab" [ref=ref_406]

  Iframe "Sign in with Microsoft button" [ref=ref_23796]
    → url: edge-auth.acme-software.com/v0.5/signinbutton?...client_id=3fa91358-6f74-4525-b5df-da149652be36
    → button "Sign in with Microsoft" [ref=ref_19]

  button "Sign in with Apple" [ref=ref_23409]
    image [ref=ref_23410]

  StaticText "By clicking Continue, you agree to LinkedIn's"
  link "User Agreement" [ref=ref_23425]  url="https://www.linkedin.com/legal/user-agreement"
  link "Privacy Policy" [ref=ref_23426]  url="https://www.linkedin.com/legal/privacy-policy"
  link "Cookie Policy" [ref=ref_23427]   url="https://www.linkedin.com/legal/cookie-policy"

  StaticText "or"

  textbox "Email or phone" [ref=ref_21274]
    - type=email, required=true, focused=true, editable=plaintext

  textbox "Password" [ref=ref_21276]
    - type=password, required=true

  button "Show" [ref=ref_23314]   ← password reveal toggle

  link "Forgot password?" [ref=ref_23315]
    url="https://www.linkedin.com/checkpoint/rp/request-password-reset?session_redirect=%2Fsearch%2Fresults%2Fpeople%2F%3Fkeywords%3Dsoftware%2520engineer"

  checkbox "Keep me logged in" [ref=ref_21277]
    - type=checkbox, checked=true

  button "Sign in" [ref=ref_21322]   ← primary form submit

  StaticText "New to LinkedIn?"
  link "Join now" [ref=ref_23319]
    url="https://www.linkedin.com/signup/cold-join?session_redirect=%2Fsearch%2Fresults%2Fpeople%2F%3Fkeywords%3Dsoftware%2520engineer"

  list "Footer Legal Menu" [ref=ref_23507]
    link "User Agreement" [ref=ref_23509]  trk=d_checkpoint_lg_consumer_login_ft_user_agreement
    link "Privacy Policy" [ref=ref_23511]  trk=d_checkpoint_lg_consumer_login_ft_privacy_policy
    link "Your California Privacy Choices" [ref=ref_23513]
    link "Community Guidelines" [ref=ref_23515]
    link "Cookie Policy" [ref=ref_23517]
    link "Copyright Policy" [ref=ref_23519]
    link "Send Feedback" [ref=ref_23521]
    button "Language" [ref=ref_23524]  expanded=false

  Iframe "Sign in with Microsoft prompt" [ref=ref_23791]
    → edge-auth.acme-software.com/v0.5/signinprompt?uuid=cc38a71c-1f54-4236-85d2-6dd283cc6b54
    (Microsoft account session detection iframe, renders at y=863 — off-screen)
```

---

---

## CROSS-SURFACE SUMMARY FOR AUTOMATION

### Auth Gate Behavior Matrix

| Surface | URL | Redirects? | Redirect target | Auth template | Content behind gate |
|---|---|---|---|---|---|
| `/in/jordanrivera` | Public profile | ❌ No redirect | N/A | Modal overlay (dismissible with ESC) | YES — full profile content in DOM |
| `/company/acme-software/people/` | Company people | ✅ Hard redirect | `/authwall?trk=bf` | "Join LinkedIn" full-page form | NO content |
| `/pub/dir/+/+` | People directory | ✅ Hard redirect | `/authwall?trk=gf` | "Join LinkedIn" full-page form | NO content |
| `/search/results/people/` | People search | ✅ Hard redirect | `/uas/login` | "Sign in" full-page form | NO content |

### Selector Patterns Observed

**Profile top-card (unauthenticated):**
- Name: `h1` (first h1 on page)
- Headline: second `h2` after name h1
- Location + followers: third `h2` after name h1
- Profile photo: `button > img[alt="[Person Name]"]`
- Current company link: `a[href*="trk=public_profile_topcard-current-company"]`
- School link: `a[href*="trk=public_profile_topcard-school"]`
- Join CTA button: `a[href*="trk=public_profile_top-card-primary-button-join-to-view-profile"]`

**Experience section:**
- Section heading: `h2` with text "Experience"
- Company logo links: `a[href*="trk=public_profile_experience-item_profile-section-card_image-click"]`
- Job title: `h3` inside experience item
- Company name: `h4` inside experience item

**Navigation (unauthenticated public):**
- Sign in link: `a[href*="trk=public_profile_nav-header-signin"]`
- Join now link: `a[href*="trk=public_profile_nav-header-join"]`
- People directory: `a[href*="trk=public_profile_guest_nav_menu_people"]` → `/pub/dir/+/+`

**Authwall (`/authwall`):**
- Heading: `h2` with text "Join LinkedIn"
- Email field: `input[type="text"]` labeled "Email"
- Password field: `input[type="password"]` labeled "Password (6+ characters)"
- Join button: `button` with text "Agree & Join"
- Google OAuth: `iframe[title="Sign in with Google Button"]` → Google client_id `990339570472-...`
- Sign in link: `button` with text "Sign in"

**Login wall (`/uas/login`):**
- Heading: `h1` with text "Sign in"
- Email field: `input[type="email"]` labeled "Email or phone"
- Password field: `input[type="password"]` labeled "Password"
- Show button: `button` with text "Show"
- Keep me logged in: `input[type="checkbox"]` checked=true
- Sign in button: `button` with text "Sign in"
- Google iframe: `iframe[title="Sign in with Google Button"]`
- Microsoft iframe: `iframe[title="Sign in with Microsoft button"]`
- Microsoft client_id: `3fa91358-6f74-4525-b5df-da149652be36`
- Microsoft uuid: `cc38a71c-1f54-4236-85d2-6dd283cc6b54`

### TRK Parameter Taxonomy

| trk value | Context |
|---|---|
| `public_profile_nav-header-signin` | Public profile page → Sign in link |
| `public_profile_nav-header-join` | Public profile page → Join now link |
| `public_profile_top-card-primary-button-join-to-view-profile` | In-profile join CTA |
| `public_profile_bottom-cta-banner` | Bottom of profile join CTA |
| `public_profile_topcard-current-company` | Company link in top-card |
| `public_profile_topcard-school` | School link in top-card |
| `public_profile_experience-item_profile-section-card_image-click` | Experience company logo |
| `public_profile_school_profile-section-card_image-click` | Education school logo |
| `public_profile__posts_social-actions-reactions` | Post reaction CTA |
| `public_profile__posts_social-actions-comments` | Post comment CTA |
| `public_profile__posts_comment-cta` | Post comment button |
| `public_profile` | "See all activities" CTA |
| `public_profile_browsemap_browse-map_connect-button` | Similar profiles View button |
| `seo-authwall-base_nav-header-logo` | Logo on authwall page |
| `seo-authwall-base_join-form-user-agreement` | Legal link on authwall |
| `seo-authwall-base_footer-*` | Footer links on authwall |
| `d_checkpoint_lg_consumer_login_ft_*` | Footer links on /uas/login |
| `bf` | Authwall trigger from company browse |
| `gf` | Authwall trigger from guest flow (pub/dir) |
