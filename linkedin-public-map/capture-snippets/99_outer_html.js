// 99 — Full document outerHTML snapshot
//
// Run on:  any page after the page has fully loaded and any modal/dialog you
//          want captured is open.
// Goal:    Capture the complete rendered DOM for archival. Sanitize before
//          committing to the repo (see notes below).
//
// SANITIZATION CHECKLIST (do this before saving to fixtures/dom-snapshots/):
//   • Replace any `li_at`, `JSESSIONID`, `lidc`, `bcookie`, `bscookie`,
//     `li_g_view` cookie values in <script> blobs with REDACTED.
//   • Strip the `csrfToken`, `clientApplicationInstance`, `serviceTrace`,
//     and any `urn:li:fs_*` URNs that include your account id.
//     (Search/replace any URN starting with `urn:li:fs_member:`,
//     `urn:li:fs_profile:`, `urn:li:identity:` that contain numbers you
//     don't want preserved.)
//   • Replace beacon URLs (`https://www.linkedin.com/li/track`,
//     `https://px.ads.linkedin.com/...`) with `https://example.test/REDACTED`.
//   • If the snapshot is of YOUR OWN profile/account, replace your name,
//     photo URL, vanity, and contact info with placeholder values.

(function capture() {
  if (!window.__cloris) throw new Error('Load _lib.js first');
  const __ = window.__cloris;
  // We don't push the whole HTML through the JSON serializer — too large for
  // the clipboard helper. Print directly to console instead.
  const html = document.documentElement.outerHTML;
  console.log(`[cloris-public] outerHTML length=${html.length}`);
  console.log('[cloris-public] Use "Copy element" on the <html> node in Elements panel for the cleanest copy.');
  console.log(html.slice(0, 200) + '…');
  // Also report the page state metadata so we can tag the snapshot.
  return __.dump('99_outer_html_metadata', {
    url: location.href,
    title: document.title,
    bytes: html.length,
    meta: __.metaTags(),
    cookies_keys: document.cookie.split(';').map((c) => c.trim().split('=')[0]),
  });
})();
