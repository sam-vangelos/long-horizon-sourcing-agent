// 01 — Authenticated profile (top card + outline)
//
// Run on:  https://www.linkedin.com/in/{vanity}/   (logged-in seat)
// Goal:    Capture the authenticated top card + section anchors so we can
//          confirm whether the public selectors (h1 / first h2 / second h2 /
//          trk=public_profile_*) still pin the same elements when logged in,
//          and discover the authenticated-only nodes (Connect / Message /
//          More buttons, Open To frame, Verifications badge).
//
// Safety:  READ-ONLY. Do not click anything during capture.

(function capture() {
  if (!window.__cloris) throw new Error('Load _lib.js first');
  const __ = window.__cloris;

  const payload = {
    url: location.href,
    title: document.title,
    meta: __.metaTags(),
    jsonLd: __.jsonLd(),
    top_card: {
      // Same anchors used by PublicProfilePage.readTopCard()
      h1: __.walk(document, 'main h1'),
      h2_all: __.walk(document, 'main h2'),
      photo: __.walk(document, 'main button[aria-label] > img[alt]'),
      buttons_main: __.walk(document, 'main section button'),
    },
    // Authenticated-only — none of these exist on the guest view.
    auth_only: {
      connect_button: __.walk(document, 'main button:has-text("Connect")'),
      message_button: __.walk(document, 'main button:has-text("Message")'),
      follow_button: __.walk(document, 'main button:has-text("Follow")'),
      more_button: __.walk(document, 'main button[aria-label*="More actions"]'),
      open_to_card: __.walk(document, 'section:has(h2:has-text("Open to"))'),
      verifications_card: __.walk(document, 'section:has(h2:has-text("Verifications"))'),
    },
    // Section headings — should be unchanged from guest view.
    section_headings: __.walk(document, 'main section > h2, main section > div > h2'),
    // Activity / Featured / Highlights are sometimes only present when logged in.
    activity_section: __.walk(document, 'section:has(h2:has-text("Activity"))'),
    featured_section: __.walk(document, 'section:has(h2:has-text("Featured"))'),
    // Top of right rail, if any.
    right_rail_aside: __.walk(document, 'aside'),
  };

  return __.dump('01_authenticated_profile', payload);
})();
