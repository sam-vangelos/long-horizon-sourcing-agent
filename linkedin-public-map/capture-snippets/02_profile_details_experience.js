// 02 — Profile details/experience overlay
//
// Run on:  https://www.linkedin.com/in/{vanity}/details/experience/
// Goal:    Capture the full-list overlay route so we can map experience-item
//          children deterministically (title, company, dates, location, bullets).
//          On the guest view this route redirects to the auth wall, so this is
//          authenticated-only.
//
// Safety:  READ-ONLY.

(function capture() {
  if (!window.__cloris) throw new Error('Load _lib.js first');
  const __ = window.__cloris;

  // Each experience entry is wrapped in a list item.
  const items = [...document.querySelectorAll('main ul > li, main section ul > li')]
    .filter(__.isVisible);

  const payload = {
    url: location.href,
    list_root_candidates: __.walk(document, 'main section ul, main ul'),
    item_count: items.length,
    items: items.slice(0, 12).map((li) => ({
      outer: __.serializeNode(li),
      anchors: [...li.querySelectorAll('a[href]')].map(__.serializeNode),
      spans_visible: [...li.querySelectorAll('span[aria-hidden="true"]')]
        .filter(__.isVisible)
        .slice(0, 12)
        .map(__.serializeNode),
      spans_sr_only: [...li.querySelectorAll('span.visually-hidden, span[class*="sr-only"]')]
        .slice(0, 12)
        .map(__.serializeNode),
    })),
    // Pagination / show-more — useful to confirm whether overlay is virtualized.
    show_more: __.walk(document, 'main button:has-text("Show more"), main button:has-text("Show all")'),
    pagination: __.walk(document, 'main nav[aria-label*="agination"], main button[aria-label*="ext page"]'),
  };

  return __.dump('02_profile_details_experience', payload);
})();
