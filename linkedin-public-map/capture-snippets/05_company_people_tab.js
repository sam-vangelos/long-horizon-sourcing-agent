// 05 — Company /people/ tab (authenticated)
//
// Run on:  https://www.linkedin.com/company/{slug}/people/
// Goal:    Map the employee browse-by-X facets (function, location, school, skill,
//          studied, what they do) and the rendered employee cards. Auth wall on
//          this route is the most aggressive on public LinkedIn so we only
//          capture once we're logged in.
//
// Safety:  READ-ONLY. Do NOT click Connect / Message / Follow on any tile.

(function capture() {
  if (!window.__cloris) throw new Error('Load _lib.js first');
  const __ = window.__cloris;

  const payload = {
    url: location.href,
    breadcrumb: __.walk(document, 'nav[aria-label*="readcrumb"]'),
    facets: __.walk(document, 'main section:has(h2):has(button)'),
    facet_headings: __.walk(document, 'main section h2, main section h3'),
    // Search-within-company input.
    search_input: __.walk(document, 'main input[placeholder*="earch employees"], main input[aria-label*="earch"]'),
    employee_tile_candidates: __.walk(document, 'main ul > li:has(a[href*="/in/"])'),
    show_more: __.walk(document, 'main button:has-text("Show more")'),
    employee_count_text: (() => {
      const el = document.querySelector('main h2, main h3');
      return el?.innerText ?? null;
    })(),
  };

  return __.dump('05_company_people_tab', payload);
})();
