// 04 — People search results
//
// Run on:  https://www.linkedin.com/search/results/people/?keywords=...&origin=FACETED_SEARCH
//          (after applying ≥1 filter such as geoUrn / currentCompany / network)
// Goal:    Capture the result-card layout so we can promote
//          search_results_people.* selectors out of `unknown`.
//
// Safety:  READ-ONLY. Do NOT click Connect / Message / Follow on any card.

(function capture() {
  if (!window.__cloris) throw new Error('Load _lib.js first');
  const __ = window.__cloris;

  // The current authenticated search uses a list of <li> items inside the main
  // content column. Each item has a link to /in/{vanity} as its primary anchor.
  const cards = [...document.querySelectorAll('main ul[role="list"] > li, main ul > li')]
    .filter((li) => li.querySelector('a[href*="/in/"]'))
    .filter(__.isVisible);

  const payload = {
    url: location.href,
    facet_chips: __.walk(document, 'main button[aria-label*="ilter"]'),
    pagination: __.walk(document, 'main nav[aria-label*="agination"] button, main button[aria-label*="ext page"]'),
    cluster_heading: __.walk(document, 'main h2, main h3'),
    result_count_text: (() => {
      // LinkedIn surfaces "About X results" in a heading or in an aria-live region.
      const live = document.querySelector('[aria-live]');
      return live ? live.innerText : null;
    })(),
    card_count: cards.length,
    cards: cards.slice(0, 8).map((li) => ({
      outer: __.serializeNode(li),
      vanity_anchor: __.walk(li, 'a[href*="/in/"]'),
      headline_lines: [...li.querySelectorAll('div, span')]
        .filter(__.isVisible)
        .slice(0, 24)
        .map(__.serializeNode),
      action_buttons: __.walk(li, 'button'),
      degree_pill: __.walk(li, 'span:has-text("· 1st"), span:has-text("· 2nd"), span:has-text("· 3rd")'),
    })),
  };

  return __.dump('04_search_results_people', payload);
})();
