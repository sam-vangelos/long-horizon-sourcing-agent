// 02 — capture search results after running a benign keyword (e.g. "java").
// Prereq: paste _lib.js first.
// Run AFTER results load. Do not click any card-hover buttons.

(() => {
  const c = window.__cloris;

  // Find the result-list container — prefer role=list with profile links inside.
  const resultLists = [...document.querySelectorAll('[role="list"], ul, ol')]
    .filter(el => el.querySelector('a[href*="/talent/profile/"]'));
  const resultRoot = resultLists.sort((a, b) => b.querySelectorAll('a[href*="/talent/profile/"]').length
                                              - a.querySelectorAll('a[href*="/talent/profile/"]').length)[0] || document.body;

  const cardEls = [...resultRoot.querySelectorAll('[role="listitem"], li')]
    .filter(el => el.querySelector('a[href*="/talent/profile/"]'));

  const cards = cardEls.slice(0, 25).map(card => {
    const profileLink = card.querySelector('a[href*="/talent/profile/"]');
    const href = profileLink?.getAttribute('href') || '';
    const memberId = (href.match(/\/talent\/profile\/([A-Za-z0-9_-]+)/) || [])[1] || null;
    return {
      memberId,
      name: profileLink?.innerText?.trim() || null,
      profileHref: href,
      cardOuterHTMLSample: card.outerHTML.slice(0, 1200),
      savedBadge: !!card.querySelector('[aria-label*="saved" i], [aria-label*="in your pipeline" i]'),
      savedText: !!card.innerText.match(/^saved$/im),
      viewedText: !!card.innerText.match(/viewed/i),
      redacted: profileLink?.innerText?.trim() === 'LinkedIn Member',
      buttons: c.walk(card, 'button, [role="button"]').map(b => ({ name: b.name, attrs: b.attrs })),
    };
  });

  const payload = {
    route: location.pathname,
    result_count_status: c.walk(document, '[role="status"], [aria-live="polite"]')
      .filter(n => /results?|of\s+\d/i.test(n.name)),
    pagination: c.walk(document, '[role="navigation"][aria-label*="pagination" i] button, nav[aria-label*="pagination" i] button, button[aria-label*="next" i], button[aria-label*="previous" i]'),
    active_filter_chips: c.walk(document, 'button[aria-label*="remove" i][aria-label*="filter" i], [data-test-active-filter] button'),
    sidebar_summary: c.walk(document, '[aria-label*="filter" i] [role="region"], aside [role="region"]').slice(0, 30),
    show_filters_btn: c.walk(document, 'button').filter(b => /show filters|advanced search/i.test(b.name)),
    cards,
    result_root_path: c.shortPath(resultRoot),
  };
  c.dump('search_results', payload);
})();
