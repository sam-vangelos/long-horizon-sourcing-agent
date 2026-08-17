// 01 — capture the search-idle surface (project Talent Pool, empty query).
// Prereq: paste _lib.js first.
// Run from a project's Recruiter Search route. Do not type anything.

(() => {
  const c = window.__cloris;
  const payload = {
    route: location.pathname,
    title: document.title,
    keyword_input: c.walk(document, 'input[type="search"], [role="combobox"], [role="textbox"], [contenteditable="true"]'),
    buttons_visible: c.walk(document, 'button, [role="button"]').slice(0, 100),
    navigation: c.walk(document, '[role="navigation"]'),
    headings: c.walk(document, 'h1, h2, h3, [role="heading"]'),
    aria_live: c.walk(document, '[aria-live], [role="status"]'),
    breadcrumb_candidates: c.walk(document, '[aria-label*="breadcrumb" i], header [aria-label*="project" i]'),
    project_id_from_url: (location.pathname.match(/\/talent\/hire\/(\d+)\//) || [])[1] || null,
  };
  c.dump('search_idle', payload);
})();
