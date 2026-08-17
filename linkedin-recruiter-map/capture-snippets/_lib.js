// Shared DOM-walk + a11y-serialize helpers.
// Paste this in DevTools console BEFORE running any 0X_*.js script.

window.__cloris = window.__cloris || {};

window.__cloris.STABLE_ATTRS = [
  'role', 'aria-label', 'aria-labelledby', 'aria-describedby',
  'aria-checked', 'aria-selected', 'aria-expanded', 'aria-disabled',
  'aria-controls', 'aria-haspopup', 'aria-live', 'aria-current',
  'aria-pressed', 'aria-required',
  'data-test-id', 'data-test', 'data-control-name', 'data-tracking-control-name',
  'data-view-name', 'data-test-search-result', 'data-test-active-filter',
  'name', 'type', 'placeholder', 'href', 'for', 'title', 'alt',
];

window.__cloris.isVisible = function(el) {
  if (!(el instanceof Element)) return false;
  const r = el.getBoundingClientRect();
  if (r.width === 0 && r.height === 0) return false;
  const cs = getComputedStyle(el);
  return cs.visibility !== 'hidden' && cs.display !== 'none';
};

window.__cloris.accessibleName = function(el) {
  // Cheap approximation; for ground truth, prefer Chrome's accessibility tree
  // via chrome.debugger or the Accessibility tab. This is good enough for selectors.
  return (
    el.getAttribute('aria-label') ||
    el.getAttribute('aria-labelledby') &&
      [...el.getAttribute('aria-labelledby').split(/\s+/)
        .map(id => document.getElementById(id)?.innerText).filter(Boolean)].join(' ') ||
    el.getAttribute('alt') ||
    el.getAttribute('title') ||
    el.getAttribute('placeholder') ||
    (el.tagName === 'INPUT' && el.labels && [...el.labels].map(l => l.innerText).join(' ')) ||
    el.innerText?.slice(0, 200) ||
    ''
  ).trim().replace(/\s+/g, ' ');
};

window.__cloris.serializeNode = function(el) {
  const attrs = {};
  for (const a of window.__cloris.STABLE_ATTRS) {
    const v = el.getAttribute(a);
    if (v != null) attrs[a] = v;
  }
  return {
    tag: el.tagName.toLowerCase(),
    role: el.getAttribute('role') || el.tagName.toLowerCase(),
    name: window.__cloris.accessibleName(el),
    attrs,
    visible: window.__cloris.isVisible(el),
    rect: (() => { const r = el.getBoundingClientRect(); return { x: r.x|0, y: r.y|0, w: r.width|0, h: r.height|0 }; })(),
    path: window.__cloris.shortPath(el),
  };
};

window.__cloris.shortPath = function(el) {
  // Compact, durable-ish path: ancestors with role/data-test only.
  const segs = [];
  let cur = el;
  while (cur && cur !== document.body && segs.length < 8) {
    const role = cur.getAttribute('role');
    const dt = cur.getAttribute('data-test-id') || cur.getAttribute('data-test');
    const al = cur.getAttribute('aria-label');
    if (role || dt || al) {
      segs.unshift([cur.tagName.toLowerCase(),
                    role ? `[role="${role}"]` : '',
                    dt ? `[data-test-id="${dt}"]` : '',
                    al ? `[aria-label="${al.slice(0,40)}"]` : ''].join(''));
    } else {
      segs.unshift(cur.tagName.toLowerCase());
    }
    cur = cur.parentElement;
  }
  return segs.join(' > ');
};

window.__cloris.walk = function(root, selector) {
  return [...(root || document).querySelectorAll(selector)]
    .filter(window.__cloris.isVisible)
    .map(window.__cloris.serializeNode);
};

window.__cloris.dump = function(label, payload) {
  const out = JSON.stringify({
    capturedAt: new Date().toISOString(),
    url: location.href,
    label,
    payload,
  }, null, 2);
  console.log(`[cloris] captured ${label} — ${out.length} bytes`);
  if (navigator.clipboard?.writeText) {
    navigator.clipboard.writeText(out).then(
      () => console.log('[cloris] copied to clipboard'),
      () => console.warn('[cloris] clipboard failed; copy from console')
    );
  }
  return out;
};

console.log('[cloris] _lib loaded. Run a 0X_*.js script next.');
