// Shared DOM-walk + a11y-serialize helpers for public-LinkedIn capture.
// Paste this in the DevTools console BEFORE running any 0X_*.js script.
//
// IMPORTANT: public LinkedIn pages have ZERO data-test-* attributes. The capture
// scripts therefore lean on role / aria-* / href substrings (`trk=...`) and
// visible text. This _lib is a shrunk, public-LI-friendly version of the
// Recruiter _lib.js.

window.__cloris = window.__cloris || {};

// Stable attributes worth keeping on every captured node.
// Note: data-test-* are intentionally included even though public LI doesn't
// emit them — if LinkedIn ever re-introduces them, we want to spot it.
window.__cloris.STABLE_ATTRS = [
  'role', 'aria-label', 'aria-labelledby', 'aria-describedby',
  'aria-checked', 'aria-selected', 'aria-expanded', 'aria-disabled',
  'aria-controls', 'aria-haspopup', 'aria-live', 'aria-current',
  'aria-pressed', 'aria-required', 'aria-hidden',
  'data-test-id', 'data-test', 'data-tracking-control-name',
  'name', 'type', 'placeholder', 'href', 'for', 'title', 'alt', 'rel',
];

window.__cloris.isVisible = function (el) {
  if (!(el instanceof Element)) return false;
  const r = el.getBoundingClientRect();
  if (r.width === 0 && r.height === 0) return false;
  const cs = getComputedStyle(el);
  return cs.visibility !== 'hidden' && cs.display !== 'none' && cs.opacity !== '0';
};

window.__cloris.accessibleName = function (el) {
  const labelledby = el.getAttribute('aria-labelledby');
  const fromLabelledBy = labelledby
    ? labelledby.split(/\s+/).map((id) => document.getElementById(id)?.innerText).filter(Boolean).join(' ')
    : '';
  return (
    el.getAttribute('aria-label') ||
    fromLabelledBy ||
    el.getAttribute('alt') ||
    el.getAttribute('title') ||
    el.getAttribute('placeholder') ||
    (el.tagName === 'INPUT' && el.labels && [...el.labels].map((l) => l.innerText).join(' ')) ||
    (el.innerText || '').slice(0, 200) ||
    ''
  ).trim().replace(/\s+/g, ' ');
};

// `trk=` substring is the most durable LinkedIn-public signal because the
// company uses it for analytics and won't break it lightly.
window.__cloris.trkOf = function (el) {
  const href = el.getAttribute('href') || '';
  const m = href.match(/[?&]trk=([^&#]+)/);
  return m ? decodeURIComponent(m[1]) : null;
};

window.__cloris.serializeNode = function (el) {
  const attrs = {};
  for (const a of window.__cloris.STABLE_ATTRS) {
    const v = el.getAttribute(a);
    if (v != null) attrs[a] = v;
  }
  const trk = window.__cloris.trkOf(el);
  if (trk) attrs['__trk'] = trk;
  return {
    tag: el.tagName.toLowerCase(),
    role: el.getAttribute('role') || el.tagName.toLowerCase(),
    name: window.__cloris.accessibleName(el),
    attrs,
    visible: window.__cloris.isVisible(el),
    rect: (() => { const r = el.getBoundingClientRect(); return { x: r.x | 0, y: r.y | 0, w: r.width | 0, h: r.height | 0 }; })(),
    path: window.__cloris.shortPath(el),
  };
};

window.__cloris.shortPath = function (el) {
  const segs = [];
  let cur = el;
  while (cur && cur !== document.body && segs.length < 8) {
    const role = cur.getAttribute('role');
    const al = cur.getAttribute('aria-label');
    const trk = window.__cloris.trkOf(cur);
    if (role || al || trk) {
      segs.unshift([
        cur.tagName.toLowerCase(),
        role ? `[role="${role}"]` : '',
        al ? `[aria-label="${al.slice(0, 40)}"]` : '',
        trk ? `[trk="${trk}"]` : '',
      ].join(''));
    } else {
      segs.unshift(cur.tagName.toLowerCase());
    }
    cur = cur.parentElement;
  }
  return segs.join(' > ');
};

window.__cloris.walk = function (root, selector) {
  return [...(root || document).querySelectorAll(selector)]
    .filter(window.__cloris.isVisible)
    .map(window.__cloris.serializeNode);
};

// Read the JSON-LD <script type="application/ld+json"> blocks on a public profile.
window.__cloris.jsonLd = function () {
  return [...document.querySelectorAll('script[type="application/ld+json"]')]
    .map((s) => {
      try { return JSON.parse(s.textContent || 'null'); } catch { return null; }
    })
    .filter(Boolean);
};

// og:* / canonical meta tags.
window.__cloris.metaTags = function () {
  const out = {};
  for (const m of document.querySelectorAll('meta[property^="og:"], meta[name="description"], link[rel="canonical"]')) {
    const k = m.getAttribute('property') || m.getAttribute('name') || m.getAttribute('rel');
    const v = m.getAttribute('content') || m.getAttribute('href') || '';
    out[k] = v;
  }
  return out;
};

window.__cloris.dump = function (label, payload) {
  const out = JSON.stringify({
    capturedAt: new Date().toISOString(),
    url: location.href,
    label,
    payload,
  }, null, 2);
  console.log(`[cloris-public] captured ${label} — ${out.length} bytes`);
  if (navigator.clipboard?.writeText) {
    navigator.clipboard.writeText(out).then(
      () => console.log('[cloris-public] copied to clipboard'),
      () => console.warn('[cloris-public] clipboard failed; copy from console'),
    );
  }
  return out;
};

console.log('[cloris-public] _lib loaded. Run a 0X_*.js script next.');
