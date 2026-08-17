// 04 — capture an open profile drawer (or full-page profile).
// Prereq:
//   1) paste _lib.js
//   2) click a candidate name in the results to open the drawer
//   3) do NOT click Save, Message, or any three-dot item

(() => {
  const c = window.__cloris;

  const dialog = [...document.querySelectorAll('[role="dialog"]')].filter(c.isVisible)[0];
  const isDrawer = !!dialog;
  const root = dialog || document.querySelector('[role="main"]') || document.body;

  const regions = [...root.querySelectorAll('[role="region"], section, [data-test-section]')]
    .filter(c.isVisible)
    .map(reg => {
      const heading = reg.querySelector('[role="heading"], h1, h2, h3');
      return {
        heading: heading?.innerText?.trim() || null,
        path: c.shortPath(reg),
        attrs: Object.fromEntries(c.STABLE_ATTRS.map(a => [a, reg.getAttribute(a)]).filter(([,v]) => v != null)),
        textSample: reg.innerText.slice(0, 300),
      };
    });

  const memberId = (location.pathname.match(/\/talent\/profile\/([A-Za-z0-9_-]+)/) || [])[1] || null;

  const payload = {
    surface: isDrawer ? 'drawer' : 'full_page',
    dialog_aria_label: dialog?.getAttribute('aria-label') || null,
    member_id: memberId,
    header_h1: c.walk(root, '[role="heading"][aria-level="1"], h1'),
    header_buttons: c.walk(root, 'header button, [role="dialog"] header button, [data-test-profile-header] button').slice(0, 30),
    open_to_work_indicator: c.walk(root, '[aria-label*="open to work" i]'),
    save_btn_candidates: c.walk(root, 'button, [role="button"]').filter(b => /save to project|save/i.test(b.name)),
    message_btn_candidates: c.walk(root, 'button, [role="button"]').filter(b => /message|inmail/i.test(b.name)),
    more_menu_candidates: c.walk(root, 'button, [role="button"]').filter(b => /^more$/i.test(b.name)),
    redacted_marker: /^LinkedIn Member$/m.test(root.innerText),
    profile_unavailable: /no longer available|cannot find/i.test(root.innerText),
    regions,
  };
  c.dump('drawer_profile', payload);
})();
