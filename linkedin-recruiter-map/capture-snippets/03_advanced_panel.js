// 03 — capture the Advanced Search panel fully expanded.
// Prereq:
//   1) paste _lib.js
//   2) click "Show filters" then "Advanced search" (the ONLY clicks allowed for this capture)
//   3) scroll the panel to materialize all filter sections (LinkedIn lazy-renders)

(() => {
  const c = window.__cloris;

  // Filter panel is usually a dialog or aside with many region/group children.
  const candidates = [
    ...document.querySelectorAll('[role="dialog"], aside, [aria-label*="filter" i], [aria-label*="advanced search" i]')
  ].filter(c.isVisible);
  const panel = candidates.sort((a, b) => b.querySelectorAll('[role="region"], [role="group"], fieldset').length
                                        - a.querySelectorAll('[role="region"], [role="group"], fieldset').length)[0] || document.body;

  const sections = [...panel.querySelectorAll('[role="region"], [role="group"], fieldset, section, details')]
    .filter(c.isVisible)
    .map(sec => {
      const heading = sec.querySelector('[role="heading"], h1, h2, h3, h4, summary, legend');
      const inputs = c.walk(sec, 'input, textarea, select, [role="combobox"], [role="textbox"], [contenteditable="true"]');
      const buttons = c.walk(sec, 'button, [role="button"], [role="checkbox"], [role="switch"], [role="radio"]');
      const dropdowns = c.walk(sec, '[aria-haspopup], [role="listbox"]');
      return {
        heading: heading?.innerText?.trim() || null,
        path: c.shortPath(sec),
        attrs: Object.fromEntries(c.STABLE_ATTRS.map(a => [a, sec.getAttribute(a)]).filter(([,v]) => v != null)),
        inputs,
        buttons: buttons.slice(0, 60),
        dropdowns,
        textSample: sec.innerText.slice(0, 400),
      };
    });

  const payload = {
    panel_path: c.shortPath(panel),
    panel_role: panel.getAttribute('role'),
    panel_aria_label: panel.getAttribute('aria-label'),
    section_count: sections.length,
    sections,
  };
  c.dump('advanced_panel', payload);
})();
