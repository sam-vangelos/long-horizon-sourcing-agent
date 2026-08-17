// 05 — capture the project picker dialog WITHOUT saving.
// Prereq:
//   1) paste _lib.js
//   2) open a candidate drawer in your scratch project
//   3) click "Save to project" — the dialog opens
//   4) RUN THIS SNIPPET IMMEDIATELY
//   5) press Escape to close. DO NOT click the Save button in the dialog.

(() => {
  const c = window.__cloris;

  const dialog = [...document.querySelectorAll('[role="dialog"]')]
    .filter(c.isVisible)
    .find(d => /save to project|add to project/i.test(d.innerText));

  if (!dialog) {
    console.warn('[cloris] project picker dialog not found — did you open it?');
    return;
  }

  const checkboxes = c.walk(dialog, '[role="checkbox"], input[type="checkbox"]');
  const buttons = c.walk(dialog, 'button, [role="button"]');
  const projectRows = [...dialog.querySelectorAll('label, [role="row"], li')]
    .filter(c.isVisible)
    .map(row => ({
      text: row.innerText.slice(0, 200),
      hasCheckbox: !!row.querySelector('input[type="checkbox"], [role="checkbox"]'),
      checked: !!row.querySelector('[aria-checked="true"]'),
      path: c.shortPath(row),
    }));

  const payload = {
    dialog_aria_label: dialog.getAttribute('aria-label'),
    dialog_path: c.shortPath(dialog),
    header_text: dialog.querySelector('[role="heading"], h1, h2, h3')?.innerText?.trim(),
    checkboxes,
    buttons: buttons.map(b => ({ name: b.name, attrs: b.attrs })),
    project_rows: projectRows,
    pre_checked_count: projectRows.filter(r => r.checked).length,
  };
  c.dump('project_picker', payload);
  console.log('[cloris] >>> NOW PRESS ESCAPE TO CLOSE THE DIALOG. DO NOT CLICK SAVE. <<<');
})();
