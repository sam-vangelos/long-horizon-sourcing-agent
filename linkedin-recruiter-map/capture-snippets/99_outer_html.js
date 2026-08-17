// 99 — dump full document.documentElement.outerHTML to clipboard.
// Use after each capture to also save the raw HTML for fixture replay.
// File is large; clipboard should handle several MB.

(() => {
  const html = document.documentElement.outerHTML;
  console.log(`[cloris] outerHTML ${html.length} bytes`);
  navigator.clipboard?.writeText(html).then(
    () => console.log('[cloris] copied outerHTML to clipboard'),
    () => console.warn('[cloris] clipboard failed; use Save As')
  );
})();
