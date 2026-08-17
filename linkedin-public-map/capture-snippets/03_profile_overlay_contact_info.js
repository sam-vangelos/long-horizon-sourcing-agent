// 03 — Contact-info overlay
//
// Run on:  https://www.linkedin.com/in/{vanity}/overlay/contact-info/
// Goal:    Capture the contact-info dialog as it appears to a *first-degree*
//          authenticated viewer. The unauthenticated route is a hard auth wall,
//          and second/third-degree viewers see a redacted dialog.
//          Note which fields appear (email, phone, websites, twitter, IM).
//
// Safety:  READ-ONLY. Do NOT click any email/phone link inside the dialog.

(function capture() {
  if (!window.__cloris) throw new Error('Load _lib.js first');
  const __ = window.__cloris;

  const dialog = document.querySelector('[role="dialog"]');
  if (!dialog) {
    return __.dump('03_profile_overlay_contact_info', { dialog_found: false, url: location.href });
  }

  const payload = {
    url: location.href,
    dialog_root: __.serializeNode(dialog),
    headings: [...dialog.querySelectorAll('h1,h2,h3,h4')].map(__.serializeNode),
    sections: [...dialog.querySelectorAll('section, [data-section]')].map(__.serializeNode),
    anchors: [...dialog.querySelectorAll('a[href]')].map(__.serializeNode),
    spans: [...dialog.querySelectorAll('span')]
      .filter(__.isVisible)
      .slice(0, 50)
      .map(__.serializeNode),
    close_button: __.walk(dialog, 'button[aria-label*="lose" i], button[aria-label*="ismiss" i]'),
  };

  return __.dump('03_profile_overlay_contact_info', payload);
})();
