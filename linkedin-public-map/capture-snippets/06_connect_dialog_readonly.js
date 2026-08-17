// 06 — Connect dialog (READ-ONLY map)
//
// Run on:  any authenticated /in/{vanity} or search/results/people page where
//          a "Connect" button is visible.
// Goal:    Map the Connect modal so we know what selectors exist for the
//          "Send without a note" path vs the "Add a note" path. This is the
//          most safety-critical mutating surface on public LinkedIn.
//
// PROCEDURE:
//   1. With the profile loaded, RIGHT-CLICK the Connect button and pick
//      "Inspect" — do NOT left-click it (left-click sends the invite if the
//      person has "send without a note" enabled).
//   2. From DevTools console, trigger the dialog via keyboard:
//        document.querySelector('main button:has-text("Connect")')
//          ?.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter' }));
//      (If the button is wrapped in a menu, open the menu first and use
//      Tab to focus "Connect" then run the same dispatchEvent.)
//   3. WHEN THE DIALOG IS VISIBLE, run this capture.
//   4. PRESS ESC to dismiss. DO NOT CLICK "Send" or "Send now".
//
// Safety:  READ-ONLY. The keyboard path is used because hovering/clicking
//          the Connect button can trigger an immediate send on some account
//          types.

(function capture() {
  if (!window.__cloris) throw new Error('Load _lib.js first');
  const __ = window.__cloris;

  const dialog = document.querySelector('[role="dialog"]');
  if (!dialog) {
    return __.dump('06_connect_dialog_readonly', {
      dialog_found: false,
      note: 'Open the Connect dialog (see procedure in script header), then re-run.',
      url: location.href,
    });
  }

  // Hard sanity check — refuse to capture if a Send button is already focused.
  const active = document.activeElement;
  const activeName = active ? __.accessibleName(active).toLowerCase() : '';
  if (activeName.startsWith('send')) {
    return __.dump('06_connect_dialog_readonly', {
      aborted: true,
      reason: 'A "Send" button is focused; press Escape and re-run to avoid accidental keystrokes.',
      url: location.href,
    });
  }

  const payload = {
    url: location.href,
    dialog_root: __.serializeNode(dialog),
    headings: [...dialog.querySelectorAll('h1,h2,h3')].map(__.serializeNode),
    buttons: [...dialog.querySelectorAll('button')].map(__.serializeNode),
    add_note_button: __.walk(dialog, 'button:has-text("Add a note")'),
    send_now_button: __.walk(dialog, 'button:has-text("Send now"), button:has-text("Send without a note"), button:has-text("Send")'),
    dismiss_button: __.walk(dialog, 'button[aria-label*="ismiss" i], button[aria-label*="lose" i]'),
    textarea_or_note_input: __.walk(dialog, 'textarea, [contenteditable="true"]'),
    char_counter: __.walk(dialog, 'div:has-text("/200"), span:has-text("/200")'),
  };

  console.warn(
    '[cloris-public] Capture complete. PRESS ESC NOW to dismiss the dialog. ' +
    'Do not click the Send button.',
  );

  return __.dump('06_connect_dialog_readonly', payload);
})();
