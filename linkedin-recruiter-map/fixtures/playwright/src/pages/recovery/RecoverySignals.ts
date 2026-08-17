// RecoverySignals — implements the manifest's recovery_signals table.
// Each detector returns a RecoveryError when triggered; the worker decides recovery.

import type { Page } from '@playwright/test';
import { RecoveryError } from '../../types.js';
import { ROUTES, extractProjectIdFromUrl } from '../../selectors/manifest.js';

export class RecoverySignals {
  constructor(private readonly page: Page) {}

  /**
   * Runs all detectors in fastest-to-slowest order. Throws RecoveryError on first match.
   * Workers should call this before any mutating action and after navigation.
   */
  async assertHealthy(args?: { expectedProjectId?: string }): Promise<void> {
    const url = this.page.url();

    if (ROUTES.checkpoint_challenge.test(url)) {
      throw new RecoveryError('blocked', 'backoff_24h');
    }
    if (ROUTES.login_wall.test(url)) {
      throw new RecoveryError('logged_out', 'halt_emit_reauth');
    }

    if (args?.expectedProjectId) {
      const projectIdFromUrl = extractProjectIdFromUrl(url);
      if (!projectIdFromUrl) {
        throw new RecoveryError('no_project_context', 'halt_refuse_mutations');
      }
      if (projectIdFromUrl !== args.expectedProjectId) {
        throw new RecoveryError('lost_project_context', 'halt_mutating_step');
      }
    }

    // wrong_seat: Advanced search button absent OR filter count too low
    const hasAdvancedLink = await this.page.getByRole('link', { name: /^Advanced search$/i }).count();
    if (hasAdvancedLink === 0) {
      // Not always wrong-seat — only flag if we're also on a search route. Otherwise leave alone.
      if (ROUTES.project_search.test(url)) {
        throw new RecoveryError('wrong_seat', 'halt_refuse_mutations');
      }
    }

    // browser_crash: document.body sanity
    const bodyOk = await this.page.evaluate(() => {
      return !!document.body && document.body.children.length >= 2 && !!document.querySelector('[role="main"]');
    });
    if (!bodyOk) {
      throw new RecoveryError('browser_crash', 'restart_session');
    }
  }

  /** Polls result_count_text — if unchanged within 5s of a filter apply, signal stale_search. */
  async assertResultCountChanged(prevText: string | null, timeoutMs = 5_000): Promise<string> {
    const sel = 'form :text-matches("^[\\\\d.]+[KM]?\\\\+?\\\\s+results?$", "i")';
    const start = Date.now();
    while (Date.now() - start < timeoutMs) {
      const el = this.page.locator(sel).first();
      if ((await el.count()) > 0) {
        const t = (await el.innerText()).trim();
        if (t !== prevText) return t;
      }
      await this.page.waitForTimeout(250);
    }
    throw new RecoveryError('stale_search', 're_apply_once');
  }
}
