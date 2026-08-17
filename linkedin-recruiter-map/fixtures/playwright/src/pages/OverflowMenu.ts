// OverflowMenu — the "More actions" menu next to "Hide filters" on the search surface.
// All items classified `defer` in the manifest; all mutating items require explicit intent.

import type { Locator, Page } from '@playwright/test';
import { OVERFLOW_MENU } from '../selectors/manifest.js';
import { extractProjectIdFromUrl } from '../selectors/manifest.js';
import type { SafetyGuard } from '../safety/SafetyGuard.js';

export class OverflowMenu {
  constructor(
    private readonly page: Page,
    public readonly menu: Locator,
    private readonly guard: SafetyGuard,
  ) {}

  /** Read all menuitem accessible names. classification: stable_now (read-only). */
  async readItemNames(): Promise<string[]> {
    const items = this.menu.getByRole('menuitem');
    const count = await items.count();
    const out: string[] = [];
    for (let i = 0; i < count; i++) {
      const it = items.nth(i);
      out.push(((await it.getAttribute('aria-label')) ?? (await it.innerText())).trim());
    }
    return out;
  }

  /** Close without selecting any item. classification: stable_now. */
  async close(): Promise<void> {
    await this.page.keyboard.press('Escape');
  }

  // -------------------------- mutating (gated) --------------------------

  async saveAsNewSearch(): Promise<void> {
    const projectId = extractProjectIdFromUrl(this.page.url());
    this.guard.requireIntent('save_as_new_search', { projectIdFromUrl: projectId, surface: 'project_search' });
    await this.menu.getByRole('menuitem', { name: OVERFLOW_MENU.save_as_new_search.name! }).click();
  }

  async clearSearch(): Promise<void> {
    const projectId = extractProjectIdFromUrl(this.page.url());
    this.guard.requireIntent('clear_search', { projectIdFromUrl: projectId, surface: 'project_search' });
    await this.menu.getByRole('menuitem', { name: OVERFLOW_MENU.clear_search.name! }).click();
  }

  async saveAsNewCustomFilter(): Promise<void> {
    const projectId = extractProjectIdFromUrl(this.page.url());
    this.guard.requireIntent('save_as_new_custom_filter', { projectIdFromUrl: projectId, surface: 'project_search' });
    await this.menu.getByRole('menuitem', { name: OVERFLOW_MENU.save_as_new_custom_filter.name! }).click();
  }

  async deleteCustomFilters(): Promise<void> {
    const projectId = extractProjectIdFromUrl(this.page.url());
    this.guard.requireIntent('delete_custom_filters', { projectIdFromUrl: projectId, surface: 'project_search' });
    await this.menu.getByRole('menuitem', { name: OVERFLOW_MENU.delete_custom_filters.name! }).click();
  }
}
