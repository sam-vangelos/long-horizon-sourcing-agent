// AdvancedSearchPage — sibling route /discover/recruiterSearch/advanced.
// Pass 2 verified the URL pattern and section headings; per-filter pill mechanics still
// need Pass-4 dump to verify editor types per filter. Treat this page as a navigation
// surface only until then.

import type { Locator, Page } from '@playwright/test';
import { ROUTES, extractProjectIdFromUrl } from '../selectors/manifest.js';
import type { SafetyGuard } from '../safety/SafetyGuard.js';
import { FilterPill } from './components/FilterPill.js';

export const ADVANCED_SECTIONS = [
  'Candidate details',
  'Education & experience',
  'Company',
  'Recruiting & candidate activity',
] as const;

export type AdvancedSection = typeof ADVANCED_SECTIONS[number];

export class AdvancedSearchPage {
  constructor(private readonly page: Page, private readonly guard: SafetyGuard) {}

  async waitForReady(): Promise<void> {
    await this.page.waitForURL(ROUTES.project_search_advanced, { timeout: 15_000 });
    await this.page.getByRole('heading', { name: 'Candidate details' }).first().waitFor({ timeout: 10_000 });
  }

  projectId(): string | null {
    return extractProjectIdFromUrl(this.page.url());
  }

  /** Returns the section container Locator anchored on its heading. */
  section(name: AdvancedSection): Locator {
    return this.page.getByRole('heading', { name }).first().locator('xpath=ancestor::*[self::section or self::div][1]');
  }

  /**
   * Returns a FilterPill scoped to the given Add-button label inside a section.
   * The Add-button label IS the canonical filter id on this surface — same string
   * the manifest's `filters.<id>.add_button_name` field records.
   */
  filterPill(section: AdvancedSection, addButtonName: string): FilterPill {
    return new FilterPill(this.page, this.section(section), addButtonName, this.guard);
  }
}
