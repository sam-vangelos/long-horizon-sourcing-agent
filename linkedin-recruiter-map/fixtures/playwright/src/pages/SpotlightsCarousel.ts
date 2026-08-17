// SpotlightsCarousel — Spotlights region inside the results form.
// Each spotlight button name parses as "{count} are {label}". Clicking a spotlight
// is treated as `apply_filter` and requires intent. Reading is always allowed.

import type { Locator, Page } from '@playwright/test';
import { extractProjectIdFromUrl, PROJECT_SEARCH } from '../selectors/manifest.js';
import type { SafetyGuard } from '../safety/SafetyGuard.js';

export interface SpotlightSummary {
  /** Raw button accessible name. */
  rawName: string;
  /** Parsed count, e.g. "3.2M+", "650K+", "45". */
  count: string;
  /** Parsed label, e.g. "Open to work", "Active talent". */
  label: string;
}

export class SpotlightsCarousel {
  constructor(private readonly page: Page, private readonly guard: SafetyGuard) {}

  region(): Locator {
    return this.page.locator(`role=region[name="${PROJECT_SEARCH.spotlights_region.name}"]`).first();
  }

  /** Enumerates currently visible spotlight buttons in the carousel viewport. */
  async readVisibleSpotlights(): Promise<SpotlightSummary[]> {
    const region = this.region();
    if (!(await region.count())) return [];
    const buttons = region.getByRole('button');
    const total = await buttons.count();
    const out: SpotlightSummary[] = [];
    for (let i = 0; i < total; i++) {
      const btn = buttons.nth(i);
      const name = ((await btn.getAttribute('aria-label')) ?? (await btn.innerText())).trim();
      // Carousel controls "Previous" / "Next" are siblings — filter them out.
      if (name === 'Previous' || name === 'Next') continue;
      const m = name.match(/^(\S+)\s+are\s+(.+)$/);
      if (!m) continue;
      out.push({ rawName: name, count: m[1]!, label: m[2]! });
    }
    return out;
  }

  async carouselNext(): Promise<void> {
    this.guard.allowReadOnly('paginate_next');
    await this.region().getByRole('button', { name: 'Next' }).click();
  }

  async carouselPrev(): Promise<void> {
    this.guard.allowReadOnly('paginate_next');
    await this.region().getByRole('button', { name: 'Previous' }).click();
  }

  /**
   * Applies a spotlight as a filter. classification: defer (alters the search).
   * Requires intent.action === 'apply_filter'.
   */
  async applySpotlight(label: string): Promise<void> {
    const projectId = extractProjectIdFromUrl(this.page.url());
    this.guard.requireIntent('apply_filter', { projectIdFromUrl: projectId, surface: 'project_search' });
    const btn = this.region().getByRole('button', { name: new RegExp(`\\bare\\s+${escapeRegExp(label)}\\b`, 'i') }).first();
    await btn.click();
  }
}

function escapeRegExp(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}
