// Pagination — thin wrapper around the two pagination navs on the results form.
// Header pagination: nav[aria-label="Profile list header pagination"] — next-only
// Footer pagination: nav[aria-label="Profile list pagination"] — page-link list + Next
//
// Both are read-only navigation. classification: stable_now.

import type { Locator, Page } from '@playwright/test';
import { PROJECT_SEARCH } from '../../selectors/manifest.js';
import type { SafetyGuard } from '../../safety/SafetyGuard.js';

export class HeaderPagination {
  constructor(private readonly page: Page, private readonly guard: SafetyGuard) {}

  nav(): Locator {
    return this.page.locator(`role=navigation[name="${PROJECT_SEARCH.pagination_header.name}"]`).first();
  }

  async currentRangeText(): Promise<string | null> {
    const nav = this.nav();
    if (!(await nav.count())) return null;
    // The range is rendered as a text node inside the nav; we match the first "N - M" pattern.
    const txt = await nav.innerText();
    return txt.match(/(\d+\s*[-–]\s*\d+)/)?.[1] ?? null;
  }

  async next(): Promise<void> {
    this.guard.allowReadOnly('paginate_next');
    await this.nav().getByRole('link', { name: /^Go to next page \d+$/ }).first().click();
  }
}

export class FooterPagination {
  constructor(private readonly page: Page, private readonly guard: SafetyGuard) {}

  nav(): Locator {
    return this.page.locator(`role=navigation[name="${PROJECT_SEARCH.pagination_footer.name}"]`).first();
  }

  async availablePages(): Promise<number[]> {
    const links = this.nav().getByRole('link', { name: /^Page \d+$/ });
    const count = await links.count();
    const out: number[] = [];
    for (let i = 0; i < count; i++) {
      const name = (await links.nth(i).getAttribute('aria-label')) ?? (await links.nth(i).innerText());
      const m = name.match(/^Page (\d+)$/);
      if (m) out.push(parseInt(m[1]!, 10));
    }
    return out;
  }

  async goTo(n: number): Promise<void> {
    this.guard.allowReadOnly('paginate_page');
    await this.nav().getByRole('link', { name: new RegExp(`^Page ${n}$`) }).first().click();
  }

  async next(): Promise<void> {
    this.guard.allowReadOnly('paginate_next');
    await this.nav().getByRole('link', { name: /^Next$/ }).first().click();
  }
}
