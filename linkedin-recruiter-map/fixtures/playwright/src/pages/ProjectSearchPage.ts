// ProjectSearchPage — the /discover/recruiterSearch surface (idle + populated).
// Pass-3-verified selectors only; defer/unknown rows route through SafetyGuard.

import type { Locator, Page } from '@playwright/test';
import { ROUTES, extractProjectIdFromUrl, PROJECT_SEARCH } from '../selectors/manifest.js';
import type { SafetyGuard } from '../safety/SafetyGuard.js';
import type { ResultTotal } from '../types.js';
import { ResultCard } from './components/ResultCard.js';
import { ProfileDrawer } from './ProfileDrawer.js';
import { OverflowMenu } from './OverflowMenu.js';
import { SpotlightsCarousel } from './SpotlightsCarousel.js';

export class ProjectSearchPage {
  constructor(private readonly page: Page, private readonly guard: SafetyGuard) {}

  // -------------------------- route + readiness --------------------------

  /** Resolves once the URL matches the search route AND either the empty-state probe or the results form is visible. */
  async waitForReady(): Promise<void> {
    await this.page.waitForURL(ROUTES.project_search, { timeout: 15_000 });
    await this.page.waitForFunction(() => {
      const hasEmpty = !!document.body.querySelector('main')?.textContent?.includes('Start a search');
      const hasForm = !!document.querySelector('form');
      return hasEmpty || hasForm;
    }, undefined, { timeout: 15_000 });
  }

  projectId(): string | null {
    return extractProjectIdFromUrl(this.page.url());
  }

  isEmptyState(): Promise<boolean> {
    return this.page.locator(PROJECT_SEARCH.empty_search_probe.selector!).first().isVisible();
  }

  // -------------------------- read-only reads --------------------------

  /** Parses the "17M+ results" / "4,231 results" / "45 results" header. */
  async readResultTotal(): Promise<ResultTotal | null> {
    const el = this.page.locator(PROJECT_SEARCH.result_count_text.selector!).first();
    if (!(await el.count())) return null;
    const t = (await el.innerText()).trim();
    const m = t.match(/^([\d.,]+)([KM])?(\+)?\s+results?$/i);
    if (!m) return null;
    const num = parseFloat((m[1] ?? '0').replace(/,/g, ''));
    const mult = m[2] === 'M' ? 1_000_000 : m[2] === 'K' ? 1_000 : 1;
    return {
      total: Math.round(num * mult),
      approximate: !!m[3],
      unit: (m[2] as 'K' | 'M' | undefined) ?? null,
    };
  }

  /**
   * Enumerates primary result cards, excluding the "All recommended matches" sub-region.
   * Order matches DOM order = visual rank. classification: stable_now (Pass 3).
   */
  async cards(): Promise<ResultCard[]> {
    const form = this.page.locator(PROJECT_SEARCH.results_form.selector!).first();
    const recommendedRegion = form.locator(`role=region[name="${PROJECT_SEARCH.embedded_recommended_matches.name}"]`);
    const allArticles = form.locator('role=article');
    const total = await allArticles.count();
    const out: ResultCard[] = [];
    for (let i = 0; i < total; i++) {
      const article = allArticles.nth(i);
      const inRecommended = await isDescendantOf(article, recommendedRegion);
      if (inRecommended) continue;
      out.push(new ResultCard(this.page, article, this.guard));
    }
    return out;
  }

  filterPane(): Locator {
    return this.page.locator(`role=complementary[name="${PROJECT_SEARCH.filter_pane.name}"]`).first();
  }

  // -------------------------- navigation (read-only) --------------------------

  /**
   * Clicks the header pagination "Go to next page N" link.
   * classification: stable_now — read-only (just paginates, no candidate state change).
   */
  async paginateNextHeader(): Promise<void> {
    this.guard.allowReadOnly('paginate_next');
    const nav = this.page.locator(`role=navigation[name="${PROJECT_SEARCH.pagination_header.name}"]`).first();
    await nav.getByRole('link', { name: /^Go to next page \d+$/ }).click();
  }

  /**
   * Clicks a numbered page in the footer pagination.
   * classification: stable_now — read-only.
   */
  async paginateToPage(n: number): Promise<void> {
    this.guard.allowReadOnly('paginate_page');
    const nav = this.page.locator(`role=navigation[name="${PROJECT_SEARCH.pagination_footer.name}"]`).first();
    await nav.getByRole('link', { name: new RegExp(`^Page ${n}$`) }).click();
  }

  /** Opens the profile drawer for a candidate by card index (0-based, excluding recommended). */
  async openCandidateDrawer(index: number): Promise<ProfileDrawer> {
    const all = await this.cards();
    const card = all[index];
    if (!card) throw new Error(`no card at index ${index} (have ${all.length})`);
    return card.openDrawer();
  }

  // -------------------------- safe overflow read --------------------------

  /**
   * Opens the "More actions" menu next to "Hide filters", reads each menuitem's name,
   * then presses Escape. classification: stable_now (read-only probe).
   * Pass-4 hook: this is the safest way to verify menu contents over time.
   */
  async openOverflowMenuReadOnly(): Promise<OverflowMenu> {
    this.guard.allowReadOnly('open_overflow_menu_read_only');
    const trigger = this.filterPane()
      .getByRole('button', { name: PROJECT_SEARCH.overflow_trigger.name! })
      .first();
    await trigger.click();
    const menu = this.page.locator('role=menu').first();
    await menu.waitFor({ state: 'visible', timeout: 3_000 });
    return new OverflowMenu(this.page, menu, this.guard);
  }

  spotlights(): SpotlightsCarousel {
    return new SpotlightsCarousel(this.page, this.guard);
  }
}

async function isDescendantOf(child: Locator, ancestor: Locator): Promise<boolean> {
  if ((await ancestor.count()) === 0) return false;
  const handle = await child.elementHandle();
  if (!handle) return false;
  try {
    return await ancestor.evaluate(
      (anc, childEl) => !!childEl && anc.contains(childEl as Node),
      handle,
    );
  } finally {
    await handle.dispose();
  }
}
