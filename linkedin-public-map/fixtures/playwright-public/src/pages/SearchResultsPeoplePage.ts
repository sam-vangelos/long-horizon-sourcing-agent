// SearchResultsPeoplePage — placeholder for the authenticated /search/results/people/
// surface. All selectors are classification=unknown until Pass 2b runs on the
// user's authenticated seat (see ../../../docs/pass-2b-followup-prompt.md).
//
// Worker should NOT instantiate this page with acceptUnverified=false (the default);
// every method below throws via SafetyGuard.assertVerifiedSelector() until rows are
// promoted to stable_now in the manifest.

import type { Locator, Page } from '@playwright/test';
import { ROUTES, SEARCH_RESULTS_PEOPLE, GEO_URNS, INDUSTRY_URNS, NETWORK_CODES } from '../selectors/manifest.js';
import type { SafetyGuard } from '../safety/SafetyGuard.js';
import { assertOnSurface } from './recovery/RecoverySignals.js';

export interface PublicSearchParams {
  keywords: string;
  /** Geo names from GEO_URNS map (e.g. ['US', 'Canada']) or raw URN ids. */
  geos?: Array<keyof typeof GEO_URNS | number>;
  currentCompanyUrns?: number[];
  pastCompanyUrns?: number[];
  schoolUrns?: number[];
  industries?: Array<keyof typeof INDUSTRY_URNS | number>;
  network?: Array<keyof typeof NETWORK_CODES>;
  profileLanguage?: string;
  firstName?: string;
  lastName?: string;
  title?: string;
  origin?: 'FACETED_SEARCH' | 'GLOBAL_SEARCH_HEADER' | 'SWITCH_SEARCH_VERTICAL';
}

/** Build the canonical /search/results/people/?... URL from worker-side params. */
export function buildPublicPeopleSearchUrl(p: PublicSearchParams): string {
  const sp = new URLSearchParams();
  sp.set('keywords', p.keywords);
  sp.set('origin', p.origin ?? 'FACETED_SEARCH');

  const geoIds = (p.geos ?? []).map((g) =>
    typeof g === 'number' ? g : GEO_URNS[g],
  ).filter((n): n is number => typeof n === 'number');
  if (geoIds.length) sp.set('geoUrn', JSON.stringify(geoIds.map(String)));

  if (p.currentCompanyUrns?.length) {
    sp.set('currentCompany', JSON.stringify(p.currentCompanyUrns.map(String)));
  }
  if (p.pastCompanyUrns?.length) {
    sp.set('pastCompany', JSON.stringify(p.pastCompanyUrns.map(String)));
  }
  if (p.schoolUrns?.length) {
    sp.set('schoolFilter', JSON.stringify(p.schoolUrns.map(String)));
  }
  const industryIds = (p.industries ?? []).map((i) =>
    typeof i === 'number' ? i : INDUSTRY_URNS[i],
  ).filter((n): n is number => typeof n === 'number');
  if (industryIds.length) sp.set('industry', JSON.stringify(industryIds.map(String)));

  if (p.network?.length) {
    sp.set('network', p.network.map((n) => NETWORK_CODES[n]).join(','));
  }
  if (p.profileLanguage) sp.set('profileLanguage', p.profileLanguage);
  if (p.firstName) sp.set('firstName', p.firstName);
  if (p.lastName) sp.set('lastName', p.lastName);
  if (p.title) sp.set('title', p.title);

  return `https://www.linkedin.com/search/results/people/?${sp.toString()}`;
}

export class SearchResultsPeoplePage {
  constructor(private readonly page: Page, private readonly guard: SafetyGuard) {}

  /**
   * Navigate to a public people-search URL. Will throw AuthWallSignInError
   * immediately for unauthenticated sessions (Pass 2a confirmed redirect to /uas/login).
   */
  async goto(params: PublicSearchParams): Promise<void> {
    this.guard.allowReadOnly('navigate_to_public_profile'); // closest read-only action
    const url = buildPublicPeopleSearchUrl(params);
    await this.page.goto(url, { waitUntil: 'domcontentloaded' });
    await assertOnSurface(this.page);
    await this.waitForReady();
  }

  async waitForReady(): Promise<void> {
    await this.page.waitForURL(ROUTES.search_results_people, { timeout: 15_000 });
  }

  /** Returns the result-card locator. All rows are CLASSIFICATION=UNKNOWN — Pass 2b required. */
  resultCards(): Locator {
    this.guard.assertVerifiedSelector(SEARCH_RESULTS_PEOPLE.result_card.id, SEARCH_RESULTS_PEOPLE.result_card.classification);
    // Best-effort placeholder selector; not yet verified live.
    return this.page.locator('main [data-chameleon-result-urn], main [data-test-search-result]');
  }

  /** Paginate to next results page. authenticated only. */
  async paginateNext(): Promise<void> {
    this.guard.allowReadOnly('paginate_search_next');
    this.guard.assertVerifiedSelector(
      SEARCH_RESULTS_PEOPLE.pagination_next.id,
      SEARCH_RESULTS_PEOPLE.pagination_next.classification,
    );
    await this.page.getByRole('button', { name: SEARCH_RESULTS_PEOPLE.pagination_next.nameRegex! }).click();
  }
}
