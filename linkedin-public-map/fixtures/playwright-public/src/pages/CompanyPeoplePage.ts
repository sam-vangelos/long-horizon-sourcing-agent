// CompanyPeoplePage — placeholder for /company/{slug}/people/.
// Pass 2a confirmed: unauthenticated GET hard-redirects to /authwall?trk=bf.
// All selectors classification=unknown until Pass 2b runs authenticated.

import type { Locator, Page } from '@playwright/test';
import { COMPANY_PEOPLE, ROUTES } from '../selectors/manifest.js';
import type { SafetyGuard } from '../safety/SafetyGuard.js';
import { assertOnSurface } from './recovery/RecoverySignals.js';

export class CompanyPeoplePage {
  constructor(private readonly page: Page, private readonly guard: SafetyGuard) {}

  /** Navigate to a company People tab. Throws AuthWallJoinError on unauthenticated sessions. */
  async goto(slug: string): Promise<void> {
    this.guard.allowReadOnly('navigate_to_company_landing');
    await this.page.goto(`https://www.linkedin.com/company/${slug}/people/`, {
      waitUntil: 'domcontentloaded',
    });
    await assertOnSurface(this.page);
    await this.waitForReady();
  }

  async waitForReady(): Promise<void> {
    await this.page.waitForURL(ROUTES.company_people, { timeout: 15_000 });
  }

  employeeCards(): Locator {
    this.guard.assertVerifiedSelector(COMPANY_PEOPLE.employee_card.id, COMPANY_PEOPLE.employee_card.classification);
    return this.page.locator('main [data-test-employee-card]');
  }

  async showMore(): Promise<void> {
    this.guard.allowReadOnly('paginate_search_next');
    this.guard.assertVerifiedSelector(
      COMPANY_PEOPLE.show_more_pagination.id,
      COMPANY_PEOPLE.show_more_pagination.classification,
    );
    await this.page.getByRole('button', { name: COMPANY_PEOPLE.show_more_pagination.nameRegex! }).click();
  }
}
