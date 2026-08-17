// AuthWallPage — read-only wrapper around the TWO auth-wall templates Pass 2a
// verified live: /authwall (Join LinkedIn) and /uas/login (Sign in).
//
// The worker never types into either; AuthWallPage exists only so the worker
// can confirm "we are on an auth wall" via DOM evidence (in addition to URL),
// and so test fixtures can assert the right page rendered.

import type { Locator, Page } from '@playwright/test';
import {
  AUTHWALL_JOIN,
  AUTHWALL_SIGNIN,
  ROUTES,
  extractAuthWallTrk,
  extractSessionRedirect,
} from '../selectors/manifest.js';
import type { SafetyGuard } from '../safety/SafetyGuard.js';

export type AuthWallTemplate = 'join' | 'signin' | 'none';

export class AuthWallPage {
  constructor(private readonly page: Page, private readonly guard: SafetyGuard) {}

  /** Identify which auth-wall template the page currently shows, from URL + DOM. */
  async detect(): Promise<{
    template: AuthWallTemplate;
    trk: string | null;
    sessionRedirect: string | null;
  }> {
    this.guard.allowReadOnly('read_url_state');
    const url = this.page.url();
    const sessionRedirect = extractSessionRedirect(url);
    if (ROUTES.authwall_join.test(url)) {
      // Confirm with DOM heading.
      const ok = await this.page
        .getByRole('heading', { name: AUTHWALL_JOIN.heading.nameRegex! })
        .count();
      return {
        template: ok > 0 ? 'join' : 'none',
        trk: extractAuthWallTrk(url),
        sessionRedirect,
      };
    }
    if (ROUTES.authwall_signin.test(url)) {
      const ok = await this.page
        .getByRole('heading', { name: AUTHWALL_SIGNIN.heading.nameRegex! })
        .count();
      return { template: ok > 0 ? 'signin' : 'none', trk: null, sessionRedirect };
    }
    return { template: 'none', trk: null, sessionRedirect: null };
  }

  /** Dismiss the bottom-right "LinkedIn is better on the app" toast if present. */
  async dismissAppUpsellIfPresent(): Promise<void> {
    this.guard.allowReadOnly('dismiss_app_toast');
    const btn = this.page.getByRole('button', { name: AUTHWALL_JOIN.app_upsell_dismiss.nameRegex! }).first();
    if (await btn.isVisible().catch(() => false)) {
      await btn.click();
    }
  }

  // Locators (read-only) for tests to assert the templates rendered correctly.
  joinHeading(): Locator {
    return this.page.getByRole('heading', { name: AUTHWALL_JOIN.heading.nameRegex! }).first();
  }

  signInHeading(): Locator {
    return this.page.getByRole('heading', { name: AUTHWALL_SIGNIN.heading.nameRegex! }).first();
  }

  joinGoogleIframe(): Locator {
    return this.page.locator(AUTHWALL_JOIN.google_iframe.selector!).first();
  }

  signInGoogleIframe(): Locator {
    return this.page.locator(AUTHWALL_SIGNIN.google_iframe.selector!).first();
  }

  signInMicrosoftIframe(): Locator {
    return this.page.locator(AUTHWALL_SIGNIN.acme-software_iframe.selector!).first();
  }
}
