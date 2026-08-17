// PublicProfilePage — the /in/{vanity} surface (unauthenticated guest view).
// Pass-2a-verified selectors only; defer/unknown rows route through SafetyGuard.
//
// Worker contract:
//   const guard = new SafetyGuard({ mode: 'read_only', guestViewBudget: 3 });
//   const page = await ctx.newPage();
//   const profile = new PublicProfilePage(page, guard);
//   await profile.goto('jordanrivera');
//   await profile.dismissSignInModalIfPresent();
//   const top = await profile.readTopCard();
//   const exp = await profile.readExperience();
//   const ed  = await profile.readEducation();

import type { Locator, Page } from '@playwright/test';
import {
  ROUTES,
  PUBLIC_PROFILE,
  extractVanityFromUrl,
} from '../selectors/manifest.js';
import type { SafetyGuard } from '../safety/SafetyGuard.js';
import type {
  EducationItem,
  ExperienceItem,
  PersonJsonLd,
  PublicProfileTopCard,
} from '../types.js';
import { assertOnSurface } from './recovery/RecoverySignals.js';

export class PublicProfilePage {
  constructor(private readonly page: Page, private readonly guard: SafetyGuard) {}

  // -------------------------- route + readiness --------------------------

  /**
   * Navigate to a vanity URL with guest-view-budget accounting.
   * Throws GuestViewLimitError before navigating if the budget is exhausted.
   */
  async goto(vanity: string): Promise<void> {
    this.guard.allowReadOnly('navigate_to_public_profile');
    const status = this.guard.noteGuestProfileLoad();
    if (status.remaining < 0) {
      // Caller will see the deny log; throw the explicit recovery signal.
      const { GuestViewLimitError } = await import('../types.js');
      throw new GuestViewLimitError(status.observed, status.budget);
    }
    const url = `https://www.linkedin.com/in/${encodeURIComponent(vanity)}`;
    await this.page.goto(url, { waitUntil: 'domcontentloaded' });
    await assertOnSurface(this.page);
    await this.waitForReady();
  }

  /** Resolves once the URL matches /in/{vanity} AND the name h1 is in the DOM. */
  async waitForReady(): Promise<void> {
    await this.page.waitForURL(ROUTES.public_profile, { timeout: 15_000 });
    await this.page.locator(PUBLIC_PROFILE.name_heading.selector!).first().waitFor({
      state: 'attached',
      timeout: 15_000,
    });
  }

  vanity(): string | null {
    return extractVanityFromUrl(this.page.url());
  }

  // -------------------------- read-only reads --------------------------

  /**
   * Dismiss the "View {firstName}'s full profile" modal if present.
   * No-op if the modal is not on screen. Always allowed.
   */
  async dismissSignInModalIfPresent(): Promise<void> {
    this.guard.allowReadOnly('dismiss_signin_modal');
    const dialog = this.page.locator(PUBLIC_PROFILE.signin_modal.selector!).first();
    if (await dialog.isVisible().catch(() => false)) {
      await this.page.keyboard.press('Escape');
      // Best-effort wait for it to detach; ignore timeout (some modal variants linger).
      await dialog.waitFor({ state: 'hidden', timeout: 2_000 }).catch(() => {});
    }
  }

  /**
   * Extract the top-card region: name, headline, location, followers, connections,
   * current company slug, current school slug, photo URL.
   * Falls back gracefully when individual fields are missing.
   */
  async readTopCard(): Promise<PublicProfileTopCard> {
    this.guard.allowReadOnly('read_public_profile_dom');
    const vanity = this.vanity() ?? '';
    const name = (await this.page.locator(PUBLIC_PROFILE.name_heading.selector!).first().innerText()).trim();
    const headline = (await this.page.locator(PUBLIC_PROFILE.headline_heading.selector!).first().innerText().catch(() => '')).trim();
    const locLine = (await this.page.locator(PUBLIC_PROFILE.location_followers_line.selector!).first().innerText().catch(() => '')).trim();

    // Pass 2a verified shape: "<location> · Contact Info <followers> followers · <connections> connections"
    const m = locLine.match(/^(?<location>.+?) · Contact Info (?<followers>[\d.KM]+) followers · (?<connections>\d+)\+? connections?$/);
    const location = m?.groups?.['location'] ?? null;
    const followers = m?.groups?.['followers'] ?? null;
    const connections = m?.groups?.['connections'] ?? null;

    const companyHref = await this.page
      .locator(`a[href*="${PUBLIC_PROFILE.topcard_current_company_link.hrefIncludes}"]`)
      .first()
      .getAttribute('href')
      .catch(() => null);
    const currentCompanySlug = companyHref?.match(/\/company\/([^/?]+)/)?.[1] ?? null;

    const schoolHref = await this.page
      .locator(`a[href*="${PUBLIC_PROFILE.topcard_school_link.hrefIncludes}"]`)
      .first()
      .getAttribute('href')
      .catch(() => null);
    const currentSchoolSlug = schoolHref?.match(/\/school\/([^/?]+)/)?.[1] ?? null;

    const photoUrl = await this.page
      .locator(PUBLIC_PROFILE.photo_button.selector!)
      .first()
      .getAttribute('src')
      .catch(() => null);

    return {
      vanity,
      name,
      headline,
      location,
      followers,
      connections,
      currentCompanySlug,
      currentSchoolSlug,
      photoUrl,
    };
  }

  /**
   * Read the JSON-LD Person/ProfilePage schema embedded in <head>. This is the
   * preferred extraction surface for the About text (which is DOM-truncated to ~69 chars
   * on the unauthenticated view) and for normalized location/works-for/alumni-of fields.
   */
  async readJsonLd(): Promise<PersonJsonLd | null> {
    this.guard.allowReadOnly('read_public_profile_jsonld');
    return this.page.evaluate(() => {
      const scripts = document.querySelectorAll('script[type="application/ld+json"]');
      for (const s of Array.from(scripts)) {
        try {
          const txt = s.textContent || '';
          const data = JSON.parse(txt);
          const arr: any[] = Array.isArray(data) ? data : (data['@graph'] || [data]);
          for (const node of arr) {
            if (!node) continue;
            if (node['@type'] === 'Person' || node['@type'] === 'ProfilePage') return node;
            if (node.mainEntityOfPage?.['@type'] === 'ProfilePage') return node;
          }
        } catch {}
      }
      return null;
    });
  }

  /** Iterate experience items via the trk= anchor; each anchor wraps the item subtree. */
  async readExperience(): Promise<ExperienceItem[]> {
    this.guard.allowReadOnly('read_public_profile_dom');
    const anchors = this.page.locator(
      `a[href*="${PUBLIC_PROFILE.experience_item_anchor.hrefIncludes}"]`,
    );
    const count = await anchors.count();
    const out: ExperienceItem[] = [];
    for (let i = 0; i < count; i++) {
      const a = anchors.nth(i);
      const href = await a.getAttribute('href');
      const companySlug = href?.match(/\/company\/([^/?]+)/)?.[1] ?? null;
      const item = await a.evaluateHandle((el) => el.closest('li, section, div'));
      const root = item.asElement();
      const title = root
        ? await root.evaluate((r) => r.querySelector('h3')?.textContent?.trim() ?? null)
        : null;
      const companyName = root
        ? await root.evaluate((r) => r.querySelector('h4')?.textContent?.trim() ?? null)
        : null;
      const dateText = root
        ? await root.evaluate((r) => {
            const t = (r.textContent || '').match(/(\d{4}.*?(Present|\d{4})[^·\n]*)(·\s*([^\n]+))?/);
            const dateRangeText = t?.[1]?.trim();
            return dateRangeText
              ? { dateRangeText, durationText: t?.[4]?.trim() ?? null }
              : null;
          })
        : null;
      out.push({
        title,
        companyName,
        companySlug,
        dateRangeText: dateText?.dateRangeText ?? null,
        durationText: dateText?.durationText ?? null,
      });
      await item.dispose();
    }
    return out;
  }

  /** Iterate education items via the trk= anchor. */
  async readEducation(): Promise<EducationItem[]> {
    this.guard.allowReadOnly('read_public_profile_dom');
    const anchors = this.page.locator(
      `a[href*="${PUBLIC_PROFILE.education_item_anchor.hrefIncludes}"]`,
    );
    const count = await anchors.count();
    const out: EducationItem[] = [];
    for (let i = 0; i < count; i++) {
      const a = anchors.nth(i);
      const href = await a.getAttribute('href');
      const schoolSlug = href?.match(/\/school\/([^/?]+)/)?.[1] ?? null;
      const item = await a.evaluateHandle((el) => el.closest('li, section, div'));
      const root = item.asElement();
      const schoolName = root
        ? await root.evaluate((r) => r.querySelector('h3')?.textContent?.trim() ?? null)
        : null;
      const degree = root
        ? await root.evaluate((r) => r.querySelector('h4')?.textContent?.trim() ?? null)
        : null;
      const dateText = root
        ? await root.evaluate((r) => {
            const t = (r.textContent || '').match(/(\d{4}\s*[-–]\s*\d{4})/);
            return t?.[1] ?? null;
          })
        : null;
      out.push({ schoolName, schoolSlug, degree, dateRangeText: dateText });
      await item.dispose();
    }
    return out;
  }

  /** Returns the modal dialog locator (read-only). Returns null if not present. */
  modal(): Locator {
    return this.page.locator(PUBLIC_PROFILE.signin_modal.selector!).first();
  }

  /** Read-only presence check for the join-wall CTAs (for telemetry only — never click). */
  async readJoinWallTelemetry(): Promise<{
    headerSignInPresent: boolean;
    headerJoinPresent: boolean;
    topCardJoinPresent: boolean;
    bottomBannerPresent: boolean;
  }> {
    this.guard.allowReadOnly('read_url_state');
    return {
      headerSignInPresent: (await this.page.locator(`a[href*="${PUBLIC_PROFILE.header_signin_link.hrefIncludes}"]`).count()) > 0,
      headerJoinPresent: (await this.page.locator(`a[href*="${PUBLIC_PROFILE.header_join_link.hrefIncludes}"]`).count()) > 0,
      topCardJoinPresent: (await this.page.locator(`a[href*="${PUBLIC_PROFILE.topcard_join_button.hrefIncludes}"]`).count()) > 0,
      bottomBannerPresent: (await this.page.locator(`a[href*="${PUBLIC_PROFILE.bottom_cta_banner.hrefIncludes}"]`).count()) > 0,
    };
  }
}
