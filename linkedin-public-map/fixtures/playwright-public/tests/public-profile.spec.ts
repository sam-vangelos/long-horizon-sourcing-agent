// Offline tests for PublicProfilePage against the Pass-2a placeholder DOM snapshot.
// Run with: PLAYWRIGHT_MODE=offline npm test

import { test, expect } from '@playwright/test';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { PublicProfilePage } from '../src/pages/PublicProfilePage.js';
import { SafetyGuard } from '../src/safety/SafetyGuard.js';
import { PUBLIC_PROFILE } from '../src/selectors/manifest.js';

const snapshot = readFileSync(
  resolve('tests/fixtures/dom-snapshots/public_profile_jordanrivera.html'),
  'utf8',
);

test.describe('PublicProfilePage — offline snapshot', () => {
  test('top-card selectors resolve uniquely', async ({ page }) => {
    await page.setContent(snapshot);
    await expect(page.locator(PUBLIC_PROFILE.name_heading.selector!).first()).toHaveText('Jordan Rivera');
    await expect(page.locator(PUBLIC_PROFILE.headline_heading.selector!).first()).toHaveText(
      /Chair, Rivera Foundation/,
    );
    await expect(page.locator(PUBLIC_PROFILE.location_followers_line.selector!).first()).toHaveText(
      /Austin.*Contact Info.*followers.*connections/,
    );
  });

  test('experience anchors match the trk= pattern', async ({ page }) => {
    await page.setContent(snapshot);
    const anchors = page.locator(
      `a[href*="${PUBLIC_PROFILE.experience_item_anchor.hrefIncludes}"]`,
    );
    await expect(anchors).toHaveCount(3);
  });

  test('education anchors match the trk= pattern', async ({ page }) => {
    await page.setContent(snapshot);
    const anchors = page.locator(
      `a[href*="${PUBLIC_PROFILE.education_item_anchor.hrefIncludes}"]`,
    );
    await expect(anchors).toHaveCount(1);
  });

  test('sign-in modal heading matches regex', async ({ page }) => {
    await page.setContent(snapshot);
    const heading = page.getByRole('heading', { name: PUBLIC_PROFILE.signin_modal_heading.nameRegex! });
    await expect(heading).toBeVisible();
  });

  test('JSON-LD Person block extractable from <head>', async ({ page }) => {
    await page.setContent(snapshot);
    const guard = new SafetyGuard({ mode: 'read_only' });
    const profile = new PublicProfilePage(page, guard);
    const ld = await profile.readJsonLd();
    expect(ld).not.toBeNull();
    expect(ld?.name ?? (ld?.mainEntity as any)?.name).toBe('Jordan Rivera');
  });

  test('readTopCard parses location/followers/connections', async ({ page }) => {
    await page.setContent(snapshot);
    const guard = new SafetyGuard({ mode: 'read_only' });
    const profile = new PublicProfilePage(page, guard);
    const top = await profile.readTopCard();
    expect(top.name).toBe('Jordan Rivera');
    expect(top.location).toBe('Austin, Texas, United States');
    expect(top.followers).toBe('12K');
    expect(top.connections).toBe('8');
    expect(top.currentCompanySlug).toBe('rivera-foundation');
    expect(top.currentSchoolSlug).toBe('state-university');
  });

  test('readExperience returns three items with company slugs', async ({ page }) => {
    await page.setContent(snapshot);
    const guard = new SafetyGuard({ mode: 'read_only' });
    const profile = new PublicProfilePage(page, guard);
    const exp = await profile.readExperience();
    expect(exp).toHaveLength(3);
    expect(exp[0]!.companySlug).toBe('rivera-foundation');
    expect(exp[1]!.companySlug).toBe('bright-energy');
    expect(exp[2]!.companySlug).toBe('acme-software');
  });

  test('join-wall CTAs detected for telemetry (read-only)', async ({ page }) => {
    await page.setContent(snapshot);
    const guard = new SafetyGuard({ mode: 'read_only' });
    const profile = new PublicProfilePage(page, guard);
    const t = await profile.readJoinWallTelemetry();
    expect(t.headerSignInPresent).toBe(true);
    expect(t.headerJoinPresent).toBe(true);
    expect(t.topCardJoinPresent).toBe(true);
    expect(t.bottomBannerPresent).toBe(true);
  });
});
