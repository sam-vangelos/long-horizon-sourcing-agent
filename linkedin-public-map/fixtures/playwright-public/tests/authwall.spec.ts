// Offline tests for AuthWallPage against the Pass-2a placeholder DOM snapshots.
// Verifies both auth-wall templates render with distinct headings + SSO iframes.

import { test, expect } from '@playwright/test';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { AuthWallPage } from '../src/pages/AuthWallPage.js';
import { SafetyGuard } from '../src/safety/SafetyGuard.js';

const joinHtml = readFileSync(
  resolve('tests/fixtures/dom-snapshots/authwall_join_company_people.html'),
  'utf8',
);
const signinHtml = readFileSync(
  resolve('tests/fixtures/dom-snapshots/authwall_signin_people_search.html'),
  'utf8',
);

test.describe('AuthWallPage — offline templates', () => {
  test('/authwall (Join) heading + Google iframe', async ({ page }) => {
    await page.setContent(joinHtml);
    const guard = new SafetyGuard({ mode: 'read_only' });
    const wall = new AuthWallPage(page, guard);
    await expect(wall.joinHeading()).toHaveText('Join LinkedIn');
    await expect(wall.joinGoogleIframe()).toHaveCount(1);
  });

  test('/uas/login (Sign in) heading + Apple/Google/Microsoft buttons', async ({ page }) => {
    await page.setContent(signinHtml);
    const guard = new SafetyGuard({ mode: 'read_only' });
    const wall = new AuthWallPage(page, guard);
    await expect(wall.signInHeading()).toHaveText('Sign in');
    await expect(wall.signInGoogleIframe()).toHaveCount(1);
    await expect(wall.signInMicrosoftIframe()).toHaveCount(1);
    await expect(page.getByRole('button', { name: /^Sign in with Apple$/ })).toBeVisible();
  });

  test('Join template has Dismiss for the app-upsell toast', async ({ page }) => {
    await page.setContent(joinHtml);
    await expect(page.getByRole('button', { name: /^Dismiss$/ })).toBeVisible();
  });
});
