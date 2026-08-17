// authenticated-page.ts — Playwright test fixture that supplies a Page suitable for
// driving the page objects. Two modes:
//
//   1. cdp     — connects to a live, authenticated LinkedIn Recruiter Chrome over CDP.
//                Set CDP_ENDPOINT to point at the running Chrome (e.g.
//                'http://localhost:9222'). Use this for Pass-3+ verification work.
//
//   2. offline — loads a recorded DOM snapshot from ./dom-snapshots/*.html via
//                page.setContent(). No network, no auth, no LinkedIn risk. Used by
//                CI and by refactor-time test runs.
//
// Mode is selected by the PLAYWRIGHT_MODE env var; see playwright.config.ts.

import { test as base, chromium, type Page } from '@playwright/test';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

type Mode = 'cdp' | 'offline';

interface Fixtures {
  mode: Mode;
  loadSnapshot: (name: SnapshotName) => Promise<Page>;
  authenticatedPage: Page;
}

export type SnapshotName =
  | 'empty-search'
  | 'results-page'
  | 'advanced-panel'
  | 'profile-drawer';

const SNAPSHOT_DIR = path.resolve(__dirname, 'dom-snapshots');

export const test = base.extend<Fixtures>({
  // eslint-disable-next-line no-empty-pattern
  mode: async ({}, use) => {
    await use((process.env.PLAYWRIGHT_MODE ?? 'cdp') as Mode);
  },

  loadSnapshot: async ({ page }, use) => {
    await use(async (name: SnapshotName) => {
      const filePath = path.join(SNAPSHOT_DIR, `${name}.html`);
      if (!fs.existsSync(filePath)) {
        throw new Error(
          `[snapshot missing] ${filePath}\n` +
            `Run capture-snippets/99_outer_html.js on a live authenticated seat ` +
            `and paste the output into ${filePath}.`,
        );
      }
      const html = fs.readFileSync(filePath, 'utf8');
      if (html.trim().startsWith('<!-- PLACEHOLDER')) {
        throw new Error(
          `[snapshot placeholder] ${filePath} still contains the placeholder header. ` +
            `Capture and paste real DOM before running this test.`,
        );
      }
      // setContent simulates the URL using the routing layer below — we set a base URL
      // matching the snapshot so URL-derived assertions pass.
      const baseUrl = baseUrlForSnapshot(name);
      await page.goto('about:blank');
      await page.evaluate((url) => history.replaceState(null, '', url), baseUrl);
      await page.setContent(html);
      return page;
    });
  },

  authenticatedPage: async ({ mode, page }, use) => {
    if (mode === 'cdp') {
      const endpoint = process.env.CDP_ENDPOINT;
      if (!endpoint) {
        throw new Error('PLAYWRIGHT_MODE=cdp requires CDP_ENDPOINT (e.g. http://localhost:9222)');
      }
      const cdp = await chromium.connectOverCDP(endpoint);
      const ctx = cdp.contexts()[0] ?? (await cdp.newContext());
      const liveP = ctx.pages()[0] ?? (await ctx.newPage());
      await use(liveP);
      await cdp.close();
      return;
    }
    // offline: hand back the stock Playwright page; tests will call loadSnapshot themselves.
    await use(page);
  },
});

export { expect } from '@playwright/test';

function baseUrlForSnapshot(name: SnapshotName): string {
  // Project id matches Pass-3 observed seat.
  const PID = '2035258290';
  switch (name) {
    case 'empty-search':
      return `https://www.linkedin.com/talent/hire/${PID}/discover/recruiterSearch`;
    case 'results-page':
      return `https://www.linkedin.com/talent/hire/${PID}/discover/recruiterSearch?searchContextId=fixture`;
    case 'advanced-panel':
      return `https://www.linkedin.com/talent/hire/${PID}/discover/recruiterSearch/advanced`;
    case 'profile-drawer':
      return `https://www.linkedin.com/talent/hire/${PID}/discover/recruiterSearch/profile/AAA_fixture_AAA?trk=SEARCH_CONTEXTUAL`;
  }
}
