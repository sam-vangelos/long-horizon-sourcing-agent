import { defineConfig } from '@playwright/test';

// PLAYWRIGHT_MODE:
//   - 'cdp'     → attach to a live, authenticated LinkedIn Recruiter session over CDP
//                 (set CDP_ENDPOINT env var). Default.
//   - 'offline' → load recorded DOM snapshots from tests/fixtures/dom-snapshots/
//                 via page.setContent(). No network, no auth. CI-safe.
const mode = process.env.PLAYWRIGHT_MODE ?? 'cdp';

export default defineConfig({
  testDir: './tests',
  timeout: 30_000,
  expect: { timeout: 5_000 },
  fullyParallel: false, // CDP-attached sessions are single-tab
  retries: 0,           // mutating retries are forbidden by manifest policy
  reporter: [['list']],
  use: {
    headless: true,
    actionTimeout: 5_000,
    navigationTimeout: 15_000,
    // tracing/video off by default — recordings of authenticated Recruiter sessions
    // are sensitive (candidate data). Enable locally only.
    trace: 'off',
    video: 'off',
    screenshot: 'off',
  },
  projects: [
    {
      name: 'recruiter',
      metadata: { mode },
    },
  ],
});
