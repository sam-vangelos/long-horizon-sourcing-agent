import { defineConfig } from '@playwright/test';

// PLAYWRIGHT_MODE:
//   - 'cdp'     → attach to a live LinkedIn session over CDP. Public LinkedIn
//                 worker may run with NO session (guest view) or with an authenticated
//                 session — surface detection routes accordingly. Default.
//   - 'offline' → load recorded DOM snapshots from tests/fixtures/dom-snapshots/
//                 via page.setContent(). No network, no auth. CI-safe.
const mode = process.env.PLAYWRIGHT_MODE ?? 'cdp';

export default defineConfig({
  testDir: './tests',
  timeout: 30_000,
  expect: { timeout: 5_000 },
  fullyParallel: false, // CDP-attached sessions are single-tab; guest-view budget is per-session
  retries: 0,           // mutating retries are forbidden by manifest policy
  reporter: [['list']],
  use: {
    headless: true,
    actionTimeout: 5_000,
    navigationTimeout: 15_000,
    // tracing/video off by default — recordings even of public profiles can leak
    // identifying info. Enable locally only.
    trace: 'off',
    video: 'off',
    screenshot: 'off',
    userAgent: process.env.LINKEDIN_PUBLIC_UA, // override only if explicitly set
  },
  projects: [
    {
      name: 'public_linkedin',
      metadata: { mode },
    },
  ],
});
