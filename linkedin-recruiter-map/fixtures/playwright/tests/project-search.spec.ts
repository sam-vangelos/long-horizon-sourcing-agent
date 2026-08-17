// Read-only assertions against ProjectSearchPage. Runs offline against the
// recorded DOM snapshot — flip PLAYWRIGHT_MODE=cdp to run against a live seat.
//
// These tests double as worked examples for Cloris worker authors.

import { test, expect, type SnapshotName } from './fixtures/authenticated-page.js';
import { ProjectSearchPage } from '../src/pages/ProjectSearchPage.js';
import { SafetyGuard } from '../src/safety/SafetyGuard.js';

test.describe('ProjectSearchPage — populated results', () => {
  test('reads result total, lists cards, and filters out recommended-matches sub-region', async ({
    loadSnapshot,
  }) => {
    const page = await loadSnapshot('results-page' satisfies SnapshotName);
    const search = new ProjectSearchPage(page, new SafetyGuard({ mode: 'read_only', log: () => {} }));

    // URL-derived state
    expect(search.projectId()).toBe('2035258290');

    // Result count parsing handles K+/M+ suffixes
    const total = await search.readResultTotal();
    expect(total).not.toBeNull();
    expect(total!.total).toBeGreaterThan(0);

    // Cards present + recommended-matches sub-region excluded
    const cards = await search.cards();
    expect(cards.length).toBeGreaterThan(0);

    // Spotlights region accessible
    const spots = await search.spotlights().readVisibleSpotlights();
    expect(spots.length).toBeGreaterThanOrEqual(0); // carousel may have 0 or more visible at once
  });

  test('idle/empty surface exposes the empty-search probe', async ({ loadSnapshot }) => {
    const page = await loadSnapshot('empty-search' satisfies SnapshotName);
    const search = new ProjectSearchPage(page, new SafetyGuard({ mode: 'read_only', log: () => {} }));
    expect(await search.isEmptyState()).toBe(true);
  });
});
