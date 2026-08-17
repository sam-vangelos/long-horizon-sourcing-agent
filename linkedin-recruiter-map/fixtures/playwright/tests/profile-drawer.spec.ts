// Read-only assertions against ProfileDrawer. Runs offline against the recorded
// DOM snapshot. Demonstrates the save-safety read path: open drawer → readInProjects
// → readPositionCounter → escape. No mutating clicks performed.

import { test, expect } from './fixtures/authenticated-page.js';
import { ProfileDrawer } from '../src/pages/ProfileDrawer.js';
import { SafetyGuard } from '../src/safety/SafetyGuard.js';

test.describe('ProfileDrawer — read-only contract', () => {
  test('reads candidateId, project scope, position counter, and In-N-projects list', async ({
    loadSnapshot,
  }) => {
    const page = await loadSnapshot('profile-drawer');
    const drawer = new ProfileDrawer(page, new SafetyGuard({ mode: 'read_only', log: () => {} }));

    expect(await drawer.isOpen()).toBe(true);
    expect(drawer.candidateId()).not.toBeNull();

    const fromProject = await drawer.readFromProjectName();
    expect(fromProject).not.toBeNull();
    expect(fromProject!.length).toBeGreaterThan(0);

    const pos = await drawer.readPositionCounter();
    if (pos) {
      expect(pos.index).toBeGreaterThanOrEqual(1);
      expect(pos.total).toBeGreaterThanOrEqual(pos.index);
    }

    const memberships = await drawer.readInProjects();
    // Empty array is valid (candidate not in any other project); contract is "no throw".
    expect(Array.isArray(memberships)).toBe(true);
    for (const m of memberships) {
      expect(['contacted', 'uncontacted']).toContain(m.status);
    }
  });

  test('mutating methods throw under read-only guard', async ({ loadSnapshot }) => {
    const page = await loadSnapshot('profile-drawer');
    const drawer = new ProfileDrawer(page, new SafetyGuard({ mode: 'read_only', log: () => {} }));

    await expect(drawer.saveToDefaultStage()).rejects.toThrow(/envelope/);
    await expect(drawer.addEmail()).rejects.toThrow(/envelope/);
    await expect(drawer.addPhone()).rejects.toThrow(/envelope/);
  });
});
