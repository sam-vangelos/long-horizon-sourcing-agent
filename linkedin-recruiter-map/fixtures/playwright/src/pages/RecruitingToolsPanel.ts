// RecruitingToolsPanel — right rail of the profile drawer.
// All "Add ..." buttons are mutating. classification: defer.
// Pass-3 verified the row labels: Notes / Reminders / Links / Tags.

import type { Locator, Page } from '@playwright/test';
import { extractProjectIdFromUrl } from '../selectors/manifest.js';
import type { SafetyGuard } from '../safety/SafetyGuard.js';

export class RecruitingToolsPanel {
  constructor(private readonly page: Page, private readonly guard: SafetyGuard) {}

  panel(): Locator {
    return this.page.locator('role=dialog').first().getByRole('region', { name: /^Recruiting Tools$/ });
  }

  async addNote(): Promise<void> {
    const projectId = extractProjectIdFromUrl(this.page.url());
    this.guard.requireIntent('add_note', { projectIdFromUrl: projectId, surface: 'profile_drawer' });
    await this.panel().getByRole('button', { name: /^Add Note about / }).click();
  }

  async addReminder(): Promise<void> {
    const projectId = extractProjectIdFromUrl(this.page.url());
    this.guard.requireIntent('add_note', { projectIdFromUrl: projectId, surface: 'profile_drawer' });
    await this.panel().getByRole('button', { name: 'Add new reminder' }).click();
  }

  async addLink(): Promise<void> {
    const projectId = extractProjectIdFromUrl(this.page.url());
    this.guard.requireIntent('add_note', { projectIdFromUrl: projectId, surface: 'profile_drawer' });
    await this.panel().getByRole('button', { name: 'Add new link' }).click();
  }

  async addTag(): Promise<void> {
    const projectId = extractProjectIdFromUrl(this.page.url());
    this.guard.requireIntent('add_tag', { projectIdFromUrl: projectId, surface: 'profile_drawer' });
    await this.panel().getByRole('button', { name: 'Add new tags' }).click();
  }
}
