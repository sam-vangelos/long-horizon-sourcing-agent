// ResultCard — scoped to a single role=article inside the results form.
// Every mutating method calls guard.requireIntent; read-only methods call guard.allowReadOnly.

import type { Locator, Page } from '@playwright/test';
import { RESULT_CARD } from '../../selectors/manifest.js';
import { extractProjectIdFromUrl } from '../../selectors/manifest.js';
import type { SafetyGuard } from '../../safety/SafetyGuard.js';
import { ProfileDrawer } from '../ProfileDrawer.js';

export class ResultCard {
  constructor(
    private readonly page: Page,
    public readonly article: Locator,
    private readonly guard: SafetyGuard,
  ) {}

  // -------------------------- read-only --------------------------

  /** Recruiter member id parsed from the candidate name link href. */
  async recruiterMemberId(): Promise<string | null> {
    const link = this.article.locator(RESULT_CARD.candidate_name_link.selector!).first();
    const href = await link.getAttribute('href');
    if (!href) return null;
    return href.match(/\/talent\/profile\/([A-Za-z0-9_-]+)/)?.[1] ?? null;
  }

  async candidateName(): Promise<string | null> {
    const link = this.article.locator(RESULT_CARD.candidate_name_link.selector!).first();
    if (!(await link.count())) return null;
    return (await link.innerText()).trim();
  }

  /** True if this card is the "LinkedIn Member" redacted form (no /talent/profile/ link). */
  async isRedacted(): Promise<boolean> {
    const name = await this.candidateName();
    if (name !== 'LinkedIn Member') return false;
    const link = this.article.locator(RESULT_CARD.candidate_name_link.selector!);
    return (await link.count()) === 0;
  }

  /** Reads the default stage embedded in the "Save to '{stage}'" button name. */
  async readDefaultStage(): Promise<string | null> {
    const btn = this.article.getByRole('button', { name: RESULT_CARD.save_to_pipeline.nameRegex! }).first();
    if (!(await btn.count())) return null;
    const name = (await btn.getAttribute('aria-label')) ?? (await btn.innerText());
    return name.match(/^Save to '([^']+)'$/)?.[1] ?? null;
  }

  /** Opens the candidate drawer via the name link. classification: stable_now. */
  async openDrawer(): Promise<ProfileDrawer> {
    this.guard.allowReadOnly('open_drawer');
    const link = this.article.locator(RESULT_CARD.candidate_name_link.selector!).first();
    await link.click();
    const dialog = this.page.locator('role=dialog').first();
    await dialog.waitFor({ state: 'visible', timeout: 5_000 });
    return new ProfileDrawer(this.page, this.guard);
  }

  // -------------------------- read-only probes for Pass 4 --------------------------

  /**
   * Opens the per-card stage chooser dropdown, reads option names, then closes via Escape.
   * Reading is non-mutating; selecting an option IS mutating, so this method never selects.
   */
  async openStagePickerReadOnly(): Promise<string[]> {
    this.guard.allowReadOnly('open_stage_picker_read_only');
    const chooser = this.article
      .getByRole('button', { name: RESULT_CARD.save_stage_chooser.name! })
      .first();
    await chooser.click();
    const listbox = this.page.locator('role=listbox, role=menu').last();
    await listbox.waitFor({ state: 'visible', timeout: 3_000 });
    const optionLocators = listbox.getByRole('option');
    const menuItems = listbox.getByRole('menuitem');
    const useOptions = (await optionLocators.count()) > 0;
    const items = useOptions ? optionLocators : menuItems;
    const count = await items.count();
    const names: string[] = [];
    for (let i = 0; i < count; i++) {
      const item = items.nth(i);
      names.push(((await item.getAttribute('aria-label')) ?? (await item.innerText())).trim());
    }
    await this.page.keyboard.press('Escape');
    return names;
  }

  /**
   * Opens the per-card "More actions for {Name}" menu, reads menuitem names, presses Escape.
   * Pass-4 candidate.
   */
  async openMoreActionsReadOnly(): Promise<string[]> {
    this.guard.allowReadOnly('open_more_actions_read_only');
    const trigger = this.article
      .getByRole('button', { name: RESULT_CARD.more_actions.nameRegex! })
      .first();
    await trigger.click();
    const menu = this.page.locator('role=menu').last();
    await menu.waitFor({ state: 'visible', timeout: 3_000 });
    const items = menu.getByRole('menuitem');
    const count = await items.count();
    const names: string[] = [];
    for (let i = 0; i < count; i++) {
      const item = items.nth(i);
      names.push(((await item.getAttribute('aria-label')) ?? (await item.innerText())).trim());
    }
    await this.page.keyboard.press('Escape');
    return names;
  }

  // -------------------------- mutating (gated) --------------------------

  /** Saves to the current project's default stage. classification: defer. */
  async saveToDefaultStage(): Promise<void> {
    const projectId = extractProjectIdFromUrl(this.page.url());
    this.guard.requireIntent('save_to_project', { projectIdFromUrl: projectId, surface: 'project_search' });
    const btn = this.article
      .getByRole('button', { name: RESULT_CARD.save_to_pipeline.nameRegex! })
      .first();
    await btn.click();
  }

  /** classification: defer. */
  async hide(): Promise<void> {
    const projectId = extractProjectIdFromUrl(this.page.url());
    this.guard.requireIntent('hide', { projectIdFromUrl: projectId, surface: 'project_search' });
    await this.article.getByRole('button', { name: RESULT_CARD.hide_candidate.nameRegex! }).first().click();
  }

  /** classification: defer. */
  async openMessageDialog(): Promise<void> {
    const projectId = extractProjectIdFromUrl(this.page.url());
    this.guard.requireIntent('message', { projectIdFromUrl: projectId, surface: 'project_search' });
    await this.article.getByRole('button', { name: RESULT_CARD.message_candidate.nameRegex! }).first().click();
  }
}
