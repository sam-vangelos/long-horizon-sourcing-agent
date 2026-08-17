// ProfileDrawer — the role=dialog mounted over the search route when a candidate is opened.
// Hybrid route + dialog (Pass 3 finding): URL pushes to /profile/{candidateId} AND dialog mounts.
// Either signal verifies open state; we prefer dialog presence.

import type { Locator, Page } from '@playwright/test';
import { extractCandidateIdFromUrl, extractProjectIdFromUrl, PROFILE_DRAWER } from '../selectors/manifest.js';
import type { SafetyGuard } from '../safety/SafetyGuard.js';
import type { ProjectMembership } from '../types.js';

export class ProfileDrawer {
  constructor(private readonly page: Page, private readonly guard: SafetyGuard) {}

  dialog(): Locator {
    return this.page.locator('role=dialog').first();
  }

  candidateId(): string | null {
    return extractCandidateIdFromUrl(this.page.url());
  }

  async isOpen(): Promise<boolean> {
    return (await this.dialog().count()) > 0 && (await this.dialog().isVisible());
  }

  /** "From {projectName}" heading content — source of truth for the drawer's project scope. */
  async readFromProjectName(): Promise<string | null> {
    const heading = this.dialog().getByRole('heading', { name: PROFILE_DRAWER.from_project_heading.nameRegex! }).first();
    if (!(await heading.count())) return null;
    const name = ((await heading.getAttribute('aria-label')) ?? (await heading.innerText())).trim();
    return name.replace(/^From /, '');
  }

  /** "1 of 17,151,103" position counter. */
  async readPositionCounter(): Promise<{ index: number; total: number } | null> {
    const el = this.dialog().locator(PROFILE_DRAWER.position_counter.selector!).first();
    if (!(await el.count())) return null;
    const t = (await el.innerText()).trim();
    const m = t.match(/^(\d+)\s+of\s+([\d,]+)$/);
    if (!m) return null;
    return { index: parseInt(m[1]!, 10), total: parseInt(m[2]!.replace(/,/g, ''), 10) };
  }

  /** Reads the "In N projects" section. Returns empty array when section absent. KEY SAVE-SAFETY READ. */
  async readInProjects(): Promise<ProjectMembership[]> {
    const heading = this.dialog()
      .getByRole('heading', { name: PROFILE_DRAWER.in_projects_section.nameRegex! })
      .first();
    if (!(await heading.count())) return [];
    // Heading parent contains the link list. We anchor via heading then locate sibling links.
    const region = heading.locator('xpath=ancestor::*[self::section or self::div][1]');
    const links = region.getByRole('link');
    const total = await links.count();
    const out: ProjectMembership[] = [];
    for (let i = 0; i < total; i++) {
      const link = links.nth(i);
      const text = ((await link.getAttribute('aria-label')) ?? (await link.innerText())).trim();
      const m = text.match(/^(.+?)\s+(contacted|uncontacted)$/i);
      if (m) {
        out.push({ projectName: m[1]!.trim(), status: m[2]!.toLowerCase() as 'contacted' | 'uncontacted' });
      }
    }
    return out;
  }

  /**
   * Resolves the candidate's public profile URL via the Public-profile button → tooltip → link href.
   * classification: stable_now (read_public_profile_link). Pass-3 verified.
   */
  async readPublicProfileVanity(): Promise<{ vanity: string; url: string } | null> {
    this.guard.allowReadOnly('read_public_profile_link');
    const btn = this.dialog().getByRole('button', { name: PROFILE_DRAWER.public_profile_button.name! }).first();
    if (!(await btn.count())) return null;
    await btn.click();
    // tooltip mounts as a popover; the link "Open link in new tab" lives inside it.
    const link = this.page.getByRole('link', { name: PROFILE_DRAWER.public_profile_open_link.nameRegex! }).first();
    await link.waitFor({ state: 'visible', timeout: 3_000 });
    const href = await link.getAttribute('href');
    // close tooltip by clicking the button again or pressing Escape — safer to press Escape.
    await this.page.keyboard.press('Escape');
    if (!href) return null;
    const m = href.match(/^https:\/\/www\.linkedin\.com\/in\/([^/?#]+)/);
    return m ? { vanity: m[1]!, url: href } : null;
  }

  /** Closes the drawer via Escape (safer than the role=presentation exit affordance). */
  async escape(): Promise<void> {
    this.guard.allowReadOnly('escape_drawer');
    await this.page.keyboard.press('Escape');
  }

  /** Drawer-internal next-candidate navigation. classification: stable_now (read-only). */
  async nextCandidate(): Promise<void> {
    this.guard.allowReadOnly('paginate_next');
    await this.dialog()
      .getByRole('link', { name: PROFILE_DRAWER.drawer_next_candidate.nameRegex! })
      .first()
      .click();
  }

  // -------------------------- mutating (gated) --------------------------

  /** Saves to current project's default stage. classification: defer. */
  async saveToDefaultStage(): Promise<void> {
    const projectId = extractProjectIdFromUrl(this.page.url());
    this.guard.requireIntent('save_to_project', { projectIdFromUrl: projectId, surface: 'profile_drawer' });
    await this.dialog()
      .getByRole('button', { name: PROFILE_DRAWER.save_to_pipeline_drawer.nameRegex! })
      .first()
      .click();
  }

  async addEmail(): Promise<void> {
    const projectId = extractProjectIdFromUrl(this.page.url());
    this.guard.requireIntent('add_email', { projectIdFromUrl: projectId, surface: 'profile_drawer' });
    await this.dialog().getByRole('button', { name: PROFILE_DRAWER.add_email.name! }).first().click();
  }

  async addPhone(): Promise<void> {
    const projectId = extractProjectIdFromUrl(this.page.url());
    this.guard.requireIntent('add_phone', { projectIdFromUrl: projectId, surface: 'profile_drawer' });
    await this.dialog().getByRole('button', { name: PROFILE_DRAWER.add_phone.name! }).first().click();
  }
}
