// FilterPill — encodes the two-step add-button + editor pattern (Pass 2 correction).
//   step 1: button name='Add {label}' (role=button) — click to reveal editor
//   step 2: editor is one of: combobox+listbox, multi-select list, range, toggle, free-text
//
// Chips render INSIDE the filter group as inline items prefixed with '+'.
// The numeric suffix '(N.NM+)' is LinkedIn's candidate-pool estimate.
//
// Applying a filter is `defer` (mutates the search). Reading current chips is read-only.

import type { Locator, Page } from '@playwright/test';
import { extractProjectIdFromUrl } from '../../selectors/manifest.js';
import type { SafetyGuard } from '../../safety/SafetyGuard.js';

export interface FilterChip {
  label: string;
  poolSize?: string; // e.g. "8.9M+"
}

export class FilterPill {
  constructor(
    private readonly page: Page,
    private readonly section: Locator,
    public readonly addButtonName: string,
    private readonly guard: SafetyGuard,
  ) {}

  /** Reads currently-applied chips inside this filter group. */
  async readChips(): Promise<FilterChip[]> {
    // Chips are typically buttons inside the section with name starting "Remove {label} filter"
    // OR text nodes prefixed with '+'. We support both.
    const chips: FilterChip[] = [];
    const removeButtons = this.section.getByRole('button', { name: /^Remove .+ filter$/i });
    const rbCount = await removeButtons.count();
    for (let i = 0; i < rbCount; i++) {
      const btn = removeButtons.nth(i);
      const name = ((await btn.getAttribute('aria-label')) ?? (await btn.innerText())).trim();
      const m = name.match(/^Remove (.+) filter$/i);
      if (!m) continue;
      const inner = m[1]!;
      const pool = inner.match(/\(([\d.]+[KM]\+?)\)\s*$/);
      const labelOnly = pool ? inner.replace(/\s*\([\d.]+[KM]\+?\)\s*$/, '') : inner;
      const chip: FilterChip = { label: labelOnly };
      if (pool) chip.poolSize = pool[1]!;
      chips.push(chip);
    }
    return chips;
  }

  /** Opens the editor. classification: stable_now (this alone does not change the search). */
  async openEditor(): Promise<Locator> {
    this.guard.allowReadOnly('open_overflow_menu_read_only'); // closest read-only verb
    const btn = this.section.getByRole('button', { name: this.addButtonName }).first();
    await btn.click();
    // Heuristic: the editor is the next role=combobox / role=listbox / role=textbox / role=group within the section.
    const editor = this.section
      .locator('role=combobox, role=textbox, role=listbox, role=group')
      .first();
    await editor.waitFor({ state: 'visible', timeout: 3_000 });
    return editor;
  }

  /**
   * Adds a value to a typeahead-style filter. classification: defer (mutates search).
   * Intent action must be 'apply_filter'.
   */
  async addTypeaheadValue(value: string): Promise<void> {
    const projectId = extractProjectIdFromUrl(this.page.url());
    this.guard.requireIntent('apply_filter', {
      projectIdFromUrl: projectId,
      surface: 'project_search_advanced',
    });
    const editor = await this.openEditor();
    await editor.fill(value);
    const listbox = this.page.locator('role=listbox').last();
    await listbox.waitFor({ state: 'visible', timeout: 3_000 });
    await listbox.getByRole('option').first().click();
  }

  /**
   * Removes a chip by clicking its "Remove {label} filter" button. classification: defer.
   */
  async removeChip(label: string): Promise<void> {
    const projectId = extractProjectIdFromUrl(this.page.url());
    this.guard.requireIntent('apply_filter', {
      projectIdFromUrl: projectId,
      surface: 'project_search_advanced',
    });
    const btn = this.section
      .getByRole('button', { name: new RegExp(`^Remove ${escapeRegExp(label)}.* filter$`, 'i') })
      .first();
    await btn.click();
  }
}

function escapeRegExp(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}
