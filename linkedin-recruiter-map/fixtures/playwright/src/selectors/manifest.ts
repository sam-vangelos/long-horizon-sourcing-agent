// Tiny typed wrapper around the YAML manifest, scoped to the rows the page objects use.
// We don't autogenerate the whole tree — Pass 6 will. For now, hand-curated views keep
// the surface area small and the test diffs human-readable.
//
// To add a selector: copy its row from the YAML, paste under the matching surface below.
// Each entry carries `classification` so SafetyGuard can gate `unknown` rows.

export interface SelectorSpec {
  /** Manifest row id (e.g. 'results_form'). */
  id: string;
  /** ARIA role hint, when available. */
  role?: string;
  /** Accessible name or regex source string. */
  name?: string;
  nameRegex?: RegExp;
  /** Playwright selector expression — fallback when role+name is insufficient. */
  selector?: string;
  classification: 'stable_now' | 'mock_only' | 'defer' | 'unknown';
  /** True if this row's mutation column is non-null/non-read_only in the manifest. */
  mutating?: boolean;
}

// ---------------------------------------------------------------------------
// A. ROUTES
// ---------------------------------------------------------------------------
export const ROUTES = {
  project_search: /^https:\/\/www\.linkedin\.com\/talent\/hire\/(?<projectId>\d+)\/discover\/recruiterSearch(?:$|\?)/,
  project_search_advanced: /^https:\/\/www\.linkedin\.com\/talent\/hire\/(?<projectId>\d+)\/discover\/recruiterSearch\/advanced/,
  profile_drawer: /^https:\/\/www\.linkedin\.com\/talent\/hire\/(?<projectId>\d+)\/discover\/recruiterSearch\/profile\/(?<candidateId>[A-Za-z0-9_-]+)/,
  login_wall: /^https:\/\/www\.linkedin\.com\/(login|checkpoint\/)/,
  checkpoint_challenge: /^https:\/\/www\.linkedin\.com\/checkpoint\/challenge\//,
} as const;

export function extractProjectIdFromUrl(url: string): string | null {
  for (const re of [ROUTES.project_search, ROUTES.project_search_advanced, ROUTES.profile_drawer]) {
    const m = url.match(re);
    if (m?.groups?.['projectId']) return m.groups['projectId'];
  }
  return null;
}

export function extractCandidateIdFromUrl(url: string): string | null {
  return url.match(ROUTES.profile_drawer)?.groups?.['candidateId'] ?? null;
}

// ---------------------------------------------------------------------------
// B. PROJECT SEARCH (idle + populated)
// ---------------------------------------------------------------------------
export const PROJECT_SEARCH = {
  filter_pane: {
    id: 'filter_complementary_pane',
    role: 'complementary',
    name: 'Search filters and AI chat',
    classification: 'stable_now',
  },
  keyword_input: {
    id: 'keyword_input',
    role: 'combobox',
    nameRegex: /job title.*ideal candidate.*keyword.*boolean/i,
    classification: 'stable_now',
  },
  empty_search_probe: {
    id: 'empty_search_probe',
    selector: 'main :has-text("Start a search")',
    classification: 'stable_now',
  },
  results_form: {
    id: 'results_form',
    selector: 'form:has(:text-matches("Select all \\\\d+ profiles"))',
    classification: 'stable_now',
  },
  result_count_text: {
    id: 'result_count_text',
    selector: 'form :text-matches("^[\\\\d.]+[KM]?\\\\+?\\\\s+results?$", "i")',
    classification: 'stable_now',
  },
  result_card: {
    id: 'result_card',
    role: 'article',
    selector: 'role=article >> has=a[href*="/talent/profile/"]',
    classification: 'stable_now',
  },
  embedded_recommended_matches: {
    id: 'embedded_recommended_matches',
    role: 'region',
    name: 'All recommended matches',
    classification: 'stable_now',
  },
  pagination_header: {
    id: 'pagination_header',
    role: 'navigation',
    name: 'Profile list header pagination',
    classification: 'stable_now',
  },
  pagination_footer: {
    id: 'pagination_footer',
    role: 'navigation',
    name: 'Profile list pagination',
    classification: 'stable_now',
  },
  overflow_trigger: {
    id: 'overflow_menu.trigger',
    role: 'button',
    name: 'More actions',
    classification: 'stable_now',
  },
  spotlights_region: {
    id: 'spotlights_region',
    role: 'region',
    name: 'Spotlights',
    classification: 'stable_now',
  },
} satisfies Record<string, SelectorSpec>;

// ---------------------------------------------------------------------------
// C. RESULT CARD ACTION RAIL (all mutating)
// ---------------------------------------------------------------------------
export const RESULT_CARD = {
  candidate_name_link: {
    id: 'candidate_name_link',
    role: 'link',
    selector: 'a[href*="/talent/profile/"]',
    classification: 'stable_now',
  },
  save_to_pipeline: {
    id: 'save_to_pipeline',
    role: 'button',
    nameRegex: /^Save to '[^']+'$/,
    classification: 'defer',
    mutating: true,
  },
  save_stage_chooser: {
    id: 'save_stage_chooser',
    role: 'button',
    name: 'Select pipeline stage to save to',
    classification: 'defer',
    mutating: true,
  },
  hide_candidate: {
    id: 'hide_candidate',
    role: 'button',
    nameRegex: /^Hide$/,
    classification: 'defer',
    mutating: true,
  },
  message_candidate: {
    id: 'message_candidate',
    role: 'button',
    nameRegex: /^Message /,
    classification: 'defer',
    mutating: true,
  },
  more_actions: {
    id: 'more_actions',
    role: 'button',
    nameRegex: /^More actions for /,
    classification: 'defer',
    mutating: true,
  },
} satisfies Record<string, SelectorSpec>;

// ---------------------------------------------------------------------------
// D. PROFILE DRAWER
// ---------------------------------------------------------------------------
export const PROFILE_DRAWER = {
  dialog: {
    id: 'drawer',
    role: 'dialog',
    classification: 'stable_now',
  },
  from_project_heading: {
    id: 'drawer_from_project_heading',
    role: 'heading',
    nameRegex: /^From /,
    classification: 'stable_now',
  },
  position_counter: {
    id: 'drawer_position_counter',
    selector: 'text=/^\\d+ of [\\d,]+$/',
    classification: 'stable_now',
  },
  in_projects_section: {
    id: 'in_projects_section',
    role: 'heading',
    nameRegex: /^In \d+ projects?$/,
    classification: 'stable_now',
  },
  public_profile_button: {
    id: 'public_profile_resolver',
    role: 'button',
    name: 'Public profile',
    classification: 'stable_now',
  },
  public_profile_open_link: {
    id: 'public_profile_open_link',
    role: 'link',
    nameRegex: /^Open link in new tab$/,
    classification: 'stable_now',
  },
  save_to_pipeline_drawer: {
    id: 'save_to_pipeline_drawer',
    role: 'button',
    nameRegex: /^Save to '[^']+'$/,
    classification: 'defer',
    mutating: true,
  },
  save_stage_chooser_drawer: {
    id: 'save_stage_chooser_drawer',
    role: 'button',
    name: 'Select pipeline stage to save to',
    classification: 'defer',
    mutating: true,
  },
  add_email: {
    id: 'add_email',
    role: 'button',
    name: 'Add email',
    classification: 'defer',
    mutating: true,
  },
  add_phone: {
    id: 'add_phone',
    role: 'button',
    name: 'Add phone number',
    classification: 'defer',
    mutating: true,
  },
  drawer_prev_candidate: {
    id: 'drawer_prev_candidate',
    role: 'button',
    nameRegex: /Previous candidate/,
    classification: 'stable_now',
  },
  drawer_next_candidate: {
    id: 'drawer_next_candidate',
    role: 'link',
    nameRegex: /^Go forward to profile \d+$/,
    classification: 'stable_now',
  },
} satisfies Record<string, SelectorSpec>;

// ---------------------------------------------------------------------------
// E. OVERFLOW MENU
// ---------------------------------------------------------------------------
export const OVERFLOW_MENU = {
  view_saved_searches:        { id: 'view_saved_searches',        role: 'menuitem', name: 'View saved searches',        classification: 'defer' },
  save_as_new_search:         { id: 'save_as_new_search',         role: 'menuitem', name: 'Save as new search',         classification: 'defer', mutating: true },
  view_search_history:        { id: 'view_search_history',        role: 'menuitem', name: 'View search history',        classification: 'defer' },
  clear_search:               { id: 'clear_search',               role: 'menuitem', name: 'Clear search',               classification: 'defer', mutating: true },
  save_as_new_custom_filter:  { id: 'save_as_new_custom_filter',  role: 'menuitem', name: 'Save as new custom filter',  classification: 'defer', mutating: true },
  delete_custom_filters:      { id: 'delete_custom_filters',      role: 'menuitem', name: 'Delete custom filters',      classification: 'defer', mutating: true },
} satisfies Record<string, SelectorSpec>;
