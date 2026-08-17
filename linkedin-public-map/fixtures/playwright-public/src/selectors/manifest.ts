// Tiny typed wrapper around the YAML manifest, scoped to the rows the page objects use.
// We don't autogenerate the whole tree — Pass 3+ will. For now, hand-curated views keep
// the surface area small and the test diffs human-readable.
//
// To add a selector: copy its row from the YAML, paste under the matching surface below.
// Each entry carries `classification` so SafetyGuard can gate `unknown` rows.

export interface SelectorSpec {
  /** Manifest row id (e.g. 'name_heading'). */
  id: string;
  /** ARIA role hint, when available. */
  role?: string;
  /** Accessible name or regex source string. */
  name?: string;
  nameRegex?: RegExp;
  /** Substring to match against an anchor's href (preferred over class on public LI). */
  hrefIncludes?: string;
  hrefRegex?: RegExp;
  /** Playwright selector expression — fallback when role+name+href is insufficient. */
  selector?: string;
  classification: 'stable_now' | 'mock_only' | 'defer' | 'unknown';
  /** True if this row's mutation column is non-null/non-read_only in the manifest. */
  mutating?: boolean;
}

// ---------------------------------------------------------------------------
// A. ROUTES
// ---------------------------------------------------------------------------
export const ROUTES = {
  public_profile: /^https:\/\/www\.linkedin\.com\/in\/(?<vanity>[A-Za-z0-9_%.\-]+)\/?(?:\?|$)/,
  public_profile_details_experience:
    /^https:\/\/www\.linkedin\.com\/in\/(?<vanity>[A-Za-z0-9_%.\-]+)\/details\/experience\/?(?:\?|$)/,
  public_profile_details_education:
    /^https:\/\/www\.linkedin\.com\/in\/(?<vanity>[A-Za-z0-9_%.\-]+)\/details\/education\/?(?:\?|$)/,
  public_profile_details_skills:
    /^https:\/\/www\.linkedin\.com\/in\/(?<vanity>[A-Za-z0-9_%.\-]+)\/details\/skills\/?(?:\?|$)/,
  public_profile_overlay_contact_info:
    /^https:\/\/www\.linkedin\.com\/in\/(?<vanity>[A-Za-z0-9_%.\-]+)\/overlay\/contact-info\/?(?:\?|$)/,
  company_landing: /^https:\/\/www\.linkedin\.com\/company\/(?<slug>[A-Za-z0-9\-_]+)\/?(?:\?|$)/,
  company_people: /^https:\/\/www\.linkedin\.com\/company\/(?<slug>[A-Za-z0-9\-_]+)\/people\/?(?:\?|$)/,
  pub_dir_legacy: /^https:\/\/www\.linkedin\.com\/pub\/dir\//,
  search_results_people: /^https:\/\/www\.linkedin\.com\/search\/results\/people\/?\?/,
  authwall_join: /^https:\/\/www\.linkedin\.com\/authwall(?:\?|$)/,
  authwall_signin: /^https:\/\/www\.linkedin\.com\/uas\/login(?:\?|$)/,
  checkpoint: /^https:\/\/www\.linkedin\.com\/checkpoint\//,
  signup_cold_join: /^https:\/\/www\.linkedin\.com\/signup\/cold-join(?:\?|$)/,
  signup_public_profile_join: /^https:\/\/www\.linkedin\.com\/signup\/public-profile-join(?:\?|$)/,
} as const;

export function extractVanityFromUrl(url: string): string | null {
  const m = url.match(ROUTES.public_profile);
  return m?.groups?.['vanity'] ?? null;
}

export function extractCompanySlugFromUrl(url: string): string | null {
  const m = url.match(ROUTES.company_landing) ?? url.match(ROUTES.company_people);
  return m?.groups?.['slug'] ?? null;
}

export function extractAuthWallTrk(url: string): string | null {
  if (!ROUTES.authwall_join.test(url)) return null;
  try {
    return new URL(url).searchParams.get('trk');
  } catch {
    return null;
  }
}

export function extractSessionRedirect(url: string): string | null {
  try {
    const sp = new URL(url).searchParams;
    return sp.get('sessionRedirect') || sp.get('session_redirect');
  } catch {
    return null;
  }
}

// ---------------------------------------------------------------------------
// B. PUBLIC PROFILE — Pass 2a verified
// ---------------------------------------------------------------------------
export const PUBLIC_PROFILE = {
  name_heading: {
    id: 'name_heading',
    role: 'heading',
    selector: 'main h1',
    classification: 'stable_now',
  },
  headline_heading: {
    id: 'headline_heading',
    role: 'heading',
    // First h2 inside main, immediately after the h1.
    selector: 'main h2 >> nth=0',
    classification: 'stable_now',
  },
  location_followers_line: {
    id: 'location_followers_connections_line',
    role: 'heading',
    selector: 'main h2 >> nth=1',
    classification: 'stable_now',
  },
  photo_button: {
    id: 'profile_photo',
    selector: 'main button[aria-label] > img[alt]',
    classification: 'stable_now',
  },
  topcard_current_company_link: {
    id: 'top_card_current_company_link',
    role: 'link',
    hrefIncludes: 'trk=public_profile_topcard-current-company',
    classification: 'stable_now',
  },
  topcard_school_link: {
    id: 'top_card_school_link',
    role: 'link',
    hrefIncludes: 'trk=public_profile_topcard-school',
    classification: 'stable_now',
  },
  about_heading: {
    id: 'about_heading',
    role: 'heading',
    selector: 'section:has(h2:has-text("About")) h2',
    classification: 'stable_now',
  },
  experience_heading: {
    id: 'experience_section_heading',
    role: 'heading',
    selector: 'section:has(h2:has-text("Experience")) h2',
    classification: 'stable_now',
  },
  experience_item_anchor: {
    id: 'experience_item_anchor',
    role: 'link',
    hrefIncludes: 'trk=public_profile_experience-item_profile-section-card_image-click',
    classification: 'stable_now',
  },
  education_heading: {
    id: 'education_section_heading',
    role: 'heading',
    selector: 'section:has(h2:has-text("Education")) h2',
    classification: 'stable_now',
  },
  education_item_anchor: {
    id: 'education_item_anchor',
    role: 'link',
    hrefIncludes: 'trk=public_profile_school_profile-section-card_image-click',
    classification: 'stable_now',
  },
  activity_heading: {
    id: 'activity_heading',
    role: 'heading',
    selector: 'section:has(h2:has-text("Activity")) h2:has-text("Activity")',
    classification: 'stable_now',
  },
  articles_heading: {
    id: 'articles_heading',
    role: 'heading',
    nameRegex: /^Articles by /i,
    classification: 'stable_now',
  },
  signin_modal: {
    id: 'signin_modal_dialog',
    role: 'dialog',
    // Heading inside the dialog matches /^View .+'s full profile$/.
    selector: 'role=dialog',
    classification: 'stable_now',
  },
  signin_modal_heading: {
    id: 'signin_modal_heading',
    role: 'heading',
    nameRegex: /^View .+'s full profile$/,
    classification: 'stable_now',
  },
  signin_modal_email_button: {
    id: 'signin_modal_email_button',
    role: 'button',
    nameRegex: /^Sign in with Email$/,
    classification: 'stable_now',
    mutating: true,
  },
  signin_modal_join_link: {
    id: 'signin_modal_join_link',
    role: 'link',
    nameRegex: /^Join now$/,
    classification: 'stable_now',
    mutating: true,
  },
  header_signin_link: {
    id: 'header_signin_link',
    role: 'link',
    hrefIncludes: 'trk=public_profile_nav-header-signin',
    classification: 'stable_now',
    mutating: true,
  },
  header_join_link: {
    id: 'header_join_link',
    role: 'link',
    hrefIncludes: 'trk=public_profile_nav-header-join',
    classification: 'stable_now',
    mutating: true,
  },
  topcard_join_button: {
    id: 'topcard_join_button',
    role: 'link',
    hrefIncludes: 'trk=public_profile_top-card-primary-button-join-to-view-profile',
    classification: 'stable_now',
    mutating: true,
  },
  bottom_cta_banner: {
    id: 'bottom_cta_banner',
    role: 'link',
    hrefIncludes: 'trk=public_profile_bottom-cta-banner',
    classification: 'stable_now',
    mutating: true,
  },
} satisfies Record<string, SelectorSpec>;

// ---------------------------------------------------------------------------
// C. AUTHWALL JOIN (/authwall) — Pass 2a verified
// ---------------------------------------------------------------------------
export const AUTHWALL_JOIN = {
  heading: {
    id: 'authwall_join.heading',
    role: 'heading',
    nameRegex: /^Join LinkedIn$/,
    classification: 'stable_now',
  },
  email_input: {
    id: 'authwall_join.email_input',
    role: 'textbox',
    nameRegex: /^Email$/,
    classification: 'stable_now',
    mutating: true,
  },
  password_input: {
    id: 'authwall_join.password_input',
    role: 'textbox',
    nameRegex: /^Password \(6\+ characters\)$/,
    classification: 'stable_now',
    mutating: true,
  },
  agree_join_button: {
    id: 'authwall_join.agree_join_button',
    role: 'button',
    nameRegex: /^Agree & Join$/,
    classification: 'stable_now',
    mutating: true,
  },
  signin_pivot_button: {
    id: 'authwall_join.signin_pivot_button',
    role: 'button',
    nameRegex: /^Sign in$/,
    classification: 'stable_now',
    mutating: true,
  },
  google_iframe: {
    id: 'authwall_join.google_iframe',
    selector: 'iframe[title="Sign in with Google Button"]',
    classification: 'stable_now',
    mutating: true,
  },
  app_upsell_dismiss: {
    id: 'authwall_join.app_upsell_dismiss',
    role: 'button',
    nameRegex: /^Dismiss$/,
    classification: 'stable_now',
  },
} satisfies Record<string, SelectorSpec>;

// ---------------------------------------------------------------------------
// D. AUTHWALL SIGN-IN (/uas/login) — Pass 2a verified
// ---------------------------------------------------------------------------
export const AUTHWALL_SIGNIN = {
  heading: {
    id: 'authwall_signin.heading',
    role: 'heading',
    nameRegex: /^Sign in$/,
    classification: 'stable_now',
  },
  email_input: {
    id: 'authwall_signin.email_input',
    role: 'textbox',
    nameRegex: /^Email or phone$/,
    classification: 'stable_now',
    mutating: true,
  },
  password_input: {
    id: 'authwall_signin.password_input',
    role: 'textbox',
    nameRegex: /^Password$/,
    classification: 'stable_now',
    mutating: true,
  },
  signin_submit: {
    id: 'authwall_signin.signin_submit',
    role: 'button',
    nameRegex: /^Sign in$/,
    classification: 'stable_now',
    mutating: true,
  },
  apple_signin: {
    id: 'authwall_signin.apple_signin',
    role: 'button',
    nameRegex: /^Sign in with Apple$/,
    classification: 'stable_now',
    mutating: true,
  },
  google_iframe: {
    id: 'authwall_signin.google_iframe',
    selector: 'iframe[title="Sign in with Google Button"]',
    classification: 'stable_now',
    mutating: true,
  },
  acme-software_iframe: {
    id: 'authwall_signin.acme-software_iframe',
    selector: 'iframe[title="Sign in with Microsoft button"]',
    classification: 'stable_now',
    mutating: true,
  },
  forgot_password_link: {
    id: 'authwall_signin.forgot_password_link',
    role: 'link',
    nameRegex: /^Forgot password\?$/,
    classification: 'stable_now',
  },
  join_now_link: {
    id: 'authwall_signin.join_now_link',
    role: 'link',
    nameRegex: /^Join now$/,
    classification: 'stable_now',
    mutating: true,
  },
} satisfies Record<string, SelectorSpec>;

// ---------------------------------------------------------------------------
// E. SEARCH RESULTS (/search/results/people/) — all UNKNOWN until Pass 2b
// ---------------------------------------------------------------------------
export const SEARCH_RESULTS_PEOPLE = {
  keyword_input: {
    id: 'search_results_people.keyword_input',
    role: 'combobox',
    classification: 'unknown',
  },
  result_card: {
    id: 'search_results_people.result_card',
    classification: 'unknown',
  },
  result_card_name_link: {
    id: 'search_results_people.result_card_name_link',
    role: 'link',
    classification: 'unknown',
  },
  pagination_next: {
    id: 'search_results_people.pagination_next',
    role: 'button',
    nameRegex: /^Next$/,
    classification: 'unknown',
  },
  result_count_text: {
    id: 'search_results_people.result_count_text',
    classification: 'unknown',
  },
  connect_button: {
    id: 'search_results_people.result_card_connect',
    role: 'button',
    nameRegex: /^Connect/,
    classification: 'unknown',
    mutating: true,
  },
  message_button: {
    id: 'search_results_people.result_card_message',
    role: 'button',
    nameRegex: /^Message/,
    classification: 'unknown',
    mutating: true,
  },
} satisfies Record<string, SelectorSpec>;

// ---------------------------------------------------------------------------
// F. COMPANY PEOPLE (/company/{slug}/people/) — all UNKNOWN until Pass 2b
// ---------------------------------------------------------------------------
export const COMPANY_PEOPLE = {
  company_name_heading: { id: 'company_people.company_name_heading', role: 'heading', classification: 'unknown' },
  headcount_line: { id: 'company_people.headcount_line', classification: 'unknown' },
  employee_card: { id: 'company_people.employee_card', classification: 'unknown' },
  employee_card_name_link: { id: 'company_people.employee_card_name_link', role: 'link', classification: 'unknown' },
  employee_card_connect: {
    id: 'company_people.employee_card_connect',
    role: 'button',
    nameRegex: /^Connect/,
    classification: 'unknown',
    mutating: true,
  },
  show_more_pagination: { id: 'company_people.show_more_pagination', role: 'button', nameRegex: /^Show more/, classification: 'unknown' },
} satisfies Record<string, SelectorSpec>;

// ---------------------------------------------------------------------------
// G. STATIC URN MAPS (worker uses these to build /search/results/people/ URLs)
// ---------------------------------------------------------------------------
export const GEO_URNS: Record<string, number> = {
  US: 103644278,
  UK: 101165590,
  Canada: 101174742,
  Germany: 101282230,
  France: 105015875,
  Netherlands: 102890719,
  Spain: 105646813,
  Brazil: 106057199,
  India: 102713980,
  Australia: 101452733,
  Mexico: 103323778,
  Japan: 101355337,
  Singapore: 102454443,
  UAE: 104305776,
};

export const INDUSTRY_URNS: Record<string, number> = {
  ComputerSoftware: 4,
  InformationTechnologyAndServices: 96,
  InternetSoftwareAndServices: 6,
  ComputerHardware: 3,
  ComputerNetworking: 5,
  TelecommunicationsServices: 8,
  ResearchServices: 70,
  HigherEducation: 68,
  FinancialServices: 43,
  Banking: 41,
  VentureCapitalAndPrivateEquity: 106,
  ManagementConsulting: 11,
};

export const NETWORK_CODES = {
  first_degree: 'F',
  second_degree: 'S',
  third_plus_degree: 'O',
} as const;
