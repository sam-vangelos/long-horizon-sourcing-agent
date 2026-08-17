// Cloris LinkedIn Public (linkedin.com fallback) — shared types.
// Mirrors classification and policy fields from manifests/linkedin-public-selectors.yaml v1.
//
// Naming convention: this file is intentionally parallel to the Recruiter fixtures' types.ts.
// SafetyGuard, Intent, EnvelopeError semantics are identical; the surface taxonomy and
// the action vocabulary are different.

export type Classification = 'stable_now' | 'mock_only' | 'defer' | 'unknown';

/** Surface families for public LinkedIn. */
export type Surface =
  | 'public_profile'
  | 'public_profile_details_experience'
  | 'public_profile_details_education'
  | 'public_profile_details_skills'
  | 'public_profile_overlay_contact_info'
  | 'public_profile_recent_activity'
  | 'company_landing'
  | 'company_people'
  | 'pub_dir_legacy'
  | 'search_results_people'
  | 'posts_activity'
  | 'pulse_article'
  | 'authwall_join'
  | 'authwall_signin'
  | 'checkpoint'
  | 'signup_forbidden'
  | 'unknown_surface';

/**
 * Action vocabulary for the public-LinkedIn worker. Two buckets:
 *   - read-only: always allowed in default mode
 *   - mutating:  requires Intent with humanConfirmed=true (see SafetyGuard)
 *
 * The mutating set is much smaller than Recruiter (no project/pipeline concept),
 * but the per-action trust signals are higher (Connect / Follow / React are
 * heavily monitored by LinkedIn's anti-automation systems).
 */
export type ActionName =
  // read-only / always allowed
  | 'read_public_profile_dom'
  | 'read_public_profile_jsonld'
  | 'read_meta_tags'
  | 'read_canonical_url'
  | 'dismiss_signin_modal'
  | 'dismiss_app_toast'
  | 'navigate_to_public_profile'
  | 'navigate_to_company_landing'
  | 'read_url_state'
  | 'paginate_search_next'        // allowed read-only only in authenticated mode
  // mutating (require Intent envelope)
  | 'connect'
  | 'follow'
  | 'message'
  | 'send_inmail'
  | 'endorse'
  | 'recommend'
  | 'report'
  | 'block'
  | 'post_react'
  | 'post_comment'
  | 'post_share'
  | 'post_repost'
  | 'post_save'
  | 'article_react'
  | 'article_comment'
  | 'article_share'
  | 'apply_to_job'
  | 'im_interested_on_job'
  | 'subscribe_newsletter'
  | 'subscribe_hashtag'
  | 'signin_modal_email_click'
  | 'signin_modal_join_click'
  | 'header_signin_click'
  | 'header_join_click'
  | 'topcard_join_click'
  | 'bottom_cta_banner_click'
  | 'see_all_activities_click'
  | 'signin_with_apple'
  | 'signin_with_google'
  | 'signin_with_acme-software'
  | 'open_the_app_deeplink'
  | 'cold_join_form_submit'
  | 'public_profile_join_form_submit'
  | 'authwall_input_any'
  | 'authwall_button_any';

/**
 * The envelope every mutating call requires. Unlike Recruiter (which uses
 * targetProjectId from URL), public LinkedIn mutations are scoped to either:
 *   - a specific person (targetVanity / targetMemberUrn), OR
 *   - a specific content item (targetPostPath / targetArticleSlug).
 * SafetyGuard verifies the live URL or DOM-extracted vanity matches the intent.
 */
export interface Intent {
  action: ActionName;
  /** Vanity slug of the candidate the action affects (e.g. 'jordanrivera'). */
  targetVanity?: string;
  /** Opaque member URN (e.g. 'urn:li:fs_profile:ACoA...') when known. */
  targetMemberUrn?: string;
  /** Post or article path when action is content-scoped. */
  targetContentPath?: string;
  /** Single-use token to prevent double-fire on retry. SafetyGuard tracks consumed tokens in-process. */
  idempotencyToken: string;
  /** When true, human-in-the-loop has approved this specific click. */
  humanConfirmed: boolean;
  /** Optional surface override for actions that span multiple surfaces. */
  surface?: Surface;
  /** Free-form metadata for audit logging. */
  notes?: string;
}

export interface GuardOptions {
  mode: 'read_only' | 'mutating';
  intent?: Intent;
  /** When true, page objects may interact with `unknown`-classification selectors. Default false. */
  acceptUnverified?: boolean;
  /** Logger used for envelope decisions. Defaults to console.warn. */
  log?: (event: GuardEvent) => void;
}

export interface GuardEvent {
  kind: 'allow' | 'deny' | 'verify_failed' | 'unverified_blocked';
  action: ActionName | 'unknown';
  reason?: string;
  url?: string;
  surface?: Surface;
}

export class EnvelopeError extends Error {
  constructor(public readonly reason: string, public readonly action: ActionName | 'unknown') {
    super(`[envelope] ${action}: ${reason}`);
    this.name = 'EnvelopeError';
  }
}

export class UnverifiedSelectorError extends Error {
  constructor(public readonly selectorId: string) {
    super(`[unverified] selector '${selectorId}' is classification=unknown and acceptUnverified=false`);
    this.name = 'UnverifiedSelectorError';
  }
}

export class RecoveryError extends Error {
  constructor(public readonly signal: string, public readonly recommendedAction: string) {
    super(`[recovery] ${signal} → ${recommendedAction}`);
    this.name = 'RecoveryError';
  }
}

/** Specialization of RecoveryError for the /authwall (Join LinkedIn) template. */
export class AuthWallJoinError extends RecoveryError {
  constructor(public readonly trk: 'bf' | 'gf' | string | null, sessionRedirect: string | null) {
    super('recovery.authwall_join', 'halt_emit_reauth');
    this.name = 'AuthWallJoinError';
    this.message = `[recovery] authwall_join trk=${trk ?? 'null'} sessionRedirect=${sessionRedirect ?? 'null'}`;
  }
}

/** Specialization of RecoveryError for the /uas/login template. */
export class AuthWallSignInError extends RecoveryError {
  constructor(sessionRedirect: string | null) {
    super('recovery.authwall_signin', 'halt_emit_reauth');
    this.name = 'AuthWallSignInError';
    this.message = `[recovery] authwall_signin sessionRedirect=${sessionRedirect ?? 'null'}`;
  }
}

/** Specialization of RecoveryError for the guest-view limit (3-5 profiles per session). */
export class GuestViewLimitError extends RecoveryError {
  constructor(public readonly observedViews: number, public readonly budget: number) {
    super('recovery.guest_view_limit', 'halt_cooldown_30min');
    this.name = 'GuestViewLimitError';
    this.message = `[recovery] guest_view_limit observed=${observedViews} budget=${budget}`;
  }
}

/** Specialization of RecoveryError for /checkpoint/challenge/ — IP/account flagged. */
export class BlockedError extends RecoveryError {
  constructor(public readonly challengeUrl: string) {
    super('recovery.blocked', 'halt_cooldown_24h');
    this.name = 'BlockedError';
    this.message = `[recovery] blocked challengeUrl=${challengeUrl}`;
  }
}

/** Output of the JSON-LD Person extractor — superset of common public-profile fields. */
export interface PersonJsonLd {
  '@type'?: string;
  name?: string;
  url?: string;
  jobTitle?: string;
  description?: string;
  image?: string | { url?: string };
  address?: { addressLocality?: string; addressRegion?: string; addressCountry?: string };
  worksFor?: Array<{ name?: string; member?: unknown; url?: string }>;
  alumniOf?: Array<{ name?: string; url?: string }>;
  sameAs?: string[];
  /** Any unknown additional fields. */
  [k: string]: unknown;
}

/** Parsed top-card metadata extracted from /in/{vanity} DOM. */
export interface PublicProfileTopCard {
  vanity: string;
  name: string;
  headline: string;
  location: string | null;
  followers: string | null;       // raw text e.g. "12K"
  connections: string | null;     // raw text e.g. "500"
  currentCompanySlug: string | null;
  currentSchoolSlug: string | null;
  photoUrl: string | null;
}

/** A single experience item extracted from the public profile DOM. */
export interface ExperienceItem {
  title: string | null;
  companyName: string | null;
  companySlug: string | null;
  dateRangeText: string | null;
  durationText: string | null;
}

/** A single education item extracted from the public profile DOM. */
export interface EducationItem {
  schoolName: string | null;
  schoolSlug: string | null;
  degree: string | null;
  dateRangeText: string | null;
}
