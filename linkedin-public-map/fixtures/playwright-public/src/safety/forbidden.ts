// Mirror of manifests/linkedin-public-selectors.yaml → worker_policy.forbidden_in_default_mode (v1).
// Source-of-truth comparison happens at SafetyGuard construction time; keep this list synced.

import type { ActionName } from '../types.js';

export const FORBIDDEN_IN_DEFAULT_MODE: ReadonlySet<ActionName> = new Set<ActionName>([
  // Social / mutating on people
  'connect',
  'follow',
  'message',
  'send_inmail',
  'endorse',
  'recommend',
  'report',
  'block',
  // Social / mutating on content
  'post_react',
  'post_comment',
  'post_share',
  'post_repost',
  'post_save',
  'article_react',
  'article_comment',
  'article_share',
  // Jobs
  'apply_to_job',
  'im_interested_on_job',
  // Subscriptions
  'subscribe_newsletter',
  'subscribe_hashtag',
  // Modal / join-wall CTAs (specific to public LinkedIn)
  'signin_modal_email_click',
  'signin_modal_join_click',
  'header_signin_click',
  'header_join_click',
  'topcard_join_click',
  'bottom_cta_banner_click',
  'see_all_activities_click',
  // Auth-wall surface clicks
  'signin_with_apple',
  'signin_with_google',
  'signin_with_acme-software',
  'open_the_app_deeplink',
  'cold_join_form_submit',
  'public_profile_join_form_submit',
  'authwall_input_any',
  'authwall_button_any',
]);

export const READ_ONLY_ALWAYS_ALLOWED: ReadonlySet<ActionName> = new Set<ActionName>([
  'read_public_profile_dom',
  'read_public_profile_jsonld',
  'read_meta_tags',
  'read_canonical_url',
  'dismiss_signin_modal',
  'dismiss_app_toast',
  'navigate_to_public_profile',
  'navigate_to_company_landing',
  'read_url_state',
  'paginate_search_next',
]);
