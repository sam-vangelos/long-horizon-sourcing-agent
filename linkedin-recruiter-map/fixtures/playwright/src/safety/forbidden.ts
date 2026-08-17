// Mirror of manifests/linkedin-recruiter-selectors.yaml → worker_policy.forbidden_in_default_mode (v3).
// Source-of-truth comparison happens at SafetyGuard construction time; keep this list synced.

import type { ActionName } from '../types.js';

export const FORBIDDEN_IN_DEFAULT_MODE: ReadonlySet<ActionName> = new Set<ActionName>([
  'save_to_project',
  'save_to_stage',
  'remove_from_project',
  'move_pipeline_stage',
  'add_note',
  'edit_note',
  'add_tag',
  'add_email',
  'add_phone',
  'reject',
  'hide',
  'block',
  'report',
  'endorse',
  'connect',
  'follow',
  'inmail',
  'message',
  'share',
  'export',
  'bulk_action_any',
  'project_switcher',
  'apply_filter',
  'clear_search',
  'save_as_new_search',
  'save_as_new_custom_filter',
  'delete_custom_filters',
]);

export const READ_ONLY_ALWAYS_ALLOWED: ReadonlySet<ActionName> = new Set<ActionName>([
  'open_drawer',
  'escape_drawer',
  'paginate_next',
  'paginate_page',
  'open_overflow_menu_read_only',
  'open_stage_picker_read_only',
  'open_more_actions_read_only',
  'read_public_profile_link',
]);
