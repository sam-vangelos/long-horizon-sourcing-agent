// Cloris LinkedIn Recruiter — shared types.
// Mirrors classification and policy fields from manifests/linkedin-recruiter-selectors.yaml v3.

export type Classification = 'stable_now' | 'mock_only' | 'defer' | 'unknown';

/** Surface families used to scope which intents are valid where. */
export type Surface =
  | 'project_search'
  | 'project_search_advanced'
  | 'profile_drawer'
  | 'pipeline'
  | 'unknown';

/**
 * Mutation actions Cloris may attempt. Mirrors manifest worker_policy.forbidden_in_default_mode
 * plus read-only navigation actions. Any action in the forbidden list requires an Intent
 * with humanConfirmed=true OR an explicitly configured automation policy override.
 */
export type ActionName =
  // read-only navigation (always allowed)
  | 'open_drawer'
  | 'escape_drawer'
  | 'paginate_next'
  | 'paginate_page'
  | 'open_overflow_menu_read_only'
  | 'open_stage_picker_read_only'
  | 'open_more_actions_read_only'
  | 'read_public_profile_link'
  // mutating (require Intent)
  | 'save_to_project'
  | 'save_to_stage'
  | 'remove_from_project'
  | 'move_pipeline_stage'
  | 'add_note'
  | 'edit_note'
  | 'add_tag'
  | 'add_email'
  | 'add_phone'
  | 'reject'
  | 'hide'
  | 'block'
  | 'report'
  | 'endorse'
  | 'connect'
  | 'follow'
  | 'inmail'
  | 'message'
  | 'share'
  | 'export'
  | 'bulk_action_any'
  | 'project_switcher'
  | 'apply_filter'
  | 'clear_search'
  | 'save_as_new_search'
  | 'save_as_new_custom_filter'
  | 'delete_custom_filters';

/**
 * The envelope every mutating call requires. Stage-aware saves are project-scoped by URL,
 * so `targetProjectId` is non-optional — SafetyGuard verifies URL.projectId === targetProjectId
 * before allowing the click. See manifest §E save_flow and pass-3 doc §I.1 for the algorithm.
 */
export interface Intent {
  action: ActionName;
  targetProjectId: string;
  /** Single-use token to prevent double-save on retry. SafetyGuard tracks consumed tokens in-process. */
  idempotencyToken: string;
  /** When true, human-in-the-loop has approved this specific click. */
  humanConfirmed: boolean;
  /** Optional stage when saving to a specific (non-default) pipeline stage. */
  stageName?: string;
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

/** Result of result_count_text parser. Mirrors manifest page_state.result_count_text.parse. */
export interface ResultTotal {
  total: number;
  approximate: boolean;
  unit: 'K' | 'M' | null;
}

/** "In N projects" row parser output. */
export interface ProjectMembership {
  projectName: string;
  status: 'contacted' | 'uncontacted';
}
