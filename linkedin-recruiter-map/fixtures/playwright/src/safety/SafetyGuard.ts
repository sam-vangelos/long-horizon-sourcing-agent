// SafetyGuard — single chokepoint for any page-object call that could mutate state.
//
// Contract:
//   - read-only methods call guard.allowReadOnly(action) and proceed.
//   - mutating methods call guard.requireIntent(action, { projectIdFromUrl }) before clicking.
//   - guard returns the validated Intent on success, throws EnvelopeError on failure.
//   - idempotency tokens are consumed exactly once per process.
//
// Mirrors manifest worker_policy. See pass-3-live-observations.md §I.1 for the save-safety
// algorithm this enforces.

import {
  EnvelopeError,
  type ActionName,
  type GuardEvent,
  type GuardOptions,
  type Intent,
  type Surface,
} from '../types.js';
import { FORBIDDEN_IN_DEFAULT_MODE, READ_ONLY_ALWAYS_ALLOWED } from './forbidden.js';

interface RequireIntentArgs {
  projectIdFromUrl: string | null;
  surface?: Surface;
}

export class SafetyGuard {
  private readonly mode: 'read_only' | 'mutating';
  private readonly intent: Intent | undefined;
  private readonly acceptUnverified: boolean;
  private readonly log: (event: GuardEvent) => void;
  private readonly consumedTokens = new Set<string>();

  constructor(opts: GuardOptions) {
    this.mode = opts.mode;
    this.intent = opts.intent;
    this.acceptUnverified = opts.acceptUnverified ?? false;
    this.log = opts.log ?? defaultLog;

    if (this.mode === 'mutating' && !this.intent) {
      throw new EnvelopeError('mode=mutating requires an intent', 'unknown');
    }
  }

  /** Always permits the action; logs the decision for audit. */
  allowReadOnly(action: ActionName): void {
    if (!READ_ONLY_ALWAYS_ALLOWED.has(action)) {
      // Even in read-only mode, never silently allow an action that the policy classifies
      // as mutating. The caller used the wrong helper.
      throw new EnvelopeError(
        `action '${action}' is not read-only; use requireIntent()`,
        action,
      );
    }
    this.log({ kind: 'allow', action });
  }

  /**
   * Verifies the configured Intent covers `action` and matches the live page state.
   * Returns the validated Intent for the caller to thread into the click (e.g. for
   * idempotency-token-in-post-data scenarios, future).
   */
  requireIntent(action: ActionName, args: RequireIntentArgs): Intent {
    if (this.mode !== 'mutating') {
      this.log({ kind: 'deny', action, reason: 'guard mode is read_only' });
      throw new EnvelopeError('guard mode is read_only', action);
    }
    const intent = this.intent;
    if (!intent) {
      this.log({ kind: 'deny', action, reason: 'no intent attached' });
      throw new EnvelopeError('no intent attached', action);
    }
    if (FORBIDDEN_IN_DEFAULT_MODE.has(action) && intent.action !== action) {
      this.log({ kind: 'deny', action, reason: `intent.action mismatch (was ${intent.action})` });
      throw new EnvelopeError(`intent.action mismatch (was ${intent.action})`, action);
    }
    if (!intent.humanConfirmed) {
      this.log({ kind: 'deny', action, reason: 'humanConfirmed is false' });
      throw new EnvelopeError('humanConfirmed is false', action);
    }
    if (args.projectIdFromUrl !== intent.targetProjectId) {
      this.log({
        kind: 'verify_failed',
        action,
        reason: `URL projectId=${args.projectIdFromUrl ?? 'null'} != intent.targetProjectId=${intent.targetProjectId}`,
        ...(args.surface !== undefined ? { surface: args.surface } : {}),
      });
      throw new EnvelopeError(
        `URL projectId mismatch: ${args.projectIdFromUrl ?? 'null'} != ${intent.targetProjectId}`,
        action,
      );
    }
    if (this.consumedTokens.has(intent.idempotencyToken)) {
      this.log({ kind: 'deny', action, reason: 'idempotency token already consumed' });
      throw new EnvelopeError('idempotency token already consumed', action);
    }
    this.consumedTokens.add(intent.idempotencyToken);
    this.log({ kind: 'allow', action, ...(args.surface !== undefined ? { surface: args.surface } : {}) });
    return intent;
  }

  /** Gate for interacting with `classification: unknown` selectors. */
  assertVerifiedSelector(selectorId: string, classification: string): void {
    if (classification === 'unknown' && !this.acceptUnverified) {
      this.log({ kind: 'unverified_blocked', action: 'unknown', reason: selectorId });
      throw new EnvelopeError(`selector '${selectorId}' is unverified`, 'unknown');
    }
  }

  /** Helper for callers that need to know whether they're allowed to mutate at all. */
  get isMutating(): boolean {
    return this.mode === 'mutating';
  }
}

function defaultLog(event: GuardEvent): void {
  // Use console.warn so it surfaces in CI logs without being stripped by test reporters
  // that swallow console.log.
  // eslint-disable-next-line no-console
  console.warn(`[SafetyGuard] ${JSON.stringify(event)}`);
}
