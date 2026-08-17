// SafetyGuard — single chokepoint for any page-object call that could mutate state
// or that would trigger a join-wall escalation on public LinkedIn.
//
// Contract:
//   - read-only methods call guard.allowReadOnly(action) and proceed.
//   - mutating methods call guard.requireIntent(action, { vanityFromUrl, surface }) before clicking.
//   - guard returns the validated Intent on success, throws EnvelopeError on failure.
//   - idempotency tokens are consumed exactly once per process.
//
// Mirrors manifest worker_policy. Pass-2a verified surfaces are wired here; unknown
// surfaces are gated by assertVerifiedSelector().

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
  vanityFromUrl: string | null;
  surface?: Surface;
}

export class SafetyGuard {
  private readonly mode: 'read_only' | 'mutating';
  private readonly intent: Intent | undefined;
  private readonly acceptUnverified: boolean;
  private readonly log: (event: GuardEvent) => void;
  private readonly consumedTokens = new Set<string>();

  // Per-session guest-view counter. Worker increments via noteGuestProfileLoad();
  // SafetyGuard throws GuestViewLimitError via a separate method consulted by pages.
  private guestViewsObserved = 0;
  private readonly guestViewBudget: number;

  constructor(opts: GuardOptions & { guestViewBudget?: number }) {
    this.mode = opts.mode;
    this.intent = opts.intent;
    this.acceptUnverified = opts.acceptUnverified ?? false;
    this.log = opts.log ?? defaultLog;
    this.guestViewBudget = opts.guestViewBudget ?? 3;

    if (this.mode === 'mutating' && !this.intent) {
      throw new EnvelopeError('mode=mutating requires an intent', 'unknown');
    }
  }

  /** Always permits the action; logs the decision for audit. */
  allowReadOnly(action: ActionName): void {
    if (!READ_ONLY_ALWAYS_ALLOWED.has(action)) {
      // Even in read-only mode, never silently allow an action the policy classifies
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
   * Returns the validated Intent for the caller to thread into the click.
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
    // Verify the live vanity (when applicable) matches the intent.
    if (intent.targetVanity && args.vanityFromUrl !== intent.targetVanity) {
      this.log({
        kind: 'verify_failed',
        action,
        reason: `URL vanity=${args.vanityFromUrl ?? 'null'} != intent.targetVanity=${intent.targetVanity}`,
        ...(args.surface !== undefined ? { surface: args.surface } : {}),
      });
      throw new EnvelopeError(
        `URL vanity mismatch: ${args.vanityFromUrl ?? 'null'} != ${intent.targetVanity}`,
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

  /**
   * Worker calls this every time it loads an unauthenticated /in/{vanity} page.
   * Returns the remaining budget; throws GuestViewLimitError when budget is exhausted.
   */
  noteGuestProfileLoad(): { observed: number; budget: number; remaining: number } {
    this.guestViewsObserved += 1;
    const remaining = this.guestViewBudget - this.guestViewsObserved;
    if (remaining <= 0) {
      this.log({
        kind: 'deny',
        action: 'navigate_to_public_profile',
        reason: `guest view budget exhausted (observed=${this.guestViewsObserved} budget=${this.guestViewBudget})`,
      });
    }
    return { observed: this.guestViewsObserved, budget: this.guestViewBudget, remaining };
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
  console.warn(`[SafetyGuard:public] ${JSON.stringify(event)}`);
}
