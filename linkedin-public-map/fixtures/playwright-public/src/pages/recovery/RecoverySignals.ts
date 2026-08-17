// RecoverySignals — pure functions that classify the live URL/DOM into one of
// the canonical recovery.* signals defined in manifests/linkedin-public-selectors.yaml
// → worker_policy.recovery_signals.
//
// Pages call detect() before/after any navigation; on a non-null result they throw
// the matching RecoveryError subclass so the worker loop can route into its
// recovery branch.

import type { Page } from '@playwright/test';
import {
  AuthWallJoinError,
  AuthWallSignInError,
  BlockedError,
  RecoveryError,
} from '../../types.js';
import { ROUTES, extractAuthWallTrk, extractSessionRedirect } from '../../selectors/manifest.js';

export interface DetectResult {
  signal: string;
  recommendedAction: string;
  surface: string | null;
  url: string;
}

/** Snap-classify the live URL. Returns null when on a known good surface. */
export function detectFromUrl(url: string): DetectResult | null {
  if (ROUTES.authwall_join.test(url)) {
    const trk = extractAuthWallTrk(url);
    return {
      signal: 'recovery.authwall_join',
      recommendedAction: 'halt_emit_reauth',
      surface: 'authwall_join',
      url,
    };
  }
  if (ROUTES.authwall_signin.test(url)) {
    return {
      signal: 'recovery.authwall_signin',
      recommendedAction: 'halt_emit_reauth',
      surface: 'authwall_signin',
      url,
    };
  }
  if (/^https:\/\/www\.linkedin\.com\/checkpoint\/challenge\//.test(url)) {
    return {
      signal: 'recovery.blocked',
      recommendedAction: 'halt_cooldown_24h',
      surface: 'checkpoint',
      url,
    };
  }
  if (ROUTES.checkpoint.test(url)) {
    return {
      signal: 'recovery.authwall_signin',
      recommendedAction: 'halt_emit_reauth',
      surface: 'checkpoint',
      url,
    };
  }
  if (ROUTES.signup_cold_join.test(url) || ROUTES.signup_public_profile_join.test(url)) {
    return {
      signal: 'recovery.authwall_join',
      recommendedAction: 'halt_emit_reauth',
      surface: 'signup_forbidden',
      url,
    };
  }
  return null;
}

/**
 * Async check that combines URL classification with a DOM probe for the
 * "modal not dismissible" guest-view-limit symptom.
 * Throws the appropriate RecoveryError on detection.
 */
export async function assertOnSurface(page: Page): Promise<void> {
  const url = page.url();
  const detected = detectFromUrl(url);
  if (!detected) return;
  const sessionRedirect = extractSessionRedirect(url);
  switch (detected.signal) {
    case 'recovery.authwall_join': {
      const trk = extractAuthWallTrk(url) ?? null;
      throw new AuthWallJoinError(trk, sessionRedirect);
    }
    case 'recovery.authwall_signin':
      throw new AuthWallSignInError(sessionRedirect);
    case 'recovery.blocked':
      throw new BlockedError(url);
    default:
      throw new RecoveryError(detected.signal, detected.recommendedAction);
  }
}
