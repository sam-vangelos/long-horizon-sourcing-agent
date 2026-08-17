// Offline unit tests for detectFromUrl() — the pure-function URL classifier
// that drives every recovery branch the worker loop has to handle.

import { expect, test } from '@playwright/test';
import { detectFromUrl } from '../src/pages/recovery/RecoverySignals.js';

test.describe('RecoverySignals.detectFromUrl', () => {
  test('classifies /authwall?trk=bf as authwall_join', () => {
    const d = detectFromUrl(
      'https://www.linkedin.com/authwall?trk=bf&trkInfo=AQH&sessionRedirect=https%3A%2F%2Fwww.linkedin.com%2Fcompany%2Facme-software%2Fpeople%2F',
    );
    expect(d).not.toBeNull();
    expect(d!.signal).toBe('recovery.authwall_join');
    expect(d!.surface).toBe('authwall_join');
    expect(d!.recommendedAction).toBe('halt_emit_reauth');
  });

  test('classifies /authwall?trk=gf as authwall_join', () => {
    const d = detectFromUrl('https://www.linkedin.com/authwall?trk=gf&original_referer=');
    expect(d).not.toBeNull();
    expect(d!.signal).toBe('recovery.authwall_join');
    expect(d!.surface).toBe('authwall_join');
  });

  test('classifies /uas/login as authwall_signin', () => {
    const d = detectFromUrl(
      'https://www.linkedin.com/uas/login?session_redirect=%2Fsearch%2Fresults%2Fpeople%2F%3Fkeywords%3Dengineer',
    );
    expect(d).not.toBeNull();
    expect(d!.signal).toBe('recovery.authwall_signin');
    expect(d!.surface).toBe('authwall_signin');
    expect(d!.recommendedAction).toBe('halt_emit_reauth');
  });

  test('classifies /checkpoint/challenge/* as blocked', () => {
    const d = detectFromUrl(
      'https://www.linkedin.com/checkpoint/challenge/AgF7?ut=2_abc',
    );
    expect(d).not.toBeNull();
    expect(d!.signal).toBe('recovery.blocked');
    expect(d!.recommendedAction).toBe('halt_cooldown_24h');
  });

  test('classifies non-challenge /checkpoint/* as authwall_signin', () => {
    const d = detectFromUrl('https://www.linkedin.com/checkpoint/rm/sign-in');
    expect(d).not.toBeNull();
    expect(d!.signal).toBe('recovery.authwall_signin');
    expect(d!.surface).toBe('checkpoint');
  });

  test('classifies /signup/cold-join as authwall_join', () => {
    const d = detectFromUrl('https://www.linkedin.com/signup/cold-join?trk=guest_homepage-basic_nav-header-join');
    expect(d).not.toBeNull();
    expect(d!.signal).toBe('recovery.authwall_join');
    expect(d!.surface).toBe('signup_forbidden');
  });

  test('returns null for a known-good public profile URL', () => {
    expect(detectFromUrl('https://www.linkedin.com/in/jordanrivera/')).toBeNull();
    expect(detectFromUrl('https://www.linkedin.com/in/satyanadella')).toBeNull();
  });

  test('returns null for the company landing page', () => {
    expect(detectFromUrl('https://www.linkedin.com/company/acme-software/')).toBeNull();
  });
});
