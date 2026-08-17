// Offline unit tests for SafetyGuard semantics on the public LinkedIn fallback path.
//
// These tests never touch a browser and never instantiate a Page — they exercise
// the envelope contract directly. Run with:
//   PLAYWRIGHT_MODE=offline npx playwright test tests/safety-guard.spec.ts

import { expect, test } from '@playwright/test';
import { SafetyGuard } from '../src/safety/SafetyGuard.js';
import { EnvelopeError, type Intent } from '../src/types.js';

function makeIntent(overrides: Partial<Intent> = {}): Intent {
  return {
    action: 'connect',
    targetVanity: 'jordanrivera',
    idempotencyToken: 'tok-' + Math.random().toString(36).slice(2),
    humanConfirmed: true,
    ...overrides,
  };
}

test.describe('SafetyGuard — read-only mode', () => {
  test('allows known read-only actions', () => {
    const events: any[] = [];
    const g = new SafetyGuard({ mode: 'read_only', log: (e) => events.push(e) });
    expect(() => g.allowReadOnly('read_public_profile_dom')).not.toThrow();
    expect(() => g.allowReadOnly('read_public_profile_jsonld')).not.toThrow();
    expect(() => g.allowReadOnly('navigate_to_public_profile')).not.toThrow();
    expect(events.every((e) => e.kind === 'allow')).toBe(true);
  });

  test('rejects mutating actions even via allowReadOnly()', () => {
    const g = new SafetyGuard({ mode: 'read_only' });
    expect(() => g.allowReadOnly('connect' as any)).toThrow(EnvelopeError);
    expect(() => g.allowReadOnly('follow' as any)).toThrow(EnvelopeError);
    expect(() => g.allowReadOnly('message' as any)).toThrow(EnvelopeError);
  });

  test('requireIntent() throws when mode=read_only', () => {
    const g = new SafetyGuard({ mode: 'read_only' });
    expect(() =>
      g.requireIntent('connect', { vanityFromUrl: 'jordanrivera' }),
    ).toThrow(/guard mode is read_only/);
  });
});

test.describe('SafetyGuard — mutating mode envelope', () => {
  test('constructor throws when mode=mutating but no intent attached', () => {
    expect(() => new SafetyGuard({ mode: 'mutating' })).toThrow(EnvelopeError);
  });

  test('allows action matching intent.action with humanConfirmed=true and matching vanity', () => {
    const intent = makeIntent({ action: 'connect', targetVanity: 'jordanrivera' });
    const g = new SafetyGuard({ mode: 'mutating', intent });
    const result = g.requireIntent('connect', { vanityFromUrl: 'jordanrivera' });
    expect(result.idempotencyToken).toBe(intent.idempotencyToken);
  });

  test('rejects when intent.humanConfirmed=false', () => {
    const intent = makeIntent({ humanConfirmed: false });
    const g = new SafetyGuard({ mode: 'mutating', intent });
    expect(() =>
      g.requireIntent('connect', { vanityFromUrl: 'jordanrivera' }),
    ).toThrow(/humanConfirmed is false/);
  });

  test('rejects when intent.action mismatches requested action', () => {
    const intent = makeIntent({ action: 'connect' });
    const g = new SafetyGuard({ mode: 'mutating', intent });
    expect(() =>
      g.requireIntent('follow', { vanityFromUrl: 'jordanrivera' }),
    ).toThrow(/intent\.action mismatch/);
  });

  test('rejects when live URL vanity does not match intent.targetVanity', () => {
    const intent = makeIntent({ targetVanity: 'jordanrivera' });
    const g = new SafetyGuard({ mode: 'mutating', intent });
    expect(() =>
      g.requireIntent('connect', { vanityFromUrl: 'satyanadella' }),
    ).toThrow(/URL vanity mismatch/);
  });

  test('idempotency token is consumed exactly once', () => {
    const intent = makeIntent();
    const g = new SafetyGuard({ mode: 'mutating', intent });
    g.requireIntent('connect', { vanityFromUrl: 'jordanrivera' });
    expect(() =>
      g.requireIntent('connect', { vanityFromUrl: 'jordanrivera' }),
    ).toThrow(/idempotency token already consumed/);
  });
});

test.describe('SafetyGuard — guest view budget', () => {
  test('reports remaining budget and goes negative after exhaustion', () => {
    const events: any[] = [];
    const g = new SafetyGuard({
      mode: 'read_only',
      guestViewBudget: 3,
      log: (e) => events.push(e),
    });
    const r1 = g.noteGuestProfileLoad();
    expect(r1).toMatchObject({ observed: 1, budget: 3, remaining: 2 });
    const r2 = g.noteGuestProfileLoad();
    expect(r2).toMatchObject({ observed: 2, budget: 3, remaining: 1 });
    const r3 = g.noteGuestProfileLoad();
    expect(r3).toMatchObject({ observed: 3, budget: 3, remaining: 0 });
    // remaining<=0 triggers a deny audit log entry
    expect(
      events.some(
        (e) => e.kind === 'deny' && /guest view budget exhausted/.test(e.reason ?? ''),
      ),
    ).toBe(true);
    const r4 = g.noteGuestProfileLoad();
    expect(r4.remaining).toBeLessThan(0);
  });

  test('honors caller-supplied budget override', () => {
    const g = new SafetyGuard({ mode: 'read_only', guestViewBudget: 5 });
    for (let i = 0; i < 4; i++) g.noteGuestProfileLoad();
    expect(g.noteGuestProfileLoad()).toMatchObject({ observed: 5, remaining: 0 });
  });
});

test.describe('SafetyGuard — assertVerifiedSelector', () => {
  test('blocks unknown rows when acceptUnverified=false (default)', () => {
    const g = new SafetyGuard({ mode: 'read_only' });
    expect(() =>
      g.assertVerifiedSelector('search_results_people.result_card_root', 'unknown'),
    ).toThrow(EnvelopeError);
  });

  test('permits unknown rows when acceptUnverified=true (Pass 2b development)', () => {
    const g = new SafetyGuard({ mode: 'read_only', acceptUnverified: true });
    expect(() =>
      g.assertVerifiedSelector('search_results_people.result_card_root', 'unknown'),
    ).not.toThrow();
  });

  test('always permits stable_now rows', () => {
    const g = new SafetyGuard({ mode: 'read_only' });
    expect(() => g.assertVerifiedSelector('public_profile.h1_name', 'stable_now')).not.toThrow();
  });
});
