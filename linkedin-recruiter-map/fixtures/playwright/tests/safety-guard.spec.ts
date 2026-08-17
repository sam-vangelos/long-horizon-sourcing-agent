// Proves SafetyGuard refuses to perform mutating actions without a proper Intent.
// These tests do NOT require a LinkedIn session — they exercise the guard contract directly.

import { describe, it } from 'node:test';
import { strict as assert } from 'node:assert';
import { SafetyGuard } from '../src/safety/SafetyGuard.js';
import { EnvelopeError, type Intent } from '../src/types.js';

describe('SafetyGuard', () => {
  it('read-only mode rejects requireIntent', () => {
    const guard = new SafetyGuard({ mode: 'read_only' });
    assert.throws(
      () => guard.requireIntent('save_to_project', { projectIdFromUrl: '2035258290' }),
      EnvelopeError,
    );
  });

  it('mutating mode without humanConfirmed rejects', () => {
    const intent: Intent = {
      action: 'save_to_project',
      targetProjectId: '2035258290',
      idempotencyToken: 't1',
      humanConfirmed: false,
    };
    const guard = new SafetyGuard({ mode: 'mutating', intent, log: () => {} });
    assert.throws(
      () => guard.requireIntent('save_to_project', { projectIdFromUrl: '2035258290' }),
      EnvelopeError,
    );
  });

  it('URL projectId mismatch rejects', () => {
    const intent: Intent = {
      action: 'save_to_project',
      targetProjectId: '2035258290',
      idempotencyToken: 't2',
      humanConfirmed: true,
    };
    const guard = new SafetyGuard({ mode: 'mutating', intent, log: () => {} });
    assert.throws(
      () => guard.requireIntent('save_to_project', { projectIdFromUrl: '9999999999' }),
      EnvelopeError,
    );
  });

  it('intent.action mismatch rejects', () => {
    const intent: Intent = {
      action: 'save_to_project',
      targetProjectId: '2035258290',
      idempotencyToken: 't3',
      humanConfirmed: true,
    };
    const guard = new SafetyGuard({ mode: 'mutating', intent, log: () => {} });
    assert.throws(
      () => guard.requireIntent('hide', { projectIdFromUrl: '2035258290' }),
      EnvelopeError,
    );
  });

  it('idempotency token cannot be reused', () => {
    const intent: Intent = {
      action: 'save_to_project',
      targetProjectId: '2035258290',
      idempotencyToken: 't4',
      humanConfirmed: true,
    };
    const guard = new SafetyGuard({ mode: 'mutating', intent, log: () => {} });
    // first call permitted
    guard.requireIntent('save_to_project', { projectIdFromUrl: '2035258290' });
    // second call with same token must reject
    assert.throws(
      () => guard.requireIntent('save_to_project', { projectIdFromUrl: '2035258290' }),
      EnvelopeError,
    );
  });

  it('allowReadOnly refuses non-read-only actions', () => {
    const guard = new SafetyGuard({ mode: 'read_only' });
    assert.throws(() => guard.allowReadOnly('save_to_project' as never), EnvelopeError);
  });

  it('happy path allows save with full envelope', () => {
    const intent: Intent = {
      action: 'save_to_project',
      targetProjectId: '2035258290',
      idempotencyToken: 't5',
      humanConfirmed: true,
    };
    const guard = new SafetyGuard({ mode: 'mutating', intent, log: () => {} });
    const out = guard.requireIntent('save_to_project', { projectIdFromUrl: '2035258290' });
    assert.equal(out.targetProjectId, '2035258290');
  });
});
