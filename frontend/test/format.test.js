import test from 'node:test';
import assert from 'node:assert/strict';
import { formatConfidence, formatTopic, errorText } from '../src/format.js';

test('formats confidence without claiming missing predictions', () => {
  assert.equal(formatConfidence(0.843), '84.3%');
  assert.equal(formatConfidence(null), 'Unavailable');
  assert.equal(formatConfidence(NaN), 'Unavailable');
});

test('renders topic labels safely', () => {
  assert.equal(formatTopic('account_access'), 'Account access');
  assert.equal(formatTopic(null), 'Not classified');
});

test('describes validation errors and falls back safely', () => {
  assert.equal(errorText({ detail: 'Subject required' }), 'Subject required');
  assert.equal(errorText({ detail: [{ msg: 'Required' }] }), 'Required');
  assert.equal(errorText({}), 'Request failed. Please try again.');
});

test('formats counts without inventing missing measurements', async () => {
  const { formatRatio } = await import('../src/format.js');
  assert.equal(formatRatio(52,66), '52 / 66');
  assert.equal(formatRatio(0,26), '0 / 26');
  assert.equal(formatRatio(undefined,26), 'Unavailable');
});
