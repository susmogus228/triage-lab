import test from 'node:test';
import assert from 'node:assert/strict';
import { itV2View } from '../src/it-v2-report.js';

test('missing or unavailable report never fabricates model values', () => {
  assert.deepEqual(itV2View(null), { available:false, baseline:null, candidate:null, limitations:[] });
  assert.deepEqual(itV2View({available:false}), { available:false, baseline:null, candidate:null, limitations:[] });
});
test('normalization exposes only supported aggregates and explicit activation', () => {
  const result=itV2View({available:true, active:false, dataset:{known_dev_n:7,external_unknown_n:0,provenance:'Local report'}, baseline:{name:'current',known_accuracy:.6,unknown_recall:.5,coverage:.2,unsafe_count:3,cases:[{text:'private'}]},candidate:{name:'candidate',known_accuracy:.8,known_macro_f1:.7,known_reject_rate:0},limitations:['Diagnostic only']});
  assert.equal(result.active,false);assert.equal(result.dataset.known_dev_n,7);assert.equal(result.dataset.external_unknown_n,0);
  assert.equal(result.baseline.known_accuracy,.6);assert.equal(result.baseline.unknown_recall,.5);assert.equal(result.candidate.known_reject_rate,0);
  assert.equal(JSON.stringify(result).includes('private'),false);
});
test('invalid or missing metrics stay unavailable', () => {
  const result=itV2View({available:true,baseline:{known_accuracy:2,unknown_recall:-1,coverage:NaN},candidate:{}});
  assert.equal(result.baseline.known_accuracy,null);assert.equal(result.baseline.unknown_recall,null);assert.equal(result.baseline.coverage,null);assert.equal(result.candidate.unknown_recall,null);
});

test('unknown recall is never displayed for an external slice with zero labeled unknowns', () => {
  const view=itV2View({available:true,evaluation_scope:'external_test',dataset:{external_unknown_n:0},baseline:{unknown_recall:.95},candidate:{unknown_recall:.9}});
  assert.equal(view.baseline.unknown_recall,null);
  assert.equal(view.candidate.unknown_recall,null);
});


test('historical diagnostic exposes unique aggregate counts only', () => {
  const view=itV2View({available:true,historical_diagnostic:{known:{rows:1000,exact_unique:551,flagged_exact_unique:3},eol:{rows:58,exact_unique:2,flagged_exact_unique:1,text:'private'},limitations:['Previously viewed']}});
  assert.equal(view.historical_diagnostic.eol.exact_unique,2);
  assert.equal(view.historical_diagnostic.eol.flagged_exact_unique,1);
  assert.equal(view.historical_diagnostic.known.flagged_exact_unique,3);
  assert.equal(JSON.stringify(view).includes('private'),false);
  assert.equal('rows' in view.historical_diagnostic.eol,false);
});
