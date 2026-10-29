import test from 'node:test';
import assert from 'node:assert/strict';
import { itExperimentView } from '../src/it-report.js';

test('missing report never invents model results or active status', () => {
  assert.deepEqual(itExperimentView(null), {available:false, models:[]});
  assert.deepEqual(itExperimentView({available:false}), {available:false, models:[]});
});

test('only explicit selected and active flags are shown, and partial scores remain unknown', () => {
  const view = itExperimentView({available:true, selected:'candidate', models:[
    {name:'baseline', selected:false, active:false, dev:{accuracy:0.5,macro_f1:0.4}, test:{accuracy:0.3}},
    {name:'candidate', selected:true, dev:{accuracy:0}, test:null},
  ]});
  assert.equal(view.models[0].active, false);
  assert.equal(view.models[1].selected, true);
  assert.equal(view.models[1].active, false);
  assert.equal(view.models[1].dev.accuracy, 0);
  assert.equal(view.models[1].test, null);
  assert.equal(view.models[0].test.macro_f1, null);
});

test('IT trial input requires a real message and joins the optional subject', async () => {
  const {prepareItTrial} = await import('../src/it-report.js');
  assert.equal(prepareItTrial('Subject','   '), null);
  assert.deepEqual(prepareItTrial('  Login issue ','  Cannot sign in  '), {text:'Login issue\nCannot sign in'});
  assert.deepEqual(prepareItTrial('  ','  Cannot sign in  '), {text:'Cannot sign in'});
  assert.equal(prepareItTrial('x'.repeat(200), 'y'.repeat(9900)),null);
});

test('known-category metrics and grouped results stay separate from full test', () => {
  const view=itExperimentView({available:true,models:[{name:'candidate',known_test:{rows:599,accuracy:.72,macro_f1:.5,exact_unique_groups:551,exact_accuracy:.68,exact_macro_f1:.49},test:{rows:657,accuracy:.65,macro_f1:.4,exact_unique_groups:553}}]});
  assert.equal(view.models[0].known_test.rows,599);
  assert.equal(view.models[0].known_test.accuracy,.72);
  assert.equal(view.models[0].known_test.exact_unique_groups,551);
  assert.equal(view.models[0].known_test.exact_accuracy,.68);
  assert.equal(view.models[0].test.rows,657);
  assert.equal(view.models[0].test.accuracy,.65);
});
