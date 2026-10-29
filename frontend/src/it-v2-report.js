const metric = value => typeof value === 'number' && Number.isFinite(value) && value >= 0 && value <= 1 ? value : null;
const count = value => Number.isInteger(value) && value >= 0 ? value : null;
const model = value => value && typeof value === 'object' ? {
  name: typeof value.name === 'string' ? value.name : 'Model',
  known_accuracy:metric(value.known_accuracy), known_macro_f1:metric(value.known_macro_f1),
  unknown_recall:metric(value.unknown_recall), known_reject_rate:metric(value.known_reject_rate),
  coverage:metric(value.coverage), unsafe_count:count(value.unsafe_count), total:count(value.total),
} : null;

/** Expose only supported aggregate measurements, never ticket text or case lists. */
export function itV2View(report) {
  if (report?.available !== true) return { available:false, baseline:null, candidate:null, limitations:[] };
  const dataset = report.dataset && typeof report.dataset === 'object' ? report.dataset : {};
  const unknownUnavailable = report.evaluation_scope === 'external_test' && dataset.external_unknown_n === 0;
  const safeModel = value => { const result = model(value); if (result && unknownUnavailable) result.unknown_recall = null; return result; };
  return {
    available:true,
    active:report.active === true,
    dataset:{name:typeof dataset.name === 'string' ? dataset.name : '', known_dev_n:count(dataset.known_dev_n),unknown_dev_n:count(dataset.unknown_dev_n),
      external_known_n:count(dataset.external_known_n),external_unknown_n:count(dataset.external_unknown_n),
      external_ambiguous_n:count(dataset.external_ambiguous_n),
      provenance:typeof dataset.provenance === 'string' ? dataset.provenance : ''},
    baseline:safeModel(report.baseline),candidate:safeModel(report.candidate),
    selection:report.selection && typeof report.selection === 'object' ? {
      chosen_model:typeof report.selection.chosen_model === 'string' ? report.selection.chosen_model : null,
      neural_status:typeof report.selection.neural_192?.status === 'string' ? report.selection.neural_192.status : null,
      neural_reason:typeof report.selection.neural_192?.reason === 'string' ? report.selection.neural_192.reason : null,
      ood_policy:typeof report.selection.ood_policy === 'string' ? report.selection.ood_policy : null,
    } : {},
    unlabelled_external:report.unlabelled_external && typeof report.unlabelled_external === 'object' ? {count:count(report.unlabelled_external.count),review_flagged:count(report.unlabelled_external.review_flagged),baseline_review_flagged:count(report.unlabelled_external.baseline_review_flagged),flag_rate:metric(report.unlabelled_external.flag_rate),warning:typeof report.unlabelled_external.warning === 'string' ? report.unlabelled_external.warning : ''} : null,
    historical_diagnostic:report.historical_diagnostic && typeof report.historical_diagnostic === 'object' ? {
      description:typeof report.historical_diagnostic.description === 'string' ? report.historical_diagnostic.description : '',
      known:{exact_unique:count(report.historical_diagnostic.known?.exact_unique),flagged_exact_unique:count(report.historical_diagnostic.known?.flagged_exact_unique)},
      eol:{exact_unique:count(report.historical_diagnostic.eol?.exact_unique),flagged_exact_unique:count(report.historical_diagnostic.eol?.flagged_exact_unique)},
      limitations:Array.isArray(report.historical_diagnostic.limitations) ? report.historical_diagnostic.limitations.filter(x=>typeof x==='string') : [],
    } : null,
    evaluation_scope:typeof report.evaluation_scope === 'string' ? report.evaluation_scope : null,
    limitations:Array.isArray(report.limitations) ? report.limitations.filter(x => typeof x === 'string') : [],
  };
}
