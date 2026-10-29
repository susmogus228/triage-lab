const metric = value => typeof value === 'number' && Number.isFinite(value) && value >= 0 && value <= 1 ? value : null;
const metrics = value => value && typeof value === 'object' ? {
  accuracy: metric(value.accuracy), macro_f1: metric(value.macro_f1),
  rows: Number.isInteger(value.rows) && value.rows >= 0 ? value.rows : null,
  exact_unique_groups: Number.isInteger(value.exact_unique_groups) && value.exact_unique_groups >= 0 ? value.exact_unique_groups : null,
  exact_conflicting_groups: Number.isInteger(value.exact_conflicting_groups) && value.exact_conflicting_groups >= 0 ? value.exact_conflicting_groups : null,
  exact_accuracy: metric(value.exact_accuracy), exact_macro_f1: metric(value.exact_macro_f1),
  labels: Array.isArray(value.labels) ? value.labels.filter(x => typeof x === 'string') : [],
  confusion_matrix: Array.isArray(value.confusion_matrix) ? value.confusion_matrix : [],
  per_class: value.per_class && typeof value.per_class === 'object' ? value.per_class : {},
} : null;

/** Normalize only explicitly supported aggregate fields; never expose raw ticket text. */
export function itExperimentView(report) {
  if (report?.available !== true) return {available:false, models:[]};
  return {
    available: true,
    dataset: report.dataset && typeof report.dataset === 'object' ? report.dataset : {},
    models: Array.isArray(report.models) ? report.models.map((model,index) => ({
      name: typeof model.name === 'string' ? model.name : `Model ${index + 1}`,
      kind: typeof model.kind === 'string' ? model.kind : '',
      selected: model.selected === true,
      active: model.active === true,
      dev: metrics(model.dev), test: metrics(model.test), known_test: metrics(model.known_test),
    })) : [],
    limitations: Array.isArray(report.limitations) ? report.limitations.filter(x => typeof x === 'string') : [],
  };
}

export function prepareItTrial(subject, message) {
  if (typeof subject !== 'string' || typeof message !== 'string' || !message.trim()) return null;
  const text = [subject.trim(), message.trim()].filter(Boolean).join('\n');
  return text.length <= 10000 ? {text} : null;
}
