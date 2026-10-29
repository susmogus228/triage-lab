export function formatConfidence(value) {
  if (typeof value !== 'number' || !Number.isFinite(value)) return 'Unavailable';
  return `${(value * 100).toFixed(1)}%`;
}
export function formatTopic(value) {
  if (!value || typeof value !== 'string') return 'Not classified';
  const clean = value.replace(/[_-]+/g, ' ').trim();
  return clean.charAt(0).toUpperCase() + clean.slice(1).toLowerCase();
}
export function errorText(payload) {
  if (typeof payload?.detail === 'string') return payload.detail;
  if (Array.isArray(payload?.detail)) return payload.detail.map((item) => item.msg).filter(Boolean).join('; ') || 'Request failed. Please try again.';
  return 'Request failed. Please try again.';
}
export function formatDate(value) {
  if (!value) return 'Unknown time';
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString(undefined, {dateStyle:'medium', timeStyle:'short'});
}
export function formatRatio(numerator, denominator) {
  return Number.isInteger(numerator) && Number.isInteger(denominator) && numerator >= 0 && denominator > 0 && numerator <= denominator
    ? `${numerator} / ${denominator}` : 'Unavailable';
}
