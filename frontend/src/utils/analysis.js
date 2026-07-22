export function uploadErrorDetails(error) {
  const data = error?.response?.data || {};
  return {
    status: data.status || 'error',
    reasonCode: data.reason_code || 'upload_failed',
    message: data.message || data.error || error?.message || 'Analysis failed.',
    guidance: Array.isArray(data.guidance) ? data.guidance : [],
  };
}

export const isConclusive = (analysis) => analysis?.analysis_status === 'completed';

export const CROP_OPTIONS = [
  { value: 'Apple', label: 'Apple' },
  { value: 'Cherry_(including_sour)', label: 'Cherry (including sour)' },
  { value: 'Corn_(maize)', label: 'Corn (maize)' },
  { value: 'Grape', label: 'Grape' },
  { value: 'Peach', label: 'Peach' },
  { value: 'Pepper,_bell', label: 'Bell pepper' },
  { value: 'Potato', label: 'Potato' },
  { value: 'Strawberry', label: 'Strawberry' },
  { value: 'Tomato', label: 'Tomato' },
];
