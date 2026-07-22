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
