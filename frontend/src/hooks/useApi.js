import { useCallback, useEffect, useState } from 'react';

export function useApi(fetcher, dependencies = []) {
  const [data, setData] = useState(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      const response = await fetcher();
      setData(response.data);
    } catch (err) {
      setError(err.response?.data?.error || err.message || 'Request failed.');
    } finally {
      setLoading(false);
    }
  }, dependencies);

  useEffect(() => {
    load();
  }, [load]);

  return { data, error, loading, reload: load };
}
