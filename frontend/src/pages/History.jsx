import { useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import { Eye, Filter } from 'lucide-react';
import { ErrorBanner, GlassCard, PageHeader, Skeleton, StatusPill } from '../components/Primitives.jsx';
import { useApi } from '../hooks/useApi.js';
import { api, assetUrl, stressColors } from '../utils/api.js';

export default function History() {
  const [filter, setFilter] = useState('All');
  const { data, loading, error } = useApi(() => api.get('/history?page=1&limit=30'), []);
  const items = useMemo(() => {
    const rows = data?.items || [];
    return filter === 'All' ? rows : rows.filter((item) => item.predicted_class === filter);
  }, [data, filter]);

  return (
    <div>
      <ErrorBanner message={error} />
      <PageHeader
        title="Prediction History"
        subtitle="Every analysis is persisted to SQLite with model probabilities, upload metadata, sensor sequence, and explainability summaries."
        actions={
          <div className="flex items-center gap-2 rounded-2xl border border-white/10 bg-white/[0.04] p-2">
            <Filter size={16} />
            <select value={filter} onChange={(event) => setFilter(event.target.value)} className="bg-transparent text-sm font-bold outline-none">
              {['All', 'Healthy', 'Low', 'Medium', 'High'].map((item) => <option key={item}>{item}</option>)}
            </select>
          </div>
        }
      />
      {loading ? <Skeleton className="h-[620px]" /> : (
        <GlassCard className="overflow-hidden p-0">
          <div className="grid gap-px bg-white/10">
            {items.map((item) => (
              <div key={item.prediction_id} className="grid gap-4 bg-pine/80 p-4 md:grid-cols-[88px_1fr_auto] md:items-center">
                <img src={assetUrl(item.image_url)} alt="" className="h-20 w-20 rounded-2xl object-cover" />
                <div>
                  <div className="mb-2 flex flex-wrap items-center gap-3">
                    <StatusPill color={stressColors[item.predicted_class]}>{item.predicted_class}</StatusPill>
                    <span className="text-sm text-emerald-50/55">{new Date(item.timestamp).toLocaleString()}</span>
                  </div>
                  <p className="font-display text-xl font-black">{item.plant_type || 'Unknown crop'} analysis</p>
                  <p className="mt-1 text-sm text-emerald-50/55">
                    Confidence {(item.confidence * 100).toFixed(1)}% · Trend {item.lstm_trend?.direction}
                  </p>
                </div>
                <Link to={`/prediction/${item.prediction_id}`} className="btn-secondary justify-center">
                  <Eye size={17} /> Open
                </Link>
              </div>
            ))}
          </div>
        </GlassCard>
      )}
    </div>
  );
}
