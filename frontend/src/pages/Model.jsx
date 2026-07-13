import { FileText, Network } from 'lucide-react';
import { ErrorBanner, GlassCard, MetricCard, PageHeader, Skeleton } from '../components/Primitives.jsx';
import { useApi } from '../hooks/useApi.js';
import { api, assetUrl } from '../utils/api.js';

export default function Model() {
  const { data, loading, error } = useApi(() => api.get('/model-info'), []);

  return (
    <div>
      <ErrorBanner message={error} />
      <PageHeader
        title="Model Architecture"
        subtitle="The deployed model follows the project specification exactly: TimeDistributed CNN features, sensor fusion, stacked LSTM layers, and a four-class softmax stress head."
      />
      {loading ? <Skeleton className="h-[760px]" /> : (
        <>
          <div className="grid gap-4 md:grid-cols-3">
            <MetricCard label="Best validation accuracy" value={`${(data.best_validation_accuracy * 100).toFixed(1)}%`} detail="Restored best checkpoint" icon={<Network />} />
            <MetricCard label="Parameters" value={data.parameter_count.toLocaleString()} detail="Trainable CNN-LSTM weights" tone="blue" icon={<FileText />} />
            <MetricCard label="Completed epochs" value={data.hyperparameters.completed_epochs} detail={`Requested ${data.hyperparameters.requested_epochs}`} tone="amber" icon={<Network />} />
          </div>

          <GlassCard className="mt-5 p-5">
            <h2 className="section-title">CNN → Sensor Fusion → LSTM → Softmax</h2>
            <div className="mt-5 grid gap-4 lg:grid-cols-5">
              {[
                ['Image sequence', '7 × 128 × 128 × 3'],
                ['CNN backbone', 'Conv32 → Conv64 → Conv128'],
                ['Fusion', 'CNN features + 4 sensors'],
                ['Temporal layers', 'LSTM 128 → LSTM 64'],
                ['Classifier', 'Dense 64 → Softmax 4'],
              ].map(([title, body], index) => (
                <div key={title} className="relative rounded-3xl border border-white/10 bg-white/[0.04] p-5">
                  <p className="text-xs font-bold text-electric">0{index + 1}</p>
                  <p className="mt-4 font-display text-xl font-black">{title}</p>
                  <p className="mt-2 text-sm text-emerald-50/60">{body}</p>
                </div>
              ))}
            </div>
          </GlassCard>

          <div className="mt-5 grid gap-5 xl:grid-cols-2">
            {[
              ['Accuracy curve', data.report_urls.accuracy_curve],
              ['Loss curve', data.report_urls.loss_curve],
              ['Confusion matrix', data.report_urls.confusion_matrix],
              ['ROC curve', data.report_urls.roc_curve],
            ].map(([title, url]) => (
              <GlassCard key={title} className="p-5">
                <h2 className="section-title">{title}</h2>
                <img src={assetUrl(url)} alt={title} className="mt-4 w-full rounded-3xl border border-white/10 bg-white" />
              </GlassCard>
            ))}
          </div>

          <GlassCard className="mt-5 p-5">
            <h2 className="section-title">Hyperparameters</h2>
            <div className="mt-4 grid gap-3 md:grid-cols-3">
              {Object.entries(data.hyperparameters).map(([key, value]) => (
                <div key={key} className="rounded-2xl border border-white/10 bg-white/[0.04] p-4">
                  <p className="text-xs text-emerald-50/45">{key.replaceAll('_', ' ')}</p>
                  <p className="mt-2 font-bold text-white">{String(value)}</p>
                </div>
              ))}
            </div>
          </GlassCard>
        </>
      )}
    </div>
  );
}
