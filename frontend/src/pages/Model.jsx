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
        subtitle="The deployed model uses a fine-tuned MobileNetV2 image encoder, crop evidence, a seven-reading sensor LSTM, calibrated reliability, and image-first probability fusion."
      />
      {loading ? <Skeleton className="h-[760px]" /> : (
        <>
          <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
            <MetricCard label="Held-out test accuracy" value={`${(data.staged_evaluation.normal.accuracy * 100).toFixed(1)}%`} detail={`${data.staged_evaluation.test_samples} unseen windows`} icon={<Network />} />
            <MetricCard label="Reliable-result accuracy" value={`${(data.staged_evaluation.selective.accuracy * 100).toFixed(1)}%`} detail={`${(data.staged_evaluation.selective.coverage * 100).toFixed(1)}% coverage; otherwise inconclusive`} icon={<Network />} />
            <MetricCard label="Image-only macro F1" value={`${(data.staged_evaluation.image_only.f1_macro * 100).toFixed(1)}%`} detail={`Sensors alone: ${(data.staged_evaluation.sensor_only.f1_macro * 100).toFixed(1)}%`} tone="blue" icon={<FileText />} />
            <MetricCard label="Crop recognition" value={`${(data.staged_evaluation.crop.accuracy * 100).toFixed(1)}%`} detail={`${data.staged_evaluation.crop.class_count} supported crops`} tone="blue" icon={<FileText />} />
            <MetricCard label="Real-leaf rejection" value={`${(data.staged_evaluation.leaf_validation.false_rejection_rate * 100).toFixed(1)}%`} detail={`${data.staged_evaluation.negative_suite.blocked}/${data.staged_evaluation.negative_suite.samples} non-leaf cases blocked`} tone="amber" icon={<FileText />} />
            <MetricCard label="Parameters" value={data.parameter_count.toLocaleString()} detail={`${data.hyperparameters.completed_epochs} completed epochs`} tone="amber" icon={<FileText />} />
          </div>

          <GlassCard className="mt-5 p-5">
            <h2 className="section-title">Image and telemetry branches → image-first fusion</h2>
            <div className="mt-5 grid gap-4 lg:grid-cols-5">
              {[
                ['Current leaf', '1 × 128 × 128 × 3'],
                ['Image encoder', 'Fine-tuned MobileNetV2 → stress + crop heads'],
                ['Sensor sequence', '7 × 4 normalized readings'],
                ['Temporal branch', 'LSTM 64 → Softmax 4'],
                ['Decision policy', 'Fusion → calibrated result or inconclusive'],
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
