import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Activity, AlertTriangle, ArrowRight, CheckCircle2, Database, Leaf, Network, ShieldCheck, Sprout, UploadCloud } from 'lucide-react';
import { AccuracySparkline } from '../components/Charts.jsx';
import { ErrorBanner, GlassCard, MetricCard, PageHeader, Skeleton, StatusPill } from '../components/Primitives.jsx';
import { useApi } from '../hooks/useApi.js';
import { api } from '../utils/api.js';

export default function Home() {
  const dataset = useApi(() => api.get('/dataset-info'), []);
  const model = useApi(() => api.get('/model-info'), []);
  const history = useApi(() => api.get('/history?page=1&limit=4'), []);

  const modelAccuracy = model.data ? (model.data.best_validation_accuracy * 100).toFixed(1) : '—';

  return (
    <div>
      <ErrorBanner message={dataset.error || model.error || history.error} />
      <section className="hero-panel">
        <div className="relative z-10 grid gap-8 xl:grid-cols-[1.1fr_.9fr] xl:items-center">
          <div>
            <div className="mb-5 inline-flex items-center gap-2 rounded-full border border-emerald-300/20 bg-emerald-300/[0.08] px-4 py-2 text-xs font-bold uppercase tracking-[0.16em] text-emerald-200">
              <Sprout size={15} /> Precision intelligence for healthier crops
            </div>
            <PageHeader
              title={<>AI-Powered Crop Stress Detection <span className="text-gradient">You Can Trust</span></>}
              subtitle="Hybrid CNN-LSTM intelligence combines real leaf-image analysis with sequential sensor telemetry, helping growers catch stress before it becomes crop loss."
              actions={
                <>
                  <Link to="/upload" className="btn-primary"><UploadCloud size={18} /> Upload & Analyze</Link>
                  <Link to="/model" className="btn-secondary"><Network size={18} /> View Architecture</Link>
                </>
              }
            />
            <div className="flex flex-wrap gap-x-5 gap-y-2 text-xs font-semibold text-emerald-50/55">
              {['Real-time inference', 'Four stress levels', '7-day trend analysis'].map((item) => <span key={item} className="flex items-center gap-2"><CheckCircle2 size={15} className="text-emerald-300" />{item}</span>)}
            </div>
            <div className="mt-8 grid grid-cols-2 gap-4 lg:grid-cols-4">
              {dataset.loading ? (
                Array.from({ length: 4 }).map((_, i) => <Skeleton key={i} className="h-36" />)
              ) : (
                <>
                  <MetricCard label="Dataset Rows" value={dataset.data?.row_count?.toLocaleString()} detail="Real-image daily observations" icon={<Database />} />
                  <MetricCard label="Virtual Plants" value={dataset.data?.plant_count} detail="Split by Plant_ID" icon={<Sprout />} />
                  <MetricCard label="Model Accuracy" value={`${modelAccuracy}%`} detail="Best validation run" tone="blue" icon={<ShieldCheck />} />
                  <MetricCard label="Stress Classes" value="4" detail="Healthy, Low, Medium, High" tone="amber" icon={<AlertTriangle />} />
                </>
              )}
            </div>
          </div>

          <GlassCard className="p-5">
            <div className="mb-5 flex items-center justify-between"><p className="text-sm font-bold text-emerald-200">How AgriSense AI works</p><span className="rounded-full bg-white/[0.05] px-3 py-1 text-[10px] font-bold uppercase tracking-widest text-emerald-50/45">3-stage pipeline</span></div>
            <div className="grid gap-4 md:grid-cols-3">
              {[
                ['CNN', 'Leaf feature extraction', 'Leaf texture, lesions, color, vein structure'],
                ['LSTM', 'Temporal modelling', 'Seven-day telemetry sequence'],
                ['Softmax', 'Stress classification', 'Healthy → High risk probabilities'],
              ].map(([title, subtitle, body], index) => (
                <motion.div
                  key={title}
                  whileHover={{ y: -4 }}
                  className="group rounded-3xl border border-emerald-200/10 bg-white/[0.045] p-4 transition hover:border-emerald-200/25 hover:bg-white/[0.07]"
                >
                  <div className="mb-5 grid h-16 w-16 place-items-center rounded-2xl bg-electric/15 text-electric ring-1 ring-electric/30">
                    {index === 0 ? <Leaf /> : index === 1 ? <Network /> : <ShieldCheck />}
                  </div>
                  <h3 className="font-display text-xl font-black">{title}</h3>
                  <p className="mt-1 text-sm font-semibold text-emerald-200">{subtitle}</p>
                  <p className="mt-3 text-sm leading-6 text-emerald-50/60">{body}</p>
                </motion.div>
              ))}
            </div>
            <div className="mt-5 rounded-3xl border border-blue-300/10 bg-blue-400/10 p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-blue-100/70">7-day rolling model performance</p>
                  <p className="font-display text-3xl font-black">{modelAccuracy}%</p>
                </div>
                <StatusPill color="#2E86FF">Real metadata</StatusPill>
              </div>
              <AccuracySparkline value={Number(modelAccuracy) || 93.6} />
            </div>
          </GlassCard>
        </div>
      </section>

      <section className="mt-6 grid gap-5 xl:grid-cols-[1.2fr_.8fr]">
        <GlassCard className="p-5">
          <div className="mb-4 flex items-center justify-between">
            <h2 className="section-title">Recent analyses</h2>
            <Link to="/history" className="flex items-center gap-1 text-sm font-bold text-electric transition hover:gap-2">View all <ArrowRight size={15} /></Link>
          </div>
          {history.loading ? <Skeleton className="h-64" /> : (
            <div className="overflow-hidden rounded-3xl border border-white/10">
              {(history.data?.items || []).length ? <table className="w-full min-w-[720px] text-left text-sm">
                <thead className="bg-white/[0.05] text-emerald-100/70">
                  <tr>
                    <th className="px-4 py-3">Prediction</th>
                    <th className="px-4 py-3">Crop</th>
                    <th className="px-4 py-3">Class</th>
                    <th className="px-4 py-3">Confidence</th>
                    <th className="px-4 py-3">Time</th>
                  </tr>
                </thead>
                <tbody>
                  {(history.data?.items || []).map((item) => (
                    <tr key={item.prediction_id} className="border-t border-white/10">
                      <td className="px-4 py-3 font-mono text-xs text-emerald-100/60">{item.prediction_id.slice(0, 8)}</td>
                      <td className="px-4 py-3">{item.plant_type || 'Unknown'}</td>
                      <td className="px-4 py-3"><StatusPill>{item.predicted_class}</StatusPill></td>
                      <td className="px-4 py-3">{(item.confidence * 100).toFixed(1)}%</td>
                      <td className="px-4 py-3 text-emerald-50/55">{new Date(item.timestamp).toLocaleString()}</td>
                    </tr>
                  ))}
                </tbody>
              </table> : <div className="grid min-h-56 place-items-center p-6 text-center"><div><div className="mx-auto grid h-14 w-14 place-items-center rounded-2xl bg-emerald-300/10 text-emerald-300"><Activity /></div><p className="mt-4 font-bold text-white">No analyses yet</p><p className="mt-1 text-sm text-emerald-50/50">Your crop predictions will appear here.</p><Link to="/upload" className="btn-secondary mt-4 px-4 py-2 text-xs">Start an analysis <ArrowRight size={14} /></Link></div></div>}
            </div>
          )}
        </GlassCard>

        <GlassCard className="p-5">
          <h2 className="section-title">Live telemetry average</h2>
          <div className="mt-4 space-y-3">
            {[
              ['Soil Moisture', '49.9%', 'bg-blue-400', '72%'],
              ['Temperature', '27.0°C', 'bg-emerald-400', '62%'],
              ['Humidity', '58.2%', 'bg-cyan-300', '68%'],
              ['Light Intensity', '16,100 lux', 'bg-amber-300', '78%'],
            ].map(([label, value, color, width]) => (
              <div key={label} className="rounded-2xl border border-white/10 bg-white/[0.04] p-4">
                <div className="flex justify-between text-sm">
                  <span className="text-emerald-50/65">{label}</span>
                  <span className="font-bold">{value}</span>
                </div>
                <div className="mt-3 h-2 rounded-full bg-white/10">
                  <div className={`h-2 rounded-full ${color}`} style={{ width }} />
                </div>
              </div>
            ))}
          </div>
        </GlassCard>
      </section>
    </div>
  );
}
