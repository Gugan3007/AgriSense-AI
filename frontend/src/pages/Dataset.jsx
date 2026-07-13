import { Database, Image as ImageIcon, Sprout } from 'lucide-react';
import { ClassBars, DistributionChart } from '../components/Charts.jsx';
import { ErrorBanner, GlassCard, MetricCard, PageHeader, Skeleton } from '../components/Primitives.jsx';
import { useApi } from '../hooks/useApi.js';
import { api, assetUrl } from '../utils/api.js';

export default function Dataset() {
  const { data, loading, error } = useApi(() => api.get('/dataset-info'), []);

  return (
    <div>
      <ErrorBanner message={error} />
      <PageHeader
        title="Dataset Information"
        subtitle="A transparent hybrid dataset: real PlantVillage leaf images and disease labels, with seeded temporal placement and correlated sensor telemetry."
      />
      {loading ? <Skeleton className="h-[720px]" /> : (
        <>
          <div className="grid gap-4 md:grid-cols-3">
            <MetricCard label="Rows" value={data.row_count.toLocaleString()} detail="One row per plant-day observation" icon={<Database />} />
            <MetricCard label="Unique images" value={data.image_count.toLocaleString()} detail="Referenced from PlantVillage" tone="blue" icon={<ImageIcon />} />
            <MetricCard label="Virtual plants" value={data.plant_count} detail={`${data.date_range.start} → ${data.date_range.end}`} tone="amber" icon={<Sprout />} />
          </div>

          <div className="mt-5 grid gap-5 xl:grid-cols-[.85fr_1.15fr]">
            <GlassCard className="p-5">
              <h2 className="section-title">Stress class distribution</h2>
              <DistributionChart distribution={data.stress_distribution} />
              <div className="grid grid-cols-2 gap-3">
                {Object.entries(data.stress_distribution).map(([label, value]) => (
                  <div key={label} className="rounded-2xl border border-white/10 bg-white/[0.04] p-3">
                    <p className="text-xs text-emerald-50/50">{label}</p>
                    <p className="font-display text-2xl font-black">{value}</p>
                  </div>
                ))}
              </div>
            </GlassCard>

            <GlassCard className="p-5">
              <h2 className="section-title">Top disease classes</h2>
              <ClassBars distribution={data.disease_class_distribution} />
            </GlassCard>
          </div>

          <GlassCard className="mt-5 p-5">
            <div className="mb-5 flex flex-col justify-between gap-3 md:flex-row md:items-center">
              <div>
                <h2 className="section-title">Real dataset samples</h2>
                <p className="text-sm text-emerald-50/60">Images are served from the actual generated CSV paths.</p>
              </div>
              <p className="text-sm font-bold text-emerald-200">{data.plant_types.length} crop types available</p>
            </div>
            <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
              {data.sample_rows.map((row) => (
                <div key={`${row.Plant_ID}-${row.Timestamp}`} className="overflow-hidden rounded-3xl border border-white/10 bg-white/[0.04]">
                  <img src={assetUrl(row.image_url)} alt={`${row.Plant_Type} ${row.Stress_Level}`} className="h-44 w-full object-cover" />
                  <div className="p-4">
                    <p className="font-bold">{row.Plant_Type}</p>
                    <p className="mt-1 text-sm text-emerald-50/55">{row.Stress_Level} · {row.Disease_Class}</p>
                  </div>
                </div>
              ))}
            </div>
          </GlassCard>

          <GlassCard className="mt-5 p-5">
            <h2 className="section-title">Preprocessing pipeline</h2>
            <div className="mt-5 grid gap-4 md:grid-cols-4">
              {[
                ['Resize', 'Images become 128×128 RGB tensors.'],
                ['Normalize', 'Pixel values are scaled to [0,1].'],
                ['Augment', 'Training-only rotation, flip, brightness, contrast.'],
                ['Sequence', 'Sliding windows of 7 days grouped by Plant_ID.'],
              ].map(([title, body]) => (
                <div key={title} className="rounded-3xl border border-white/10 bg-white/[0.04] p-5">
                  <p className="font-display text-xl font-black">{title}</p>
                  <p className="mt-3 text-sm leading-6 text-emerald-50/60">{body}</p>
                </div>
              ))}
            </div>
          </GlassCard>
        </>
      )}
    </div>
  );
}
