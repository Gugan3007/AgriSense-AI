import { Link, useParams } from 'react-router-dom';
import { ArrowLeft, Clock, Download, Share2, Zap } from 'lucide-react';
import { ProbabilityBars, TrendLine } from '../components/Charts.jsx';
import { ErrorBanner, GlassCard, PageHeader, Skeleton, StatusPill } from '../components/Primitives.jsx';
import { useApi } from '../hooks/useApi.js';
import { api, assetUrl, stressColors } from '../utils/api.js';

function HeatmapGrid({ regions = [] }) {
  const top = new Map(regions.map((region) => [`${region.row}-${region.column}`, region.score]));
  return (
    <div className="grid aspect-square grid-cols-4 gap-2 rounded-[1.7rem] border border-white/10 bg-gradient-to-br from-blue-500/20 via-emerald-400/10 to-amber-300/20 p-3">
      {Array.from({ length: 16 }).map((_, index) => {
        const row = Math.floor(index / 4) + 1;
        const column = (index % 4) + 1;
        const score = top.get(`${row}-${column}`) || 0.35 + index * 0.015;
        return (
          <div
            key={`${row}-${column}`}
            className="rounded-2xl"
            style={{
              background: `rgba(${score > 0.8 ? '244,180,0' : '46,134,255'}, ${Math.min(score, 1)})`,
              boxShadow: score > 0.8 ? '0 0 28px rgba(244,180,0,.45)' : 'none',
            }}
          />
        );
      })}
    </div>
  );
}

function LeafWithActivation({ imageUrl, heatmap = [] }) {
  const cells = heatmap.flat();
  return (
    <div className="relative overflow-hidden rounded-[1.8rem]">
      <img src={imageUrl} alt="Uploaded leaf" className="h-[420px] w-full object-cover" />
      {cells.length > 0 && (
        <div className="pointer-events-none absolute inset-0 grid grid-cols-7 opacity-70 mix-blend-screen">
          {cells.map((value, index) => (
            <div
              key={index}
              style={{
                background: `radial-gradient(circle, rgba(244,180,0,${Math.min(0.9, value)}) 0%, rgba(46,134,255,${Math.max(0.15, value * 0.45)}) 48%, transparent 78%)`,
                filter: 'blur(7px)',
              }}
            />
          ))}
        </div>
      )}
      <div className="absolute bottom-4 right-4 rounded-full border border-white/15 bg-black/35 px-3 py-1 text-xs font-bold text-white backdrop-blur">
        Real CNN activation overlay
      </div>
    </div>
  );
}

export default function Prediction() {
  const { id } = useParams();
  const { data, loading, error } = useApi(() => api.get(`/history/${id}`), [id]);

  if (loading) return <Skeleton className="h-[720px]" />;

  return (
    <div>
      <ErrorBanner message={error} />
      {!data ? null : (
        <>
          <PageHeader
            title="Prediction Dashboard"
            subtitle="A live record from the trained CNN-LSTM model, including class probabilities, activation summary, and seven-day sensor trend."
            actions={
              <>
                <Link to="/history" className="btn-secondary"><ArrowLeft size={18} /> Back to Predictions</Link>
                <button className="btn-secondary"><Share2 size={18} /> Share</button>
                <button className="btn-primary"><Download size={18} /> Export Record</button>
              </>
            }
          />

          <div className="grid gap-5 xl:grid-cols-[.9fr_1.1fr]">
            <GlassCard className="p-5">
              <p className="mb-4 text-sm font-bold text-emerald-200">Uploaded leaf</p>
              <LeafWithActivation imageUrl={assetUrl(data.image_url)} heatmap={data.cnn_feature_summary?.heatmap_grid || []} />
            </GlassCard>

            <GlassCard className="p-5">
              <div className="flex flex-col justify-between gap-4 md:flex-row">
                <div>
                  <StatusPill color={stressColors[data.predicted_class]}>{data.predicted_class}</StatusPill>
                  <h2 className="mt-4 font-display text-5xl font-black text-white">
                    {(data.confidence * 100).toFixed(1)}%
                  </h2>
                  <p className="mt-2 text-emerald-50/60">model confidence</p>
                </div>
                <div className="rounded-3xl border border-white/10 bg-white/[0.04] p-4 text-sm text-emerald-50/65">
                  <p className="flex items-center gap-2 font-bold text-white"><Clock size={16} /> Prediction time</p>
                  <p className="mt-2">{data.prediction_time_ms?.toFixed?.(1) || data.prediction_time_ms} ms</p>
                  <p className="mt-1">{new Date(data.timestamp).toLocaleString()}</p>
                </div>
              </div>
              <ProbabilityBars probabilities={data.class_probabilities} />
            </GlassCard>
          </div>

          <div className="mt-5 grid gap-5 xl:grid-cols-2">
            <GlassCard className="p-5">
              <div className="mb-4 flex items-center justify-between">
                <h2 className="section-title">CNN activation heatmap</h2>
                <StatusPill color="#2E86FF">{data.cnn_feature_summary?.source_layer || 'image_encoder'}</StatusPill>
              </div>
              <div className="grid gap-5 md:grid-cols-[.8fr_1fr] md:items-center">
                <HeatmapGrid regions={data.cnn_feature_summary?.top_regions} />
                <div>
                  <p className="text-sm leading-6 text-emerald-50/65">
                    Activation energy summarizes the final convolutional response used by the temporal model. Top regions mark the strongest visual evidence in the uploaded leaf.
                  </p>
                  <div className="mt-4 rounded-3xl border border-white/10 bg-white/[0.04] p-4">
                    <p className="text-sm text-emerald-50/60">Activation energy</p>
                    <p className="font-display text-3xl font-black">{data.cnn_feature_summary?.activation_energy}</p>
                  </div>
                </div>
              </div>
            </GlassCard>

            <GlassCard className="p-5">
              <div className="mb-4 flex items-center justify-between">
                <h2 className="section-title">LSTM temporal trend</h2>
                <StatusPill color="#F4B400">{data.lstm_trend?.direction}</StatusPill>
              </div>
              <p className="mb-3 text-xs text-emerald-50/50">Sensor stress score (0–100)</p>
              <TrendLine values={data.lstm_trend?.stress_score || []} />
            </GlassCard>
          </div>

          <GlassCard className="mt-5 p-5">
            <h2 className="section-title">Sample information</h2>
            <div className="mt-4 grid gap-3 md:grid-cols-4">
              {[
                ['Crop', data.plant_type || 'Unknown'],
                ['Upload ID', data.upload_id],
                ['Prediction ID', data.prediction_id],
                ['Sequence handling', data.sequence_adjustment?.strategy],
              ].map(([label, value]) => (
                <div key={label} className="rounded-3xl border border-white/10 bg-white/[0.04] p-4">
                  <p className="text-xs text-emerald-50/45">{label}</p>
                  <p className="mt-2 break-words text-sm font-bold text-white">{value}</p>
                </div>
              ))}
            </div>
          </GlassCard>
        </>
      )}
    </div>
  );
}
