import { Leaf, Target, Users } from 'lucide-react';
import { GlassCard, PageHeader } from '../components/Primitives.jsx';

export default function About() {
  return (
    <div>
      <PageHeader
        title="About AgriSense AI"
        subtitle="A final-year deep learning case study built as a real, runnable product: honest dataset construction, trained image-first inference, Flask persistence, and a polished React interface."
      />
      <div className="grid gap-5 xl:grid-cols-3">
        {[
          [Target, 'Problem statement', 'Farm stress often becomes visible after yield loss has already started. AgriSense uses early image and telemetry signals to classify plant stress earlier.'],
          [Leaf, 'Sustainable impact', 'The project supports SDG 2 Zero Hunger and SDG 12 Responsible Consumption by helping teams reduce avoidable crop waste.'],
          [Users, 'Review-ready build', 'Every layer is executable locally: dataset, model, API, UI, and smoke tests. No fake live predictions are presented as real.'],
        ].map(([Icon, title, body]) => (
          <GlassCard key={title} className="p-6">
            <Icon className="text-emerald-300" size={34} />
            <h2 className="mt-5 font-display text-2xl font-black">{title}</h2>
            <p className="mt-3 leading-7 text-emerald-50/65">{body}</p>
          </GlassCard>
        ))}
      </div>
      <GlassCard className="mt-5 p-6">
        <h2 className="section-title">What is real vs simulated?</h2>
        <div className="mt-5 grid gap-4 md:grid-cols-2">
          <div className="rounded-3xl border border-emerald-300/15 bg-emerald-400/10 p-5">
            <p className="font-display text-xl font-black text-emerald-200">Real</p>
            <p className="mt-3 text-emerald-50/65">Leaf images, disease class labels, trained model inference, API responses, saved predictions, report images, and model metadata.</p>
          </div>
          <div className="rounded-3xl border border-amber-300/15 bg-amber-400/10 p-5">
            <p className="font-display text-xl font-black text-amber-200">Simulated honestly</p>
            <p className="mt-3 text-emerald-50/65">Daily timeline placement and sensor telemetry are generated with a seeded, documented process because PlantVillage has no time/sensor dimension.</p>
          </div>
        </div>
      </GlassCard>
    </div>
  );
}
