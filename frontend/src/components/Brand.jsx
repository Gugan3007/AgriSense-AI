import { Leaf } from 'lucide-react';

export default function Brand({ compact = false }) {
  return (
    <div className="flex items-center gap-3">
      <div className="grid h-10 w-10 place-items-center rounded-2xl bg-emerald-400/15 text-emerald-300 ring-1 ring-emerald-300/25">
        <Leaf size={22} fill="currentColor" />
      </div>
      {!compact && (
        <div>
          <p className="font-display text-lg font-black tracking-tight text-white">AgriSense AI</p>
          <p className="text-xs font-medium text-emerald-200/70">Predict Today. Protect Tomorrow.</p>
        </div>
      )}
    </div>
  );
}
