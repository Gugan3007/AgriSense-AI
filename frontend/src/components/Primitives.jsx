import { motion } from 'framer-motion';

export function PageHeader({ title, subtitle, actions }) {
  return (
    <div className="mb-6 flex flex-col justify-between gap-4 md:flex-row md:items-end">
      <div>
        <h1 className="max-w-4xl font-display text-4xl font-black leading-tight tracking-tight text-white md:text-6xl">
          {title}
        </h1>
        {subtitle && <p className="mt-3 max-w-2xl text-base leading-7 text-emerald-50/70 md:text-lg">{subtitle}</p>}
      </div>
      {actions && <div className="flex flex-wrap gap-3">{actions}</div>}
    </div>
  );
}

export function GlassCard({ children, className = '', delay = 0 }) {
  return (
    <motion.section
      initial={{ opacity: 0, y: 18 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay, duration: 0.45, ease: 'easeOut' }}
      className={`glass-card ${className}`}
    >
      {children}
    </motion.section>
  );
}

export function MetricCard({ label, value, detail, tone = 'emerald', icon }) {
  const toneClass = {
    emerald: 'text-emerald-300',
    blue: 'text-blue-300',
    amber: 'text-amber-300',
    red: 'text-red-300',
  }[tone];
  return (
    <GlassCard className="glass-card-hover p-5">
      <div className={`mb-4 ${toneClass}`}>{icon}</div>
      <p className="text-sm font-medium text-emerald-50/60">{label}</p>
      <p className="mt-1 font-display text-3xl font-black text-white">{value}</p>
      <p className="mt-2 text-xs text-emerald-50/50">{detail}</p>
    </GlassCard>
  );
}

export function Skeleton({ className = '' }) {
  return <div className={`animate-pulse rounded-3xl bg-white/10 ${className}`} />;
}

export function ErrorBanner({ message }) {
  if (!message) return null;
  return (
    <div className="mb-5 rounded-3xl border border-red-300/25 bg-red-500/10 px-5 py-4 text-sm text-red-100">
      {message}
    </div>
  );
}

export function StatusPill({ children, color = '#6FE27D' }) {
  return (
    <span className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/[0.06] px-3 py-1 text-xs font-bold text-white">
      <span className="h-2 w-2 rounded-full" style={{ background: color }} />
      {children}
    </span>
  );
}
