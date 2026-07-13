import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { stressColors } from '../utils/api.js';

const tooltipStyle = {
  background: 'rgba(4, 22, 17, 0.94)',
  border: '1px solid rgba(167, 243, 208, 0.18)',
  borderRadius: '16px',
  color: '#fff',
};

export function ProbabilityBars({ probabilities = {} }) {
  const rows = Object.entries(probabilities).map(([name, value]) => ({ name, value: Number(value) * 100 }));
  return (
    <ResponsiveContainer width="100%" height={220}>
      <BarChart data={rows} layout="vertical" margin={{ left: 18, right: 16 }}>
        <CartesianGrid stroke="rgba(255,255,255,.08)" horizontal={false} />
        <XAxis type="number" domain={[0, 100]} hide />
        <YAxis type="category" dataKey="name" width={78} tick={{ fill: '#d8fbe7', fontSize: 12 }} />
        <Tooltip contentStyle={tooltipStyle} formatter={(value) => `${value.toFixed(1)}%`} />
        <Bar dataKey="value" radius={[0, 10, 10, 0]} isAnimationActive={false}>
          {rows.map((row) => <Cell key={row.name} fill={stressColors[row.name]} />)}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

export function TrendLine({ values = [] }) {
  const data = values.map((value, index) => ({ day: `D${index + 1}`, stress: value }));
  return (
    <ResponsiveContainer width="100%" height={220}>
      <LineChart data={data}>
        <CartesianGrid stroke="rgba(255,255,255,.08)" />
        <XAxis dataKey="day" tick={{ fill: '#d8fbe7', fontSize: 12 }} />
        <YAxis tick={{ fill: '#d8fbe7', fontSize: 12 }} />
        <Tooltip contentStyle={tooltipStyle} />
        <Line type="monotone" dataKey="stress" stroke="#F4B400" strokeWidth={3} dot={{ r: 4, fill: '#F4B400' }} isAnimationActive={false} />
      </LineChart>
    </ResponsiveContainer>
  );
}

export function DistributionChart({ distribution = {} }) {
  const rows = Object.entries(distribution);
  const total = rows.reduce((sum, [, value]) => sum + Number(value), 0) || 1;
  let cursor = 0;
  const gradient = rows.map(([name, value]) => {
    const start = cursor;
    const size = (Number(value) / total) * 100;
    cursor += size;
    return `${stressColors[name] || '#2E86FF'} ${start}% ${cursor}%`;
  }).join(', ');
  return (
    <div className="grid min-h-[260px] place-items-center">
      <div className="relative grid h-56 w-56 place-items-center rounded-full" style={{ background: `conic-gradient(${gradient})` }}>
        <div className="grid h-32 w-32 place-items-center rounded-full bg-pine text-center shadow-glass">
          <div>
            <p className="font-display text-3xl font-black">{total.toLocaleString()}</p>
            <p className="text-xs text-emerald-50/55">rows</p>
          </div>
        </div>
      </div>
    </div>
  );
}

export function ClassBars({ distribution = {} }) {
  const rows = Object.entries(distribution).map(([name, value]) => ({ name, value: Number(value) })).slice(0, 10);
  return (
    <ResponsiveContainer width="100%" height={320}>
      <BarChart data={rows} margin={{ bottom: 60 }}>
        <CartesianGrid stroke="rgba(255,255,255,.08)" />
        <XAxis dataKey="name" tick={{ fill: '#d8fbe7', fontSize: 10 }} angle={-28} textAnchor="end" interval={0} />
        <YAxis tick={{ fill: '#d8fbe7', fontSize: 12 }} />
        <Tooltip contentStyle={tooltipStyle} />
        <Bar dataKey="value" fill="#2E86FF" radius={[10, 10, 0, 0]} isAnimationActive={false} />
      </BarChart>
    </ResponsiveContainer>
  );
}

export function AccuracySparkline({ value = 93.6 }) {
  const data = [72, 81, 76, 88, 91, 90, value].map((score, index) => ({ index, score }));
  return (
    <ResponsiveContainer width="100%" height={120}>
      <AreaChart data={data}>
        <defs>
          <linearGradient id="accuracyGlow" x1="0" x2="0" y1="0" y2="1">
            <stop offset="5%" stopColor="#2E86FF" stopOpacity={0.7} />
            <stop offset="95%" stopColor="#2E86FF" stopOpacity={0} />
          </linearGradient>
        </defs>
        <Area dataKey="score" stroke="#2E86FF" fill="url(#accuracyGlow)" strokeWidth={3} isAnimationActive={false} />
      </AreaChart>
    </ResponsiveContainer>
  );
}
