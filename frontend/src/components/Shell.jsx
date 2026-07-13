import { NavLink, Outlet } from 'react-router-dom';
import { motion } from 'framer-motion';
import {
  Activity,
  BarChart3,
  Database,
  History,
  Home,
  Info,
  Mail,
  Network,
  UploadCloud,
} from 'lucide-react';
import Brand from './Brand.jsx';

const navItems = [
  { to: '/', label: 'Home', icon: Home },
  { to: '/upload', label: 'Upload & Analyze', icon: UploadCloud },
  { to: '/history', label: 'Predictions', icon: History },
  { to: '/dataset', label: 'Datasets', icon: Database },
  { to: '/model', label: 'Model', icon: Network },
  { to: '/about', label: 'About', icon: Info },
  { to: '/contact', label: 'Contact', icon: Mail },
];

export default function Shell() {
  return (
    <div className="min-h-screen overflow-x-hidden bg-night text-white">
      <div className="leaf-orbit" />
      <div className="mx-auto flex min-h-screen max-w-[1500px] gap-5 px-4 py-4 lg:px-6">
        <aside className="hidden w-72 shrink-0 flex-col rounded-[2rem] border border-emerald-200/10 bg-pine/75 p-4 shadow-glass backdrop-blur-2xl lg:sticky lg:top-4 lg:flex lg:h-[calc(100vh-2rem)]">
          <Brand />
          <nav className="mt-8 space-y-2">
            {navItems.map(({ to, label, icon: Icon }) => (
              <NavLink key={to} to={to} className={({ isActive }) => `nav-item ${isActive ? 'nav-item-active' : ''}`}>
                <Icon size={18} />
                <span>{label}</span>
              </NavLink>
            ))}
          </nav>
          <div className="mt-auto rounded-3xl border border-emerald-200/10 bg-white/[0.04] p-4 text-sm text-emerald-50/75">
            <div className="mb-3 flex items-center gap-2 text-emerald-300">
              <Activity size={18} />
              <span className="font-semibold">Live ML Stack</span>
            </div>
            CNN feature extraction, LSTM trend modelling, and real Flask inference are connected.
          </div>
        </aside>

        <main className="min-w-0 flex-1">
          <header className="sticky top-3 z-30 mb-5 flex items-center justify-between rounded-[1.5rem] border border-emerald-200/10 bg-pine/70 px-4 py-3 shadow-glass backdrop-blur-2xl lg:hidden">
            <Brand compact />
            <NavLink to="/upload" className="btn-primary px-4 py-2 text-xs">Analyze</NavLink>
          </header>

          <div className="mb-4 grid grid-cols-4 gap-2 lg:hidden">
            {navItems.slice(0, 4).map(({ to, label, icon: Icon }) => (
              <NavLink key={to} to={to} className={({ isActive }) => `mobile-nav ${isActive ? 'mobile-nav-active' : ''}`}>
                <Icon size={16} />
                <span>{label.split(' ')[0]}</span>
              </NavLink>
            ))}
          </div>

          <motion.div
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.45, ease: 'easeOut' }}
          >
            <Outlet />
          </motion.div>
        </main>
      </div>
    </div>
  );
}
