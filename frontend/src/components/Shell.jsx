import { NavLink, Outlet, useLocation } from 'react-router-dom';
import { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import {
  Activity,
  BarChart3,
  Database,
  History,
  Home,
  Info,
  Mail,
  Menu,
  Network,
  UploadCloud,
  X,
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
  const [menuOpen, setMenuOpen] = useState(false);
  const location = useLocation();

  useEffect(() => setMenuOpen(false), [location.pathname]);

  return (
    <div className="min-h-screen overflow-x-hidden bg-night text-white">
      <div className="leaf-orbit" />
      <div className="mx-auto flex min-h-screen max-w-[1500px] gap-5 px-4 py-4 lg:px-6">
        <aside className="hidden w-72 shrink-0 flex-col rounded-[2rem] border border-emerald-200/10 bg-pine/75 p-4 shadow-glass backdrop-blur-2xl lg:sticky lg:top-4 lg:flex lg:h-[calc(100vh-2rem)]">
          <div className="px-2 pt-1"><Brand /></div>
          <div className="mx-2 mt-6 flex items-center gap-2 rounded-full border border-emerald-300/15 bg-emerald-300/[0.06] px-3 py-2 text-[11px] font-bold uppercase tracking-[0.16em] text-emerald-200">
            <span className="relative flex h-2 w-2"><span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-300 opacity-60" /><span className="relative h-2 w-2 rounded-full bg-emerald-300" /></span>
            System operational
          </div>
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
          <header className="sticky top-3 z-40 mb-5 flex items-center justify-between rounded-[1.5rem] border border-emerald-200/10 bg-pine/90 px-4 py-3 shadow-glass backdrop-blur-2xl lg:hidden">
            <Brand />
            <button type="button" onClick={() => setMenuOpen((value) => !value)} className="grid h-11 w-11 place-items-center rounded-2xl border border-white/10 bg-white/[0.05] text-white" aria-label="Toggle navigation" aria-expanded={menuOpen}>
              {menuOpen ? <X size={20} /> : <Menu size={20} />}
            </button>
          </header>

          <div className={`${menuOpen ? 'grid' : 'hidden'} absolute left-4 right-4 top-20 z-30 grid-cols-2 gap-2 rounded-[1.75rem] border border-emerald-200/10 bg-pine/95 p-3 shadow-glass backdrop-blur-2xl lg:hidden`}>
            {navItems.map(({ to, label, icon: Icon }) => (
              <NavLink key={to} to={to} className={({ isActive }) => `mobile-nav ${isActive ? 'mobile-nav-active' : ''}`}>
                <Icon size={16} />
                <span>{label}</span>
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
