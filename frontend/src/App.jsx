import { Route, Routes } from 'react-router-dom';
import { lazy, Suspense } from 'react';
import Shell from './components/Shell.jsx';
import { Skeleton } from './components/Primitives.jsx';

const About = lazy(() => import('./pages/About.jsx'));
const Contact = lazy(() => import('./pages/Contact.jsx'));
const Dataset = lazy(() => import('./pages/Dataset.jsx'));
const History = lazy(() => import('./pages/History.jsx'));
const Home = lazy(() => import('./pages/Home.jsx'));
const Model = lazy(() => import('./pages/Model.jsx'));
const Prediction = lazy(() => import('./pages/Prediction.jsx'));
const Upload = lazy(() => import('./pages/Upload.jsx'));

function RouteFallback() {
  return <Skeleton className="h-[70vh]" />;
}

export default function App() {
  return (
    <Suspense fallback={<RouteFallback />}>
      <Routes>
        <Route element={<Shell />}>
          <Route index element={<Home />} />
          <Route path="/upload" element={<Upload />} />
          <Route path="/prediction/:id" element={<Prediction />} />
          <Route path="/history" element={<History />} />
          <Route path="/dataset" element={<Dataset />} />
          <Route path="/model" element={<Model />} />
          <Route path="/about" element={<About />} />
          <Route path="/contact" element={<Contact />} />
        </Route>
      </Routes>
    </Suspense>
  );
}
