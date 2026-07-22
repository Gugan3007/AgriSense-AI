import { useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { CloudUpload, FileImage, Leaf, Loader2, Plus, WandSparkles } from 'lucide-react';
import { ErrorBanner, GlassCard, PageHeader } from '../components/Primitives.jsx';
import { api, sensorColumns } from '../utils/api.js';

const initialReadings = Array.from({ length: 7 }, (_, index) => ({
  day: index + 1,
  Soil_Moisture: 58 - index * 1.4,
  Temperature: 25.8 + index * 0.35,
  Humidity: 64 - index * 1.1,
  Light_Intensity: 14500 + index * 180,
}));

export default function Upload() {
  const navigate = useNavigate();
  const [file, setFile] = useState(null);
  const [plantType, setPlantType] = useState('Tomato');
  const [useSimulated, setUseSimulated] = useState(true);
  const [readings, setReadings] = useState(initialReadings);
  const [stage, setStage] = useState('');
  const [error, setError] = useState('');

  const preview = useMemo(() => (file ? URL.createObjectURL(file) : ''), [file]);
  const busy = Boolean(stage);

  const updateReading = (index, key, value) => {
    setReadings((current) => current.map((row, rowIndex) => (
      rowIndex === index ? { ...row, [key]: Number(value) } : row
    )));
  };

  const analyze = async () => {
    if (!file) {
      setError('Upload one leaf image before analyzing.');
      return;
    }
    setError('');
    try {
      setStage('Uploading leaf image…');
      const form = new FormData();
      form.append('image', file);
      const upload = await api.post('/upload', form, { headers: { 'Content-Type': 'multipart/form-data' } });

      setStage('Extracting CNN features…');
      await new Promise((resolve) => setTimeout(resolve, 450));
      setStage('Fusing image and sensor probabilities…');
      const prediction = await api.post('/predict', {
        upload_id: upload.data.upload_id,
        plant_type: plantType,
        recent_sensor_readings: readings.map(({ day, ...row }) => row),
      });
      setStage('Finalizing dashboard…');
      await new Promise((resolve) => setTimeout(resolve, 350));
      navigate(`/prediction/${prediction.data.prediction_id}`);
    } catch (err) {
      setError(err.response?.data?.error || err.message || 'Analysis failed.');
    } finally {
      setStage('');
    }
  };

  return (
    <div>
      <ErrorBanner message={error} />
      <PageHeader
        title="Upload & Analyze"
        subtitle="Submit one current leaf image and recent sensor telemetry. AgriSense will run the trained image-first model and open a live prediction dashboard."
      />

      <div className="grid gap-5 xl:grid-cols-[1.1fr_.9fr]">
        <GlassCard className="p-5">
          <label className="dropzone">
            <input
              type="file"
              accept="image/png,image/jpeg,image/webp"
              className="sr-only"
              onChange={(event) => setFile(event.target.files?.[0] || null)}
            />
            {preview ? (
              <img src={preview} alt="Uploaded leaf preview" className="h-full w-full rounded-[1.7rem] object-cover" />
            ) : (
              <div className="text-center">
                <CloudUpload className="mx-auto mb-4 text-electric" size={50} />
                <p className="font-display text-2xl font-black">Drag & drop a leaf image here</p>
                <p className="mt-2 text-sm text-emerald-50/60">or click to choose JPG, PNG, or WebP</p>
              </div>
            )}
          </label>

          <div className="mt-5 grid gap-4 md:grid-cols-2">
            <label className="field-label">
              Plant type
              <select value={plantType} onChange={(event) => setPlantType(event.target.value)} className="field-input">
                {['Apple', 'Tomato', 'Potato', 'Corn', 'Grape', 'Pepper', 'Strawberry', 'Soybean'].map((crop) => (
                  <option key={crop}>{crop}</option>
                ))}
              </select>
            </label>
            <div className="rounded-3xl border border-emerald-200/10 bg-white/[0.04] p-4">
              <p className="text-sm font-bold text-white">Capture tips</p>
              <ul className="mt-3 space-y-2 text-sm text-emerald-50/60">
                <li>Use clear, well-lit images.</li>
                <li>Keep one leaf centered.</li>
                <li>Avoid shadows and glare.</li>
              </ul>
            </div>
          </div>
        </GlassCard>

        <GlassCard className="p-5">
          <div className="flex flex-col justify-between gap-3 md:flex-row md:items-center">
            <div>
              <h2 className="section-title">Recent sensor telemetry</h2>
              <p className="text-sm text-emerald-50/60">Seven readings are ideal for the model sequence.</p>
            </div>
            <button className="btn-secondary text-xs" onClick={() => setUseSimulated((value) => !value)}>
              <Plus size={16} /> {useSimulated ? 'Edit custom data' : 'Use sample trend'}
            </button>
          </div>

          <div className="mt-5 max-h-[420px] space-y-3 overflow-auto pr-2">
            {readings.map((row, index) => (
              <div key={row.day} className="rounded-3xl border border-white/10 bg-white/[0.035] p-4">
                <p className="mb-3 text-sm font-bold text-emerald-200">Day {row.day}</p>
                <div className="grid grid-cols-2 gap-3">
                  {sensorColumns.map((column) => (
                    <label key={column} className="text-xs text-emerald-50/55">
                      {column.replace('_', ' ')}
                      <input
                        disabled={useSimulated}
                        type="number"
                        value={row[column]}
                        onChange={(event) => updateReading(index, column, event.target.value)}
                        className="field-input mt-1 py-2 text-sm disabled:opacity-65"
                      />
                    </label>
                  ))}
                </div>
              </div>
            ))}
          </div>

          <motion.button
            whileTap={{ scale: 0.98 }}
            onClick={analyze}
            disabled={busy}
            className="btn-primary mt-5 w-full justify-center py-4"
          >
            {busy ? <Loader2 className="animate-spin" size={18} /> : <WandSparkles size={18} />}
            {busy ? stage : 'Analyze Now'}
          </motion.button>
          <p className="mt-3 text-center text-xs text-emerald-50/45">Analysis usually takes 5–20 seconds after the model is warm.</p>
        </GlassCard>
      </div>
    </div>
  );
}
