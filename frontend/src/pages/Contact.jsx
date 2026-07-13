import { useState } from 'react';
import { Mail, Send } from 'lucide-react';
import { ErrorBanner, GlassCard, PageHeader } from '../components/Primitives.jsx';
import { api } from '../utils/api.js';

export default function Contact() {
  const [form, setForm] = useState({ name: '', email: '', message: '' });
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [busy, setBusy] = useState(false);

  const submit = async (event) => {
    event.preventDefault();
    setBusy(true);
    setError('');
    setSuccess('');
    try {
      const response = await api.post('/contact', form);
      setSuccess(`Message saved. Reference: ${response.data.message_id.slice(0, 8)}`);
      setForm({ name: '', email: '', message: '' });
    } catch (err) {
      setError(err.response?.data?.error || err.message || 'Could not send message.');
    } finally {
      setBusy(false);
    }
  };

  return (
    <div>
      <ErrorBanner message={error} />
      <PageHeader
        title="Contact"
        subtitle="Send a message into the same Flask backend and SQLite database used by the rest of the demo."
      />
      <div className="grid gap-5 xl:grid-cols-[.75fr_1.25fr]">
        <GlassCard className="p-6">
          <Mail className="text-electric" size={38} />
          <h2 className="mt-5 font-display text-3xl font-black">Project review channel</h2>
          <p className="mt-3 leading-7 text-emerald-50/65">
            Use this form during the review demo to prove the frontend can write to the backend, validate input, and show real success/error states.
          </p>
          {success && <div className="mt-5 rounded-3xl border border-emerald-300/20 bg-emerald-400/10 p-4 text-emerald-100">{success}</div>}
        </GlassCard>

        <GlassCard className="p-6">
          <form onSubmit={submit} className="space-y-4">
            <label className="field-label">
              Name
              <input className="field-input" value={form.name} onChange={(event) => setForm({ ...form, name: event.target.value })} />
            </label>
            <label className="field-label">
              Email
              <input className="field-input" type="email" value={form.email} onChange={(event) => setForm({ ...form, email: event.target.value })} />
            </label>
            <label className="field-label">
              Message
              <textarea className="field-input min-h-40 resize-y" value={form.message} onChange={(event) => setForm({ ...form, message: event.target.value })} />
            </label>
            <button disabled={busy} className="btn-primary w-full justify-center py-4">
              <Send size={18} /> {busy ? 'Saving…' : 'Send Message'}
            </button>
          </form>
        </GlassCard>
      </div>
    </div>
  );
}
