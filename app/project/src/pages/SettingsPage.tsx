import { useState } from 'react';
import { Save, Loader2 } from 'lucide-react';
import { useSettings } from '../contexts/SettingsContext';

const TIME_CONTROLS = [
  { value: 'blitz', label: 'Blitz' },
  { value: 'rapid_classical', label: 'Rapid/Classical' }
];

export default function SettingsPage() {
  const { settings, updateSettings, isLoading } = useSettings();
  const [saving, setSaving] = useState(false);

  const saveSettings = async () => {
    setSaving(true);
    try {
      // Settings are automatically saved via the context
      // This just provides user feedback
      await new Promise(resolve => setTimeout(resolve, 500)); // Simulate save time
      console.log('✅ Settings saved:', settings);
    } catch (error) {
      console.error('Error saving settings:', error);
    } finally {
      setSaving(false);
    }
  };

  if (isLoading) {
    return (
      <div className="flex-1 bg-gradient-to-br from-orange-50 via-pink-50 to-rose-50 p-8 flex items-center justify-center">
        <Loader2 className="animate-spin text-orange-400" size={40} />
      </div>
    );
  }

  return (
    <div className="flex-1 bg-gradient-to-br from-orange-50 via-pink-50 to-rose-50 p-8">
      <div className="max-w-2xl mx-auto">
        <h2 className="text-3xl font-bold bg-gradient-to-r from-orange-600 to-rose-600 bg-clip-text text-transparent mb-8">Settings</h2>

        <div className="bg-white rounded-lg shadow-lg p-6 space-y-6">
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-2">
              🏆 Player ELO Rating
            </label>
            <div className="flex items-center gap-4">
              <input
                type="range"
                min="800"
                max="3000"
                value={settings.playerElo}
                onChange={(e) => updateSettings({ playerElo: parseInt(e.target.value) })}
                className="flex-1 h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-orange-600"
              />
              <input
                type="number"
                min="800"
                max="3000"
                value={settings.playerElo}
                onChange={(e) => updateSettings({ playerElo: parseInt(e.target.value) || 1500 })}
                className="w-24 px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-orange-500 focus:border-transparent"
              />
            </div>
            <p className="text-sm text-slate-500 mt-2">
              Your chess rating affects which ML models are used for analysis (800 - 3000)
            </p>
          </div>

          <div>
            <label className="block text-sm font-medium text-slate-700 mb-2">
              ⏱️ Time Control
            </label>
            <select
              value={settings.timeControl}
              onChange={(e) => updateSettings({ timeControl: e.target.value })}
              className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-orange-500 focus:border-transparent"
            >
              {TIME_CONTROLS.map((tc) => (
                <option key={tc.value} value={tc.value}>
                  {tc.label}
                </option>
              ))}
            </select>
            <p className="text-sm text-slate-500 mt-2">
              Choose the time control for ML model selection (Blitz vs Rapid/Classical)
            </p>
          </div>

          <button
            onClick={saveSettings}
            disabled={saving}
            className="w-full py-3 px-4 bg-gradient-to-r from-orange-600 to-rose-600 hover:from-orange-700 hover:to-rose-700 disabled:from-orange-400 disabled:to-rose-400 text-white rounded-lg font-medium transition-colors flex items-center justify-center gap-2 shadow-lg"
          >
            {saving ? (
              <>
                <Loader2 className="animate-spin" size={18} />
                Saving...
              </>
            ) : (
              <>
                <Save size={18} />
                Save Settings
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}
