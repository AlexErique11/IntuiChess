import { Info, X } from 'lucide-react';
import { useState } from 'react';

interface PositionInfoProps {
  features: Record<string, number | string>;
  eloRange: string;
  timeControl: string;
}

export default function PositionInfo({ features, eloRange, timeControl }: PositionInfoProps) {
  const [showModal, setShowModal] = useState(false);

  // Select most important features to display (no scrolling needed)
  const importantFeatures = [
    'material_imbalance', 'mobility', 'king_safety', 'king_exposure',
    'center_control', 'space_control', 'piece_coordination', 'pawn_structure',
    'stockfish_eval', 'volatility', 'hanging_pieces', 'pins'
  ];

  const displayFeatures = importantFeatures
    .filter(key => key in features)
    .slice(0, 12) // Limit to 12 features to fit without scrolling
    .map(key => ({ key, value: features[key] }));

  return (
    <>
      {/* Info Button */}
      <button
        onClick={() => setShowModal(true)}
        className="p-2 bg-white hover:bg-slate-50 rounded-full shadow-lg border border-slate-200 transition-all hover:shadow-xl"
        title="View Position Features"
      >
        <Info size={20} className="text-slate-600" />
      </button>

      {/* Modal */}
      {showModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-xl shadow-2xl p-6 w-96 max-w-[90vw] max-h-[90vh]">
            {/* Header */}
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-xl font-bold text-slate-800">🧠 Position Analysis</h3>
              <button
                onClick={() => setShowModal(false)}
                className="p-1 hover:bg-slate-100 rounded-full transition-colors"
              >
                <X size={20} className="text-slate-500" />
              </button>
            </div>

            {/* Model Info */}
            <div className="mb-4 p-3 bg-gradient-to-r from-blue-50 to-cyan-50 rounded-lg border border-blue-200">
              <div className="text-sm space-y-1">
                <div><strong>🏆 Elo Range:</strong> {eloRange}</div>
                <div><strong>⏱️ Time Control:</strong> {timeControl === 'blitz' ? 'Blitz' : 'Rapid/Classical'}</div>
              </div>
            </div>

            {/* Features Grid */}
            <div className="space-y-2">
              <h4 className="text-sm font-semibold text-slate-700 mb-3">📊 Key Position Features:</h4>
              <div className="grid grid-cols-2 gap-2 text-sm">
                {displayFeatures.map(({ key, value }) => (
                  <div key={key} className="flex justify-between p-2 bg-slate-50 rounded">
                    <span className="text-slate-600 truncate" title={key}>
                      {key.replace(/_/g, ' ')}:
                    </span>
                    <span className="font-mono text-slate-800 ml-2">
                      {typeof value === 'number' ? value.toFixed(2) : value}
                    </span>
                  </div>
                ))}
              </div>
            </div>

            {/* Footer */}
            <div className="mt-4 pt-3 border-t border-slate-200 text-xs text-slate-500 text-center">
              Showing key features from ML model analysis
            </div>
          </div>
        </div>
      )}
    </>
  );
}