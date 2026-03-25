import { ExternalLink } from 'lucide-react';

export default function BrowserPage() {
  const openChessCom = () => {
    // Open chess.com in the default browser
    window.open('https://www.chess.com', '_blank', 'noopener,noreferrer');
  };

  return (
    <div className="flex-1 bg-gradient-to-br from-green-50 via-emerald-50 to-teal-50 p-8">
      <div className="max-w-2xl mx-auto">
        <h2 className="text-3xl font-bold bg-gradient-to-r from-green-600 to-teal-600 bg-clip-text text-transparent mb-8">Open Chess.com</h2>

        <div className="bg-white rounded-lg shadow-lg p-8 text-center space-y-6">
          <div className="w-20 h-20 bg-gradient-to-br from-green-100 to-teal-100 rounded-full flex items-center justify-center mx-auto">
            <ExternalLink className="text-green-600" size={40} />
          </div>

          <div>
            <h3 className="text-xl font-semibold text-slate-800 mb-2">
              Launch Chess.com
            </h3>
            <p className="text-slate-600">
              Click the button below to open chess.com in your default browser
            </p>
          </div>

          <button
            onClick={openChessCom}
            className="px-8 py-4 bg-gradient-to-r from-green-600 to-teal-600 hover:from-green-700 hover:to-teal-700 text-white rounded-lg font-medium transition-colors flex items-center justify-center gap-3 mx-auto text-lg shadow-lg hover:shadow-xl"
          >
            <ExternalLink size={24} />
            Open Chess.com
          </button>
        </div>
      </div>
    </div>
  );
}
