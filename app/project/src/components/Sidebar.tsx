import { Home, Settings, Globe } from 'lucide-react';

interface SidebarProps {
  currentPage: string;
  onNavigate: (page: string) => void;
}

export default function Sidebar({ currentPage, onNavigate }: SidebarProps) {
  const navItems = [
    { id: 'home', label: 'Chess Board', icon: Home },
    { id: 'settings', label: 'Settings', icon: Settings },
    { id: 'browser', label: 'Open Chess.com', icon: Globe },
  ];

  return (
    <div className="w-64 bg-gradient-to-b from-slate-900 via-slate-800 to-slate-900 text-white h-screen flex flex-col shadow-2xl">
      <div className="p-6 border-b border-cyan-500/30">
        <h1 className="text-2xl font-bold tracking-tight bg-gradient-to-r from-cyan-400 to-teal-400 bg-clip-text text-transparent">IntuiChess</h1>
      </div>

      <nav className="flex-1 p-4">
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = currentPage === item.id;

          return (
            <button
              key={item.id}
              onClick={() => onNavigate(item.id)}
              className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg mb-2 transition-all ${
                isActive
                  ? 'bg-gradient-to-r from-cyan-600 to-teal-600 text-white shadow-lg shadow-cyan-500/50'
                  : 'text-slate-300 hover:bg-slate-700/50 hover:text-white'
              }`}
            >
              <Icon size={20} />
              <span className="font-medium">{item.label}</span>
            </button>
          );
        })}
      </nav>

      <div className="p-4 border-t border-cyan-500/30">
        <p className="text-xs text-cyan-400/80">Analyse your position with IntuiChess</p>
      </div>
    </div>
  );
}
