import { useState } from 'react';
import Sidebar from './components/Sidebar';
import HomePage from './pages/HomePage';
import SettingsPage from './pages/SettingsPage';
import BrowserPage from './pages/BrowserPage';
import { SettingsProvider } from './contexts/SettingsContext';

function App() {
  const [currentPage, setCurrentPage] = useState('home');

  const renderPage = () => {
    switch (currentPage) {
      case 'home':
        return <HomePage />;
      case 'settings':
        return <SettingsPage />;
      case 'browser':
        return <BrowserPage />;
      default:
        return <HomePage />;
    }
  };

  return (
    <SettingsProvider>
      <div className="flex h-screen overflow-hidden">
        <Sidebar currentPage={currentPage} onNavigate={setCurrentPage} />
        {renderPage()}
      </div>
    </SettingsProvider>
  );
}

export default App;
