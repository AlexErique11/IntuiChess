import { createContext, useContext, useState, useEffect, ReactNode } from 'react';

interface Settings {
  playerElo: number;
  timeControl: string;
}

interface SettingsContextType {
  settings: Settings;
  updateSettings: (newSettings: Partial<Settings>) => void;
  isLoading: boolean;
}

const defaultSettings: Settings = {
  playerElo: 1500,
  timeControl: 'blitz'
};

const SettingsContext = createContext<SettingsContextType | undefined>(undefined);

export function SettingsProvider({ children }: { children: ReactNode }) {
  const [settings, setSettings] = useState<Settings>(defaultSettings);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    // Load settings from localStorage as a fallback (since Supabase might not work in all environments)
    loadSettings();
  }, []);

  const loadSettings = async () => {
    try {
      // Try localStorage first (for Electron app)
      const stored = localStorage.getItem('chess-analyzer-settings');
      if (stored) {
        const parsed = JSON.parse(stored);
        setSettings({ ...defaultSettings, ...parsed });
      }
      
      // Note: If you want to use Supabase, you can add that logic here
      // For now, we'll use localStorage which works in Electron
    } catch (error) {
      console.error('Error loading settings:', error);
      // Use defaults on error
    } finally {
      setIsLoading(false);
    }
  };

  const updateSettings = (newSettings: Partial<Settings>) => {
    const updatedSettings = { ...settings, ...newSettings };
    setSettings(updatedSettings);
    
    // Save to localStorage
    try {
      localStorage.setItem('chess-analyzer-settings', JSON.stringify(updatedSettings));
    } catch (error) {
      console.error('Error saving settings:', error);
    }
  };

  return (
    <SettingsContext.Provider value={{ settings, updateSettings, isLoading }}>
      {children}
    </SettingsContext.Provider>
  );
}

export function useSettings() {
  const context = useContext(SettingsContext);
  if (context === undefined) {
    throw new Error('useSettings must be used within a SettingsProvider');
  }
  return context;
}