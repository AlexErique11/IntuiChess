// Ambient declaration for the preload-exposed API
export {}; // ensure this is a module

declare global {
  interface Window {
    electron: {
      ping: () => string;
      analyzeChessPosition: (fen: string, avgElo: number, timeControl: string) => Promise<{
        success: boolean;
        position_quality: number;
        move_ease: number;
        features: Record<string, number | string>;
        elo_range: string;
        time_control: string;
        raw_scores?: {
          position_quality: number;
          move_ease: number;
        };
        error?: string;
      }>;
    };
  }
}
