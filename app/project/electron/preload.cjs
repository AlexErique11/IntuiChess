// Electron preload script
// Runs in isolated context; expose safe APIs only
const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electron', {
  // Example: no-op ping for wiring test
  ping: () => 'pong',
  
  // Chess analysis using ML models
  analyzeChessPosition: async (fen, avgElo, timeControl) => {
    try {
      const result = await ipcRenderer.invoke('analyze-chess-position', { fen, avgElo, timeControl });
      return result;
    } catch (error) {
      console.error('Chess analysis failed:', error);
      throw error;
    }
  }
});
