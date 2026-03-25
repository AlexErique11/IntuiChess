// Electron main process
// CommonJS to align with package.json "main": "electron/main.cjs"
const { app, BrowserWindow, shell, ipcMain } = require('electron');
const { spawn, exec } = require('child_process');
const path = require('path');
const fs = require('fs');

const isDev = process.env.ELECTRON_START_URL || process.env.VITE_DEV_SERVER_URL || process.env.NODE_ENV === 'development';

/**
 * Create the main application window
 */
function createWindow() {
  const win = new BrowserWindow({
    width: 1600,
    height: 1000,
    minWidth: 1400,
    minHeight: 900,
    show: false,
    webPreferences: {
      preload: path.join(__dirname, 'preload.cjs'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false,
    },
  });

  win.once('ready-to-show', () => win.show());

  if (isDev) {
    const devServerURL = process.env.ELECTRON_START_URL || process.env.VITE_DEV_SERVER_URL || 'http://localhost:5173';
    win.loadURL(devServerURL);
    win.webContents.openDevTools({ mode: 'detach' });
  } else {
    const indexPath = path.join(__dirname, '..', 'dist', 'index.html');
    if (fs.existsSync(indexPath)) {
      win.loadFile(indexPath);
    } else {
      // Fallback if path differs
      win.loadURL(`file://${path.resolve(process.cwd(), 'dist', 'index.html')}`);
    }
  }

  // Open URLs in user's browser rather than in-app
  win.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url);
    return { action: 'deny' };
  });
}

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});

// Chess analysis IPC handler
ipcMain.handle('analyze-chess-position', async (event, { fen, avgElo, timeControl }) => {
  return new Promise((resolve, reject) => {
    console.log(`🧠 Analyzing chess position: Elo=${avgElo}, TimeControl=${timeControl}`);
    
    // Path to Python wrapper (adjust relative to the Electron app location)
    const scriptPath = path.join(__dirname, '..', '..', '..', 'chess_analyzer_wrapper.py');
    
    const pythonProcess = spawn('python', [scriptPath, fen, avgElo.toString(), timeControl], {
      cwd: path.join(__dirname, '..', '..', '..')
    });
    
    let outputData = '';
    let errorData = '';

    pythonProcess.stdout.on('data', (data) => {
      outputData += data.toString();
    });

    pythonProcess.stderr.on('data', (data) => {
      errorData += data.toString();
    });

    pythonProcess.on('close', (code) => {
      if (code === 0) {
        try {
          const result = JSON.parse(outputData);
          console.log(`✅ Chess analysis complete: PosQuality=${result.position_quality.toFixed(2)}, MoveEase=${result.move_ease.toFixed(2)}`);
          resolve(result);
        } catch (error) {
          console.error('❌ Failed to parse Python output:', error);
          reject({ error: 'Failed to parse analysis results', details: error.message });
        }
      } else {
        console.error('❌ Python analysis failed:', errorData);
        reject({ error: 'Python script failed', details: errorData || 'Unknown error' });
      }
    });

    pythonProcess.on('error', (error) => {
      console.error('❌ Failed to start Python process:', error);
      reject({ error: 'Failed to start analysis', details: error.message });
    });
  });
});

app.whenReady().then(() => {
  createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});
