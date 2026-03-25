# 🚀 How to Start Chess Analyzer

## 📋 **Two Options Available:**

### 1. 🖥️ **Desktop App (Recommended) - Real ML Analysis**
**✅ This version runs your actual ML models and shows real eval bars!**

#### Quick Start:
**Double-click:** `start_desktop_app.bat`

#### Manual Start:
```bash
# Terminal 1: Start dev server
node node_modules/vite/bin/vite.js

# Terminal 2: Start Electron app (after dev server is running)
node_modules/.bin/electron.cmd .
```

#### What you get:
- ✅ **Real ML predictions** from your trained models
- ✅ **Elo-specific analysis** (800-, 800-1100, 1100-1400, etc.)
- ✅ **Time control sensitivity** (blitz vs rapid_classical)
- ✅ **Eval bars show actual values** from position_quality & move_ease
- ✅ **Feature tooltips** with all chess metrics
- ✅ **No API needed** - direct Python integration

---

### 2. 🌐 **Web Browser (localhost) - Mock Data Only**
**⚠️ This version shows fake/mock eval bars - not your real ML models!**

#### Start:
```bash
# Just the dev server
node node_modules/vite/bin/vite.js
# Then visit: http://localhost:5173
```

#### What you get:
- ❌ **Mock data only** - not your real analysis
- ❌ **Eval bars show fake values**
- ✅ **UI works** for testing interface
- ❌ **No Python execution** (browser security)

---

## 🎯 **For Real Chess Analysis:**

**Use the Desktop App!** The web version can't run Python, so it falls back to mock data.

## 🔧 **Requirements:**
- Python with your ML models installed
- Node.js (already have v22.13.0 ✅)
- Stockfish executable at: `C:\Users\alexa\Downloads\stockfish\stockfish-windows-x86-64-avx2.exe`

## 🐛 **Troubleshooting:**
If the batch file doesn't work, run the manual commands in PowerShell:
```powershell
# Terminal 1
Set-Location "C:\Users\alexa\OneDrive\Desktop\projects\ChessAnalyser\app\project"
node node_modules/vite/bin/vite.js

# Terminal 2 (after first one starts)
Set-Location "C:\Users\alexa\OneDrive\Desktop\projects\ChessAnalyser\app\project"  
node_modules/.bin/electron.cmd .
```

## 📊 **Expected Results in Desktop App:**
- **Left eval bar**: Position Quality (-10 to +10)
- **Right eval bar**: Move Ease (-10 to +10)  
- **Info icons**: Hover to see features like mobility, king_safety, center_control, etc.
- **Settings panel**: Change Elo (800-3000) and time control to see different model predictions