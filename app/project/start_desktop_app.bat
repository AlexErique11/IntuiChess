@echo off
echo ========================================================
echo 🚀 Chess Analyzer Desktop App with Real ML Models
echo ========================================================
echo.
echo ✅ Automatic re-analysis on every change:
echo    • Position changes (moves, FEN input, navigation)
echo    • Elo rating changes (800- to 2200+)
echo    • Time control changes (blitz vs rapid_classical)
echo.
echo 📊 Real eval bars show:
echo    • Position Quality (-10 to +10)
echo    • Move Ease (-10 to +10) 
echo    • All chess features in tooltips
echo.

REM Change to the correct directory
cd /d "C:\Users\alexa\OneDrive\Desktop\projects\ChessAnalyser\app\project"

REM Start the Vite development server (renderer process)
echo 📡 Starting development server...
start /min cmd /k "node node_modules/vite/bin/vite.js"

REM Wait a moment for the dev server to start
timeout /t 5 /nobreak

REM Start the Electron app (main process)
echo 🖥️  Launching desktop app...
node_modules/.bin/electron.cmd .

echo.
echo ✅ Desktop app should now be running!
echo   - Real ML models will analyze positions
echo   - Eval bars show position_quality and move_ease
echo   - Info tooltips show all features
echo.
pause