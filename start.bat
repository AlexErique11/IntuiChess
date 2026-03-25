@echo off
echo Starting Chess Analyser with Real ML Analysis...
echo.
cd app\project
echo Installing dependencies (if needed)...
call npm install --silent
echo.
echo Starting Node.js backend (port 3001) and React frontend (port 5173)...
echo Backend will call Python ML models with Stockfish
echo.
start "Backend" cmd /k "npm run backend"
timeout /t 3
echo.
echo Starting frontend...
call npm run dev
