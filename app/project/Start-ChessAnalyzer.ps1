# Chess Analyzer Desktop App Startup Script
# Run this in PowerShell to start the desktop app with real ML analysis

Write-Host "🚀 Starting Chess Analyzer Desktop App..." -ForegroundColor Cyan
Write-Host ""
Write-Host "This will launch the Electron desktop app with real ML analysis." -ForegroundColor Green
Write-Host "✅ Real eval bars from your trained models" -ForegroundColor Green
Write-Host "✅ Elo-specific predictions (800-, 800-1100, 1100-1400, etc.)" -ForegroundColor Green
Write-Host "✅ Time control sensitivity (blitz vs rapid_classical)" -ForegroundColor Green
Write-Host ""

# Change to correct directory
Set-Location "C:\Users\alexa\OneDrive\Desktop\projects\ChessAnalyser\app\project"

# Start Vite dev server in background
Write-Host "📡 Starting development server..." -ForegroundColor Yellow
$ViteProcess = Start-Process -FilePath "node" -ArgumentList "node_modules/vite/bin/vite.js" -PassThru -WindowStyle Minimized

# Wait for dev server to start
Write-Host "⏳ Waiting for dev server to initialize..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

# Check if Vite is running
if ($ViteProcess -and !$ViteProcess.HasExited) {
    Write-Host "✅ Dev server started successfully" -ForegroundColor Green
    
    # Start Electron app
    Write-Host "🖥️  Launching desktop app..." -ForegroundColor Yellow
    try {
        & "node_modules/.bin/electron.cmd" .
        Write-Host ""
        Write-Host "✅ Desktop app should now be running!" -ForegroundColor Green
        Write-Host "   - Real ML models will analyze positions" -ForegroundColor White
        Write-Host "   - Eval bars show position_quality and move_ease" -ForegroundColor White
        Write-Host "   - Info tooltips show all features" -ForegroundColor White
    }
    catch {
        Write-Host "❌ Failed to start Electron app: $($_.Exception.Message)" -ForegroundColor Red
    }
}
else {
    Write-Host "❌ Failed to start dev server" -ForegroundColor Red
}

Write-Host ""
Write-Host "Press any key to continue..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")