# Chess Analyzer System - Startup Instructions

## Overview
The system is now fully integrated! Here's how each component works:

### 🧠 Analysis Flow
1. **Frontend** (React/TypeScript) sends FEN + player settings to **Backend**
2. **Backend** (Node.js) calls **Python Wrapper** with chess position
3. **Python Wrapper** uses **chess_analyser.py logic**:
   - Extracts features using Stockfish 
   - Loads ML models based on Elo/time control
   - Calculates predicted scores for position quality & move ease
   - Converts scores to eval bar values using non-linear mapping
4. **Eval bars** display the scores, **info tooltips** show all features

### 🚀 How to Start the System

#### Terminal 1 - Start Backend:
```bash
cd "C:\Users\alexa\OneDrive\Desktop\projects\ChessAnalyser\app\project"
node chess-analyzer-backend.js
```

#### Terminal 2 - Start Frontend:
```bash
cd "C:\Users\alexa\OneDrive\Desktop\projects\ChessAnalyser\app\project"  
npm run dev
```

#### Test the Python Wrapper directly:
```bash
python "C:\Users\alexa\OneDrive\Desktop\projects\ChessAnalyser\chess_analyzer_wrapper.py" "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" 1500 blitz
```

### 📊 What You'll See
- **Left eval bar**: Position Quality (how good/bad the position is)  
- **Right eval bar**: Move Ease (how easy it is to find good moves)
- **Info icons**: Hover to see all position features from ML analysis
- **Settings panel**: Adjust player Elo (800-3000) and time control

### 🎯 Key Features Working
✅ Real ML models from chess_analyser.py logic  
✅ Stockfish feature extraction  
✅ Elo-specific model selection  
✅ Non-linear eval bar mapping  
✅ Feature tooltips with position details  
✅ JSON API for frontend integration  

The system uses the exact same logic as chess_analyser.py but in a web interface!