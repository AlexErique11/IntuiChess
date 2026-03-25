# Chess Analyser

A human-centric chess analysis system that uses machine learning to evaluate chess positions from a player's perspective, focusing on playability and decision-making difficulty rather than traditional engine evaluation.

## 🧠 Machine Learning Research

This project explores a novel approach to chess analysis by training models on human gameplay patterns rather than engine evaluations. The goal is to understand how humans actually experience and evaluate chess positions.

### Human-Centric Position Evaluation

Traditional chess engines evaluate positions using material count and positional factors, but this project takes a different approach by modeling human decision-making patterns.

The system trains separate models for:
- **Position Quality**: How favorable a position feels for human players
- **Move Ease**: How difficult it is for players at different skill levels to find good moves

### Model Architecture

Eight distinct models are trained for different Elo ranges (800-2200+) and two time controls (blitz vs classical), each learning from millions of human game positions.

#### Advanced Feature Engineering

The models use 40+ sophisticated chess features including:

```python
# King safety network analysis
def compute_king_exposure(board):
    king_square = board.king(board.turn)
    king_zone = [sq for sq in chess.SQUARES if chess.square_distance(king_square, sq) <= 1]

    exposure_score = 0
    for sq in king_zone:
        attackers = board.attackers(not board.turn, sq)
        for attacker_sq in attackers:
            piece_type = board.piece_type_at(attacker_sq)
            exposure_score += PIECE_WEIGHTS.get(piece_type, 0.5)

    return exposure_score
```

#### Multi-Depth Analysis Pipeline

A key innovation is analyzing how move evaluations change with search depth:

```python
def analyze_move_volatility(board, engine):
    # Shallow analysis for pattern recognition
    shallow_eval = engine.analyse(board, Limit(depth=1))

    # Deep analysis for tactical assessment
    deep_eval = engine.analyse(board, Limit(depth=6))

    # Calculate volatility
    volatility = abs(deep_eval.score - shallow_eval.score)

    return volatility, deep_eval
```

## 🎯 Chess Application

Built around these ML models, the desktop application provides:

### Interactive Chess Interface
- Drag-and-drop chess board with legal move validation
- Real-time ML analysis that updates during play
- Visual evaluation bars for position quality and move ease
- Check detection with smooth animations
- Move history navigation and board orientation controls

### Analysis Features
- Detailed position information with hover tooltips
- Elo-based analysis (800-3000 rating support)
- Time control selection (blitz vs classical)
- FEN notation input/output for position sharing
- Responsive design for different screen sizes

## 🚀 Getting Started

### Desktop Application Setup

```bash
# Navigate to the app directory
cd app/project

# Install dependencies
npm install

# Launch the desktop application
npm run dev
```

The application will automatically download the ML models (~200-500MB) on first run.

### Alternative Launch Options
- Double-click `start-dev.bat` for development mode with hot-reload
- Double-click `start_desktop_app.bat` for the production desktop app

## 🛠️ Technologies Used

### Machine Learning Stack
- **Python** with scikit-learn for model training
- **Pandas & NumPy** for data processing and analysis
- **Chess library** for position manipulation
- **Stockfish** for deep tactical analysis during feature extraction
- **1B+ training positions** from Lichess human games

### Application Stack
- **React 18** with TypeScript for type-safe frontend development
- **Tailwind CSS** for responsive styling and UI components
- **Electron** for cross-platform desktop application framework
- **Vite** for fast development builds and optimized production bundles

### Chess Engine Integration
- Custom chess logic with move validation and check detection
- FEN parsing for standard chess position notation
- Real-time analysis pipeline with 300ms debouncing
- IPC communication between frontend and ML backend

## 📁 Project Structure

```
ChessAnalyser/
├── README.md                    # Project documentation
├── chess_analyser.py           # Original command-line prototype
├── app/project/                # Desktop application
│   ├── src/
│   │   ├── components/         # React UI components
│   │   ├── utils/              # Chess logic and ML integration
│   │   └── pages/              # Application screens
│   ├── electron/               # Desktop app configuration
│   │   ├── main.cjs           # Electron main process
│   │   ├── preload.cjs        # IPC communication layer
│   │   └── asset-manager.js   # ML model download management
│   └── package.json            # Node.js dependencies
└── ml_training/                # Machine learning research
    ├── feature_extraction.py   # Chess feature engineering (40+ features)
    ├── train_model.py          # Model training pipeline
    ├── features.csv            # Processed training data (1B+ positions)
    ├── elo_models/             # Trained models by skill level
    ├── feature_sets.json       # Elo-specific feature selection
    └── human_playability_model.json # Model architecture definition
```

## 🔬 Technical Implementation

### Training Methodology

1. **Data Collection**: 50M+ human games from Lichess, filtered for decisive results
2. **Feature Engineering**: 40+ features capturing human decision-making patterns
3. **Model Training**: Elo-specific models with time control variants
4. **Validation**: Cross-Elo testing achieving RMSE < 1.2 and R² > 0.85

### Real-Time Analysis Integration

```javascript
// Real-time ML analysis in the React application
const performAnalysis = async (fen) => {
  const result = await analyzePosition(fen, settings.playerElo, settings.timeControl);

  setPositionQuality(result.position_quality);
  setMoveEase(result.move_ease);
  setAnalysisFeatures(result.features);
};
```

## 📊 Performance Metrics

- **Training Data Scale**: 1B+ positions from 50M+ human games
- **Model Storage**: 200-500MB compressed ML models
- **Analysis Speed**: <100ms for position evaluation
- **Memory Usage**: ~300MB RAM for full model loading
- **Accuracy**: RMSE < 1.2, R² > 0.85 across all skill levels

## 🔮 Future Development

Planned enhancements include:
- **Reinforcement Learning**: Self-improving models through gameplay
- **Opening Analysis**: ML-powered opening preparation tools
- **Mobile Application**: React Native version for mobile devices
- **Tournament Integration**: Chess.com and Lichess API connections

## 📄 License

This project is developed for educational and research purposes. The chess analysis models and training data are available for academic and personal use.

---

*Chess Analyser* - Human-centric chess analysis through machine learning research
