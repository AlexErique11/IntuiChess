const express = require('express');
const cors = require('cors');
const { spawn } = require('child_process');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 3001;

// Middleware
app.use(cors());
app.use(express.json());

// Chess analysis endpoint
app.post('/api/analyze-chess-position', async (req, res) => {
  try {
    const { fen, avgElo, timeControl } = req.body;

    if (!fen || avgElo === undefined || !timeControl) {
      return res.status(400).json({
        error: 'Missing required parameters: fen, avgElo, timeControl'
      });
    }

    console.log(`🧠 Analyzing chess position: Elo=${avgElo}, TimeControl=${timeControl}`);

    // Path to Python wrapper (adjust relative to the server location)
    const scriptPath = path.join(__dirname, '..', '..', 'chess_analyzer_wrapper.py');

    // Spawn Python process for analysis
    const pythonProcess = spawn('python', [scriptPath, fen, avgElo.toString(), timeControl], {
      cwd: path.join(__dirname, '..', '..')
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
          res.json(result);
        } catch (error) {
          console.error('❌ Failed to parse Python output:', error);
          res.status(500).json({
            error: 'Failed to parse analysis results',
            details: error.message
          });
        }
      } else {
        console.error('❌ Python analysis failed:', errorData);
        res.status(500).json({
          error: 'Python script failed',
          details: errorData || 'Unknown error'
        });
      }
    });

    pythonProcess.on('error', (error) => {
      console.error('❌ Failed to start Python process:', error);
      res.status(500).json({
        error: 'Failed to start analysis',
        details: error.message
      });
    });

  } catch (error) {
    console.error('❌ Server error:', error);
    res.status(500).json({
      error: 'Internal server error',
      details: error.message
    });
  }
});

// Health check endpoint
app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', message: 'Chess Analyzer Backend is running' });
});

// Error handling middleware
app.use((error, req, res, next) => {
  console.error('Unhandled error:', error);
  res.status(500).json({
    error: 'Internal server error',
    details: error.message
  });
});

app.listen(PORT, () => {
  console.log(`🚀 Chess Analyzer Backend running on port ${PORT}`);
  console.log(`📊 Health check available at http://localhost:${PORT}/api/health`);
});
