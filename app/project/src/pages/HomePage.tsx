import { useState, useEffect } from 'react';
import { AlertCircle, RotateCcw } from 'lucide-react';
import ChessBoard from '../components/ChessBoard';
import MoveControls from '../components/MoveControls';
import FenInput from '../components/FenInput';
import EvalBar from '../components/EvalBar';
import PositionInfo from '../components/PositionInfo';
import { parseFEN, boardToFEN, getActivePlayerFromFEN, STARTING_FEN } from '../utils/fenParser';
import { isValidMove, makeMove, isInCheck, wouldBeInCheck } from '../utils/chessLogic';
import { Board } from '../types/chess';
import { analyzePosition } from '../utils/chessAnalyserApi';
import { useSettings } from '../contexts/SettingsContext';

export default function HomePage() {
  const { settings } = useSettings();
  const [board, setBoard] = useState<Board>(parseFEN(STARTING_FEN));
  const [currentMove, setCurrentMove] = useState(0);
  const [positions, setPositions] = useState<string[]>([STARTING_FEN]);
  const [currentFen, setCurrentFen] = useState(STARTING_FEN);
  const [currentTurn, setCurrentTurn] = useState<'white' | 'black'>('white');
  const [whiteInCheck, setWhiteInCheck] = useState(false);
  const [blackInCheck, setBlackInCheck] = useState(false);
  const [positionQuality, setPositionQuality] = useState<number>(0);
  const [moveEase, setMoveEase] = useState<number>(0);
  const [analysisFeatures, setAnalysisFeatures] = useState<Record<string, number | string>>({});
  const [eloRange, setEloRange] = useState<string>('1400-1600');
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const [analysisError, setAnalysisError] = useState<string | null>(null);
  const [boardFlipped, setBoardFlipped] = useState<boolean>(false);

  useEffect(() => {
    const fen = positions[currentMove];
    const newBoard = parseFEN(fen);
    const activePlayer = getActivePlayerFromFEN(fen);
    setBoard(newBoard);
    setCurrentFen(fen);
    setCurrentTurn(activePlayer);

    setWhiteInCheck(isInCheck(newBoard, true));
    setBlackInCheck(isInCheck(newBoard, false));
  }, [currentMove, positions]);

  // Separate useEffect for analysis with debouncing to avoid too many rapid calls
  useEffect(() => {
    const analysisTimeout = setTimeout(() => {
      console.log(`🔄 Triggering re-analysis: FEN=${currentFen.slice(0, 50)}..., Elo=${settings.playerElo}, TimeControl=${settings.timeControl}`);
      performAnalysis(currentFen);
    }, 300); // 300ms debounce to prevent rapid successive calls

    return () => clearTimeout(analysisTimeout);
  }, [currentFen, settings.timeControl, settings.playerElo]);

  const handleMove = (fromRow: number, fromCol: number, toRow: number, toCol: number): boolean => {
    const piece = board[fromRow][fromCol];
    if (!piece) return false;

    const pieceIsWhite = piece === piece.toUpperCase();
    const isCorrectTurn = (currentTurn === 'white' && pieceIsWhite) || (currentTurn === 'black' && !pieceIsWhite);

    if (!isCorrectTurn) return false;

    if (!isValidMove(board, fromRow, fromCol, toRow, toCol)) {
      return false;
    }

    if (wouldBeInCheck(board, fromRow, fromCol, toRow, toCol, pieceIsWhite)) {
      return false;
    }

    const newBoard = makeMove(board, fromRow, fromCol, toRow, toCol);
    const newTurn = currentTurn === 'white' ? 'black' : 'white';
    const newFen = boardToFEN(newBoard, newTurn);

    setBoard(newBoard);
    setPositions([...positions.slice(0, currentMove + 1), newFen]);
    setCurrentMove(currentMove + 1);
    setCurrentFen(newFen);
    setCurrentTurn(newTurn);

    setWhiteInCheck(isInCheck(newBoard, true));
    setBlackInCheck(isInCheck(newBoard, false));

    return true;
  };

  const handleFenSubmit = (fen: string) => {
    try {
      const newBoard = parseFEN(fen);
      const activePlayer = getActivePlayerFromFEN(fen);
      setBoard(newBoard);
      setPositions([...positions.slice(0, currentMove + 1), fen]);
      setCurrentMove(currentMove + 1);
      setCurrentFen(fen);
      setCurrentTurn(activePlayer);
    } catch (error) {
      console.error('Invalid FEN:', error);
    }
  };

  const handleResetBoard = () => {
    setBoard(parseFEN(STARTING_FEN));
    setPositions([STARTING_FEN]);
    setCurrentMove(0);
    setCurrentFen(STARTING_FEN);
    setCurrentTurn('white');
    setWhiteInCheck(false);
    setBlackInCheck(false);
    // Reset analysis data
    setPositionQuality(0);
    setMoveEase(0);
    setAnalysisFeatures({});
  };

  const performAnalysis = async (fen: string) => {
    try {
      setIsAnalyzing(true);
      setAnalysisError(null);
      console.log(`🧠 Starting ML analysis for Elo ${settings.playerElo} (${settings.timeControl})...`);
      
      const result = await analyzePosition(fen, settings.playerElo, settings.timeControl);
      
      console.log(`✅ Analysis complete: PosQuality=${result.position_quality.toFixed(2)}, MoveEase=${result.move_ease.toFixed(2)}, EloRange=${result.elo_range}`);
      
      setPositionQuality(result.position_quality);
      setMoveEase(result.move_ease);
      setAnalysisFeatures(result.features);
      setEloRange(result.elo_range);
    } catch (error) {
      console.error('❌ Analysis failed:', error);
      setAnalysisError(error instanceof Error ? error.message : 'Analysis failed');
      // Keep previous values on error, don't reset to 0
    } finally {
      setIsAnalyzing(false);
    }
  };

  const currentPlayerInCheck = currentTurn === 'white' ? whiteInCheck : blackInCheck;

  return (
    <div className="flex-1 bg-gradient-to-br from-blue-50 via-cyan-50 to-teal-50 p-8">
      <div className="max-w-7xl mx-auto">
        <div className="flex items-center justify-between mb-8">
          <h2 className="text-3xl font-bold bg-gradient-to-r from-cyan-600 to-teal-600 bg-clip-text text-transparent">
            Interactive Chess Board
          </h2>
          <div className="flex items-center gap-4">
            <div className={`px-4 py-2 rounded-lg font-semibold ${
              currentTurn === 'white' ? 'bg-white text-slate-800 shadow-lg' : 'bg-slate-800 text-white shadow-lg'
            }`}>
              {currentTurn === 'white' ? 'White' : 'Black'} to move
            </div>
            {currentPlayerInCheck && (
              <div className="px-4 py-2 bg-red-500 text-white rounded-lg font-semibold flex items-center gap-2 shadow-lg animate-pulse">
                <AlertCircle size={20} />
                Check!
              </div>
            )}
            <PositionInfo
              features={analysisFeatures}
              eloRange={eloRange}
              timeControl={settings.timeControl}
            />
          </div>
        </div>

        <div className="flex gap-6 items-start justify-center min-h-[600px]">
          {/* Left Evaluation Bar - Position Quality */}
          <div className="flex flex-col items-center">
            <div className={`transition-opacity duration-300 ${
              isAnalyzing ? 'opacity-60' : 'opacity-100'
            }`}>
              <EvalBar
                value={boardFlipped ? -positionQuality : positionQuality}
                label="Position Quality"
              />
            </div>
            <div className="mt-2 h-5 flex items-center justify-center w-24">
              {isAnalyzing && (
                <div className="text-xs text-blue-600 animate-pulse font-medium whitespace-nowrap">
                  🔄 Analyzing...
                </div>
              )}
            </div>
          </div>

          {/* Chess Board and Controls */}
          <div className="flex flex-col items-center gap-6">
            <ChessBoard
              board={board}
              onMove={handleMove}
              currentTurn={currentTurn}
              isInCheck={currentPlayerInCheck}
              flipped={boardFlipped}
            />
            <div className="flex items-center gap-4">
              <MoveControls
                currentMove={currentMove}
                totalMoves={positions.length - 1}
                onMoveChange={setCurrentMove}
              />
              <button
                onClick={() => setBoardFlipped(!boardFlipped)}
                className="p-2 bg-white hover:bg-slate-50 border border-slate-300 rounded-lg shadow-sm transition-colors flex items-center gap-2"
                title={boardFlipped ? "View from White's perspective" : "View from Black's perspective"}
              >
                <RotateCcw size={18} className="text-slate-600" />
                <span className="text-sm font-medium text-slate-700">
                  {boardFlipped ? 'White View' : 'Black View'}
                </span>
              </button>
            </div>
          </div>

          {/* Right Evaluation Bar - Move Ease */}
          <div className="flex flex-col items-center">
            <div className={`transition-opacity duration-300 ${
              isAnalyzing ? 'opacity-60' : 'opacity-100'
            }`}>
              <EvalBar
                value={boardFlipped ? -moveEase : moveEase}
                label="Move Ease"
              />
            </div>
            <div className="mt-2 h-5 flex items-center justify-center w-24">
              {isAnalyzing && (
                <div className="text-xs text-blue-600 animate-pulse font-medium whitespace-nowrap">
                  🔄 Analyzing...
                </div>
              )}
            </div>
          </div>

          {/* Control Panel - Responsive Width */}
          <div className="w-80 lg:w-80 md:w-72 sm:w-64 ml-8 lg:ml-8 md:ml-6 sm:ml-4 space-y-4 flex-shrink-0">
            <FenInput onFenSubmit={handleFenSubmit} currentFen={currentFen} onResetBoard={handleResetBoard} />

            {/* Analysis Status */}
            <div className="bg-white p-4 rounded-lg shadow-lg">
              <h3 className="text-lg font-semibold text-slate-800 mb-3">🧠 ML Analysis Status</h3>

              <div className={`text-sm p-3 rounded transition-colors duration-300 min-h-[200px] ${
                isAnalyzing ? 'bg-blue-50 border border-blue-200' :
                analysisError ? 'bg-red-50 border border-red-200' : 'bg-green-50 border border-green-200'
              }`}>
                <div className="space-y-2 h-full flex flex-col justify-between">
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <strong>🏆 Player Elo:</strong>
                      <span>{settings.playerElo}</span>
                    </div>
                    <div className="flex justify-between">
                      <strong>⏱️ Time Control:</strong>
                      <span>{settings.timeControl === 'blitz' ? 'Blitz' : 'Rapid/Classical'}</span>
                    </div>
                    <div className="flex justify-between">
                      <strong>📁 Model Range:</strong>
                      <span>{eloRange}</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <strong>🔄 Status:</strong>
                      <span className="min-w-[80px] text-right">{
                        isAnalyzing ? (
                          <span className="text-blue-600 animate-pulse">Analyzing...</span>
                        ) : analysisError ? (
                          <span className="text-red-600">❌ Error</span>
                        ) : (
                          <span className="text-green-600">✅ Ready</span>
                        )
                      }</span>
                    </div>
                  </div>
                </div>
                <div className="mt-3 pt-2 border-t border-slate-200 text-xs text-slate-600">
                  <div className="break-words">
                    ℹ️ Auto-updates when position, Elo, or time control changes.
                    <br/>
                    🛠️ Change settings in the Settings page.
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
