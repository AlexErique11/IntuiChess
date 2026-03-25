import { ChevronLeft, ChevronRight, ChevronsLeft, ChevronsRight } from 'lucide-react';

interface MoveControlsProps {
  currentMove: number;
  totalMoves: number;
  onMoveChange: (move: number) => void;
}

export default function MoveControls({ currentMove, totalMoves, onMoveChange }: MoveControlsProps) {
  return (
    <div className="flex items-center gap-2 bg-white p-4 rounded-lg shadow-lg border-2 border-cyan-200">
      <button
        onClick={() => onMoveChange(0)}
        disabled={currentMove === 0}
        className="p-2 rounded-lg bg-cyan-100 hover:bg-cyan-200 disabled:opacity-30 disabled:cursor-not-allowed transition-colors text-cyan-700"
        title="First move"
      >
        <ChevronsLeft size={20} />
      </button>

      <button
        onClick={() => onMoveChange(Math.max(0, currentMove - 1))}
        disabled={currentMove === 0}
        className="p-2 rounded-lg bg-cyan-100 hover:bg-cyan-200 disabled:opacity-30 disabled:cursor-not-allowed transition-colors text-cyan-700"
        title="Previous move"
      >
        <ChevronLeft size={20} />
      </button>

      <div className="px-4 py-2 bg-gradient-to-r from-cyan-50 to-teal-50 rounded-lg min-w-[100px] text-center font-medium text-cyan-900">
        Move {currentMove} / {totalMoves}
      </div>

      <button
        onClick={() => onMoveChange(Math.min(totalMoves, currentMove + 1))}
        disabled={currentMove === totalMoves}
        className="p-2 rounded-lg bg-cyan-100 hover:bg-cyan-200 disabled:opacity-30 disabled:cursor-not-allowed transition-colors text-cyan-700"
        title="Next move"
      >
        <ChevronRight size={20} />
      </button>

      <button
        onClick={() => onMoveChange(totalMoves)}
        disabled={currentMove === totalMoves}
        className="p-2 rounded-lg bg-cyan-100 hover:bg-cyan-200 disabled:opacity-30 disabled:cursor-not-allowed transition-colors text-cyan-700"
        title="Last move"
      >
        <ChevronsRight size={20} />
      </button>
    </div>
  );
}
