import { Board, PieceType } from '../types/chess';

export function isValidMove(
  board: Board,
  fromRow: number,
  fromCol: number,
  toRow: number,
  toCol: number
): boolean {
  const piece = board[fromRow][fromCol];
  if (!piece) return false;

  const targetPiece = board[toRow][toCol];
  const isWhite = piece === piece.toUpperCase();
  const targetIsWhite = targetPiece ? targetPiece === targetPiece.toUpperCase() : null;

  if (targetIsWhite !== null && isWhite === targetIsWhite) {
    return false;
  }

  const rowDiff = toRow - fromRow;
  const colDiff = toCol - fromCol;
  const absRowDiff = Math.abs(rowDiff);
  const absColDiff = Math.abs(colDiff);

  const pieceType = piece.toLowerCase();

  switch (pieceType) {
    case 'p':
      return isValidPawnMove(board, fromRow, fromCol, toRow, toCol, isWhite, rowDiff, colDiff, absRowDiff, absColDiff);
    case 'n':
      return (absRowDiff === 2 && absColDiff === 1) || (absRowDiff === 1 && absColDiff === 2);
    case 'b':
      return absRowDiff === absColDiff && isPathClear(board, fromRow, fromCol, toRow, toCol);
    case 'r':
      return (rowDiff === 0 || colDiff === 0) && isPathClear(board, fromRow, fromCol, toRow, toCol);
    case 'q':
      return (absRowDiff === absColDiff || rowDiff === 0 || colDiff === 0) && isPathClear(board, fromRow, fromCol, toRow, toCol);
    case 'k':
      return absRowDiff <= 1 && absColDiff <= 1;
    default:
      return false;
  }
}

function isValidPawnMove(
  board: Board,
  fromRow: number,
  fromCol: number,
  toRow: number,
  toCol: number,
  isWhite: boolean,
  rowDiff: number,
  colDiff: number,
  absRowDiff: number,
  absColDiff: number
): boolean {
  const direction = isWhite ? -1 : 1;
  const startRow = isWhite ? 6 : 1;
  const targetPiece = board[toRow][toCol];

  if (colDiff === 0 && !targetPiece) {
    if (rowDiff === direction) return true;
    if (fromRow === startRow && rowDiff === 2 * direction && !board[fromRow + direction][fromCol]) {
      return true;
    }
  }

  if (absColDiff === 1 && rowDiff === direction && targetPiece) {
    return true;
  }

  return false;
}

function isPathClear(
  board: Board,
  fromRow: number,
  fromCol: number,
  toRow: number,
  toCol: number
): boolean {
  const rowStep = toRow > fromRow ? 1 : toRow < fromRow ? -1 : 0;
  const colStep = toCol > fromCol ? 1 : toCol < fromCol ? -1 : 0;

  let currentRow = fromRow + rowStep;
  let currentCol = fromCol + colStep;

  while (currentRow !== toRow || currentCol !== toCol) {
    if (board[currentRow][currentCol] !== null) {
      return false;
    }
    currentRow += rowStep;
    currentCol += colStep;
  }

  return true;
}

export function makeMove(
  board: Board,
  fromRow: number,
  fromCol: number,
  toRow: number,
  toCol: number
): Board {
  const newBoard = board.map(row => [...row]);
  newBoard[toRow][toCol] = newBoard[fromRow][fromCol];
  newBoard[fromRow][fromCol] = null;
  return newBoard;
}

export function findKing(board: Board, isWhite: boolean): [number, number] | null {
  const kingPiece = isWhite ? 'K' : 'k';
  for (let row = 0; row < 8; row++) {
    for (let col = 0; col < 8; col++) {
      if (board[row][col] === kingPiece) {
        return [row, col];
      }
    }
  }
  return null;
}

export function isSquareUnderAttack(
  board: Board,
  targetRow: number,
  targetCol: number,
  byWhite: boolean
): boolean {
  for (let row = 0; row < 8; row++) {
    for (let col = 0; col < 8; col++) {
      const piece = board[row][col];
      if (!piece) continue;

      const pieceIsWhite = piece === piece.toUpperCase();
      if (pieceIsWhite !== byWhite) continue;

      if (isValidMove(board, row, col, targetRow, targetCol)) {
        return true;
      }
    }
  }
  return false;
}

export function isInCheck(board: Board, isWhite: boolean): boolean {
  const kingPos = findKing(board, isWhite);
  if (!kingPos) return false;

  return isSquareUnderAttack(board, kingPos[0], kingPos[1], !isWhite);
}

export function wouldBeInCheck(
  board: Board,
  fromRow: number,
  fromCol: number,
  toRow: number,
  toCol: number,
  isWhite: boolean
): boolean {
  const testBoard = makeMove(board, fromRow, fromCol, toRow, toCol);
  return isInCheck(testBoard, isWhite);
}
