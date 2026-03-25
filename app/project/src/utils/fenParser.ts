import { Board, PieceType } from '../types/chess';

export function parseFEN(fen: string): Board {
  const board: Board = Array(8).fill(null).map(() => Array(8).fill(null));
  const rows = fen.split(' ')[0].split('/');

  for (let i = 0; i < 8; i++) {
    let col = 0;
    for (const char of rows[i]) {
      if (char >= '1' && char <= '8') {
        col += parseInt(char);
      } else {
        board[i][col] = char as PieceType;
        col++;
      }
    }
  }

  return board;
}

export function getActivePlayerFromFEN(fen: string): 'white' | 'black' {
  const parts = fen.split(' ');
  return parts[1] === 'w' ? 'white' : 'black';
}

export function boardToFEN(board: Board, activePlayer: 'white' | 'black' = 'white'): string {
  const rows: string[] = [];

  for (let i = 0; i < 8; i++) {
    let row = '';
    let emptyCount = 0;

    for (let j = 0; j < 8; j++) {
      if (board[i][j] === null) {
        emptyCount++;
      } else {
        if (emptyCount > 0) {
          row += emptyCount;
          emptyCount = 0;
        }
        row += board[i][j];
      }
    }

    if (emptyCount > 0) {
      row += emptyCount;
    }

    rows.push(row);
  }

  const activePlayerChar = activePlayer === 'white' ? 'w' : 'b';
  return rows.join('/') + ` ${activePlayerChar} KQkq - 0 1`;
}

export const STARTING_FEN = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';
