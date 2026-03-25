export type PieceType = 'p' | 'n' | 'b' | 'r' | 'q' | 'k' | 'P' | 'N' | 'B' | 'R' | 'Q' | 'K';
export type Square = PieceType | null;
export type Board = Square[][];

export interface ChessMove {
  from: [number, number];
  to: [number, number];
  piece: PieceType;
}

export interface Settings {
  elo: number;
  defaultBrowser: string;
}
