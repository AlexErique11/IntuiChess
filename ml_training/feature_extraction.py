#feature_extraction.py
import os

import chess
import chess.engine
import statistics

# ===== CONSTANTS =====
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STOCKFISH_PATH = os.path.join(SCRIPT_DIR, "..", "stockfish-windows-x86-64-avx2.exe")
DEPTH = 6
MATE_SCORE = 100000

def evaluate_all_moves(board, engine, depth):
    legal_moves = list(board.legal_moves)
    n = len(legal_moves)
    if n == 0:
        return 0, {}

    infos = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=n)

    results = {}
    for info in infos:
        move = info["pv"][0] if "pv" in info else None
        score_obj = info["score"].pov(board.turn)
        if score_obj.mate() is not None:
            score = MATE_SCORE if score_obj.mate() > 0 else -MATE_SCORE
        else:
            score = score_obj.score(mate_score=MATE_SCORE) or 0

        results[move] = score

    # Fill missing moves with score 0
    for m in legal_moves:
        results.setdefault(m, 0)

    best_eval = max(results.values()) if results else 0
    return best_eval, results



def compute_features(board, engine, depth=DEPTH):
    """
    Compute human-playability metrics for a given board state.
    Optimized to use a single engine call for all move evaluations.
    """
    legal_moves = list(board.legal_moves)
    n_moves = len(legal_moves)


    pins = sum(
        1 for sq, piece in board.piece_map().items()
        if board.is_pinned(piece.color, sq)
    )
    pins = 0 #incomplete feature
    tactical_motifs = pins

    PIECE_WEIGHTS = {
        chess.QUEEN: 1.0,
        chess.ROOK: 0.8,
        chess.BISHOP: 0.7,
        chess.KNIGHT: 0.5,
        chess.PAWN: 0.7,
        chess.KING: 0.9
    }
    king_square = board.king(board.turn)

    def compute_king_exposure(board):

        # Define king zone (3×3 squares around king)
        king_zone = [sq for sq in chess.SQUARES if chess.square_distance(king_square, sq) <= 1]

        exposure_score = 0
        for sq in king_zone:
            attackers = board.attackers(not board.turn, sq)
            for attacker_sq in attackers:
                piece_type = board.piece_type_at(attacker_sq)
                if piece_type:
                    exposure_score += PIECE_WEIGHTS.get(piece_type, 0.5)  # Default weight 0.5

        return exposure_score
    king_exposure = compute_king_exposure(board)

    has_kingside = board.has_kingside_castling_rights(board.turn)
    has_queenside = board.has_queenside_castling_rights(board.turn)

    def compute_castling_status(board):
        if board.turn == chess.WHITE:
            kingside = board.has_kingside_castling_rights(chess.WHITE)
            queenside = board.has_queenside_castling_rights(chess.WHITE)
        else:
            kingside = board.has_kingside_castling_rights(chess.BLACK)
            queenside = board.has_queenside_castling_rights(chess.BLACK)

        if kingside and queenside:
            return 3
        elif kingside:
            return 2
        elif queenside:
            return 1
        else:
            return 0
    castling_status = compute_castling_status(board)

    def compute_defending_pieces(board, king_square):
        piece_weights = {
            chess.PAWN: 0.7,
            chess.KNIGHT: 1.0,
            chess.BISHOP: 1.2,
            chess.ROOK: 1.5,
            chess.QUEEN: 2.0
        }

        defending_score = 0
        for piece_type, weight in piece_weights.items():
            for sq in board.pieces(piece_type, board.turn):
                if chess.square_distance(sq, king_square) <= 2:
                    defending_score += weight

        return defending_score

    defending_pieces = compute_defending_pieces(board, king_square)

    # Pawn structure
    doubled_pawns = sum(
        [list(chess.square_file(sq) for sq in board.pieces(chess.PAWN, not board.turn)).count(f) - 1 for f in
         set(chess.square_file(sq) for sq in board.pieces(chess.PAWN, not board.turn))]) - sum(
        [list(chess.square_file(sq) for sq in board.pieces(chess.PAWN, board.turn)).count(f) - 1 for f in
         set(chess.square_file(sq) for sq in board.pieces(chess.PAWN, board.turn))])

    # Backward pawn detection
    backward_pawns = (
            sum(
                1 for p in board.pieces(chess.PAWN, chess.WHITE)
                if any(
                    chess.square_file(o) in [chess.square_file(p) - 1, chess.square_file(p) + 1] and chess.square_rank(
                        o) > chess.square_rank(p) for o in board.pieces(chess.PAWN, chess.WHITE))
                and not any(
                    chess.square_file(o) in [chess.square_file(p) - 1, chess.square_file(p) + 1] and chess.square_rank(
                        o) < chess.square_rank(p) for o in board.pieces(chess.PAWN, chess.WHITE))
            )
            - sum(
        1 for p in board.pieces(chess.PAWN, chess.BLACK)
        if any(chess.square_file(o) in [chess.square_file(p) - 1, chess.square_file(p) + 1] and chess.square_rank(
            o) < chess.square_rank(p) for o in board.pieces(chess.PAWN, chess.BLACK))
        and not any(chess.square_file(o) in [chess.square_file(p) - 1, chess.square_file(p) + 1] and chess.square_rank(
            o) > chess.square_rank(p) for o in board.pieces(chess.PAWN, chess.BLACK))
    )
    )
    # If it's Black's turn, invert the sign
    if not board.turn:
        backward_pawns = -backward_pawns

    def pawn_islands(pawns):
        """Return list of islands, each island is a set of files occupied."""
        if not pawns:
            return []
        files = sorted(set(chess.square_file(p) for p in pawns))
        islands = []
        current = [files[0]]

        for f in files[1:]:
            if f == current[-1] + 1:
                current.append(f)
            else:
                islands.append(set(current))
                current = [f]
        islands.append(set(current))
        return islands

    def compute_pawn_majority(board):
        side = board.turn
        opponent = not side

        side_pawns = list(board.pieces(chess.PAWN, side))
        opp_pawns = list(board.pieces(chess.PAWN, opponent))

        side_islands = pawn_islands(side_pawns)
        opp_islands = pawn_islands(opp_pawns)

        score = 0.0

        # Define the weights for single-pawn advantages
        weights = {1: 1.0, 2: 0.8, 3: 0.7, 4: 0.6, 5: 0.5}

        for isl in side_islands:
            overlap = [o for o in opp_islands if not (max(o) < min(isl) or min(o) > max(isl))]

            if overlap:
                opp_count = max(len(o) for o in overlap)
            else:
                opp_count = 0

            my_count = len(isl)
            diff = my_count - opp_count

            if diff > 0:
                # Use capped key for weights (max 5)
                base = weights.get(my_count, 0.5)

                # If advantage is 2 pawns, double the value
                if diff == 2:
                    score += 2.5 * base
                elif diff >= 3:
                    score += 4 * base
                else:
                    score += base


        for isl in opp_islands:
            overlap = [o for o in side_islands if not (max(o) < min(isl) or min(o) > max(isl))]

            if overlap:
                my_count = max(len(o) for o in overlap)
            else:
                my_count = 0

            opp_count = len(isl)
            diff = my_count - opp_count

            if diff < 0:
                # Opponent has majority → subtract
                base = weights.get(opp_count, 0.5)

                if diff == -2:
                    score -= 2.5 * base
                elif diff <= -3:
                    score -= 4 * base
                else:
                    score -= base

        return score

    pawn_majority = compute_pawn_majority(board)

    def compute_mobility(board):
        safe_moves = 0
        my_color = board.turn

        piece_safe_moves = {}

        for move in board.generate_pseudo_legal_moves():
            if board.piece_at(move.from_square).color != my_color:
                continue

            if is_safe_move(board, move.from_square, move.to_square, my_color):
                safe_moves += 1
                piece = board.piece_at(move.from_square)
                piece_name = piece.symbol()
                piece_safe_moves.setdefault(piece_name, 0)
                piece_safe_moves[piece_name] += 1

        # # Debug print: safe moves per piece
        # for piece_name, count in piece_safe_moves.items():
        #     print(f"Piece {piece_name} has {count} safe moves")
        #
        # print("Total mobility score:", safe_moves)
        return safe_moves

    def is_safe_move(board, from_square, to_square, my_color):
        piece_value_map = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 100
        }

        moving_piece = board.piece_at(from_square)
        if not moving_piece:
            return False

        moving_value = piece_value_map[moving_piece.piece_type]

        # If destination square occupied by same color → not safe
        if board.piece_at(to_square) and board.piece_at(to_square).color == my_color:
            return False

        # Check threats to target square
        attackers = list(board.attackers(not my_color, to_square))
        if not attackers:
            return True  # no threat at all

        for attacker_square in attackers:
            attacker = board.piece_at(attacker_square)
            if not attacker:
                continue

            attacker_value = piece_value_map[attacker.piece_type]

            # Captured by less important piece → not safe
            if attacker_value < moving_value:
                return False

            # Captured by more important piece → safe only if defended
            if attacker_value >= moving_value:
                defenders = list(board.attackers(my_color, to_square))
                if from_square in defenders:
                    defenders.remove(from_square)  # ignore self-attack

                if not defenders:
                    return False

        return True

    mobility = compute_mobility(board)

    # Piece coordination
    pieces = board.piece_map()
    my_pieces = {sq: piece for sq, piece in pieces.items() if piece.color == board.turn}
    connectedness = sum(
        1 for sq, piece in my_pieces.items()
        if len(board.attackers(piece.color, sq)) >= 1
    )
    piece_coordination = connectedness / max(1, len(my_pieces))

    # Hanging pieces
    hanging_pieces = 0
    major_pieces = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]

    for piece_type in major_pieces:
        for square in board.pieces(piece_type, board.turn):
            if not board.is_attacked_by(board.turn, square):
                hanging_pieces += 1
    for square in board.pieces(chess.PAWN, board.turn):
        if not board.is_attacked_by(board.turn, square):
            hanging_pieces += 0.25

    rooks = list(board.pieces(chess.ROOK, board.turn))
    rooks_connected = int(len(rooks) == 2 and board.is_attacked_by(board.turn, rooks[0]) and board.is_attacked_by(board.turn, rooks[1]))
    bishop_pair = int(len(board.pieces(chess.BISHOP, board.turn)) == 2)


    def overworked_pieces(board):
        """
            Returns a list of squares where the piece protects 2 or more
            major pieces (rook or queen) that are not defended by any other piece.
            """
        MAJOR_PIECES = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
        count = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is None:
                continue

            protected_major_count = 0
            attacked_squares = board.attacks(square)

            for target_square in attacked_squares:
                target_piece = board.piece_at(target_square)
                if target_piece is None:
                    continue
                if target_piece.color != piece.color:
                    continue
                if target_piece.piece_type not in MAJOR_PIECES:
                    continue

                # Check if target_piece is defended by any other piece
                defenders = board.attackers(piece.color, target_square)
                if len(defenders) == 1 and square in defenders:
                    protected_major_count += 1
            if piece.color is board.turn and protected_major_count >= 2:
                count += 1
            if piece.color is not board.turn and protected_major_count >= 2:
                count -= 1

        return count
    overworked = overworked_pieces(board)

    # Engine-heavy features
    best_eval, evals_dict = evaluate_all_moves(board, engine, DEPTH)
    stockfish_eval = best_eval
    evals = list(evals_dict.values())
    # print("Best eval:", best_eval)
    # print("Legal moves and their evaluations:")
    #
    # for move_uci, score in evals_dict.items():
    #     print(f"Move: {move_uci}, Eval: {score}")


    if len(evals) > 1:
        blunder_score = 0.0
        losses = []

        for ev in evals:
            loss = best_eval - ev
            losses.append(loss)

            # --- Case 1: Best move is mate ---
            if abs(best_eval) >= MATE_SCORE:
                if ev < 1000:  # only count if drops below 1 pawns
                    severity = (1000 - ev) / 1000  # scaled 0.1
                    blunder_score += severity

            # --- Case 2: Big advantage (>= +3 pawns) ---
            elif best_eval >= 300:
                if loss >= 0.8 * abs(best_eval):  # lose ≥80% of advantage
                    severity = loss / abs(best_eval)
                    blunder_score += min(severity, 3)

            # --- Case 3: Big disadvantage (<= -3 pawns) ---
            elif best_eval <= -300:
                if ev <= 2 * best_eval:  # doubles opponent’s advantage
                    severity = abs(ev - best_eval) / abs(best_eval) / 2
                    blunder_score += min(severity, 3)

            # --- Case 4: Near equal (between -1 and +1) ---
            elif -100 <= best_eval <= 100:
                if ev <= -200:  # goes to -2 or worse
                    severity = (-ev) / 200
                    blunder_score += min(severity *0.7, 3)

            # --- Case 5: Other situations (mild advantage/disadvantage) ---
            else:
                if loss >= 250:  # simple cutoff: lose 2.5 pawns
                    severity = loss / 250
                    blunder_score += min(severity, 3)

        # normalize blunder severity
        blunder_ratio = blunder_score / len(evals)

        # variance of move evals, normalized by eval range
        eval_range = max(evals) - min(evals)
        variance = statistics.variance(evals) / (eval_range ** 2 + 1e-6) if eval_range > 0 else 0

        # combine into volatility
        volatility = 0.7 * blunder_ratio + 0.3 * variance
    else:
        volatility = 0.0

    def compute_trap_susceptibility(board, engine, evals_dict, lower_depth=1):
        # Step 1: Evaluate all moves at lower depth
        lower_best_eval, lower_evals_dict = evaluate_all_moves(board, engine, lower_depth)

        trap_moves = 0
        candidate_moves = 0

        # Step 2: Define threshold for "best moves" at depth 2
        threshold = 0.7  # Keep at least 70% of advantage

        for move_uci, eval_at_lower_depth in lower_evals_dict.items():
            # Check if move is a "best move"
            if lower_best_eval >= 0:
                keeps_advantage = (eval_at_lower_depth >= lower_best_eval * threshold)
            else:
                keeps_advantage = eval_at_lower_depth >= lower_best_eval * (
                            1 + 0.2) if lower_best_eval <= -300 else eval_at_lower_depth >= lower_best_eval * (1 + 0.4)

            if eval_at_lower_depth == MATE_SCORE or -MATE_SCORE:
                keeps_advantage = True
            if keeps_advantage:
                candidate_moves += 1

                # Step 3: Check deeper evaluation
                deeper_eval = evals_dict.get(chess.Move.from_uci(str(move_uci)))
                if eval_at_lower_depth == MATE_SCORE:
                    trap_moves += 1
                    continue
                if eval_at_lower_depth == -MATE_SCORE and deeper_eval < 200:
                    trap_moves += 1
                    continue
                # Trap detection: deeper eval significantly worse
                if deeper_eval < eval_at_lower_depth - 450 and deeper_eval < 500:  # drop ≥ 1.5 pawns
                    trap_moves += 1
                elif deeper_eval < eval_at_lower_depth - 600 and deeper_eval >= 500:
                    trap_moves += 1

        # Step 4: Normalize
        return trap_moves / max(1, candidate_moves)

    trap_susceptibility = compute_trap_susceptibility(board, engine, evals_dict)


    def compute_move_ease(board, legal_moves, best_eval, evals_dict):
        top_moves, decent_moves = 0, 0

        good_moves = 0
        for move in legal_moves:
            move_eval = evals_dict.get(move, None)
            # Determine if it's a top move
            if best_eval >= 0:
                keeps_advantage = (move_eval >= best_eval * 0.7)
                increases_disadvantage = False
            else:
                if best_eval <= -300:  # big disadvantage (< -3 pawns)
                    increases_disadvantage = (move_eval >= best_eval * 1.2)
                elif best_eval <= -100:  # smaller disadvantage
                    increases_disadvantage = (move_eval >= best_eval * 1.4)
                else:
                    increases_disadvantage = (move_eval >= best_eval * 1.8)
                keeps_advantage = False

            is_top_move = keeps_advantage or increases_disadvantage
            # Determine if it's a decent move
            if not is_top_move:
                if best_eval >= 0:
                    keeps_half_advantage = (move_eval >= best_eval * 0.5)
                    increases_disadvantage_decent = False
                else:
                    if best_eval <= -300:
                        increases_disadvantage_decent = (move_eval >= best_eval * 1.6)
                    elif best_eval <= -100:  # smaller disadvantage
                        increases_disadvantage_decent = (move_eval >= best_eval * 1.8)
                    else:
                        increases_disadvantage_decent = (move_eval >= best_eval * 2.2)
                    keeps_half_advantage = False

                is_decent_move = keeps_half_advantage or increases_disadvantage_decent
            else:
                is_decent_move = False

            # Score increases
            if is_top_move:
                good_moves += 1
                if board.gives_check(move) or board.is_capture(move):
                    top_moves += 1
            elif is_decent_move:
                good_moves += 1
                if board.gives_check(move) or board.is_capture(move):
                    decent_moves += 1
        move_ease_score = (top_moves + 0.4 * decent_moves) / max(1, good_moves)

        return move_ease_score

    move_ease = compute_move_ease(board, legal_moves, best_eval, evals_dict)

    # Material imbalance
    PIECE_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}

    player_color = board.turn
    material_imbalance = sum(PIECE_VALUES[p.piece_type] if p.color == board.turn else -PIECE_VALUES[p.piece_type]
                        for p in board.piece_map().values())
    # Count total number of pieces for both sides
    pieces_left = len(board.piece_map())

    # Define phase based on number of pieces left
    if pieces_left > 20:
        phase = 0  # Opening
    elif pieces_left > 10:
        phase = 1  # Middlegame
    else:
        phase = 2  # Endgame

    # Space and passed pawns
    def compute_space_control(board):
        white_control = 0
        black_control = 0

        for square in chess.SQUARES:
            white_attacks = len(board.attackers(chess.WHITE, square))
            black_attacks = len(board.attackers(chess.BLACK, square))

            # Add occupying piece to control
            piece = board.piece_at(square)
            if piece:
                if piece.color == chess.WHITE:
                    white_attacks += 1
                else:
                    black_attacks += 1

            if white_attacks > black_attacks:
                white_control += 1
            elif black_attacks > white_attacks:
                black_control += 1

        return white_control - black_control

    space_control = compute_space_control(board)
    if board.turn == chess.BLACK:
        space_control = -space_control

    def weighted_passed_pawns(board, color):
        pawns = [p for p in board.pieces(chess.PAWN, color)]
        files = {}

        # Group passed pawns by file
        for p in pawns:
            f, r = chess.square_file(p), chess.square_rank(p)
            ranks_ahead = range(r + 1, 8) if color == chess.WHITE else range(0, r)
            blocked = any(
                board.piece_at(chess.square(f + dx, r_target)) and
                board.piece_at(chess.square(f + dx, r_target)).piece_type == chess.PAWN and
                board.piece_at(chess.square(f + dx, r_target)).color != color
                for dx in [-1, 0, 1] if 0 <= f + dx < 8
                for r_target in ranks_ahead if 0 <= r_target < 8
            )
            if not blocked:
                files.setdefault(f, []).append(p)

        score = 0
        counted_files = set()
        for f in files:
            # Check for connected passed pawns on adjacent files
            if f - 1 in files or f + 1 in files:
                score += 1.25
                counted_files.add(f)
            elif f not in counted_files:
                # Only single passed pawns on this file
                score += min(len(files[f]) * 0.6, 2.5)  # If multiple pawns on same file, cap at 2.5
                counted_files.add(f)
        return score

    passed_pawns = weighted_passed_pawns(board, chess.WHITE) - weighted_passed_pawns(board, chess.BLACK)
    if not board.turn:
        passed_pawns = -passed_pawns

    center_squares = [chess.D4, chess.E4, chess.D5, chess.E5]

    center_control = 0
    for sq in center_squares:
        white_attacks = len(board.attackers(chess.WHITE, sq))
        black_attacks = len(board.attackers(chess.BLACK, sq))
        piece = board.piece_at(sq)
        if piece:
            if piece.color == chess.WHITE:
                white_attacks += 1
            else:
                black_attacks += 1
        center_control += (white_attacks - black_attacks)/max(abs(white_attacks - black_attacks), 1)
    if board.turn == chess.BLACK:
        space_control = -space_control

    return {
        "volatility": volatility,
        "move_ease": move_ease,
        "trap_susceptibility": trap_susceptibility,
        "king_exposure": king_exposure,
        # "castling_status": castling_status,
        "defending_pieces": defending_pieces,
        "doubled_pawns": doubled_pawns,
        "backward_pawns": backward_pawns,
        "pawn_majority": pawn_majority,
        "mobility": mobility,
        "piece_coordination": piece_coordination,
        "hanging_pieces": hanging_pieces,
        "rooks_connected": rooks_connected,
        "bishop_pair": bishop_pair,
        "overworked_defenders": overworked,
        # "checks": checks,
        # "captures": captures,
        "pins": pins,
        "tactical_motifs": tactical_motifs,
        "material_imbalance": material_imbalance,
        "phase": phase,
        "space_control": space_control,
        "passed_pawns": passed_pawns,
        "center_control": center_control,
        "stockfish_eval": stockfish_eval,
        "top_moves": [m.uci() for m in legal_moves],
        "evals_dict": {m.uci(): v for m, v in evals_dict.items()}
    }
