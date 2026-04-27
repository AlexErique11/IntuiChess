# chess_analyser.py
import chess
import chess.engine
import pandas as pd
import joblib
import json
import os
import math
from ml_training.feature_extraction import compute_features


# --- Paths ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "ml_training", "elo_models")
METRICS_FILE = os.path.join(MODEL_DIR, "model_metrics.json")
FEATURE_SETS_FILE = os.path.join(SCRIPT_DIR, "ml_training", "feature_sets.json")
STOCKFISH_PATH = os.path.join(SCRIPT_DIR, "stockfish-windows-x86-64-avx2.exe")

# --- Load metrics and feature sets ---
with open(METRICS_FILE, "r") as f:
    model_metrics = json.load(f)

with open(FEATURE_SETS_FILE, "r") as f:
    FEATURE_SETS = json.load(f)

# --- Elo categorization ---
def categorize_elo(avg_elo):
    if avg_elo is None:
        return "unknown"
    if avg_elo < 800:
        return "800-"
    elif avg_elo <= 1100:
        return "800-1100"
    elif avg_elo <= 1400:
        return "1100-1400"
    elif avg_elo <= 1600:
        return "1400-1600"
    elif avg_elo <= 1800:
        return "1600-1800"
    elif avg_elo <= 2000:
        return "1800-2000"
    elif avg_elo <= 2200:
        return "2000-2200"
    else:
        return "2200+"

def cp_to_eval_bar(cp, max_val=10.0):
    """Convert centipawns (white's POV) to the -10..+10 display scale via tanh."""
    return max(-max_val, min(max_val, math.tanh(cp / 400.0) * max_val))

# --- Ask user for FEN, Elo, and time control ---
fen = input("Enter a FEN: ")
avg_elo = int(input("Enter average Elo: "))
elo_range = categorize_elo(avg_elo)

time_control = input("Enter time control ('blitz' or 'rapid_classical'): ").strip().lower()
if time_control not in ["blitz", "rapid_classical"]:
    raise ValueError("Invalid time control. Must be 'blitz' or 'rapid_classical'.")

# --- Start Stockfish & extract features ---
board = chess.Board(fen)
with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:
    features = compute_features(board, engine)

# --- Stockfish eval converted to white's POV ---
raw_eval = features.get("stockfish_eval", 0) or 0
raw_eval = max(-10000, min(10000, float(raw_eval)))
side_sign = 1 if board.turn == chess.WHITE else -1
eval_white_cp = raw_eval * side_sign

# --- Run ML models ---
predicted_scores = {}
for target in ["label_position_quality", "label_move_ease"]:
    if elo_range in FEATURE_SETS and target in FEATURE_SETS[elo_range]:
        feature_cols = FEATURE_SETS[elo_range][target]
    else:
        feature_cols = FEATURE_SETS["default"][target]

    X = pd.DataFrame([{k: features.get(k, 0) for k in feature_cols}])

    model_path = os.path.join(MODEL_DIR, f"model_{elo_range}_{time_control}_{target}.pkl")
    if not os.path.exists(model_path):
        raise ValueError(f"No trained model found for Elo range {elo_range}, time {time_control}, target {target}")
    model = joblib.load(model_path)
    predicted_scores[target] = float(model.predict(X)[0])

me_score = max(0.01, min(0.9999, predicted_scores["label_move_ease"]))
pq_score = max(0.01, min(0.9999, predicted_scores["label_position_quality"]))

# --- Move Ease: predicted eval after next human move ---
# Invert label formula: diff_expected = 100 * (1/me_score - 1)
# The human is expected to lose diff_expected cp from the engine's best.
diff_expected = 100.0 * (1.0 / me_score - 1.0)
predicted_move_ease_cp = eval_white_cp - side_sign * diff_expected
move_ease_bar = cp_to_eval_bar(predicted_move_ease_cp)

# --- Position Quality: predicted eval after ~20 moves ---
predicted_pq_cp = eval_white_cp * pq_score
position_quality_bar = cp_to_eval_bar(predicted_pq_cp)

# --- Report ---
print(f"\nStockfish eval (white's POV): {eval_white_cp:+.0f} cp")

print(f"\n[Move Ease]  — predicted eval after next human move")
print(f"  Expected cp loss vs best move : {diff_expected:.1f} cp")
print(f"  Predicted eval after move     : {predicted_move_ease_cp:+.1f} cp")
print(f"  Eval bar                      : {move_ease_bar:+.2f}")

print(f"\n[Position Quality]  — predicted eval after ~20 moves")
print(f"  Predicted eval after 10 moves : {predicted_pq_cp:+.1f} cp")
print(f"  Eval bar                      : {position_quality_bar:+.2f}")

# --- Model accuracy ---
tc_metrics = model_metrics.get(elo_range, {}).get(time_control, {})
for target, label in [("label_move_ease", "Move Ease"), ("label_position_quality", "Pos. Quality")]:
    m = tc_metrics.get(target, {})
    r2   = m.get("r2")
    corr = m.get("corr")
    n_games = m.get("n_games", "?")
    r2_str   = f"{r2*100:.0f}%" if r2   is not None else "—"
    corr_str = f"{corr:.2f}"   if corr is not None else "—"
    print(f"  {label} model accuracy : R²={r2_str}  r={corr_str}  ({n_games} games)")

# --- Raw feature values ---
print("\n--- Raw Features ---")
for k, v in features.items():
    if k not in ["top_moves", "evals_dict"]:
        try:
            print(f"{k}: {float(v):.2f}")
        except (ValueError, TypeError):
            print(f"{k}: {v}")
