#!/usr/bin/env python3
"""
Python wrapper that uses the chess_analyser.py logic and returns JSON
This script takes FEN, ELO, and time control as command line arguments
"""

import chess
import chess.engine
import pandas as pd
import joblib
import json
import os
import math
import sys
import numpy as np
from ml_training.feature_extraction import compute_features

# --- Paths (copied from chess_analyser.py) ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "ml_training", "elo_models")
METRICS_FILE = os.path.join(MODEL_DIR, "model_metrics.json")
FEATURE_SETS_FILE = os.path.join(SCRIPT_DIR, "ml_training", "feature_sets.json")
STOCKFISH_PATH = os.path.join(SCRIPT_DIR, "stockfish-windows-x86-64-avx2.exe")

# --- Load metrics and feature sets (copied from chess_analyser.py) ---
try:
    with open(METRICS_FILE, "r") as f:
        model_metrics = json.load(f)

    with open(FEATURE_SETS_FILE, "r") as f:
        FEATURE_SETS = json.load(f)
except Exception as e:
    # Fallback if files don't exist
    model_metrics = {}
    FEATURE_SETS = {"default": {"label_position_quality": [], "label_move_ease": []}}

# --- Elo categorization (copied from chess_analyser.py) ---
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

# --- Centipawn to eval bar mapping ---
def cp_to_eval_bar(cp, max_val=10.0):
    """Convert centipawns (white's POV) to the -10..+10 display scale via tanh."""
    return max(-max_val, min(max_val, math.tanh(cp / 400.0) * max_val))

def convert_to_json_serializable(obj):
    """
    Convert numpy types to native Python types for JSON serialization
    """
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, 'item'):  # numpy scalars
        return obj.item()
    else:
        return obj

def analyze_position(fen, avg_elo=1500, time_control="blitz"):
    try:
        elo_range = categorize_elo(avg_elo)

        board = chess.Board(fen)
        with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:
            features = compute_features(board, engine)

        # --- Stockfish eval, converted to white's POV ---
        # stockfish_eval is from the side-to-move's POV (positive = good for them).
        # Multiply by -1 when it's black's turn to express everything from white's perspective.
        raw_eval = features.get("stockfish_eval", 0) or 0
        raw_eval = max(-10000, min(10000, float(raw_eval)))
        side_sign = 1 if board.turn == chess.WHITE else -1
        eval_white_cp = raw_eval * side_sign

        # --- Run ML models for move ease and position quality ---
        predicted_scores = {}
        for target in ["label_position_quality", "label_move_ease"]:
            if elo_range in FEATURE_SETS and target in FEATURE_SETS[elo_range]:
                feature_cols = FEATURE_SETS[elo_range][target]
            else:
                feature_cols = FEATURE_SETS["default"][target]

            X = pd.DataFrame([{k: features.get(k, 0) for k in feature_cols}])
            model_path = os.path.join(MODEL_DIR, f"model_{elo_range}_{time_control}_{target}.pkl")
            if os.path.exists(model_path):
                model = joblib.load(model_path)
                predicted_scores[target] = float(model.predict(X)[0])
            else:
                predicted_scores[target] = 0.5

        # Clamp model outputs away from 0 to avoid division by zero
        me_score = max(0.01, min(0.9999, predicted_scores["label_move_ease"]))
        pq_score = max(0.01, min(0.9999, predicted_scores["label_position_quality"]))

        # --- Move Ease bar: predicted eval after the next human move ---
        # label_move_ease = 1 / (1 + diff/100), where diff = cp lost vs engine best.
        # Invert: diff_expected = 100 * (1/score - 1).
        # When it's white's turn the human (white) loses diff from white's advantage.
        # When it's black's turn the human (black) loses diff, which benefits white.
        diff_expected = 100.0 * (1.0 / me_score - 1.0)
        predicted_move_ease_cp = eval_white_cp - side_sign * diff_expected
        move_ease = cp_to_eval_bar(predicted_move_ease_cp)

        # --- Position Quality bar: predicted eval after ~20 moves ---
        # pq_score encodes eval stability over 20 moves (the training lookahead).
        # High score = stable; low score = volatile, eval regresses toward 0.
        predicted_pq_cp = eval_white_cp * pq_score
        position_quality = cp_to_eval_bar(predicted_pq_cp)

        # --- Prepare features for display ---
        display_features = {}
        for k, v in features.items():
            if k not in ("top_moves", "evals_dict"):
                try:
                    display_features[k] = convert_to_json_serializable(v)
                except (ValueError, TypeError):
                    display_features[k] = str(v)

        # --- Model accuracy from training metrics ---
        tc_metrics = model_metrics.get(elo_range, {}).get(time_control, {})
        model_accuracy = {
            "move_ease": {
                "r2":   tc_metrics.get("label_move_ease", {}).get("r2"),
                "corr": tc_metrics.get("label_move_ease", {}).get("corr"),
            },
            "position_quality": {
                "r2":   tc_metrics.get("label_position_quality", {}).get("r2"),
                "corr": tc_metrics.get("label_position_quality", {}).get("corr"),
            },
        }

        return {
            "success": True,
            "position_quality": position_quality,
            "move_ease": move_ease,
            "features": display_features,
            "elo_range": elo_range,
            "time_control": time_control,
            "model_accuracy": model_accuracy,
            "raw_scores": {
                "position_quality": pq_score,
                "move_ease": me_score,
                "stockfish_eval_cp": eval_white_cp,
                "predicted_move_ease_cp": predicted_move_ease_cp,
                "predicted_pq_cp": predicted_pq_cp,
            }
        }

    except Exception as e:
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"success": False, "error": "No FEN provided"}))
        sys.exit(1)
    
    fen = sys.argv[1]
    avg_elo = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
    time_control = sys.argv[3] if len(sys.argv) > 3 else "blitz"
    
    result = analyze_position(fen, avg_elo, time_control)
    print(json.dumps(result))