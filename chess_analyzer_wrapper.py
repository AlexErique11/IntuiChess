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

# --- Non-linear evaluation bar mapping (copied from chess_analyser.py) ---
def score_to_eval_bar(predicted_score, max_eval=10, extreme_scale=3):
    """
    Map 0-1 normalized score to non-linear, engine-style eval bar.
    """
    x = predicted_score - 0.5  # center at 0
    sign = math.copysign(1, x)
    abs_x = abs(x)
    base_eval = max_eval * math.log1p(10 * abs_x) / math.log1p(10)
    extreme_eval = (abs_x ** 2) * extreme_scale
    eval_bar = sign * (base_eval + extreme_eval)
    return eval_bar

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
    """
    Analyze a chess position using the exact logic from chess_analyser.py
    """
    try:
        elo_range = categorize_elo(avg_elo)
        
        # --- Start Stockfish & extract features (copied from chess_analyser.py) ---
        board = chess.Board(fen)
        with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:
            features = compute_features(board, engine)
        
        # --- Targets (copied from chess_analyser.py) ---
        targets = ["label_position_quality", "label_move_ease"]
        predicted_scores = {}
        eval_bars = {}
        
        # --- Predict each target (copied from chess_analyser.py) ---
        for target in targets:
            # Select feature columns (fall back to default if not specified for this elo range)
            if elo_range in FEATURE_SETS and target in FEATURE_SETS[elo_range]:
                feature_cols = FEATURE_SETS[elo_range][target]
            else:
                feature_cols = FEATURE_SETS["default"][target]

            # Prepare input
            X = pd.DataFrame([{k: features[k] for k in feature_cols}])

            # Load model
            model_path = os.path.join(MODEL_DIR, f"model_{elo_range}_{time_control}_{target}.pkl")
            if not os.path.exists(model_path):
                # If model doesn't exist, use a fallback value
                predicted_score = 0.5  # neutral
                eval_bar = 0.0
            else:
                model = joblib.load(model_path)
                # Predict
                predicted_score = model.predict(X)[0]
                # Convert to eval bar
                eval_bar = score_to_eval_bar(predicted_score, max_eval=10, extreme_scale=3)

            # Convert to native Python types for JSON serialization
            predicted_scores[target] = convert_to_json_serializable(predicted_score)
            eval_bars[target] = convert_to_json_serializable(eval_bar)

        # --- Prepare features for display (copied from chess_analyser.py) ---
        display_features = {}
        for k, v in features.items():
            if k not in ["top_moves", "evals_dict"]:
                try:
                    # Convert to native Python types for JSON serialization
                    display_features[k] = convert_to_json_serializable(v)
                except (ValueError, TypeError):
                    display_features[k] = str(v)

        return {
            "success": True,
            "position_quality": eval_bars.get("label_position_quality", 0.0),
            "move_ease": eval_bars.get("label_move_ease", 0.0),
            "features": display_features,
            "elo_range": elo_range,
            "time_control": time_control,
            "raw_scores": {
                "position_quality": predicted_scores.get("label_position_quality", 0.5),
                "move_ease": predicted_scores.get("label_move_ease", 0.5)
            }
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"success": False, "error": "No FEN provided"}))
        sys.exit(1)
    
    fen = sys.argv[1]
    avg_elo = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
    time_control = sys.argv[3] if len(sys.argv) > 3 else "blitz"
    
    result = analyze_position(fen, avg_elo, time_control)
    print(json.dumps(result))