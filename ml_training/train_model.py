# train_model.py
import os
import chess
import chess.pgn
import pandas as pd
import zstandard as zstd
import io
import xgboost as xgb
import json
import joblib
from tqdm import tqdm
from feature_extraction import compute_features
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import numpy as np
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- Paths & configs ---
warnings.filterwarnings("ignore", category=UserWarning)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURE_SETS_PATH = os.path.join(SCRIPT_DIR, "feature_sets.json")
with open(FEATURE_SETS_PATH, "r") as f:
    FEATURE_SETS = json.load(f)

STOCKFISH_PATH = os.path.join(SCRIPT_DIR, "..", "stockfish-windows-x86-64-avx2.exe")
DATA_PATH = os.path.join(SCRIPT_DIR, "data", "lichess_data.zst")
FEATURES_CSV = os.path.join(SCRIPT_DIR, "features.csv")
MODEL_DIR = os.path.join(SCRIPT_DIR, "elo_models")
os.makedirs(MODEL_DIR, exist_ok=True)

MAX_GAMES = 5000
N_CORES = 6
DEPTH = 6
MATE_SCORE1 = 40000

# --- Functions ---
def categorize_time_control(game_headers):
    if "TimeControl" not in game_headers:
        return "unknown"
    tc = game_headers["TimeControl"]
    try:
        tc = int(tc.split("+")[0])
    except ValueError:
        return "unknown"

    if tc < 180:
        return "bullet"
    elif tc < 600:
        return "blitz"
    else:
        return "rapid_classical"

def eval_change_score(eval_list_json, move_index, lookahead=20):
    eval_list = json.loads(eval_list_json)
    current_eval = eval_list[move_index]
    future_index = move_index + lookahead
    future_eval = eval_list[-1] if future_index >= len(eval_list) else eval_list[future_index]
    eval_diff = future_eval - current_eval
    return float(1 / (1 + abs(eval_diff) / 100))

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

def process_game(game_data):
    """Analyze a single game with its own Stockfish engine."""
    game_index, game_text = game_data
    import chess
    import chess.pgn
    import chess.engine
    import io
    import json
    from feature_extraction import compute_features

    board = chess.Board()
    game_positions = []
    eval_list = []

    game = chess.pgn.read_game(io.StringIO(game_text))
    if game is None:
        return []

    with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:
        avg_elo = None
        if "WhiteElo" in game.headers and "BlackElo" in game.headers:
            try:
                white_elo = int(game.headers["WhiteElo"])
                black_elo = int(game.headers["BlackElo"])
                avg_elo = (white_elo + black_elo) / 2
            except ValueError:
                avg_elo = None

        if avg_elo is None:
            return []

        time_control = categorize_time_control(game.headers)
        if time_control == "bullet":
            return []

        for move in game.mainline_moves():
            info = engine.analyse(board, chess.engine.Limit(depth=DEPTH))
            eval_score = info["score"].pov(board.turn).score(mate_score=MATE_SCORE1)
            eval_list.append(eval_score)

            features = compute_features(board, engine)
            human_move = move.uci()
            features.update({
                "human_move": human_move,
                "game_number": game_index,
                "eval_score": eval_score,
                "avg_elo": avg_elo,
                "time_control": time_control
            })

            # --- Move ease label ---
            best_move = info["pv"][0] if "pv" in info else None
            if best_move is not None:
                # Evaluate after the human move
                board.push(move)
                info_human = engine.analyse(board, chess.engine.Limit(depth=DEPTH))
                eval_human = info_human["score"].pov(board.turn).score(mate_score=MATE_SCORE1)
                board.pop()

                # Evaluate after Stockfish's best move
                board.push(best_move)
                info_best = engine.analyse(board, chess.engine.Limit(depth=DEPTH))
                eval_best = info_best["score"].pov(board.turn).score(mate_score=MATE_SCORE1)
                board.pop()

                # Difference between the two resulting positions
                diff = abs(eval_best - eval_human)
                move_ease = 1 / (1 + diff / 100)
            else:
                move_ease = 0.5

            features["label_move_ease"] = move_ease
            game_positions.append(features)
            board.push(move)  # finally play the human move

        for pos_features in game_positions:
            pos_features["eval_list"] = json.dumps(eval_list)

    return game_positions


# -------------------- MAIN SCRIPT --------------------
if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    warnings.filterwarnings("ignore", category=UserWarning)

    # --- Load already processed features ---
    if os.path.exists(FEATURES_CSV):
        df_features = pd.read_csv(FEATURES_CSV)
        if "game_number" in df_features.columns:
            df_features["game_number"] = pd.to_numeric(df_features["game_number"], errors="coerce")
            processed_games = int(df_features["game_number"].max())
        else:
            processed_games = 0
    else:
        df_features = pd.DataFrame()
        processed_games = 0

    print(f"Already processed games: {processed_games}")

    # --- Read games into memory ---
    games_to_process = []
    with open(DATA_PATH, "rb") as f:
        dctx = zstd.ZstdDecompressor()
        stream = dctx.stream_reader(f)
        text_stream = io.TextIOWrapper(stream, encoding="utf-8")
        current_game_index = 0

        while len(games_to_process) < MAX_GAMES:
            game = chess.pgn.read_game(text_stream)
            if game is None:
                break
            current_game_index += 1
            if current_game_index <= processed_games:
                continue
            games_to_process.append((current_game_index, str(game)))

    # --- Run parallel Stockfish analysis ---
    positions = []
    with ProcessPoolExecutor(max_workers=N_CORES) as executor:
        futures = [executor.submit(process_game, g) for g in games_to_process]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Processing games"):
            positions.extend(f.result())

    # --- Append new features ---
    if positions:
        df_new = pd.DataFrame(positions)
        if not df_features.empty:
            df_features = pd.concat([df_features, df_new], ignore_index=True)
        else:
            df_features = df_new
        df_features.to_csv(FEATURES_CSV, index=False)
        print(f"Features saved to {FEATURES_CSV}, total games now: {df_features['game_number'].max()}")

    # --- Compute position quality labels ---
    df_features["move_index"] = df_features.groupby("game_number").cumcount()
    df_features["label_position_quality"] = [
        eval_change_score(eval_list, idx)
        for eval_list, idx in zip(df_features["eval_list"], df_features["move_index"])
    ]
    df_features["elo_range"] = df_features["avg_elo"].apply(categorize_elo)

    # --- Training models ---
    targets = ["label_position_quality", "label_move_ease"]
    elo_ranges = df_features["elo_range"].unique()
    time_controls = ["blitz", "rapid_classical"]

    model_metrics = {}

    for elo_range in elo_ranges:
        for tc in time_controls:
            df_range = df_features[(df_features["elo_range"] == elo_range) &
                                   (df_features["time_control"] == tc)]
            if df_range.shape[0] < 50:
                print(f"Skipping Elo {elo_range}, Time {tc} due to insufficient data")
                continue

            print(f"\nTraining models for Elo {elo_range}, Time {tc} (games: {df_range['game_number'].nunique()})")
            for target in targets:
                feature_cols = FEATURE_SETS.get(elo_range, {}).get(target, FEATURE_SETS["default"][target])
                X = df_range[feature_cols].select_dtypes(include=[np.number]).fillna(0)
                y = df_range[target]
                X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

                param_grid = {
                    "subsample": [0.7, 0.8, 0.9],
                    "n_estimators": [200, 300],
                    "min_child_weight": [1, 3],
                    "max_depth": [6, 8],
                    "learning_rate": [0.05, 0.1],
                    "gamma": [0, 1],
                    "colsample_bytree": [0.8, 1.0]
                }

                xgb_model = xgb.XGBRegressor(objective="reg:squarederror", tree_method="hist", seed=42)
                grid_search = RandomizedSearchCV(
                    estimator=xgb_model,
                    param_distributions=param_grid,
                    n_iter=20,
                    scoring="neg_mean_squared_error",
                    cv=3,
                    verbose=1,
                    n_jobs=-1,
                    random_state=42
                )
                grid_search.fit(X_train, y_train)
                best_params = grid_search.best_params_
                print(f"Best hyperparameters for {elo_range} ({target}): {best_params}")

                final_model = xgb.XGBRegressor(
                    **best_params,
                    objective="reg:squarederror",
                    eval_metric="rmse",
                    tree_method="hist",
                    seed=42
                )
                final_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

                y_pred = final_model.predict(X_val)
                rmse = mean_squared_error(y_val, y_pred) ** 0.5
                r2 = r2_score(y_val, y_pred)
                corr = np.corrcoef(y_val, y_pred)[0, 1]

                print(f"[{target}] RMSE: {rmse:.4f} | R²: {r2:.4f} | Corr: {corr:.4f}")

                model_filename = f"model_{elo_range}_{tc}_{target}.pkl"
                joblib.dump(final_model, os.path.join(MODEL_DIR, model_filename))

                model_metrics.setdefault(elo_range, {}).setdefault(tc, {})[target] = {
                    "rmse": rmse,
                    "r2": r2,
                    "corr": corr,
                    "n_positions": df_range.shape[0],
                    "n_games": df_range["game_number"].nunique()
                }

    # --- Save metrics ---
    with open(os.path.join(MODEL_DIR, "model_metrics.json"), "w") as f:
        json.dump(model_metrics, f, indent=2)

    print("\n--- Training complete ---")
    print("Model metrics:", model_metrics)
