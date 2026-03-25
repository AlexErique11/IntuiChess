import os
import joblib
import json
import pandas as pd
import numpy as np

# --- Paths ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURES_CSV = os.path.join(SCRIPT_DIR, "features.csv")
MODEL_DIR = os.path.join(SCRIPT_DIR, "elo_models")
METRICS_FILE = os.path.join(MODEL_DIR, "model_metrics.json")
FEATURE_SETS_FILE = os.path.join(SCRIPT_DIR, "feature_sets.json")

with open(FEATURE_SETS_FILE, "r") as f:
    FEATURE_SETS = json.load(f)

# --- Load features ---
if not os.path.exists(FEATURES_CSV):
    raise FileNotFoundError(f"Features CSV not found at {FEATURES_CSV}")

df_features = pd.read_csv(FEATURES_CSV)


def categorize_elo(avg_elo):
    if pd.isna(avg_elo):
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


df_features["elo_range"] = df_features["avg_elo"].apply(categorize_elo)

# --- Load metrics ---
if not os.path.exists(METRICS_FILE):
    raise FileNotFoundError(f"Metrics file not found at {METRICS_FILE}")

with open(METRICS_FILE, "r") as f:
    model_metrics = json.load(f)

# --- Count games and positions per Elo range ---
elo_range_order = [
    "800-", "800-1100", "1100-1400", "1400-1600",
    "1600-1800", "1800-2000", "2000-2200", "2200+"
]

elo_ranges = [r for r in elo_range_order if r in df_features["elo_range"].unique()]
# --- Count games and positions per Elo range and time control ---
time_controls = ["blitz", "rapid_classical"]
elo_stats = {}

for elo_range in elo_ranges:
    df_range = df_features[df_features["elo_range"] == elo_range]
    elo_stats[elo_range] = {}
    for tc in time_controls:
        df_tc = df_range[df_range["time_control"] == tc]
        n_games = df_tc["game_number"].nunique()
        n_positions = len(df_tc)
        metrics = model_metrics.get(elo_range, {}).get(tc, {}).get("label_position_quality", {})
        elo_stats[elo_range][tc] = {
            "games": n_games,
            "positions": n_positions,
            "rmse": metrics.get("rmse", None),
            "r2": metrics.get("r2", None)
        }

print("\n--- Position Quality Stats by Elo and Time Control ---")
for elo_range, tc_stats in elo_stats.items():
    # Convert time control stats to formatted strings
    tc_lines = [
        f"{tc}: {stats['games']} games, {stats['positions']} positions, RMSE: {stats['rmse']}, R²: {stats['r2']}"
        for tc, stats in tc_stats.items()
    ]

    # Print Elo range and first time control on same line
    if tc_lines:
        print(f"\nElo range: {elo_range}  {tc_lines[0]}")
        # Print remaining time controls aligned
        for line in tc_lines[1:]:
            print(f"{' ' * (len('Elo range: ') + len(elo_range) + 2)}{line}")



# --- Ask for single Elo input ---
try:
    elo_input = int(input("\nEnter a single Elo rating (e.g. 1550): ").strip())
except ValueError:
    print("Invalid Elo value.")
    exit()

# --- Find Elo range ---
elo_range = categorize_elo(elo_input)
print(f"Elo {elo_input} falls into range '{elo_range}'")

if elo_range not in elo_stats:
    print(f"No stats found for Elo range '{elo_range}'.")
    exit()

# --- Show feature importance per time control ---
for tc in time_controls:
    model_file = os.path.join(MODEL_DIR, f"model_{elo_range}_{tc}_label_position_quality.pkl")
    if not os.path.exists(model_file):
        print(f"Model file for Elo '{elo_range}', Time '{tc}' not found.")
        continue

    model = joblib.load(model_file)
    try:
        feature_cols = FEATURE_SETS[elo_range]["label_position_quality"]
    except KeyError:
        print(f"Feature set for Elo range '{elo_range}' not found. Using default.")
        feature_cols = FEATURE_SETS["default"]["label_position_quality"]

    importance_scores = model.feature_importances_

    print(f"\n--- Feature Importance for Elo {elo_input} (Range {elo_range}, Time {tc}) ---")
    importance_df = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": importance_scores
    }).sort_values(by="Importance", ascending=False)

    print(importance_df.to_string(index=False))

    # --- Plot ---
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.barh(importance_df["Feature"], importance_df["Importance"], color="skyblue")
        plt.gca().invert_yaxis()
        plt.xlabel("Importance")
        plt.title(f"Feature Importance for Elo {elo_input} (Range {elo_range}, Time {tc})")
        plt.show()
    except ImportError:
        print("Matplotlib not installed — skipping feature importance plot.")


for elo_range in elo_ranges:
    df_range = df_features[df_features["elo_range"] == elo_range]
    n_games = df_range["game_number"].nunique()
    n_positions = len(df_range)
    metrics = model_metrics.get(elo_range, {}).get("label_position_quality", {})
    elo_stats[elo_range] = {
        "games": n_games,
        "positions": n_positions,
        "rmse": metrics.get("rmse", None),
        "r2": metrics.get("r2", None)
    }

print("\n--- Position Quality Stats ---")
for elo_range, stats in elo_stats.items():
    print(f"{elo_range}: {stats['games']} games, {stats['positions']} positions, RMSE: {stats['rmse']}, R²: {stats['r2']}")

# --- Ask for single Elo input ---
try:
    elo_input = int(input("\nEnter a single Elo rating (e.g. 1550): ").strip())
except ValueError:
    print("Invalid Elo value.")
    exit()

# --- Find Elo range ---
elo_range = categorize_elo(elo_input)
print(f"Elo {elo_input} falls into range '{elo_range}'")

if elo_range not in elo_stats:
    print(f"No stats found for Elo range '{elo_range}'.")
    exit()

model_file = os.path.join(MODEL_DIR, f"model_{elo_range}_label_position_quality.pkl")
if not os.path.exists(model_file):
    print(f"Model file for Elo range '{elo_range}' not found.")
    exit()

# --- Load model ---
model = joblib.load(model_file)

# --- Feature importance ---
try:
    feature_cols = FEATURE_SETS[elo_range]["label_position_quality"]
except KeyError:
    print(f"Feature set for Elo range '{elo_range}' not found. Using default.")
    feature_cols = FEATURE_SETS["default"]["label_position_quality"]


importance_scores = model.feature_importances_

print(f"\n--- Feature Importance for Elo {elo_input} (Range {elo_range}) ---")
importance_df = pd.DataFrame({
    "Feature": feature_cols,
    "Importance": importance_scores
}).sort_values(by="Importance", ascending=False)

print(importance_df.to_string(index=False))

# print(f"\nModel metrics for Elo range '{elo_range}':")
# print(f"RMSE: {elo_stats[elo_range]['rmse']}")
# print(f"R²: {elo_stats[elo_range]['r2']}")

# --- Show a plot ---
try:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df["Feature"], importance_df["Importance"], color="skyblue")
    plt.gca().invert_yaxis()
    plt.xlabel("Importance")
    plt.title(f"Feature Importance for Elo {elo_input} (Range {elo_range})")
    plt.show()
except ImportError:
    print("Matplotlib not installed — skipping feature importance plot.")
