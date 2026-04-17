import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
import joblib
import os

# =============================================================================
#  CHOOSE YOUR MODEL — comment out the one you don't want
# =============================================================================
MODEL_BACKEND = "random_forest"
# MODEL_BACKEND = "xgboost_optuna"

# =============================================================================
#  CONFIGURATION
# =============================================================================
CSV_FILENAME    = 'training_data_WIDER_TWIST_4A_100sam.csv'
MODEL_FILENAME  = 'uwb_random_forest_WIDER_TWIST_NEG_4A_100sam.pkl'

# XGBoost + Optuna settings (ignored when using random_forest)
OPTUNA_TRIALS   = 100       # More trials = better tuning, but slower
OPTUNA_CV_FOLDS = 5         # K-fold cross-validation folds during tuning
OPTUNA_TIMEOUT  = None      # Max seconds to spend tuning (None = no limit)
# =============================================================================

print("--- UWB Robust Model Trainer (Wide Format) ---")
print(f"Backend: {MODEL_BACKEND.upper().replace('_', ' ')}\n")

if not os.path.exists(CSV_FILENAME):
    print(f"Error: {CSV_FILENAME} not found. Please run the collector first.")
    exit()

# 1. Load wide-format data
df_wide = pd.read_csv(CSV_FILENAME)
df_wide.columns = df_wide.columns.str.strip() # Clean headers
print(f"Loaded {len(df_wide)} wide rows.")

# 2. Re-format data (Melt Wide to Long)
# Automatically detect prefixes like A0, A1, A2...
anchor_prefixes = sorted(list(set([col.split('_')[0] for col in df_wide.columns if '_Raw_Dist' in col])))
print(f"Detected Anchors in CSV: {anchor_prefixes}")

all_anchor_data = []
for prefix in anchor_prefixes:
    try:
        cols = [f'{prefix}_Raw_Dist', f'{prefix}_RX_Power', f'{prefix}_FP_Power', f'{prefix}_Quality', f'{prefix}_True_Distance']
        temp_df = df_wide[cols].copy()
        temp_df.columns = ['Raw_Dist', 'RX_Power', 'FP_Power', 'Quality', 'True_Distance']
        temp_df['Anchor_ID'] = int(prefix.replace('A', '')) # Convert "A0" to 0
        all_anchor_data.append(temp_df)
    except KeyError as e:
        print(f"Warning: Skipping {prefix} due to missing columns: {e}")

df = pd.concat(all_anchor_data, ignore_index=True)
print(f"Total samples for training: {len(df)}")

# 3. Feature engineering
df['Power_Diff'] = abs(df['RX_Power'] - df['FP_Power'])
df['Distance_Error'] = df['True_Distance'] - df['Raw_Dist']

FEATURE_COLS = ['Anchor_ID', 'Raw_Dist', 'RX_Power', 'FP_Power', 'Power_Diff', 'Quality']
X = df[FEATURE_COLS]
y = df['Distance_Error']

print(f"Error stats (m): mean={y.mean():.4f}  std={y.std():.4f}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =============================================================================
#  RANDOM FOREST
# =============================================================================
if MODEL_BACKEND == "random_forest":
    from sklearn.ensemble import RandomForestRegressor
    print("\nTraining Random Forest...")
    model = RandomForestRegressor(n_estimators=100, max_depth=8, min_samples_leaf=5, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    score = model.score(X_test, y_test)
    print(f"R² Score: {score * 100:.2f}%")

# =============================================================================
#  XGBOOST + OPTUNA
# =============================================================================
elif MODEL_BACKEND == "xgboost_optuna":
    import xgboost as xgb
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print(f"\nRunning Optuna hyperparameter search...")

    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "objective": "reg:squarederror", "tree_method": "hist", "random_state": 42, "n_jobs": -1,
        }
        cv_scores = cross_val_score(xgb.XGBRegressor(**params), X_train, y_train, cv=OPTUNA_CV_FOLDS, scoring="neg_root_mean_squared_error", n_jobs=-1)
        return -cv_scores.mean()

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=OPTUNA_TRIALS, timeout=OPTUNA_TIMEOUT)

    print(f"\nBest CV RMSE: {study.best_value * 100:.2f} cm")
    model = xgb.XGBRegressor(**study.best_params, objective="reg:squarederror", tree_method="hist", random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    print(f"R² Score (test set): {model.score(X_test, y_test) * 100:.2f}%")

# =============================================================================
#  COMMON EVALUATION & SAVE
# =============================================================================
importances = sorted(zip(FEATURE_COLS, model.feature_importances_), key=lambda x: -x[1])
print("\nFeature importances:")
for name, imp in importances:
    print(f"  {name:<12} {imp:.3f}  {'█' * int(imp * 40)}")

joblib.dump(model, MODEL_FILENAME)
print(f"\nModel saved as '{MODEL_FILENAME}'")