import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
import joblib

# =============================================================================
#  CHOOSE YOUR MODEL — comment out the one you don't want
# =============================================================================

MODEL_BACKEND = "random_forest"
# MODEL_BACKEND = "xgboost_optuna"

# =============================================================================
#  CONFIGURATION
# =============================================================================

CSV_FILENAME    = 'training_data.csv'
MODEL_FILENAME  = 'uwb_spellcasting_model.pkl'
NUM_ANCHORS     = 4

# XGBoost + Optuna settings (ignored when using random_forest)
OPTUNA_TRIALS   = 100       # More trials = better tuning, but slower
OPTUNA_CV_FOLDS = 5         # K-fold cross-validation folds during tuning
OPTUNA_TIMEOUT  = None      # Max seconds to spend tuning (None = no limit)

# =============================================================================

print("--- UWB Model Trainer ---")
print(f"Backend: {MODEL_BACKEND.upper().replace('_', ' ')}\n")

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_FILENAME)
print(f"Loaded {len(df)} samples.\n")

print("Samples per anchor:")
for aid in range(NUM_ANCHORS):
    count = len(df[df['Anchor_ID'] == aid])
    status = "✓" if count > 0 else "✗ NO DATA"
    print(f"  Anchor {aid}: {count:>5} samples  {status}")

if df['Anchor_ID'].nunique() < NUM_ANCHORS:
    print(f"\n⚠ Warning: Only {df['Anchor_ID'].nunique()}/{NUM_ANCHORS} anchors have data.")

# ── Feature engineering ───────────────────────────────────────────────────────
df['Power_Diff']      = abs(df['RX_Power'] - df['FP_Power'])
df['Distance_Error']  = df['True_Distance'] - df['Raw_Dist']

FEATURE_COLS = ['Anchor_ID', 'Raw_Dist', 'RX_Power', 'FP_Power', 'Power_Diff', 'Quality']
X = df[FEATURE_COLS]
y = df['Distance_Error']

print(f"\nError stats (m): mean={y.mean():.4f}  std={y.std():.4f}  "
      f"min={y.min():.4f}  max={y.max():.4f}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =============================================================================
#  RANDOM FOREST
# =============================================================================
if MODEL_BACKEND == "random_forest":
    from sklearn.ensemble import RandomForestRegressor

    print("\nTraining Random Forest...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=6,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)
    print(f"R² Score: {score * 100:.2f}%")

    importances = sorted(zip(FEATURE_COLS, model.feature_importances_), key=lambda x: -x[1])
    print("\nFeature importances:")
    for name, imp in importances:
        bar = '█' * int(imp * 40)
        print(f"  {name:<12} {imp:.3f}  {bar}")

# =============================================================================
#  XGBOOST + OPTUNA
# =============================================================================
elif MODEL_BACKEND == "xgboost_optuna":
    import xgboost as xgb
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)  # suppress per-trial spam

    print(f"\nRunning Optuna hyperparameter search ({OPTUNA_TRIALS} trials, "
          f"{OPTUNA_CV_FOLDS}-fold CV)...")

    def objective(trial):
        params = {
            # Tree structure
            "max_depth":        trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),

            # Learning rate & size
            "n_estimators":     trial.suggest_int("n_estimators", 100, 800),
            "learning_rate":    trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),

            # Regularisation — key for preventing overfit on noisy sensor data
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "gamma":            trial.suggest_float("gamma", 0.0, 5.0),

            # Fixed
            "objective":        "reg:squarederror",
            "tree_method":      "hist",   # fast on CPU; auto-uses GPU if available
            "random_state":     42,
            "n_jobs":           -1,
        }

        candidate = xgb.XGBRegressor(**params)

        # Negative RMSE across CV folds (Optuna minimises, so we negate)
        cv_scores = cross_val_score(
            candidate, X_train, y_train,
            cv=OPTUNA_CV_FOLDS,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
        )
        return -cv_scores.mean()   # lower RMSE = better = lower objective

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=OPTUNA_TRIALS, timeout=OPTUNA_TIMEOUT,
                   show_progress_bar=True)

    best_params = study.best_params
    best_rmse   = study.best_value
    print(f"\nBest CV RMSE: {best_rmse * 100:.2f} cm  "
          f"({best_rmse / 0.0254:.3f} in)")
    print(f"Best params: {best_params}")

    # ── Retrain final model on the full training set with best params ──────────
    print("\nRetraining on full training set with best params...")
    model = xgb.XGBRegressor(
        **best_params,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)
    print(f"R² Score (held-out test set): {score * 100:.2f}%")

    # Feature importances (gain-based — most meaningful for XGBoost)
    importances_raw = model.feature_importances_
    importances = sorted(zip(FEATURE_COLS, importances_raw), key=lambda x: -x[1])
    print("\nFeature importances (gain):")
    for name, imp in importances:
        bar = '█' * int(imp * 40)
        print(f"  {name:<12} {imp:.3f}  {bar}")

    # ── Optuna summary ────────────────────────────────────────────────────────
    print(f"\nOptuna study summary:")
    print(f"  Trials completed : {len(study.trials)}")
    print(f"  Best trial #     : {study.best_trial.number}")
    print(f"  Best RMSE        : {best_rmse * 100:.2f} cm")

else:
    print(f"Error: Unknown MODEL_BACKEND '{MODEL_BACKEND}'.")
    print("Set it to 'random_forest' or 'xgboost_optuna' at the top of the script.")
    exit(1)

# =============================================================================
#  SAVE
# =============================================================================
joblib.dump(model, MODEL_FILENAME)
print(f"\nModel saved as '{MODEL_FILENAME}'")
print("Run run_ai_tracker.py to deploy it.")
