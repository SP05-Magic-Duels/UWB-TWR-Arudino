import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

# --- CONFIGURATION ---
CSV_FILENAME   = 'training_data.csv'
MODEL_FILENAME = 'uwb_spellcasting_model.pkl'
NUM_ANCHORS    = 4
# ---------------------

print("--- UWB 4-Anchor Model Trainer ---\n")

# 1. Load collected data
df = pd.read_csv(CSV_FILENAME)
print(f"Loaded {len(df)} total samples.")

# Show a per-anchor breakdown so you can spot missing calibration data
print("\nSamples per anchor:")
for aid in range(NUM_ANCHORS):
    count = len(df[df['Anchor_ID'] == aid])
    status = "✓" if count > 0 else "✗ NO DATA"
    print(f"  Anchor {aid}: {count:>5} samples  {status}")

if df['Anchor_ID'].nunique() < NUM_ANCHORS:
    print(f"\n⚠ Warning: Only {df['Anchor_ID'].nunique()} of {NUM_ANCHORS} anchors have training data.")
    print("  Run collect_training_data.py for each anchor before deploying.\n")

# 2. Feature engineering
#    Power_Diff is the strongest indicator of body-blocking attenuation.
#    Anchor_ID lets the model learn hardware-level biases per anchor.
df['Power_Diff'] = abs(df['RX_Power'] - df['FP_Power'])

# 3. Inputs and target
X = df[['Anchor_ID', 'Raw_Dist', 'RX_Power', 'FP_Power', 'Power_Diff', 'Quality']]
df['Distance_Error'] = df['True_Distance'] - df['Raw_Dist']
y = df['Distance_Error']

print(f"\nDistance error stats (metres):")
print(f"  Mean:  {y.mean():.4f} m  ({y.mean() / 0.0254:.2f} in)")
print(f"  Std:   {y.std():.4f} m  ({y.std() / 0.0254:.2f} in)")
print(f"  Min:   {y.min():.4f} m  ({y.min() / 0.0254:.2f} in)")
print(f"  Max:   {y.max():.4f} m  ({y.max() / 0.0254:.2f} in)")

# 4. Train / test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train Random Forest
print("\nTraining Random Forest model...")
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=6,          # Prevents overfitting to noise
    min_samples_leaf=10,  # Ensures generalization across distances
    random_state=42,
    n_jobs=-1             # Use all CPU cores for faster training
)
model.fit(X_train, y_train)

# 6. Evaluate
score = model.score(X_test, y_test)
print(f"Model R² Score: {score * 100:.2f}%")

# Show feature importances so you can understand what the model learned
feature_names = ['Anchor_ID', 'Raw_Dist', 'RX_Power', 'FP_Power', 'Power_Diff', 'Quality']
importances = sorted(zip(feature_names, model.feature_importances_), key=lambda x: -x[1])
print("\nFeature importances:")
for name, imp in importances:
    bar = '█' * int(imp * 40)
    print(f"  {name:<12} {imp:.3f}  {bar}")

# 7. Save
joblib.dump(model, MODEL_FILENAME)
print(f"\nModel saved as '{MODEL_FILENAME}'")
