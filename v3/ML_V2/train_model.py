import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
import os

# --- CONFIGURATION ---
CSV_FILENAME   = 'training_data_wide.csv' # Updated to match your new format
MODEL_FILENAME = 'uwb_spellcasting_model.pkl'
NUM_ANCHORS    = 2 # Matches your current hardware setup
# ---------------------

print("--- UWB Multi-Anchor Model Trainer (Wide Format) ---\n")

if not os.path.exists(CSV_FILENAME):
    print(f"Error: {CSV_FILENAME} not found. Please run the collector first.")
    exit()

# 1. Load wide-format data
df_wide = pd.read_csv(CSV_FILENAME)
print(f"Loaded {len(df_wide)} wide synchronized rows.")

# 2. Re-format data (Melt Wide to Long)
# We turn [A0_Dist, A0_Pwr, A1_Dist, A1_Pwr] into individual rows per anchor
all_anchor_data = []

for i in range(NUM_ANCHORS):
    # Extract columns for this specific anchor
    cols = [f'A{i}_Raw_Dist', f'A{i}_RX_Power', f'A{i}_FP_Power', f'A{i}_Quality', f'A{i}_True_Distance']
    
    # Create a temporary dataframe for this anchor
    temp_df = df_wide[cols].copy()
    
    # Rename columns to standard names
    temp_df.columns = ['Raw_Dist', 'RX_Power', 'FP_Power', 'Quality', 'True_Distance']
    
    # Add the Anchor_ID so the model knows which hardware unit this was
    temp_df['Anchor_ID'] = i
    
    all_anchor_data.append(temp_df)

# Combine all anchors into one master training set
df = pd.concat(all_anchor_data, ignore_index=True)
print(f"Total training samples processed: {len(df)}")

# 3. Feature engineering
# Power_Diff is the gap between total power and first path power
df['Power_Diff'] = abs(df['RX_Power'] - df['FP_Power'])

# Target: The error the model needs to predict and subtract
df['Distance_Error'] = df['True_Distance'] - df['Raw_Dist']

# 4. Inputs (X) and Target (y)
FEATURE_COLS = ['Anchor_ID', 'Raw_Dist', 'RX_Power', 'FP_Power', 'Power_Diff', 'Quality']
X = df[FEATURE_COLS]
y = df['Distance_Error']

# Stats check
print(f"\nDistance error stats (metres):")
print(f"  Mean Error: {y.mean():.4f} m ({y.mean() / 0.0254:.2f} in)")
print(f"  Std Dev:    {y.std():.4f} m")

# 5. Train / test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Train Random Forest
print("\nTraining Random Forest model...")
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=8,          # Slightly deeper for multi-anchor complexity
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# 7. Evaluate
score = model.score(X_test, y_test)
print(f"Model R² Accuracy Score: {score * 100:.2f}%")

# Feature Importance visualization
importances = sorted(zip(FEATURE_COLS, model.feature_importances_), key=lambda x: -x[1])
print("\nFeature importances (What the AI is looking at):")
for name, imp in importances:
    bar = '█' * int(imp * 40)
    print(f"  {name:<12} {imp:.3f}  {bar}")

# 8. Save
joblib.dump(model, MODEL_FILENAME)
print(f"\nModel saved as '{MODEL_FILENAME}'")