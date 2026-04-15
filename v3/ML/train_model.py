import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

# 1. Load the data you collected
df = pd.read_csv("training_data.csv")

# 2. We engineer one extra feature: The Difference between RX and FP
# (This is the ultimate indicator of a human body blocking the signal)
df['Power_Diff'] = abs(df['RX_Power'] - df['FP_Power'])

# 3. Define our Inputs (X) and Output (y)
X = df[['Raw_Dist', 'RX_Power', 'FP_Power', 'Power_Diff', 'Quality']]
y = df['True_Distance']

# 4. Split the data to verify it works
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train the Random Forest AI
print("Training the Model...")
model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# 6. Test its accuracy
score = model.score(X_test, y_test)
print(f"Model Accuracy (R^2 Score): {score * 100:.2f}%")

# 7. Save the model to a file so your visualizer can use it
joblib.dump(model, "uwb_spellcasting_model.pkl")
print("Model saved as uwb_spellcasting_model.pkl")